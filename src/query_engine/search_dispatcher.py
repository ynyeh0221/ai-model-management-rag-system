# src/query_engine/search_dispatcher.py

import asyncio
import logging
import time
from enum import Enum
from typing import Dict, List, Any, Optional, Union


class QueryIntent(Enum):
    RETRIEVAL = "retrieval"
    COMPARISON = "comparison"
    NOTEBOOK = "notebook"
    IMAGE_SEARCH = "image_search"
    METADATA = "metadata"
    UNKNOWN = "unknown"

class SearchDispatcher:
    """
    Dispatcher that routes queries to appropriate search handlers
    based on the classified intent.
    """
    
    def __init__(self, chroma_manager, text_embedder, image_embedder, 
                 access_control_manager=None, analytics=None):
        """
        Initialize the SearchDispatcher with required dependencies.
        
        Args:
            chroma_manager: Manager for Chroma vector database interactions
            text_embedder: Component for generating text embeddings
            image_embedder: Component for generating image embeddings
            access_control_manager: Optional manager for access control
            analytics: Optional analytics collector
        """
        self.chroma_manager = chroma_manager
        self.text_embedder = text_embedder
        self.image_embedder = image_embedder
        self.access_control_manager = access_control_manager
        self.analytics = analytics
        self.logger = logging.getLogger(__name__)
        
        # Define handlers mapping for dispatching
        self.handlers = {
            QueryIntent.RETRIEVAL: self.handle_text_search,
            QueryIntent.COMPARISON: self.handle_comparison,
            QueryIntent.NOTEBOOK: self.handle_notebook_request,
            QueryIntent.IMAGE_SEARCH: self.handle_image_search,
            QueryIntent.METADATA: self.handle_metadata_search,
            QueryIntent.UNKNOWN: self.handle_fallback_search
        }
    
    async def dispatch(self, query: str, intent: Union[str, QueryIntent], 
                      parameters: Dict[str, Any], user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Dispatch a query to the appropriate search handler based on intent.
        
        Args:
            query: The processed query text
            intent: The classified intent (string or enum)
            parameters: Dictionary of extracted parameters
            user_id: Optional user identifier for access control
            
        Returns:
            Dictionary containing search results and metadata
        """
        start_time = time.time()
        self.logger.info(f"Dispatching query with intent: {intent}")
        
        # Convert string intent to enum if needed
        if isinstance(intent, str):
            try:
                intent = QueryIntent(intent)
            except ValueError:
                self.logger.warning(f"Unknown intent: {intent}, falling back to RETRIEVAL")
                intent = QueryIntent.RETRIEVAL
        
        # Apply access control if available
        if self.access_control_manager and user_id:
            parameters = self.access_control_manager.apply_access_filters(parameters, user_id)
        
        # Get the appropriate handler
        handler = self.handlers.get(intent, self.handle_fallback_search)
        
        try:
            # Call the handler
            results = await handler(query, parameters)
            
            # Calculate execution time
            execution_time = (time.time() - start_time) * 1000  # Convert to ms
            
            # Log analytics if available
            if self.analytics:
                self.analytics.log_performance_metrics(
                    query_id=parameters.get('query_id', 'unknown'),
                    total_time_ms=int(execution_time),
                    search_time_ms=int(execution_time)  # More detailed metrics would be set in handlers
                )
            
            # Add metadata to results
            results['metadata'] = {
                'intent': intent.value if isinstance(intent, QueryIntent) else intent,
                'execution_time_ms': execution_time,
                'result_count': len(results.get('items', [])),
                'parameters': self._sanitize_parameters(parameters)
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in search dispatch: {e}", exc_info=True)
            
            # Log failed search if analytics available
            if self.analytics:
                self.analytics.update_query_status(
                    query_id=parameters.get('query_id', 'unknown'),
                    status='failed'
                )
            
            # Return error information
            return {
                'success': False,
                'error': str(e),
                'metadata': {
                    'intent': intent.value if isinstance(intent, QueryIntent) else intent,
                    'execution_time_ms': (time.time() - start_time) * 1000
                }
            }

    async def handle_text_search(self, query: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a text search query for model scripts and metadata.
        Returns a single result per unique model_id.
        """
        self.logger.debug(f"Handling text search: {query}")
        start_time = time.time()

        try:
            # Generate embedding for the query
            embedding_start = time.time()
            embedding_time = (time.time() - embedding_start) * 1000

            # Extract search parameters
            try:
                requested_limit = int(parameters.get('limit', 100))
                if requested_limit <= 0:
                    requested_limit = 100
            except (TypeError, ValueError):
                requested_limit = 10  # Default to 10 for any conversion errors

            self.logger.debug(f"Using requested_limit: {requested_limit}")

            # Use a higher limit for the initial search to account for multiple chunks per model
            search_limit = requested_limit * 100
            filters = parameters.get('filters', {})

            # Prepare Chroma query for chunks
            search_params = {
                'query': query,
                'where': self._translate_filters_to_chroma(filters),
                'limit': search_limit,
                'include': ["metadatas", "documents", "distances"]
            }

            # Execute vector search on code chunks
            search_start = time.time()
            chunk_results = await self.chroma_manager.search(
                collection_name="model_scripts_chunks",
                **search_params
            )
            search_time = (time.time() - search_start) * 1000

            # Group results by model_id
            model_groups = {}

            for idx, result in enumerate(chunk_results.get('results', [])):
                # Extract model_id from metadata
                model_id = result.get('metadata', {}).get('model_id', 'unknown')
                metadata_doc_id = result.get('metadata', {}).get('metadata_doc_id', None)

                # Initialize group if this is the first chunk from this model
                if model_id not in model_groups:
                    model_groups[model_id] = {
                        'model_id': model_id,
                        'metadata_doc_id': metadata_doc_id,
                        'chunks': [],
                        'best_score': 0,
                        'best_chunk_idx': -1,
                        'metadata': None  # Will be populated later
                    }

                # Add this chunk to the model's group
                chunk_idx = len(model_groups[model_id]['chunks'])
                model_groups[model_id]['chunks'].append({
                    'id': result.get('id'),
                    'score': result.get('score', 0),
                    'metadata': result.get('metadata', {}),
                    'content': result.get('document', ""),
                    'chunk_idx': chunk_idx
                })

                # Update best score if this chunk has a higher score
                if result.get('score', 0) > model_groups[model_id]['best_score']:
                    model_groups[model_id]['best_score'] = result.get('score', 0)
                    model_groups[model_id]['best_chunk_idx'] = chunk_idx

            # Fetch metadata for all models found - directly query the metadata collection
            model_ids = list(model_groups.keys())
            if model_ids:
                # First, try to fetch by metadata_doc_ids if available
                metadata_doc_ids = [group['metadata_doc_id'] for group in model_groups.values()
                                    if group['metadata_doc_id']]

                if metadata_doc_ids:
                    metadata_results = await self.chroma_manager.get(
                        collection_name="model_scripts_metadata",
                        ids=metadata_doc_ids,
                        include=["metadatas"]
                    )

                    # Map metadata to models by metadata_doc_id
                    for result in metadata_results.get('results', []):
                        result_id = result.get('id')
                        # Find which model group this metadata belongs to
                        for model_id, group in model_groups.items():
                            if group['metadata_doc_id'] == result_id:
                                group['metadata'] = result.get('metadata', {})

                # For any model without metadata, try to fetch by model_id
                models_without_metadata = [model_id for model_id, group in model_groups.items()
                                           if group['metadata'] is None]

                if models_without_metadata:
                    # Create a where clause to fetch metadata by model_id
                    where_clause = {"model_id": {"$in": models_without_metadata}}

                    additional_metadata = await self.chroma_manager.get(
                        collection_name="model_scripts_metadata",
                        where=where_clause,
                        include=["metadatas"]
                    )

                    # Map these results by model_id
                    for result in additional_metadata.get('results', []):
                        result_model_id = result.get('metadata', {}).get('model_id')
                        if result_model_id in model_groups:
                            model_groups[result_model_id]['metadata'] = result.get('metadata', {})

            # Convert groups to a list and sort by best score
            model_list = list(model_groups.values())
            model_list.sort(key=lambda x: x['best_score'], reverse=True)

            # Limit to requested number of models
            model_list = model_list[:requested_limit]

            # Prepare unique model items list - one entry per model
            items = []
            for rank, model in enumerate(model_list):
                # Create a single result item per model
                metadata = model['metadata'] or {}
                best_chunk = None

                # Get the best chunk
                if model['chunks'] and 0 <= model['best_chunk_idx'] < len(model['chunks']):
                    best_chunk = model['chunks'][model['best_chunk_idx']]

                # Create a consolidated item
                item = {
                    'id': model['metadata_doc_id'] or (
                        best_chunk['id'] if best_chunk else f"model_{model['model_id']}"),
                    'model_id': model['model_id'],
                    'score': model['best_score'],
                    'metadata': metadata,
                    'rank': rank + 1,
                    'chunk_count': len(model['chunks'])
                }
                items.append(item)

            # Log performance metrics if analytics available
            if self.analytics and 'query_id' in parameters:
                self.analytics.log_performance_metrics(
                    query_id=parameters['query_id'],
                    embedding_time_ms=int(embedding_time),
                    search_time_ms=int(search_time),
                    total_time_ms=int((time.time() - start_time) * 1000)
                )

            return {
                'success': True,
                'type': 'text_search',
                'items': items,
                'total_found': len(items),
                'total_models': len(model_groups),
                'performance': {
                    'embedding_time_ms': embedding_time,
                    'search_time_ms': search_time,
                    'total_time_ms': (time.time() - start_time) * 1000
                }
            }

        except Exception as e:
            self.logger.error(f"Error in text search: {e}", exc_info=True)
            raise

    async def handle_image_search(self, query: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle an image search query.

        Args:
            query: The processed query text
            parameters: Dictionary of extracted parameters

        Returns:
            Dictionary containing search results for images
        """
        self.logger.debug(f"Handling image search: {query}")
        start_time = time.time()

        try:
            # Generate embedding for the query
            embedding_start = time.time()

            # Check if we're doing a text-to-image search or image-to-image search
            if 'image_data' in parameters:
                # Image-to-image search
                query_embedding = await self.image_embedder.generate_embedding(
                    image_data=parameters['image_data']
                )
            else:
                # Text-to-image search (using CLIP's multimodal capabilities)
                query_embedding = await self.image_embedder.generate_text_embedding(query)

            embedding_time = (time.time() - embedding_start) * 1000

            # Extract search parameters
            limit = parameters.get('limit', 100)
            style_tags = parameters.get('style_tags', [])
            prompt_terms = parameters.get('prompt_terms', "")
            resolution = parameters.get('resolution', None)

            # Build filters
            filters = {}

            if style_tags:
                filters['style_tags'] = {'$in': style_tags}

            if prompt_terms:
                filters['prompt'] = {'$contains': prompt_terms}

            if resolution:
                filters['resolution.width'] = {'$eq': resolution.get('width')}
                filters['resolution.height'] = {'$eq': resolution.get('height')}

            # Add any model-specific filters
            if 'model_ids' in parameters and parameters['model_ids']:
                filters['source_model_id'] = {'$in': parameters['model_ids']}

            # Prepare Chroma query - FIX: Use query parameter instead of embedding
            search_params = {
                'query': {'embedding': query_embedding},  # Changed from 'embedding' to 'query' with a nested dict
                'where': filters if filters else None,
                'limit': limit,
                'include': ["metadatas", "distances"]
            }

            # Execute vector search
            search_start = time.time()
            image_results = await self.chroma_manager.search(
                collection_name="generated_images",
                **search_params
            )
            search_time = (time.time() - search_start) * 1000

            # Process results
            items = []
            for idx, result in enumerate(image_results.get('results', [])):
                metadata = result.get('metadata', {})

                # Add the image URL/path
                image_path = metadata.get('image_path', "")
                thumbnail_path = metadata.get('thumbnail_path', "")

                items.append({
                    'id': result.get('id'),
                    'score': result.get('score', 0.0),
                    'metadata': metadata,
                    'image_path': image_path,
                    'thumbnail_path': thumbnail_path,
                    'rank': idx + 1
                })

            # Log performance metrics if analytics available
            if self.analytics and 'query_id' in parameters:
                self.analytics.log_performance_metrics(
                    query_id=parameters['query_id'],
                    embedding_time_ms=int(embedding_time),
                    search_time_ms=int(search_time),
                    total_time_ms=int((time.time() - start_time) * 1000)
                )

            return {
                'success': True,
                'type': 'image_search',
                'items': items,
                'total_found': len(items),
                'performance': {
                    'embedding_time_ms': embedding_time,
                    'search_time_ms': search_time,
                    'total_time_ms': (time.time() - start_time) * 1000
                }
            }

        except Exception as e:
            self.logger.error(f"Error in image search: {e}", exc_info=True)
            raise
    
    async def handle_comparison(self, query: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a comparison query for multiple models.
        
        Args:
            query: The processed query text
            parameters: Dictionary of extracted parameters
            
        Returns:
            Dictionary containing comparison results
        """
        self.logger.debug(f"Handling comparison: {parameters}")
        start_time = time.time()
        
        try:
            # Get model IDs to compare
            model_ids = parameters.get('model_ids', [])
            if not model_ids or len(model_ids) < 2:
                raise ValueError("Comparison requires at least two model IDs")
            
            # Get comparison dimensions
            dimensions = parameters.get('comparison_dimensions', ['architecture', 'performance'])
            
            # Fetch model data in parallel
            tasks = []
            for model_id in model_ids:
                tasks.append(self._fetch_model_data(model_id, dimensions))
                
            model_data_list = await asyncio.gather(*tasks)
            
            # Process comparison data
            comparison_results = {
                'models': {},
                'dimensions': {},
                'summary': {}
            }
            
            # Organize by model
            for model_data in model_data_list:
                model_id = model_data.get('model_id', 'unknown')
                comparison_results['models'][model_id] = model_data
            
            # Organize by dimension
            for dimension in dimensions:
                comparison_results['dimensions'][dimension] = {}
                for model_id, model_data in comparison_results['models'].items():
                    comparison_results['dimensions'][dimension][model_id] = model_data.get(dimension, {})
            
            # Generate performance comparisons
            if 'performance' in dimensions and len(model_ids) >= 2:
                perf_comparisons = self._generate_performance_comparisons(
                    [comparison_results['models'][mid] for mid in model_ids]
                )
                comparison_results['summary']['performance'] = perf_comparisons
            
            # Generate architecture comparisons if applicable
            if 'architecture' in dimensions and len(model_ids) >= 2:
                arch_comparisons = self._generate_architecture_comparisons(
                    [comparison_results['models'][mid] for mid in model_ids]
                )
                comparison_results['summary']['architecture'] = arch_comparisons
            
            return {
                'success': True,
                'type': 'comparison',
                'models': model_ids,
                'dimensions': dimensions,
                'results': comparison_results,
                'performance': {
                    'total_time_ms': (time.time() - start_time) * 1000
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in comparison: {e}", exc_info=True)
            raise
    
    async def handle_notebook_request(self, query: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a notebook generation request.
        
        Args:
            query: The processed query text
            parameters: Dictionary of extracted parameters
            
        Returns:
            Dictionary containing notebook generation results
        """
        self.logger.debug(f"Handling notebook request: {parameters}")
        start_time = time.time()
        
        try:
            # Get model IDs for notebook
            model_ids = parameters.get('model_ids', [])
            if not model_ids:
                raise ValueError("Notebook generation requires at least one model ID")
            
            # Get analysis types
            analysis_types = parameters.get('analysis_types', ['basic'])
            
            # Get dataset information if provided
            dataset = parameters.get('dataset', None)
            
            # Get resource constraints if specified
            resources = parameters.get('resources', 'standard')
            
            # Placeholder for notebook generation logic
            # In a real implementation, this would call the Colab Notebook Generator
            notebook_request = {
                'model_ids': model_ids,
                'analysis_types': analysis_types,
                'dataset': dataset,
                'resources': resources,
                'user_id': parameters.get('user_id', None)
            }
            
            # Simulate notebook generation result
            notebook_result = {
                'notebook_id': f"nb_{model_ids[0]}_{int(time.time())}",
                'title': f"Analysis of {', '.join(model_ids)}",
                'status': 'pending',
                'estimated_completion_time': int(time.time() + 300)  # 5 minutes from now
            }
            
            return {
                'success': True,
                'type': 'notebook_request',
                'request': notebook_request,
                'result': notebook_result,
                'performance': {
                    'total_time_ms': (time.time() - start_time) * 1000
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in notebook request: {e}", exc_info=True)
            raise

    async def handle_metadata_search(self, query: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a metadata-specific search query.

        Args:
            query: The processed query text
            parameters: Dictionary of extracted parameters

        Returns:
            Dictionary containing metadata search results
        """
        self.logger.debug(f"Handling metadata search: {query}")
        start_time = time.time()

        try:
            # Extract search parameters
            filters = parameters.get('filters', {})
            limit = parameters.get('limit', 100)

            # If query mentions a specific month, add it to filters
            if "april" in query.lower():
                if not filters:
                    filters = {}
                filters["created_month"] = {"$eq": "April"}
            elif "march" in query.lower():
                if not filters:
                    filters = {}
                filters["created_month"] = {"$eq": "March"}
            # Add more month checks as needed

            # Convert filters to Chroma format
            where_conditions = []

            for key, value in filters.items():
                if isinstance(value, dict) and any(op.startswith('$') for op in value.keys()):
                    # Already has operators
                    where_conditions.append({key: value})
                elif isinstance(value, list):
                    # List values become $in operators
                    where_conditions.append({key: {"$in": value}})
                else:
                    # Simple values become $eq operators
                    where_conditions.append({key: {"$eq": value}})

            # Create a proper $and query if we have multiple conditions
            chroma_filters = {}
            if len(where_conditions) > 1:
                chroma_filters = {"$and": where_conditions}
            elif len(where_conditions) == 1:
                chroma_filters = where_conditions[0]

            # Execute metadata search
            search_start = time.time()
            print(f"chroma_filters: {chroma_filters}")
            metadata_results = await self.chroma_manager.get(
                collection_name="model_scripts_metadata",
                limit=limit,
                where=chroma_filters,
                include=["metadatas"]
            )
            search_time = (time.time() - search_start) * 1000

            # Process results
            items = []
            for idx, result in enumerate(metadata_results.get('results', [])):
                items.append({
                    'id': result.get('id'),
                    'score': 100.0,
                    'metadata': result.get('metadata', {}),
                    'rank': idx + 1,
                    'model_id': result.get('metadata', {}).get('model_id', 'unknown')
                })

            return {
                'success': True,
                'type': 'metadata_search',
                'items': items,
                'total_found': len(items),
                'performance': {
                    'search_time_ms': search_time,
                    'total_time_ms': (time.time() - start_time) * 1000
                }
            }

        except Exception as e:
            self.logger.error(f"Error in metadata search: {e}", exc_info=True)
            raise

    async def handle_fallback_search(self, query: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle queries with unknown intent with a best-effort approach.

        Args:
            query: The processed query text
            parameters: Dictionary of extracted parameters

        Returns:
            Dictionary containing search results
        """
        self.logger.warning(f"Using fallback search for query: {query}")

        # Try a combination of text search and metadata search
        try:
            # Run text search
            text_results = await self.handle_text_search(query, parameters)

            # If text search yielded results, return them
            if text_results.get('success', False) and text_results.get('total_found', 0) > 0:
                return text_results

            # If no results from text search, try metadata search
            try:
                metadata_results = await self.handle_metadata_search(query, parameters)

                if metadata_results.get('success', False) and metadata_results.get('total_found', 0) > 0:
                    return metadata_results
            except Exception as e:
                self.logger.error(f"Error in metadata search during fallback: {e}", exc_info=True)

            # If still no results, return empty results with proper structure
            return {
                'success': True,
                'type': 'fallback_search',
                'items': [],
                'total_found': 0,
                'message': "No results found using various search strategies"
            }

        except Exception as e:
            self.logger.error(f"Error in fallback search: {e}", exc_info=True)

            # Return a properly structured error response
            return {
                'success': False,
                'error': "An error occurred during the search",
                'type': 'fallback_search',
                'items': [],
                'total_found': 0
            }
    
    async def _fetch_model_data(self, model_id: str, dimensions: List[str]) -> Dict[str, Any]:
        """
        Fetch data for a specific model.
        
        Args:
            model_id: The model identifier
            dimensions: List of data dimensions to fetch
            
        Returns:
            Dictionary containing model data
        """
        # Prepare filters to get model data
        filters = {'model_id': {'$eq': model_id}}
        
        # Fetch model data from Chroma
        model_data = await self.chroma_manager.get(
            collection_name="model_scripts",
            where=filters,
            include=["metadata"]
        )
        
        # Process the results
        if not model_data.get('results'):
            return {'model_id': model_id, 'found': False}
        
        result = {'model_id': model_id, 'found': True}
        
        # Extract metadata from the first document (should be the main model document)
        metadata = model_data['results'][0].get('metadata', {})
        
        # Extract dimensions
        for dimension in dimensions:
            if dimension == 'architecture' and 'architecture_type' in metadata:
                result['architecture'] = {
                    'type': metadata.get('architecture_type', {}).get('value', 'unknown'),
                    'hidden_size': metadata.get('model_dimensions', {}).get('hidden_size', {}).get('value'),
                    'num_layers': metadata.get('model_dimensions', {}).get('num_layers', {}).get('value'),
                    'num_attention_heads': metadata.get('model_dimensions', {}).get('num_attention_heads', {}).get('value'),
                    'total_parameters': metadata.get('model_dimensions', {}).get('total_parameters', {}).get('value')
                }
            
            elif dimension == 'performance' and 'performance' in metadata:
                result['performance'] = {
                    'accuracy': metadata.get('performance', {}).get('accuracy', {}).get('value'),
                    'loss': metadata.get('performance', {}).get('loss', {}).get('value'),
                    'perplexity': metadata.get('performance', {}).get('perplexity', {}).get('value'),
                    'eval_dataset': metadata.get('performance', {}).get('eval_dataset', {}).get('value')
                }
            
            elif dimension == 'training' and 'training_config' in metadata:
                result['training'] = {
                    'batch_size': metadata.get('training_config', {}).get('batch_size', {}).get('value'),
                    'learning_rate': metadata.get('training_config', {}).get('learning_rate', {}).get('value'),
                    'optimizer': metadata.get('training_config', {}).get('optimizer', {}).get('value'),
                    'epochs': metadata.get('training_config', {}).get('epochs', {}).get('value'),
                    'training_time_hours': metadata.get('training_config', {}).get('training_time_hours', {}).get('value'),
                    'hardware_used': metadata.get('training_config', {}).get('hardware_used', {}).get('value')
                }
            
            elif dimension == 'dataset' and 'dataset' in metadata:
                result['dataset'] = {
                    'name': metadata.get('dataset', {}).get('name', {}).get('value'),
                    'version': metadata.get('dataset', {}).get('version', {}).get('value'),
                    'num_samples': metadata.get('dataset', {}).get('num_samples', {}).get('value')
                }
            
            elif dimension == 'framework' and 'framework' in metadata:
                result['framework'] = {
                    'name': metadata.get('framework', {}).get('name'),
                    'version': metadata.get('framework', {}).get('version')
                }
        
        # Add basic metadata
        result['basic'] = {
            'version': metadata.get('version'),
            'creation_date': metadata.get('creation_date'),
            'last_modified_date': metadata.get('last_modified_date'),
            'predecessor_models': metadata.get('predecessor_models', [])
        }
        
        return result
    
    def _generate_performance_comparisons(self, model_data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate performance comparisons between models.
        
        Args:
            model_data_list: List of model data dictionaries
            
        Returns:
            Dictionary containing performance comparisons
        """
        comparisons = {
            'accuracy': {},
            'loss': {},
            'perplexity': {},
            'relative_improvement': {}
        }
        
        # Extract models with performance data
        models_with_perf = []
        for model_data in model_data_list:
            if model_data.get('found', False) and 'performance' in model_data:
                models_with_perf.append(model_data)
        
        if len(models_with_perf) < 2:
            return {'error': 'Not enough models with performance data for comparison'}
        
        # Compare accuracy
        accuracy_models = [(m['model_id'], m['performance']['accuracy']) 
                          for m in models_with_perf 
                          if m['performance'].get('accuracy') is not None]
        
        if accuracy_models:
            # Sort by accuracy (descending)
            accuracy_models.sort(key=lambda x: x[1], reverse=True)
            comparisons['accuracy'] = {
                'best': {'model_id': accuracy_models[0][0], 'value': accuracy_models[0][1]},
                'ranking': [{'model_id': m[0], 'value': m[1]} for m in accuracy_models]
            }
        
        # Compare loss
        loss_models = [(m['model_id'], m['performance']['loss']) 
                      for m in models_with_perf 
                      if m['performance'].get('loss') is not None]
        
        if loss_models:
            # Sort by loss (ascending, lower is better)
            loss_models.sort(key=lambda x: x[1])
            comparisons['loss'] = {
                'best': {'model_id': loss_models[0][0], 'value': loss_models[0][1]},
                'ranking': [{'model_id': m[0], 'value': m[1]} for m in loss_models]
            }
        
        # Compare perplexity
        perplexity_models = [(m['model_id'], m['performance']['perplexity']) 
                            for m in models_with_perf 
                            if m['performance'].get('perplexity') is not None]
        
        if perplexity_models:
            # Sort by perplexity (ascending, lower is better)
            perplexity_models.sort(key=lambda x: x[1])
            comparisons['perplexity'] = {
                'best': {'model_id': perplexity_models[0][0], 'value': perplexity_models[0][1]},
                'ranking': [{'model_id': m[0], 'value': m[1]} for m in perplexity_models]
            }
        
        # Calculate relative improvements
        if len(models_with_perf) >= 2:
            relative_improvements = {}
            
            # Get pairs of models to compare
            for i, model1 in enumerate(models_with_perf):
                for j, model2 in enumerate(models_with_perf):
                    if i == j:
                        continue
                    
                    model1_id = model1['model_id']
                    model2_id = model2['model_id']
                    pair_key = f"{model1_id}_vs_{model2_id}"
                    improvements = {}
                    
                    # Compare accuracy
                    acc1 = model1['performance'].get('accuracy')
                    acc2 = model2['performance'].get('accuracy')
                    if acc1 is not None and acc2 is not None and acc2 > 0:
                        improvements['accuracy'] = {
                            'absolute': acc1 - acc2,
                            'percentage': (acc1 - acc2) / acc2 * 100.0,
                            'better': acc1 > acc2
                        }
                    
                    # Compare loss
                    loss1 = model1['performance'].get('loss')
                    loss2 = model2['performance'].get('loss')
                    if loss1 is not None and loss2 is not None and loss2 > 0:
                        improvements['loss'] = {
                            'absolute': loss1 - loss2,
                            'percentage': (loss1 - loss2) / loss2 * 100.0,
                            'better': loss1 < loss2
                        }
                    
                    # Compare perplexity
                    ppl1 = model1['performance'].get('perplexity')
                    ppl2 = model2['performance'].get('perplexity')
                    if ppl1 is not None and ppl2 is not None and ppl2 > 0:
                        improvements['perplexity'] = {
                            'absolute': ppl1 - ppl2,
                            'percentage': (ppl1 - ppl2) / ppl2 * 100.0,
                            'better': ppl1 < ppl2
                        }
                    
                    relative_improvements[pair_key] = improvements
            
            comparisons['relative_improvement'] = relative_improvements
        
        return comparisons
    
    def _generate_architecture_comparisons(self, model_data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate architecture comparisons between models.
        
        Args:
            model_data_list: List of model data dictionaries
            
        Returns:
            Dictionary containing architecture comparisons
        """
        comparisons = {
            'architecture_types': {},
            'model_size': {},
            'complexity': {}
        }
        
        # Extract models with architecture data
        models_with_arch = []
        for model_data in model_data_list:
            if model_data.get('found', False) and 'architecture' in model_data:
                models_with_arch.append(model_data)
        
        if len(models_with_arch) < 2:
            return {'error': 'Not enough models with architecture data for comparison'}
        
        # Compare architecture types
        arch_types = {}
        for model in models_with_arch:
            arch_type = model['architecture'].get('type', 'unknown')
            if arch_type not in arch_types:
                arch_types[arch_type] = []
            arch_types[arch_type].append(model['model_id'])
        
        comparisons['architecture_types'] = arch_types
        
        # Compare model sizes (parameters)
        param_models = [(m['model_id'], m['architecture'].get('total_parameters', 0)) 
                       for m in models_with_arch]
        
        if param_models:
            # Sort by parameter count (descending)
            param_models.sort(key=lambda x: x[1], reverse=True)
            comparisons['model_size'] = {
                'largest': {'model_id': param_models[0][0], 'parameters': param_models[0][1]},
                'smallest': {'model_id': param_models[-1][0], 'parameters': param_models[-1][1]},
                'ranking': [{'model_id': m[0], 'parameters': m[1]} for m in param_models]
            }
            
            # Add relative size comparisons
            if len(param_models) >= 2:
                size_ratios = {}
                for i, (id1, params1) in enumerate(param_models):
                    for j, (id2, params2) in enumerate(param_models):
                        if i == j or params2 == 0:
                            continue
                        pair_key = f"{id1}_vs_{id2}"
                        size_ratios[pair_key] = params1 / params2 if params2 > 0 else float('inf')
                
                comparisons['model_size']['relative_sizes'] = size_ratios
        
        # Compare model complexity (layers, heads)
        complexity_metrics = {}
        for model in models_with_arch:
            model_id = model['model_id']
            arch = model['architecture']
            metrics = {
                'layers': arch.get('num_layers', 0),
                'attention_heads': arch.get('num_attention_heads', 0),
                'hidden_size': arch.get('hidden_size', 0)
            }
            complexity_metrics[model_id] = metrics
        
        comparisons['complexity'] = {
            'metrics': complexity_metrics,
            'comparisons': {}
        }
        
        # Compare layers
        if all('layers' in metrics and metrics['layers'] > 0 for metrics in complexity_metrics.values()):
            layer_models = [(model_id, metrics['layers']) 
                           for model_id, metrics in complexity_metrics.items()]
            layer_models.sort(key=lambda x: x[1], reverse=True)
            
            comparisons['complexity']['comparisons']['layers'] = {
                'most': {'model_id': layer_models[0][0], 'value': layer_models[0][1]},
                'least': {'model_id': layer_models[-1][0], 'value': layer_models[-1][1]},
                'ranking': [{'model_id': m[0], 'value': m[1]} for m in layer_models]
            }
        
        # Compare attention heads
        if all('attention_heads' in metrics and metrics['attention_heads'] > 0 
              for metrics in complexity_metrics.values()):
            head_models = [(model_id, metrics['attention_heads']) 
                          for model_id, metrics in complexity_metrics.items()]
            head_models.sort(key=lambda x: x[1], reverse=True)
            
            comparisons['complexity']['comparisons']['attention_heads'] = {
                'most': {'model_id': head_models[0][0], 'value': head_models[0][1]},
                'least': {'model_id': head_models[-1][0], 'value': head_models[-1][1]},
                'ranking': [{'model_id': m[0], 'value': m[1]} for m in head_models]
            }
        
        # Compare hidden size
        if all('hidden_size' in metrics and metrics['hidden_size'] > 0 
              for metrics in complexity_metrics.values()):
            size_models = [(model_id, metrics['hidden_size']) 
                          for model_id, metrics in complexity_metrics.items()]
            size_models.sort(key=lambda x: x[1], reverse=True)
            
            comparisons['complexity']['comparisons']['hidden_size'] = {
                'largest': {'model_id': size_models[0][0], 'value': size_models[0][1]},
                'smallest': {'model_id': size_models[-1][0], 'value': size_models[-1][1]},
                'ranking': [{'model_id': m[0], 'value': m[1]} for m in size_models]
            }
        
        # Calculate efficiency metrics (if possible)
        if all('total_parameters' in model['architecture'] and model['architecture']['total_parameters'] > 0
              for model in models_with_arch):
            efficiency_metrics = {}
            
            for model in models_with_arch:
                model_id = model['model_id']
                params = model['architecture'].get('total_parameters', 0)
                
                # Check if performance data is available
                if 'performance' in model and model['performance'].get('accuracy') is not None:
                    accuracy = model['performance'].get('accuracy', 0)
                    
                    # Parameter efficiency (accuracy per million parameters)
                    if params > 0:
                        efficiency_metrics[model_id] = {
                            'accuracy_per_million_params': accuracy / (params / 1_000_000)
                        }
            
            if efficiency_metrics:
                comparisons['efficiency'] = efficiency_metrics
        
        return comparisons

    def _translate_filters_to_chroma(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Translate query filters to Chroma's filter format.

        Args:
            filters: Filters in the query format

        Returns:
            Filters in Chroma's format
        """
        # Add defensive checks at the beginning
        if filters is None:
            return {}

        if isinstance(filters, list):
            self.logger.warning("Filters received as list instead of dictionary. Converting to empty dict.")
            return {}

        chroma_filters = {}

        # Handle direct mappings
        for key, value in filters.items():
            # Skip nested filters for separate handling
            if isinstance(value, dict) and any(op.startswith('$') for op in value.keys()):
                continue

            # Handle list values
            if isinstance(value, list):
                chroma_filters[key] = {"$in": value}
            else:
                chroma_filters[key] = {"$eq": value}

        # Handle operator-based filters
        for key, operators in filters.items():
            if not isinstance(operators, dict):
                continue

            for op, value in operators.items():
                if op == "$eq":
                    chroma_filters[key] = {"$eq": value}
                elif op == "$ne":
                    chroma_filters[key] = {"$ne": value}
                elif op == "$gt":
                    chroma_filters[key] = {"$gt": value}
                elif op == "$gte":
                    chroma_filters[key] = {"$gte": value}
                elif op == "$lt":
                    chroma_filters[key] = {"$lt": value}
                elif op == "$lte":
                    chroma_filters[key] = {"$lte": value}
                elif op == "$in":
                    chroma_filters[key] = {"$in": value}
                elif op == "$nin":
                    chroma_filters[key] = {"$nin": value}
                elif op == "$contains":
                    # Special handling for string contains
                    chroma_filters[key] = {"$contains": value}

        return chroma_filters

    def _sanitize_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize parameters for inclusion in response metadata.
        Remove sensitive or internal fields.
        
        Args:
            parameters: Original parameters dictionary
            
        Returns:
            Sanitized parameters dictionary
        """
        if not parameters:
            return {}
            
        # Create a copy to avoid modifying the original
        sanitized = parameters.copy()
        
        # Remove sensitive fields
        sensitive_fields = ['user_id', 'access_token', 'auth_context', 'raw_query', 'query_id']
        for field in sensitive_fields:
            if field in sanitized:
                del sanitized[field]
        
        # Remove image data (could be large)
        if 'image_data' in sanitized:
            sanitized['image_data'] = "[binary data removed]"
        
        return sanitized
