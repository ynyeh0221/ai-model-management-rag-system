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

        # Add user_id to parameters for access control in handlers
        if user_id:
            parameters['user_id'] = user_id

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
        Query multiple metadata tables and merge results.
        """
        self.logger.debug(f"Handling text search: {query}")
        start_time = time.time()

        try:
            # Get user_id from parameters for access control
            user_id = parameters.get('user_id')

            # Extract search parameters
            try:
                requested_limit = int(parameters.get('limit', 20))
                if requested_limit <= 0:
                    requested_limit = 20
            except (TypeError, ValueError):
                requested_limit = 20

            # Define a higher limit for chunk searches
            chunk_search_limit = requested_limit * 10
            filters = parameters.get('filters', {})

            # Apply access control filter if needed
            if self.access_control_manager and user_id:
                access_filter = self.access_control_manager.create_access_filter(user_id)
                if filters:
                    filters = {
                        "$and": [
                            filters if "$and" not in filters else {"$and": filters["$and"]},
                            access_filter if "$and" not in access_filter else {"$and": access_filter["$and"]}
                        ]
                    }
                else:
                    filters = access_filter

            # Translate filters to Chroma format
            chroma_filters = self._translate_filters_to_chroma(filters)

            # Define metadata tables to search
            metadata_tables = [
                "model_descriptions",
                "model_architectures",
                "model_frameworks",
                "model_datasets",
                "model_training_configs",
                "model_date",
                "model_file",
                "model_git"
            ]

            # Table weights for relevance calculation
            table_weights = {
                "model_descriptions": 0.30,
                "model_architectures": 0.20,
                "model_frameworks": 0.15,
                "model_datasets": 0.15,
                "model_training_configs": 0.10,
                "model_date": 0.05,
                "model_file": 0.03,
                "model_git": 0.02
            }

            # STEP 1: Search primary tables for model info
            # We'll focus on description and architecture first as they're most important
            metadata_search_start = time.time()

            # Start with model_descriptions as it's most relevant
            primary_results = await self.chroma_manager.search(
                collection_name="model_descriptions",
                query=query,
                where=chroma_filters,
                limit=requested_limit * 3,  # Get more results to ensure coverage
                include=["metadatas", "documents", "distances"]
            )

            # Process results and create model dictionary keyed by model_id
            model_results = {}

            # Process primary table results
            for result in primary_results.get('results', []):
                metadata = result.get('metadata', {})
                model_id = metadata.get('model_id', 'unknown')
                distance = result.get('distance', 1.0)

                # Apply access control
                if self.access_control_manager and user_id:
                    if not self.access_control_manager.check_access({'metadata': metadata}, user_id, "view"):
                        continue

                # Initialize model data
                if model_id not in model_results:
                    model_results[model_id] = {
                        'model_id': model_id,
                        'metadata': {},
                        'distance': distance * table_weights.get("model_descriptions", 0.3),
                        'tables': ["model_descriptions"],
                        'match_source': 'metadata',
                        'chunk_descriptions': []  # Initialize list to hold chunk descriptions
                    }

                # Add metadata from this table
                model_results[model_id]['metadata'].update(metadata)

            # STEP 2: For each found model, fetch metadata from other tables
            model_ids = list(model_results.keys())

            # Only fetch additional metadata if we have models
            if model_ids:
                for table in metadata_tables[1:]:  # Skip descriptions as we already searched it
                    # Skip if we have no models to search for
                    if not model_ids:
                        continue

                    table_results = await self.chroma_manager.get(
                        collection_name=table,
                        where={"model_id": {"$in": model_ids}},
                        include=["metadatas"]
                    )

                    # Process results
                    for result in table_results.get('results', []):
                        metadata = result.get('metadata', {})
                        model_id = metadata.get('model_id', 'unknown')

                        # Skip if model not in our results (shouldn't happen)
                        if model_id not in model_results:
                            continue

                        # Add metadata from this table
                        model_results[model_id]['metadata'].update(metadata)
                        model_results[model_id]['tables'].append(table)

                        # Adjust distance based on table match (more tables = better match)
                        current_distance = model_results[model_id]['distance']
                        weight = table_weights.get(table, 0.1)
                        model_results[model_id][
                            'distance'] -= weight * 0.1  # Slightly improve score for each table match

            metadata_search_time = (time.time() - metadata_search_start) * 1000

            # STEP 3: Search chunks table
            chunks_search_start = time.time()
            chunk_results = await self.chroma_manager.search(
                collection_name="model_scripts_chunks",
                query=query,
                where=chroma_filters,
                limit=chunk_search_limit,
                include=["metadatas", "documents", "distances"]
            )
            chunks_search_time = (time.time() - chunks_search_start) * 1000

            # Process chunk results
            for result in chunk_results.get('results', []):
                metadata = result.get('metadata', {})
                model_id = metadata.get('model_id', 'unknown')
                chunk_id = metadata.get('chunk_id', -1)
                distance = result.get('distance', 1.0)

                # Apply access control
                if self.access_control_manager and user_id:
                    if not self.access_control_manager.check_access({'metadata': metadata}, user_id, "view"):
                        continue

                # If we already have this model, update its match source and distance
                if model_id in model_results:
                    model_results[model_id]['match_source'] = 'metadata+chunks'
                    # Chunks are important evidence, so improve score significantly
                    model_results[model_id]['distance'] = 0.7 * model_results[model_id]['distance'] + 0.3 * (
                            distance * 0.8)

                    # Store the chunk_id and distance for later fetching descriptions
                    model_results[model_id]['chunk_descriptions'].append({
                        'chunk_id': chunk_id,
                        'distance': distance
                    })
                else:
                    # This is a new model found only in chunks
                    # We'll need to fetch its metadata
                    model_results[model_id] = {
                        'model_id': model_id,
                        'metadata': metadata,
                        'distance': distance * 0.8,  # Chunk matches are weighted at 80%
                        'tables': ["chunks"],
                        'match_source': 'chunks',
                        'chunk_descriptions': [{
                            'chunk_id': chunk_id,
                            'distance': distance
                        }]
                    }

            # STEP 4: For models found only in chunks, fetch their metadata
            chunks_only_models = [model_id for model_id, data in model_results.items()
                                  if data['match_source'] == 'chunks']

            if chunks_only_models:
                # First fetch from descriptions as it has most important metadata
                desc_results = await self.chroma_manager.get(
                    collection_name="model_descriptions",
                    where={"model_id": {"$in": chunks_only_models}},
                    include=["metadatas"]
                )

                # Process and update metadata
                for result in desc_results.get('results', []):
                    metadata = result.get('metadata', {})
                    model_id = metadata.get('model_id', 'unknown')

                    if model_id in model_results:
                        model_results[model_id]['metadata'].update(metadata)
                        model_results[model_id]['tables'].append("model_descriptions")

                # Then fetch other important metadata selectively to avoid overloading
                important_tables = ["model_architectures", "model_frameworks", "model_datasets"]

                for table in important_tables:
                    table_results = await self.chroma_manager.get(
                        collection_name=table,
                        where={"model_id": {"$in": chunks_only_models}},
                        include=["metadatas"]
                    )

                    # Process results
                    for result in table_results.get('results', []):
                        metadata = result.get('metadata', {})
                        model_id = metadata.get('model_id', 'unknown')

                        if model_id in model_results:
                            model_results[model_id]['metadata'].update(metadata)
                            model_results[model_id]['tables'].append(table)

            # STEP 5: Fetch chunk descriptions for each model
            # This is a new step to fetch chunk descriptions from model_descriptions table
            for model_id, model_data in model_results.items():
                # Sort chunk descriptions by distance (lower distance is better)
                chunk_descriptions = sorted(model_data['chunk_descriptions'], key=lambda x: x['distance'])

                # Take top 3 chunks with lowest distance
                top_chunks = chunk_descriptions[:3]

                if top_chunks:
                    # Prepare queries to fetch descriptions for each chunk
                    description_queries = []
                    for chunk_info in top_chunks:
                        chunk_id = chunk_info['chunk_id']
                        description_id = f"model_descriptions_{model_id}_chunk_{chunk_id}"

                        # Create async query for this chunk description
                        description_queries.append(self.chroma_manager.get(
                            collection_name="model_descriptions",
                            where={"id": {"$eq": description_id}},
                            include=["metadatas"]
                        ))

                    # Run all description queries concurrently
                    if description_queries:
                        description_results = await asyncio.gather(*description_queries)

                        # Process results and collect descriptions
                        descriptions = []
                        for result in description_results:
                            if result and result.get('results'):
                                for item in result.get('results', []):
                                    metadata = item.get('metadata', {})
                                    description = metadata.get('description', '')
                                    if description:
                                        descriptions.append(description)

                        # Merge descriptions and add to model data
                        if descriptions:
                            merged_description = " ".join(descriptions)
                            model_data['merged_description'] = merged_description

            # Convert to list and sort by distance (lower is better)
            output_list = list(model_results.values())
            output_list.sort(key=lambda x: x['distance'])

            # Limit to requested number of results
            output_list = output_list[:requested_limit]

            # Prepare final result items
            items = []
            for rank, model in enumerate(output_list):
                items.append({
                    'id': f"model_metadata_{model.get('model_id')}",
                    'model_id': model.get('model_id'),
                    'metadata': model.get('metadata', {}),
                    'rank': rank + 1,
                    'match_source': model.get('match_source'),
                    'distance': model.get('distance'),
                    'merged_description': model.get('merged_description', '')  # Add merged description to output
                })

            # Calculate performance metrics
            total_search_time = metadata_search_time + chunks_search_time
            total_time = (time.time() - start_time) * 1000

            # Log performance metrics if analytics available
            if self.analytics and 'query_id' in parameters:
                self.analytics.log_performance_metrics(
                    query_id=parameters['query_id'],
                    search_time_ms=int(total_search_time),
                    total_time_ms=int(total_time)
                )

            return {
                'success': True,
                'type': 'text_search',
                'items': items,
                'total_found': len(items),
                'total_models': len(model_results),
                'performance': {
                    'metadata_search_time_ms': metadata_search_time,
                    'chunks_search_time_ms': chunks_search_time,
                    'total_search_time_ms': total_search_time,
                    'total_time_ms': total_time
                }
            }

        except Exception as e:
            self.logger.error(f"Error in text search: {e}", exc_info=True)
            raise

    async def handle_metadata_search(self, query: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a metadata-specific search query across multiple metadata tables.
        """
        self.logger.debug(f"Handling metadata search: {query}")
        start_time = time.time()

        try:
            # Get user_id from parameters for access control
            user_id = parameters.get('user_id')

            # Extract search parameters
            filters = parameters.get('filters', {})
            limit = parameters.get('limit', 20)

            # If query mentions a specific month, add it to filters
            month_filter = None
            if "april" in query.lower():
                month_filter = {"created_month": {"$eq": "April"}}
            elif "march" in query.lower():
                month_filter = {"created_month": {"$eq": "March"}}
            # Add more month checks as needed

            # Convert filters to Chroma format
            chroma_filters = self._translate_filters_to_chroma(filters)

            # Apply month filter specifically to model_date table if needed
            date_specific_filters = None
            if month_filter:
                if chroma_filters:
                    date_specific_filters = {
                        "$and": [chroma_filters, month_filter]
                    }
                else:
                    date_specific_filters = month_filter

            # Apply access control filter if applicable
            if self.access_control_manager and user_id:
                access_filter = self.access_control_manager.create_access_filter(user_id)

                if chroma_filters:
                    chroma_filters = {
                        "$and": [chroma_filters, access_filter]
                    }
                else:
                    chroma_filters = access_filter

                if date_specific_filters:
                    date_specific_filters = {
                        "$and": [date_specific_filters, access_filter]
                    }

            # Define metadata tables to search with their relevance weights
            metadata_tables = {
                "model_descriptions": 0.25,
                "model_architectures": 0.15,
                "model_frameworks": 0.15,
                "model_datasets": 0.15,
                "model_training_configs": 0.15
            }

            # Add date table if we have month filters (with higher weight)
            if month_filter:
                metadata_tables["model_date"] = 0.30  # Higher priority since we're filtering by month
            else:
                metadata_tables["model_date"] = 0.10

            # Add remaining tables with lower weights
            metadata_tables["model_file"] = 0.03
            metadata_tables["model_git"] = 0.02

            # Start with most relevant table (either descriptions or dates if month filter exists)
            search_start = time.time()
            model_results = {}

            # Start with date table if we have month filters
            if month_filter:
                # Get models matching the month filter
                date_results = await self.chroma_manager.get(
                    collection_name="model_date",
                    where=date_specific_filters,
                    limit=limit * 3,
                    include=["metadatas"]
                )

                # Process models from date table
                for result in date_results.get('results', []):
                    metadata = result.get('metadata', {})
                    model_id = metadata.get('model_id', 'unknown')

                    # Apply access control check
                    if self.access_control_manager and user_id:
                        if not self.access_control_manager.check_access({'metadata': metadata}, user_id, "view"):
                            continue

                    # Add to model results
                    if model_id not in model_results:
                        model_results[model_id] = {
                            'model_id': model_id,
                            'metadata': {},
                            'relevance_score': metadata_tables["model_date"],  # Start with date table weight
                            'tables': ["model_date"],
                            'chunk_descriptions': []  # Initialize list to hold chunk descriptions
                        }

                    # Add metadata from this table
                    model_results[model_id]['metadata'].update(metadata)
            else:
                # Start with descriptions table if no month filter
                desc_results = await self.chroma_manager.get(
                    collection_name="model_descriptions",
                    where=chroma_filters,
                    limit=limit * 3,
                    include=["metadatas"]
                )

                # Process models from descriptions table
                for result in desc_results.get('results', []):
                    metadata = result.get('metadata', {})
                    model_id = metadata.get('model_id', 'unknown')
                    chunk_id = metadata.get('chunk_id', -1)

                    # Apply access control check
                    if self.access_control_manager and user_id:
                        if not self.access_control_manager.check_access({'metadata': metadata}, user_id, "view"):
                            continue

                    # Add to model results
                    if model_id not in model_results:
                        model_results[model_id] = {
                            'model_id': model_id,
                            'metadata': {},
                            'relevance_score': metadata_tables["model_descriptions"],  # Start with desc table weight
                            'tables': ["model_descriptions"],
                            'chunk_descriptions': []  # Initialize list to hold chunk descriptions
                        }

                    # Add metadata from this table
                    model_results[model_id]['metadata'].update(metadata)

                    # If this is a chunk description, add to the list
                    if chunk_id >= 0:
                        model_results[model_id]['chunk_descriptions'].append({
                            'chunk_id': chunk_id,
                            'relevance_score': model_results[model_id]['relevance_score']
                        })

            # Get list of model IDs we found
            model_ids = list(model_results.keys())

            # If we have models, fetch their metadata from other tables
            if model_ids:
                # Determine which tables we still need to fetch from
                remaining_tables = [t for t in metadata_tables.keys()
                                    if t != "model_date" and t != "model_descriptions"]

                # Fetch from each remaining table
                for table in remaining_tables:
                    table_results = await self.chroma_manager.get(
                        collection_name=table,
                        where={"model_id": {"$in": model_ids}},
                        include=["metadatas"]
                    )

                    # Process results
                    for result in table_results.get('results', []):
                        metadata = result.get('metadata', {})
                        model_id = metadata.get('model_id', 'unknown')

                        # Skip if model not in our results (shouldn't happen)
                        if model_id not in model_results:
                            continue

                        # Add metadata from this table
                        model_results[model_id]['metadata'].update(metadata)
                        model_results[model_id]['tables'].append(table)

                        # Increase relevance score based on table weight
                        model_results[model_id]['relevance_score'] += metadata_tables.get(table, 0.1)

            # STEP: Fetch chunk descriptions for each model
            for model_id, model_data in model_results.items():
                # If we don't have any chunk descriptions yet, get all of them for the model
                if not model_data['chunk_descriptions']:
                    # Query for all chunk descriptions for this model
                    all_desc_results = await self.chroma_manager.get(
                        collection_name="model_descriptions",
                        where={"model_id": {"$eq": model_id}, "chunk_id": {"$gte": 0}},
                        # Only get descriptions with chunk_id
                        include=["metadatas"]
                    )

                    # Process and add to chunk_descriptions
                    if all_desc_results and all_desc_results.get('results'):
                        for item in all_desc_results.get('results', []):
                            metadata = item.get('metadata', {})
                            chunk_id = metadata.get('chunk_id', -1)
                            if chunk_id >= 0:
                                model_data['chunk_descriptions'].append({
                                    'chunk_id': chunk_id,
                                    'relevance_score': model_data['relevance_score'] / 2
                                    # Lower priority than direct matches
                                })

                # Sort chunk descriptions by relevance score (higher is better)
                chunk_descriptions = sorted(model_data['chunk_descriptions'], key=lambda x: x['relevance_score'],
                                            reverse=True)

                # Take top 3 chunks with highest relevance
                top_chunks = chunk_descriptions[:3]

                if top_chunks:
                    # Prepare queries to fetch descriptions for each chunk
                    description_queries = []
                    for chunk_info in top_chunks:
                        chunk_id = chunk_info['chunk_id']
                        description_id = f"model_descriptions_{model_id}_chunk_{chunk_id}"

                        # Create async query for this chunk description
                        description_queries.append(self.chroma_manager.get(
                            collection_name="model_descriptions",
                            where={"id": {"$eq": description_id}},
                            include=["metadatas"]
                        ))

                    # Run all description queries concurrently
                    if description_queries:
                        description_results = await asyncio.gather(*description_queries)

                        # Process results and collect descriptions
                        descriptions = []
                        for result in description_results:
                            if result and result.get('results'):
                                for item in result.get('results', []):
                                    metadata = item.get('metadata', {})
                                    description = metadata.get('description', '')
                                    if description:
                                        descriptions.append(description)

                        # Merge descriptions and add to model data
                        if descriptions:
                            merged_description = " ".join(descriptions)
                            model_data['merged_description'] = merged_description

            search_time = (time.time() - search_start) * 1000

            # Convert to list and sort by relevance score (higher is better)
            output_list = list(model_results.values())
            output_list.sort(key=lambda x: x['relevance_score'], reverse=True)

            # Limit to requested number of results
            output_list = output_list[:limit]

            # Prepare final result items
            items = []
            for idx, model in enumerate(output_list):
                items.append({
                    'id': f"model_metadata_{model.get('model_id')}",
                    'model_id': model.get('model_id'),
                    'metadata': model.get('metadata', {}),
                    'rank': idx + 1,
                    'relevance_score': model.get('relevance_score'),
                    'merged_description': model.get('merged_description', '')  # Add merged description to output
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
            # Get user_id from parameters for access control
            user_id = parameters.get('user_id')

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
            limit = parameters.get('limit', 20)
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

            # Apply access control filter if applicable
            if self.access_control_manager and user_id:
                access_filter = self.access_control_manager.create_access_filter(user_id)

                # Merge access control filter with existing filters
                if filters:
                    # If both have conditions, combine with $and
                    filters = {
                        "$and": [
                            filters,
                            access_filter
                        ]
                    }
                else:
                    # If no existing filters, just use access filter
                    filters = access_filter

            # Prepare Chroma query
            search_params = {
                'query': {'embedding': query_embedding},
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

                # Apply additional access control check
                if self.access_control_manager and user_id:
                    # Check if user has access to this image
                    if not self.access_control_manager.check_access(
                            {'metadata': metadata}, user_id, "view"
                    ):
                        # Skip this result if user doesn't have access
                        continue

                # Add the image URL/path
                image_path = metadata.get('image_path', "")
                thumbnail_path = metadata.get('thumbnail_path', "")

                items.append({
                    'id': result.get('id'),
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
            # Get user_id from parameters for access control
            user_id = parameters.get('user_id')

            # Get model IDs to compare
            model_ids = parameters.get('model_ids', [])
            if not model_ids or len(model_ids) < 2:
                raise ValueError("Comparison requires at least two model IDs")

            # Get comparison dimensions
            dimensions = parameters.get('comparison_dimensions', ['architecture', 'performance'])

            # Fetch model data in parallel with access control
            tasks = []
            for model_id in model_ids:
                tasks.append(self._fetch_model_data(model_id, dimensions, user_id))

            model_data_list = await asyncio.gather(*tasks)

            # Filter out models the user doesn't have access to
            accessible_models = [model for model in model_data_list if model.get('found', False)]

            # Check if we still have enough models to compare
            if len(accessible_models) < 2:
                raise ValueError("Need at least two accessible models to perform comparison")

            # Process comparison data
            comparison_results = {
                'models': {},
                'dimensions': {},
                'summary': {}
            }

            # Organize by model
            for model_data in accessible_models:
                model_id = model_data.get('model_id', 'unknown')
                comparison_results['models'][model_id] = model_data

            # Organize by dimension
            for dimension in dimensions:
                comparison_results['dimensions'][dimension] = {}
                for model_id, model_data in comparison_results['models'].items():
                    comparison_results['dimensions'][dimension][model_id] = model_data.get(dimension, {})

            # Generate performance comparisons
            if 'performance' in dimensions and len(accessible_models) >= 2:
                perf_comparisons = self._generate_performance_comparisons(accessible_models)
                comparison_results['summary']['performance'] = perf_comparisons

            # Generate architecture comparisons if applicable
            if 'architecture' in dimensions and len(accessible_models) >= 2:
                arch_comparisons = self._generate_architecture_comparisons(accessible_models)
                comparison_results['summary']['architecture'] = arch_comparisons

            return {
                'success': True,
                'type': 'comparison',
                'models': [model.get('model_id') for model in accessible_models],
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
            # Get user_id from parameters for access control
            user_id = parameters.get('user_id')

            # Get model IDs for notebook
            model_ids = parameters.get('model_ids', [])
            if not model_ids:
                raise ValueError("Notebook generation requires at least one model ID")

            # Verify user has access to all requested models
            if self.access_control_manager and user_id:
                accessible_models = []
                for model_id in model_ids:
                    # Check if user has access to this model
                    model_info = await self._fetch_model_metadata(model_id)
                    if model_info and self.access_control_manager.check_access(
                            {'metadata': model_info}, user_id, "view"
                    ):
                        accessible_models.append(model_id)

                # Update model_ids to only include accessible ones
                model_ids = accessible_models

                if not model_ids:
                    raise ValueError("User does not have access to any of the requested models")

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
                'user_id': user_id
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

    # Helper methods for the search functions
    async def _search_metadata_table(self, table_name, query, filters, limit, weight):
        """Search a specific metadata table and return weighted results."""
        try:
            search_params = {
                'query': query,
                'where': filters,
                'limit': limit,
                'include': ["metadatas", "documents", "distances"]
            }

            results = await self.chroma_manager.search(
                collection_name=table_name,
                **search_params
            )

            # Add the table name and weight to the results
            return {
                'table_name': table_name,
                'weight': weight,
                'results': results.get('results', [])
            }
        except Exception as e:
            self.logger.error(f"Error searching metadata table {table_name}: {e}")
            return {
                'table_name': table_name,
                'weight': weight,
                'results': []
            }

    async def _get_metadata_table(self, table_name, filters, limit, weight):
        """Get data from a specific metadata table based on filters."""
        try:
            get_params = {
                'where': filters,
                'limit': limit,
                'include': ["metadatas"]
            }

            results = await self.chroma_manager.get(
                collection_name=table_name,
                **get_params
            )

            # Add the table name and weight to the results
            return {
                'table_name': table_name,
                'weight': weight,
                'results': results.get('results', [])
            }
        except Exception as e:
            self.logger.error(f"Error getting metadata from table {table_name}: {e}")
            return {
                'table_name': table_name,
                'weight': weight,
                'results': []
            }

    async def _fetch_model_metadata_by_id(self, table_name, model_id, user_id=None):
        """Fetch metadata for a specific model from a specific table."""
        try:
            # Prepare filters to get model metadata
            filters = {'model_id': {'$eq': model_id}}

            # Apply access control filter if applicable
            if self.access_control_manager and user_id:
                access_filter = self.access_control_manager.create_access_filter(user_id)
                filters = {
                    "$and": [
                        filters,
                        access_filter
                    ]
                }

            # Fetch model metadata from specific table
            result = await self.chroma_manager.get(
                collection_name=table_name,
                where=filters,
                include=["metadatas"]
            )

            # Return the first result if available
            if result and result.get('results'):
                metadata = result['results'][0].get('metadata', {})
                # Create a single top-level field for this table's data to avoid conflicts
                table_key = table_name.replace('model_', '')  # e.g., "model_frameworks" becomes "frameworks"
                return {'metadata': {table_key: metadata}}

        except Exception as e:
            self.logger.error(f"Error fetching model metadata for {model_id} from {table_name}: {e}")

        return None

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

    async def _fetch_model_data(self, model_id: str, dimensions: List[str], user_id: Optional[str] = None) -> Dict[
        str, Any]:
        """
        Fetch data for a specific model with access control.

        Args:
            model_id: The model identifier
            dimensions: List of data dimensions to fetch
            user_id: Optional user ID for access control

        Returns:
            Dictionary containing model data
        """
        # Prepare filters to get model data
        filters = {'model_id': {'$eq': model_id}}

        # Apply access control filter if applicable
        if self.access_control_manager and user_id:
            access_filter = self.access_control_manager.create_access_filter(user_id)

            # Combine filters with access control
            filters = {
                "$and": [
                    filters,
                    access_filter
                ]
            }

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
                    'num_attention_heads': metadata.get('model_dimensions', {}).get('num_attention_heads', {}).get(
                        'value'),
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
                    'training_time_hours': metadata.get('training_config', {}).get('training_time_hours', {}).get(
                        'value'),
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

    async def _fetch_model_metadata(self, model_id: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Fetch only the metadata for a specific model with access control.

        Args:
            model_id: The model identifier
            user_id: Optional user ID for access control

        Returns:
            Dictionary containing model metadata or None if not found or no access
        """
        # Prepare filters to get model metadata
        filters = {'model_id': {'$eq': model_id}}

        # Apply access control filter if applicable
        if self.access_control_manager and user_id:
            access_filter = self.access_control_manager.create_access_filter(user_id)

            # Combine filters with access control
            filters = {
                "$and": [
                    filters,
                    access_filter
                ]
            }

        # Fetch model metadata from Chroma
        try:
            metadata_results = await self.chroma_manager.get(
                collection_name="model_scripts_metadata",
                where=filters,
                include=["metadatas"]
            )

            # Return the first result's metadata if available
            if metadata_results and metadata_results.get('results'):
                return metadata_results['results'][0].get('metadata', {})

        except Exception as e:
            self.logger.error(f"Error fetching model metadata for {model_id}: {e}")

        return None

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

        # Handle case where we already have a properly structured filter (with $and, $or, etc.)
        if len(filters) == 1 and list(filters.keys())[0].startswith('$'):
            return filters

        # Handle multiple top-level filter conditions
        # ChromaDB expects them to be wrapped with an operator like $and
        if len(filters) > 1:
            conditions = []
            for key, value in filters.items():
                # Skip nested operators for separate handling
                if isinstance(value, dict) and any(op.startswith('$') for op in value.keys()):
                    conditions.append({key: value})
                # Handle list values
                elif isinstance(value, list):
                    conditions.append({key: {"$in": value}})
                # Handle simple values
                else:
                    conditions.append({key: {"$eq": value}})

            return {"$and": conditions}

        # Handle single filter condition
        if len(filters) == 1:
            key, value = next(iter(filters.items()))

            # Handle nested operator filters
            if isinstance(value, dict) and any(op.startswith('$') for op in value.keys()):
                return filters

            # Handle list values
            if isinstance(value, list):
                return {key: {"$in": value}}

            # Handle simple values
            return {key: {"$eq": value}}

        return {}

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
