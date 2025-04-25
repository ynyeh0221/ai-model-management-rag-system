# src/query_engine/search_dispatcher.py
import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Union, Tuple

from src.query_engine.handlers.comparison_generator import ComparisonGenerator
from src.query_engine.handlers.distance_normalizer import DistanceNormalizer
from src.query_engine.handlers.filter_translator import FilterTranslator
from src.query_engine.handlers.metadata_table_manager import MetadataTableManager
from src.query_engine.handlers.model_data_fetcher import ModelDataFetcher
from src.query_engine.handlers.performance_metrics_calculator import PerformanceMetricsCalculator
from src.query_engine.handlers.image_search_manager import ImageSearchManager
from src.query_engine.query_intent import QueryIntent


class SearchDispatcher:
    """
    Dispatcher that routes queries to appropriate search handlers
    based on the classified intent.
    """

    def __init__(self, chroma_manager, text_embedder, image_embedder,
                 access_control_manager=None, analytics=None, image_search_manager=None):
        """
        Initialize the SearchDispatcher with required dependencies.

        Args:
            chroma_manager: Manager for Chroma vector database interactions
            text_embedder: Component for generating text embeddings
            image_embedder: Component for generating image embeddings
            access_control_manager: Optional manager for access control
            analytics: Optional analytics collector
            image_search_manager: Optional manager for image searches
        """
        self.chroma_manager = chroma_manager
        self.text_embedder = text_embedder
        self.image_embedder = image_embedder
        self.access_control_manager = access_control_manager
        self.analytics = analytics
        self.logger = logging.getLogger(__name__)

        # Initialize utility classes
        self.distance_normalizer = DistanceNormalizer()
        self.filter_translator = FilterTranslator()
        self.model_data_fetcher = ModelDataFetcher(chroma_manager, access_control_manager)
        self.performance_metrics = PerformanceMetricsCalculator(analytics)
        self.comparison_generator = ComparisonGenerator()
        self.metadata_table_manager = MetadataTableManager(chroma_manager, access_control_manager)

        # Initialize the image search manager if not provided
        self.image_search_manager = image_search_manager or ImageSearchManager(
            chroma_manager=chroma_manager,
            image_embedder=image_embedder,
            access_control_manager=access_control_manager,
            analytics=analytics
        )

        # Define handlers mapping for dispatching
        self.handlers = {
            QueryIntent.RETRIEVAL: self.handle_metadata_search,
            QueryIntent.COMPARISON: {
                "model_id": self.handle_comparison,
                "cohort": self.handle_comparison_cohort
            },
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
        if intent is not QueryIntent.COMPARISON:
            handler = self.handlers.get(intent, self.handle_fallback_search)
        else:
            if len(parameters.get("filters", {}).get("model_id") or []) >= 2:
                handler = self.handlers.get(intent, self.handle_fallback_search).get('model_id')
            else:
                handler = self.handlers.get(intent, self.handle_fallback_search).get('cohort')

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
                'parameters': self.performance_metrics.sanitize_parameters(parameters)
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
        1. Query all metadata tables to find model_ids
        2. Query model_scripts_chunks to find more model_ids
        3. Fetch complete metadata for ALL found model_ids
        4. Calculate weighted distance sum using complete metadata
        5. Sort ALL results by distance
        6. Limit to requested number of results
        """
        self.logger.debug(f"Handling text search: {query}")
        start_time = time.time()

        try:
            # Extract search parameters and apply access control
            user_id, requested_limit, chroma_filters = self._extract_text_search_parameters(parameters)

            # Define metadata tables to search with their weights
            table_weights = self.metadata_table_manager.get_metadata_table_weights()

            # STEP 0: Get global min/max distances for all collections
            collections = list(table_weights.keys()) + ["model_scripts_chunks"]
            collection_stats = await self._get_collection_distance_stats_for_query(query, collections, user_id)
            self.logger.info(f"Collection stats for query: {collection_stats}")

            # STEP 1: Search all metadata tables in parallel to collect model_ids
            metadata_search_start = time.time()
            all_results = await self._search_all_metadata_tables(query, chroma_filters, requested_limit, table_weights,
                                                                 user_id)

            # STEP 2: Search chunks table to find more matching models
            chunks_search_start = time.time()
            all_results, chunks_search_time = await self._search_model_chunks_table(
                query, chroma_filters, requested_limit, all_results, user_id, chunks_search_start
            )

            # STEP 3: Fetch complete metadata for ALL found models with distances
            all_results = await self._fetch_complete_model_metadata(
                query, all_results, table_weights, user_id
            )

            # Special handling for model descriptions using chunks - process one model at a time
            all_results = await self._process_model_descriptions_text_search(query, all_results, 5)

            # STEP 4: Calculate weighted distance sum for all models using all gathered metadata
            all_results = self._calculate_model_distances(all_results, table_weights, collection_stats)

            # STEP 5 & 6: Convert to list, sort by distance, and limit results
            output_list = self._sort_and_limit_search_results(all_results, requested_limit)

            # Prepare final result items
            items = self._prepare_text_search_items(output_list)

            # Calculate performance metrics
            metadata_search_time = (time.time() - metadata_search_start) * 1000
            performance_metrics = self.performance_metrics.calculate_text_search_performance(
                start_time, metadata_search_time, chunks_search_time, parameters
            )

            return {
                'success': True,
                'type': 'text_search',
                'items': items,
                'total_found': len(items),
                'total_models': len(all_results),
                'performance': performance_metrics
            }

        except Exception as e:
            self.logger.error(f"Error in text search: {e}", exc_info=True)
            raise

    def _extract_text_search_parameters(self, parameters: Dict[str, Any]) -> Tuple[Optional[str], int, Dict[str, Any]]:
        """Extract and validate search parameters."""
        # Get user_id from parameters for access control
        user_id = parameters.get('user_id')

        # Extract search parameters
        try:
            requested_limit = int(parameters.get('limit', 10))
            if requested_limit <= 0:
                requested_limit = 10
        except (TypeError, ValueError):
            requested_limit = 10

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
        chroma_filters = self.filter_translator.translate_to_chroma(filters)

        return user_id, requested_limit, chroma_filters

    async def _search_all_metadata_tables(
            self, query: str, chroma_filters: Dict[str, Any], requested_limit: int,
            table_weights: Dict[str, float], user_id: Optional[str]
    ) -> Dict[str, Any]:
        """Search all metadata tables in parallel to collect model_ids."""
        search_tasks = []

        # Create search tasks for all tables
        for table_name in table_weights.keys():
            if table_name == "model_descriptions":
                search_limit = requested_limit * 40
            else:
                search_limit = requested_limit
            search_tasks.append(self.chroma_manager.search(
                collection_name=table_name,
                query=query,
                where=chroma_filters,
                limit=search_limit,
                include=["metadatas", "documents"]  # Don't need distances here
            ))

        # Execute all searches in parallel
        search_results = await asyncio.gather(*search_tasks)

        # Initialize dictionary to collect all models
        all_results = {}

        # Process search results from each table to identify all model_ids ONLY
        for table_idx, result in enumerate(search_results):
            table_name = list(table_weights.keys())[table_idx]

            for item in result.get('results', []):
                metadata = item.get('metadata', {})
                model_id = metadata.get('model_id', 'unknown')

                # Skip if no model_id
                if model_id == 'unknown':
                    continue

                # Apply access control
                if self.access_control_manager and user_id:
                    if not self.access_control_manager.check_access({'metadata': metadata}, user_id, "view"):
                        continue

                # If first time seeing this model, initialize its entry
                if model_id not in all_results:
                    all_results[model_id] = {
                        'model_id': model_id,
                        'tables': [],
                        'table_initial_distances': {},  # Empty dictionary for distances
                        'match_source': 'metadata'
                    }

                # Add this table to the list of matching tables if not already there
                if table_name not in all_results[model_id]['tables']:
                    all_results[model_id]['tables'].append(table_name)

        return all_results

    async def _search_model_chunks_table(
            self, query: str, chroma_filters: Dict[str, Any], requested_limit: int,
            all_results: Dict[str, Any], user_id: Optional[str], chunks_search_start: float
    ) -> Tuple[Dict[str, Any], float]:
        """Search chunks table to find more matching models."""
        try:
            chunk_results = await self.chroma_manager.search(
                collection_name="model_scripts_chunks",
                query=query,
                where=chroma_filters,
                limit=requested_limit * 100,  # Define a higher limit for chunk searches
                include=["metadatas", "documents", "distances"]
            )
            chunks_search_time = (time.time() - chunks_search_start) * 1000
        except Exception as e:
            self.logger.error(f"Error in chunks search: {e}")
            chunk_results = {'results': []}
            chunks_search_time = (time.time() - chunks_search_start) * 1000

        # Process chunk results
        for result in chunk_results.get('results', []):
            metadata = result.get('metadata', {})
            model_id = metadata.get('model_id', 'unknown')

            # Skip if no model_id
            if model_id == 'unknown':
                continue

            # Apply access control
            if self.access_control_manager and user_id:
                if not self.access_control_manager.check_access({'metadata': metadata}, user_id, "view"):
                    continue

            # Get distance
            distance = result.get('distance')
            if distance is None:
                distance = 2.0

            # If we already have this model, update its match source
            if model_id in all_results:
                all_results[model_id]['match_source'] = 'metadata+chunks'
                all_results[model_id]['chunk_initial_distance'] = distance
            else:
                # This is a new model found only in chunks
                all_results[model_id] = {
                    'model_id': model_id,
                    'tables': ["chunks"],
                    'match_source': 'chunks',
                    'chunk_initial_distance': distance
                }

        return all_results, chunks_search_time

    async def _fetch_complete_model_metadata(
            self, query: str, all_results: Dict[str, Any], table_weights: Dict[str, float], user_id: Optional[str]
    ) -> Dict[str, Any]:
        """Fetch complete metadata for ALL found models with distances."""
        try:
            # Get the list of all model IDs
            all_model_ids = [model_id for model_id in all_results.keys() if model_id != 'unknown']

            if all_model_ids:
                # Process in smaller batches to reduce memory usage
                batch_size = 1
                for i in range(0, len(all_model_ids), batch_size):
                    batch_model_ids = all_model_ids[i:i + batch_size]

                    # Create metadata fetch tasks for each table except model_descriptions
                    for table_name in table_weights.keys():
                        if table_name != 'model_descriptions':
                            try:
                                # Use search to retrieve actual distances
                                result = await self.chroma_manager.search(
                                    collection_name=table_name,
                                    query=query,  # Use the original query
                                    where={"model_id": {"$in": batch_model_ids}},
                                    include=["metadatas", "documents", "distances"]  # Include distances
                                )

                                # Process each result item
                                for idx, item in enumerate(result.get('results', [])):
                                    metadata = item.get('metadata', {})
                                    model_id = metadata.get('model_id', 'unknown')

                                    # Get the distance from the results
                                    distance = self.distance_normalizer.extract_search_distance(
                                        result, idx, item, table_name
                                    )

                                    # Find the corresponding model in our results
                                    if model_id in all_results:
                                        # Initialize metadata dictionary if it doesn't exist
                                        if 'metadata' not in all_results[model_id]:
                                            all_results[model_id]['metadata'] = {}

                                        # Update the metadata with this table's data
                                        all_results[model_id]['metadata'].update(metadata)

                                        # Update the tables list if not already present
                                        if table_name not in all_results[model_id]['tables']:
                                            all_results[model_id]['tables'].append(table_name)

                                        # Make sure table_initial_distances exists before accessing it
                                        if 'table_initial_distances' not in all_results[model_id]:
                                            all_results[model_id]['table_initial_distances'] = {}

                                        # Always store the distance from this search
                                        all_results[model_id]['table_initial_distances'][table_name] = distance
                            except Exception as e:
                                self.logger.error(f"Error searching metadata from {table_name} for batch: {e}")
        except Exception as e:
            self.logger.error(f"Error fetching complete metadata for models: {e}")

        return all_results

    async def _process_model_descriptions_text_search(self, query: str, all_results: Dict[str, Any], search_limit: int=5) -> Dict[str, Any]:
        """Process model descriptions using chunks."""
        for model_id, model_data in all_results.items():
            try:
                if not model_id or model_id == 'unknown':
                    continue

                # Ensure model_data has the required keys
                if 'metadata' not in model_data:
                    model_data['metadata'] = {}

                if 'table_initial_distances' not in model_data:
                    model_data['table_initial_distances'] = {}

                # Initialize description-related fields
                model_data['chunk_descriptions'] = []
                model_data['chunk_description_distances'] = []

                try:
                    # Search for the most relevant chunks using the query
                    model_chunks_search = await self.chroma_manager.search(
                        collection_name="model_descriptions",
                        query=query,  # Use the same query to find the most relevant chunks
                        where={"model_id": {"$eq": model_id}},  # Only search chunks for this specific model
                        limit=search_limit,
                        include=["metadatas", "distances"]  # Include distances
                    )

                    # Process chunk results to collect descriptions and distances
                    chunk_distances = []
                    chunk_descriptions = []

                    if model_chunks_search and isinstance(model_chunks_search,
                                                          dict) and 'results' in model_chunks_search:
                        for chunk_result in model_chunks_search.get('results', []):
                            if not isinstance(chunk_result, dict):
                                continue

                            # Get distance
                            distance = chunk_result.get('distance')
                            # Log the distance value
                            self.logger.debug(f"Distance for description chunk of model {model_id}: {distance}")

                            # Only add valid distances
                            if distance is not None:
                                chunk_distances.append(distance)
                            else:
                                # If no distance is provided, use a moderate default (2.0)
                                chunk_distances.append(2.0)

                            # Get description
                            description = None
                            if 'metadata' in chunk_result and isinstance(chunk_result['metadata'], dict):
                                description = chunk_result['metadata'].get('description')

                            if description and isinstance(description, str):
                                chunk_descriptions.append(description)

                    # Store collected data
                    model_data['chunk_descriptions'] = chunk_descriptions
                    model_data['chunk_description_distances'] = chunk_distances

                    # Create merged description from chunks
                    if chunk_descriptions:
                        model_data['merged_description'] = " ".join(chunk_descriptions)
                        # Calculate average distance for model_descriptions (as requested)
                        if chunk_distances:
                            avg_description_distance = sum(chunk_distances) / len(chunk_distances)
                            self.logger.debug(
                                f"Average description distance for model {model_id}: {avg_description_distance}")
                            model_data['table_initial_distances']['model_descriptions'] = avg_description_distance
                    else:
                        model_data['merged_description'] = ""
                        # Use a moderate default distance if no descriptions found
                        model_data['table_initial_distances']['model_descriptions'] = 2.0

                    # Update metadata with merged description
                    model_data['metadata']['description'] = model_data.get('merged_description', '')

                    # Add to tables list if not already there
                    if 'model_descriptions' not in model_data['tables']:
                        model_data['tables'].append('model_descriptions')

                except Exception as e:
                    self.logger.error(f"Error in chunk description search for model {model_id}: {e}")
                    # Set a moderate default distance if error occurs
                    model_data['table_initial_distances']['model_descriptions'] = 2.0
            except Exception as e:
                self.logger.error(f"Error in model description handling for model: {e}")

        return all_results

    async def _get_collection_distance_stats_for_query(self, query: str, collections: List[str],
                                                       user_id: Optional[str] = None) -> Dict[str, Dict[str, float]]:
        """
        Get min and max distances for each collection based on the actual user query.
        """
        self.logger.info(f"Getting min/max distances for all collections using query: {query}")

        # Initialize result dictionary
        collection_stats = {}

        # For each collection, run the query with a large limit to get distance distribution
        for collection_name in collections:
            try:
                # Get access control filter if needed
                filters = {}
                if self.access_control_manager and user_id:
                    filters = self.access_control_manager.create_access_filter(user_id)

                # Run the query with a large limit to get a good distribution
                result = await self.chroma_manager.search(
                    collection_name=collection_name,
                    query=query,
                    where=filters,  # Apply access control filters only
                    limit=10000,  # Very large limit to get good distribution
                    include=["distances"]
                )

                # Extract all distances
                distances = []
                for idx, item in enumerate(result.get('results', [])):
                    distance = self.distance_normalizer.extract_search_distance(
                        result, idx, item, collection_name
                    )
                    if distance is not None:
                        distances.append(distance)

                # Calculate min and max if we have distances
                if distances:
                    # Sort distances
                    distances.sort()

                    # Get min and max (with some buffer to avoid edge cases)
                    min_distance = distances[0]
                    max_distance = distances[-1]

                    # Log some statistics about the distribution
                    percentile_10 = distances[int(len(distances) * 0.1)] if len(distances) > 10 else min_distance
                    percentile_90 = distances[int(len(distances) * 0.9)] if len(distances) > 10 else max_distance
                    median = distances[int(len(distances) * 0.5)] if len(distances) > 2 else min_distance

                    collection_stats[collection_name] = {
                        'min': min_distance,
                        'max': max_distance,
                        'median': median,
                        'percentile_10': percentile_10,
                        'percentile_90': percentile_90,
                        'count': len(distances)
                    }
                else:
                    # Default values if no distances found
                    collection_stats[collection_name] = {
                        'min': 0.0,
                        'max': 2.0,  # Typical max distance in embeddings
                        'median': 1.0,
                        'percentile_10': 0.5,
                        'percentile_90': 1.5,
                        'count': 0
                    }
                    self.logger.warning(f"No distances found for collection {collection_name}, using defaults")

            except Exception as e:
                self.logger.error(f"Error getting distance stats for collection {collection_name}: {e}")
                # Use default values on error
                collection_stats[collection_name] = {
                    'min': 0.0,
                    'max': 2.0,
                    'median': 1.0,
                    'percentile_10': 0.5,
                    'percentile_90': 1.5,
                    'count': 0
                }

        return collection_stats

    def _calculate_model_distances(self, all_results: Dict[str, Any], table_weights: Dict[str, float],
                                   collection_stats: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Calculate weighted distance sum for all models using normalized distances."""
        self.logger.info(f"Calculating model distances using collection stats")

        for model_id, model_data in all_results.items():
            # Skip special keys
            if not isinstance(model_data, dict) or 'model_id' not in model_data:
                continue

            # Calculate weighted distance from metadata tables
            weighted_sum = 0.0

            # Initialize normalized distances dictionary
            model_data['table_normalized_distances'] = {}

            for table_name, table_weight in table_weights.items():
                if table_name in model_data.get('table_initial_distances', {}):
                    # Get the raw distance
                    raw_distance = model_data['table_initial_distances'][table_name]

                    # Get stats for this table from collection stats
                    table_stats = collection_stats.get(table_name, {
                        'min': 0.0,
                        'max': 2.0,
                        'percentile_10': 0.5,
                        'percentile_90': 1.5
                    })

                    # Normalize the distance using the robust method
                    normalized_distance = self.distance_normalizer.normalize_distance(raw_distance, table_stats)
                else:
                    # Missing table data should be treated as worst possible match
                    normalized_distance = 1.0

                # Store the normalized distance
                model_data['table_normalized_distances'][table_name] = normalized_distance

                # Add to weighted sum
                weighted_sum += normalized_distance * table_weight

                self.logger.debug(
                    f"Model {model_id}, table {table_name}: "
                    f"normalized={normalized_distance}, weight={table_weight}"
                )

            # Since table_weights sum to 1.0, weighted_sum is already the weighted average
            metadata_distance = weighted_sum

            # Normalize chunk distance if it exists
            if 'chunk_initial_distance' in model_data:
                chunk_stats = collection_stats.get('model_scripts_chunks', {
                    'min': 0.0,
                    'max': 2.0,
                    'percentile_10': 0.5,
                    'percentile_90': 1.5
                })

                raw_chunk_distance = model_data['chunk_initial_distance']
                normalized_chunk_distance = self.distance_normalizer.normalize_distance(raw_chunk_distance, chunk_stats)

                model_data['chunk_normalized_distance'] = normalized_chunk_distance

                # Parameters: 0.9, 0.1
                # Rationale: We significantly favor metadata matches (90%) over chunk matches (10%)
                if model_data['match_source'] == 'metadata+chunks':
                    final_distance = 0.9 * metadata_distance + 0.1 * normalized_chunk_distance
                else:  # 'chunks' only
                    final_distance = normalized_chunk_distance
            else:
                final_distance = metadata_distance

            # Store the final calculated distance
            model_data['distance'] = final_distance

            # Also store the metadata_distance for comparison
            model_data['metadata_distance'] = metadata_distance

            # Add raw distance stats for debugging
            model_data['distance_stats'] = {
                'weighted_sum': weighted_sum,
                'weight_sum': 1.0,  # Now always 1.0 since we use full weights
                'metadata_tables_count': len(model_data.get('table_normalized_distances', {})),
                'has_chunks': 'chunk_initial_distance' in model_data
            }

        return all_results

    def _sort_and_limit_search_results(self, all_results: Dict[str, Any], requested_limit: int) -> List[Dict[str, Any]]:
        """Convert to list, sort by distance, and limit to requested number of results."""
        # Convert to list
        output_list = list(all_results.values())

        # Sort by distance (lower is better)
        output_list.sort(key=lambda x: x.get('distance', 2.0))

        # Limit to requested number of results
        return output_list[:requested_limit]

    def _prepare_text_search_items(self, output_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare final result items."""
        items = []
        for rank, model in enumerate(output_list):
            items.append({
                'id': f"model_metadata_{model.get('model_id')}",
                'model_id': model.get('model_id'),
                'metadata': model.get('metadata', {}),
                'rank': rank + 1,
                'match_source': model.get('match_source', 'unknown'),
                'distance': model.get('distance', 2.0),
                'merged_description': model.get('merged_description', '')
            })

        return items

    async def handle_metadata_search(self, query: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a metadata-specific search query across multiple metadata tables.
        1. Query all metadata tables to find model_ids
        2. Fetch complete metadata for ALL found model_ids
        3. Calculate weighted distance sum using complete metadata
        4. Sort ALL results by distance
        5. Limit to requested number of results
        """
        self.logger.debug(f"Handling metadata search: {query}")
        start_time = time.time()

        try:
            # Extract search parameters and apply access control
            user_id, requested_limit, chroma_filters = self._extract_text_search_parameters(parameters)

            # Define metadata tables to search with their weights
            table_weights = self.metadata_table_manager.get_metadata_table_weights()

            # STEP 0: Get global min/max distances for all collections
            collections = list(table_weights.keys()) + ["model_scripts_chunks"]
            collection_stats = await self._get_collection_distance_stats_for_query(query, collections, user_id)
            self.logger.info(f"Collection stats for query: {collection_stats}")

            # STEP 1: Search all metadata tables in parallel to collect model_ids
            metadata_search_start = time.time()
            all_results = await self._search_all_metadata_tables(
                query, chroma_filters, requested_limit, table_weights, user_id
            )

            print(f"Model ids to fetch metadata fields: {len(all_results)}")

            # STEP 2: Fetch complete metadata for ALL found models with distances
            all_results = await self._fetch_complete_model_metadata(
                query, all_results, table_weights, user_id
            )

            # Special handling for model descriptions using chunks - process one model at a time
            all_results = await self._process_model_descriptions_text_search(query, all_results, 5)

            # STEP 3: Calculate weighted distance sum for all models using all gathered metadata
            all_results = self._calculate_model_distances(all_results, table_weights, collection_stats)

            # STEP 4 & 5: Convert to list, sort by distance, and limit results
            output_list = self._sort_and_limit_search_results(all_results, requested_limit)

            # Prepare final result items
            items = self._prepare_text_search_items(output_list)

            # Calculate performance metrics
            metadata_search_time = (time.time() - metadata_search_start) * 1000
            total_time = (time.time() - start_time) * 1000

            performance_metrics = {
                'metadata_search_time_ms': metadata_search_time,
                'total_time_ms': total_time
            }

            return {
                'success': True,
                'type': 'metadata_search',
                'items': items,
                'total_found': len(items),
                'total_models': len(all_results),
                'performance': performance_metrics
            }

        except Exception as e:
            self.logger.error(f"Error in metadata search: {e}", exc_info=True)
            raise

    async def handle_image_search(self, query: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle an image search query by delegating to the ImageSearchManager.

        Args:
            query: The processed query text
            parameters: Dictionary of extracted parameters

        Returns:
            Dictionary containing search results for images
        """
        self.logger.debug(f"Delegating image search to ImageSearchManager: {query}")
        start_time = time.time()

        try:
            # Delegate to the image search manager
            results = await self.image_search_manager.handle_image_search(query, parameters)

            # Add performance metrics if not already included
            if 'performance' not in results:
                results['performance'] = {}

            results['performance']['total_time_ms'] = (time.time() - start_time) * 1000

            # Log performance metrics if analytics available
            if self.analytics and 'query_id' in parameters:
                self.analytics.log_performance_metrics(
                    query_id=parameters['query_id'],
                    total_time_ms=int((time.time() - start_time) * 1000)
                )

            return results

        except Exception as e:
            self.logger.error(f"Error in image search: {e}", exc_info=True)

            # Return a properly structured error response
            return {
                'success': False,
                'error': str(e),
                'type': 'image_search',
                'items': [],
                'total_found': 0,
                'performance': {
                    'total_time_ms': (time.time() - start_time) * 1000
                }
            }

    async def handle_comparison(self, query: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a comparison query for multiple models by first retrieving full metadata
        for each model using the metadata search flow, then performing dimension-based comparisons.
        Returns output in the same format as handle_metadata_search.
        """
        self.logger.debug(f"Handling comparison (metadata search flow): {parameters}")
        start_time = time.time()

        # Extract model IDs to compare
        model_ids = parameters.get('filters', {}).get('model_id', [])
        if not model_ids or len(model_ids) < 2:
            raise ValueError("Comparison requires at least two model IDs")

        # Use metadata search flow to fetch complete metadata for these model_ids
        user_id, _, _ = self._extract_text_search_parameters(parameters)
        table_weights = self.metadata_table_manager.get_metadata_table_weights()
        chroma_filters = {"model_id": {"$in": model_ids}}

        # STEP 1 & 2: Search metadata tables and fetch complete metadata
        all_results = await self._search_all_metadata_tables(
            query="", chroma_filters=chroma_filters,
            requested_limit=len(model_ids), table_weights=table_weights,
            user_id=user_id
        )
        all_results = await self._fetch_complete_model_metadata(
            query="", all_results=all_results,
            table_weights=table_weights, user_id=user_id
        )
        all_results = await self._process_model_descriptions_text_search(
            query="", all_results=all_results, search_limit=5
        )

        # Prepare items list
        items = []
        for rank, model_id in enumerate(model_ids, start=1):
            data = all_results.get(model_id, {})
            metadata = data.get('metadata', {})
            items.append({
                'id': f"model_metadata_{model_id}",
                'model_id': model_id,
                'metadata': metadata,
                'rank': rank,
                'match_source': data.get('match_source', 'unknown'),
                'distance': data.get('distance', 2.0),
                'merged_description': data.get('merged_description', '')
            })

        total_found = len(items)
        total_models = len(all_results)
        total_time_ms = (time.time() - start_time) * 1000

        performance_metrics = {
            'total_time_ms': total_time_ms
        }

        return {
            'success': True,
            'type': 'comparison',
            'items': items,
            'total_found': total_found,
            'total_models': total_models,
            'performance': performance_metrics
        }

    async def handle_comparison_cohort(self, query: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare two free-text cohorts by running multiple independent retrievals
        (metadata_search) and merging their results.
        """
        self.logger.debug(f"Handling comparison (metadata search flow): {parameters}")
        start_time = time.time()

        cohorts = parameters.get("cohorts")
        if not isinstance(cohorts, list):
            raise ValueError("More than one cohorts are required for comparison_cohort")

        base_query = parameters.get("base_query")
        if base_query is None:
            raise ValueError("base_query is required for comparison_cohort")

        # Print for debug
        print(f"base_query: {base_query}")

        # Build sub-queries by appending each cohort name
        queries = [f"{base_query} find models related to {cohort}" for cohort in cohorts]

        # Run existing metadata_search function twice
        results = [await self.handle_metadata_search(query, {
            **parameters, "limit": 1  # Limit the size to control input size to LLM
        }) for query in queries]

        # Tag each result with its cohort
        for i, res in enumerate(results):
            for item in res.get("items", []):
                item["cohort"] = cohorts[i]

        # Merge and sort by distance (lower is more similar)
        combined = [item for res in results for item in res["items"]]
        combined.sort(key=lambda x: x.get("distance", float("inf")))

        total_found = len(combined)
        total_time_ms = (time.time() - start_time) * 1000

        performance_metrics = {
            'total_time_ms': total_time_ms
        }

        return {
            "success": True,
            "type": "comparison_cohort",
            "items": combined,
            'total_found': total_found,
            'performance': performance_metrics
        }

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
                    model_info = await self.model_data_fetcher.fetch_model_metadata(model_id)
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