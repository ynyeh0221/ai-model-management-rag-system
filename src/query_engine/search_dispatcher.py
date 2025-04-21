import asyncio
import logging
import time
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Tuple

from src.query_engine.image_search_manager import ImageSearchManager


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
            QueryIntent.COMPARISON: self.handle_comparison,
            QueryIntent.NOTEBOOK: self.handle_notebook_request,
            QueryIntent.IMAGE_SEARCH: self.handle_image_search,  # Now using image_search_manager
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
            table_weights = self._get_metadata_table_weights()

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

            print(f"Model ids to fetch metadata fields: {len(all_results)}")

            # STEP 3: Fetch complete metadata for ALL found models with distances
            all_results = await self._fetch_complete_model_metadata(
                query, all_results, table_weights, user_id
            )

            # Special handling for model descriptions using chunks - process one model at a time
            all_results = await self._process_model_descriptions_text_search(query, all_results)

            # STEP 4: Calculate weighted distance sum for all models using all gathered metadata
            all_results = self._calculate_model_distances(all_results, table_weights, collection_stats)

            # STEP 5 & 6: Convert to list, sort by distance, and limit results
            output_list = self._sort_and_limit_search_results(all_results, requested_limit)

            # Prepare final result items
            items = self._prepare_text_search_items(output_list)

            # Calculate performance metrics
            metadata_search_time = (time.time() - metadata_search_start) * 1000
            performance_metrics = self._calculate_text_search_performance(
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

    def _get_metadata_table_weights(self) -> Dict[str, float]:
        """Define metadata tables to search with their weights."""
        return {
            "model_descriptions": 0.30,
            "model_architectures": 0.15,
            "model_frameworks": 0.05,
            "model_datasets": 0.15,
            "model_training_configs": 0.15,
            "model_date": 0.15,
            "model_file": 0.05,
            "model_git": 0.0
        }

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
        chroma_filters = self._translate_filters_to_chroma(filters)

        return user_id, requested_limit, chroma_filters

    async def _search_all_metadata_tables(
            self, query: str, chroma_filters: Dict[str, Any], requested_limit: int,
            table_weights: Dict[str, float], user_id: Optional[str]
    ) -> Dict[str, Any]:
        """Search all metadata tables in parallel to collect model_ids."""
        search_tasks = []

        # Create search tasks for all tables
        for table_name in table_weights.keys():
            search_tasks.append(self.chroma_manager.search(
                collection_name=table_name,
                query=query,
                where=chroma_filters,
                limit=requested_limit,
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

    def _extract_search_distance(self, result: Dict[str, Any], idx: int, item: Dict[str, Any],
                                 table_name: str = 'unknown') -> float:
        """Extract distance from search results."""
        distance = None
        model_id = item.get('metadata', {}).get('model_id', 'unknown')

        if 'distances' in result and isinstance(result['distances'], list):
            if len(result['distances']) > idx:
                if isinstance(result['distances'][idx], list) and len(result['distances'][idx]) > 0:
                    distance = result['distances'][idx][0]  # ChromaDB sometimes returns nested lists
                else:
                    distance = result['distances'][idx]
        else:
            # Or try to get it directly from the item
            distance = item.get('distance')

        # Use a default if all else fails
        if distance is None:
            distance = 2.0

        # Log the distance for debugging
        self.logger.debug(f"Distance for model {model_id} in {table_name}: {distance}")

        return distance

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

                                # Log the raw results structure for debugging
                                self.logger.debug(f"Search result structure for {table_name}: {result.keys()}")

                                # Process each result item
                                for idx, item in enumerate(result.get('results', [])):
                                    metadata = item.get('metadata', {})
                                    model_id = metadata.get('model_id', 'unknown')

                                    # Get the distance from the results
                                    distance = self._extract_search_distance(result, idx, item, table_name)

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

    async def _process_model_descriptions_text_search(self, query: str, all_results: Dict[str, Any]) -> Dict[str, Any]:
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
                        limit=1,
                        include=["metadatas", "distances"]  # Include distances
                    )

                    # Log the raw search results to debug distance issues
                    self.logger.debug(f"Description search results for model {model_id}: {model_chunks_search}")

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
        This gets a much larger sample size than the actual search will use.

        Args:
            query: The user's query
            collections: List of collection names to get stats for
            user_id: Optional user ID for access control

        Returns:
            Dictionary with collection names as keys and min/max distance stats as values
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
                    distance = self._extract_search_distance(result, idx, item, collection_name)
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

                    self.logger.info(
                        f"Collection {collection_name} stats: "
                        f"min={min_distance}, max={max_distance}, "
                        f"10th={percentile_10}, median={median}, 90th={percentile_90}, "
                        f"count={len(distances)}"
                    )
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

    def _normalize_distance(self, distance: float, stats: Dict[str, float]) -> float:
        """
        Normalize a distance value using distribution statistics.
        Uses percentiles for more robust normalization.

        Args:
            distance: The distance value to normalize
            stats: Dictionary with distance statistics ('min', 'max', 'percentile_10', 'percentile_90', etc.)

        Returns:
            Normalized distance value in range [0, 1]
        """
        # Get percentile values for more robust normalization
        min_val = stats.get('percentile_10', stats.get('min', 0.0))
        max_val = stats.get('percentile_90', stats.get('max', 2.0))

        # Check for division by zero
        if max_val == min_val:
            return 0.0  # If all distances are the same, normalized distance is 0

        # Normalize to [0, 1] range where 0 is best match and 1 is worst match
        normalized = (distance - min_val) / (max_val - min_val)

        # Ensure the value stays within [0, 1] even in edge cases
        normalized = max(0.0, min(1.0, normalized))

        self.logger.debug(f"Normalized distance: {distance} -> {normalized} (range: {min_val} - {max_val})")

        return normalized

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
            weight_sum = 0.0

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
                    normalized_distance = self._normalize_distance(raw_distance, table_stats)

                    # Store the normalized distance
                    model_data['table_normalized_distances'][table_name] = normalized_distance

                    # Add to weighted sum
                    weighted_sum += normalized_distance * table_weight
                    weight_sum += table_weight

                    self.logger.debug(
                        f"Model {model_id}, table {table_name}: raw={raw_distance}, "
                        f"normalized={normalized_distance}, weight={table_weight}"
                    )

            # Calculate metadata distance (weighted average of normalized distances)
            if weight_sum > 0:
                metadata_distance = weighted_sum / weight_sum
            else:
                metadata_distance = 1.0  # Default to maximum normalized distance (1.0) if no metadata

            # Normalize chunk distance if it exists
            if 'chunk_initial_distance' in model_data:
                chunk_stats = collection_stats.get('model_scripts_chunks', {
                    'min': 0.0,
                    'max': 2.0,
                    'percentile_10': 0.5,
                    'percentile_90': 1.5
                })

                raw_chunk_distance = model_data['chunk_initial_distance']
                normalized_chunk_distance = self._normalize_distance(raw_chunk_distance, chunk_stats)

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
                'weight_sum': weight_sum,
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

    def _calculate_text_search_performance(
            self, start_time: float, metadata_search_time: float, chunks_search_time: float,
            parameters: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate performance metrics."""
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
            'metadata_search_time_ms': metadata_search_time,
            'chunks_search_time_ms': chunks_search_time,
            'total_search_time_ms': total_search_time,
            'total_time_ms': total_time
        }

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
            table_weights = self._get_metadata_table_weights()

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
            all_results = await self._process_model_descriptions_text_search(query, all_results)

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
        table_weights = self._get_metadata_table_weights()
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
            query="", all_results=all_results
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
