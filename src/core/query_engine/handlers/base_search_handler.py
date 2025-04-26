import asyncio
import logging
from typing import Dict, Any, Tuple, Optional, List

from src.core.query_engine.handlers.utils.distance_normalizer import DistanceNormalizer
from src.core.query_engine.handlers.utils.filter_translator import FilterTranslator
from src.core.vector_db.access_control import AccessControlManager
from src.core.vector_db.chroma_manager import ChromaManager


class BaseSearchHandler:

    def __init__(self, chroma_manager: ChromaManager, access_control_manager: AccessControlManager, filter_translator: FilterTranslator, distance_normalizer: DistanceNormalizer):
        self.logger = logging.getLogger(__name__)
        self.distance_normalizer = distance_normalizer
        self.chroma_manager = chroma_manager
        self.access_control_manager = access_control_manager
        self.filter_translator = filter_translator

    async def _get_collection_distance_stats_for_query(self, query: str, collections: List[str],
                                                       user_id: Optional[str] = None) -> Dict[
        str, Dict[str, float]]:
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

    def _extract_text_search_parameters(self, parameters: Dict[str, Any]) -> Tuple[
        Optional[str], int, Dict[str, Any]]:
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

    async def _process_model_descriptions_text_search(self, query: str, all_results: Dict[str, Any],
                                                      search_limit: int = 8) -> Dict[str, Any]:
        """Process model descriptions using chunks.

        Args:
            query: The search query string
            all_results: Dictionary containing all model results
            search_limit: Number of records to search (default: 8)

        Returns:
            Updated all_results dictionary with description data

        Note: This method searches for search_limit records but only uses top 5
        most similar records to calculate the distance.
        """
        # Define the max number of chunks to use for distance calculation
        top_chunks_for_distance = 5
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
                    # Note: search_limit could be higher than 5 if passed in
                    model_chunks_search = await self.chroma_manager.search(
                        collection_name="model_descriptions",
                        query=query,  # Use the same query to find the most relevant chunks
                        where={"model_id": {"$eq": model_id}},  # Only search chunks for this specific model
                        limit=search_limit,
                        include=["metadatas", "distances"]  # Include distances
                    )

                    # Process chunk results to collect descriptions and distances
                    chunk_results = []
                    all_chunk_descriptions = []

                    if model_chunks_search and isinstance(model_chunks_search,
                                                          dict) and 'results' in model_chunks_search:
                        for chunk_result in model_chunks_search.get('results', []):
                            if not isinstance(chunk_result, dict):
                                continue

                            # Get distance
                            distance = chunk_result.get('distance')
                            # Log the distance value
                            self.logger.debug(f"Distance for description chunk of model {model_id}: {distance}")

                            # Get description
                            description = None
                            if 'metadata' in chunk_result and isinstance(chunk_result['metadata'], dict):
                                description = chunk_result['metadata'].get('description')

                            # Store all valid descriptions regardless of distance
                            if description and isinstance(description, str):
                                all_chunk_descriptions.append(description)

                                # For distance calculation, only add entries with valid distances
                                if distance is not None:
                                    chunk_results.append({
                                        'distance': distance,
                                        'description': description
                                    })
                                else:
                                    # If no distance is provided, use a moderate default (2.0)
                                    chunk_results.append({
                                        'distance': 2.0,
                                        'description': description
                                    })

                        # Sort by distance (ascending) and take only top chunks for distance calculation
                        chunk_results.sort(key=lambda x: x['distance'])
                        top_chunks = chunk_results[
                                     :top_chunks_for_distance]  # Take only the top most similar chunks for distance

                        # Extract distances from top 5 chunks only
                        chunk_distances = [chunk['distance'] for chunk in top_chunks]

                        # Use all descriptions from the search
                        chunk_descriptions = all_chunk_descriptions

                        # Store collected data
                        model_data['chunk_descriptions'] = chunk_descriptions
                        model_data['chunk_description_distances'] = chunk_distances

                        # Create merged description from all chunks found
                        if chunk_descriptions:
                            model_data['merged_description'] = " ".join(chunk_descriptions)
                            # Calculate average distance using only top 5 chunks
                            if chunk_distances:
                                # Note: chunk_distances contains only the top most similar chunks
                                avg_description_distance = sum(chunk_distances) / len(chunk_distances)
                                self.logger.debug(
                                    f"Average description distance for model {model_id} (using top {top_chunks_for_distance}): {avg_description_distance}")
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
                normalized_chunk_distance = self.distance_normalizer.normalize_distance(raw_chunk_distance,
                                                                                        chunk_stats)

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

    def _sort_and_limit_search_results(self, all_results: Dict[str, Any], requested_limit: int) -> List[
        Dict[str, Any]]:
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