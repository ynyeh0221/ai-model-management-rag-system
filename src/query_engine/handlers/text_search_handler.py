import time
from typing import Dict, Any, Optional, Tuple

from src.query_engine.handlers.base_search_handler import BaseSearchHandler
from src.query_engine.handlers.utils.distance_normalizer import DistanceNormalizer
from src.query_engine.handlers.utils.filter_translator import FilterTranslator
from src.query_engine.handlers.utils.metadata_table_manager import MetadataTableManager
from src.vector_db_manager.access_control import AccessControlManager
from src.vector_db_manager.chroma_manager import ChromaManager


class TextSearchHandler(BaseSearchHandler):
    def __init__(self, metadata_table_manager: MetadataTableManager, access_control_manager: AccessControlManager, filter_translator: FilterTranslator,
                 chroma_manager: ChromaManager, distance_normalizer: DistanceNormalizer):
        super().__init__(chroma_manager, access_control_manager, filter_translator, distance_normalizer)
        self.metadata_table_manager = metadata_table_manager

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
            all_results = await self._process_model_descriptions_text_search(query, all_results)

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