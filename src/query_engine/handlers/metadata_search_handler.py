import time
from typing import Dict, Any

from src.query_engine.handlers.base_search_handler import BaseSearchHandler
from src.query_engine.handlers.utils.distance_normalizer import DistanceNormalizer
from src.query_engine.handlers.utils.filter_translator import FilterTranslator
from src.query_engine.handlers.utils.metadata_table_manager import MetadataTableManager


class MetadataSearchHandler(BaseSearchHandler):
    def __init__(self, chroma_manager, filter_translator: FilterTranslator, distance_normalizer: DistanceNormalizer, access_control_manager=None):
        super().__init__(chroma_manager, access_control_manager, filter_translator, distance_normalizer)
        self.chroma_manager = chroma_manager
        self.access_control_manager = access_control_manager
        self.metadata_table_manager = MetadataTableManager(chroma_manager, access_control_manager)

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