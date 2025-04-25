import time
from typing import Dict, Any

from src.query_engine.handlers.base_search_handler import BaseSearchHandler
from src.query_engine.handlers.metadata_search_handler import MetadataSearchHandler
from src.query_engine.handlers.utils.distance_normalizer import DistanceNormalizer
from src.query_engine.handlers.utils.filter_translator import FilterTranslator
from src.query_engine.handlers.utils.metadata_table_manager import MetadataTableManager
from src.vector_db_manager.access_control import AccessControlManager
from src.vector_db_manager.chroma_manager import ChromaManager


class ComparisonHandler(BaseSearchHandler):
    def __init__(self, metadata_table_manager: MetadataTableManager, metadata_search_manager: MetadataSearchHandler,
                 access_control_manager: AccessControlManager, filter_translator: FilterTranslator, chroma_manager: ChromaManager,
                 distance_normalizer: DistanceNormalizer):
        super().__init__(chroma_manager, access_control_manager, filter_translator, distance_normalizer)
        self.metadata_table_manager = metadata_table_manager
        self.metadata_search_manager = metadata_search_manager

    async def handle_comparison(self, query: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        if len(parameters.get("filters", {}).get("model_id") or []) >= 2:
            return await self.handle_comparison_model_id(query, parameters)
        else:
            return await self.handle_comparison_cohort(query, parameters)

    async def handle_comparison_model_id(self, query: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
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
        results = [await self.metadata_search_manager.handle_metadata_search(query, {
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