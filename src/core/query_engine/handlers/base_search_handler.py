"""
BaseSearchHandler: Intelligent Vector Database Search Engine
===========================================================

This class implements a sophisticated search system for machine learning model metadata
stored in a vector database (ChromaDB). It provides intelligent query processing with
Named Entity Recognition (NER), multi-table search capabilities, and advanced filtering.

CORE ARCHITECTURE:
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           BaseSearchHandler                                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│ Dependencies:                                                                   │
│ • ChromaManager: Vector DB operations                                           │
│ • AccessControlManager: User permission filtering                               │
│ • FilterTranslator: Query filter transformation                                 │
│ • DistanceNormalizer: Distance score normalization                              │
└─────────────────────────────────────────────────────────────────────────────────┘

SEARCH WORKFLOW:
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐    ┌──────────────┐
│   User      │    │  Parameter   │    │   NER Entity    │    │   Table      │
│   Query     │───▶│  Extraction  │───▶│   Analysis      │───▶│  Selection   │
│             │    │              │    │                 │    │              │
└─────────────┘    └──────────────┘    └─────────────────┘    └──────────────┘
                                                │
                    ┌─────────────────────────────┼─────────────────────────────┐
                    │                             ▼                             │
            ┌───────────────┐              ┌──────────────┐              ┌─────────────┐
            │   Negative    │              │   Positive   │              │   Neutral   │
            │   Entities    │              │   Entities   │              │   Search    │
            │   (Filter     │              │   (Target    │              │   (All      │
            │    Out)       │              │    Tables)   │              │   Tables)   │
            └───────────────┘              └──────────────┘              └─────────────┘
                    │                             │                             │
                    ▼                             ▼                             │
            ┌───────────────┐              ┌──────────────┐                     │
            │   Execute     │              │   Execute    │                     │
            │   Negative    │              │   Positive   │◀────────────────────┘
            │   Searches    │              │   Searches   │
            └───────────────┘              └──────────────┘
                    │                             │
                    └─────────────┬───────────────┘
                                  ▼
                    ┌─────────────────────────────┐
                    │      Filter & Combine       │
                    │        Results              │
                    └─────────────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │   Fetch Complete Metadata   │
                    │   + Process Descriptions    │
                    └─────────────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │    Calculate Normalized     │
                    │    Distances & Rank         │
                    └─────────────────────────────┘

KEY COMPONENTS AND LOGIC:

1. STATISTICS COLLECTION (_get_collection_distance_stats_for_query):
   - Samples up to 10,000 records per collection using the actual user query
   - Computes min/max/median/percentiles for distance normalization
   - Handles access control filtering during statistics gathering

2. NER ENTITY ANALYSIS (_analyze_ner_filters):
   - Processes three entity types: architecture, dataset, training_config
   - Distinguishes between positive entities (target what to find) and negative entities (filter out)
   - Maps entities to their corresponding database tables:
     * architecture → model_architectures
     * dataset → model_datasets  
     * training_config → model_training_configs

3. SEARCH STRATEGY (_search_all_metadata_tables):
   Four distinct search scenarios based on NER analysis:
   
   a) No NER entities: Search all tables with original query
   b) Only positive entities: Search specific tables using entity values as queries
   c) Only negative entities: Search all tables, then filter out negative matches
   d) Mixed entities: Search positive tables with entity values, filter negative matches

4. POSITIVE ENTITY PROCESSING:
   - Uses entity values as specialized search queries (e.g., "ResNet" for architecture)
   - Increases search limits for NER queries (3x multiplier)
   - Supports intersection logic when multiple entity types are present

5. NEGATIVE ENTITY FILTERING:
   - Executes separate searches for negative entities
   - Filters out models with similarity distance < 1.3 to negative entities
   - Provides detailed filtering reasons for debugging

6. METADATA COMPLETION (_fetch_complete_model_metadata):
   - Processes models in small batches to reduce memory usage
   - Fetches complete metadata from all relevant tables
   - Merges distance scores and metadata across tables

7. DESCRIPTION PROCESSING (_process_model_descriptions_text_search):
   - Handles chunked text descriptions stored separately
   - Searches for top 10 chunks but uses only top 5 for distance calculation
   - Reconstructs full descriptions by sorting chunks by offset
   - Computes average distance across relevant chunks

8. DISTANCE NORMALIZATION & RANKING (_calculate_model_distances):
   - Uses collection statistics to normalize distances (0.0 = perfect match, 1.0 = worst)
   - Applies weighted combination across different metadata tables
   - Combines metadata distance (90%) with chunk distance (10%) when both available
   - Handles missing data by assigning worst-case scores (1.0)

DATA STRUCTURES:

Table Weights Dictionary:
{
    "model_architectures": 0.3,
    "model_datasets": 0.25, 
    "model_training_configs": 0.25,
    "model_descriptions": 0.15,
    "model_date": 0.05
}

Result Structure:
{
    "model_id_123": {
        "model_id": "123",
        "metadata": {...},
        "tables": ["model_architectures", "model_datasets"],
        "table_initial_distances": {"model_architectures": 0.45, ...},
        "table_normalized_distances": {"model_architectures": 0.23, ...},
        "distance": 0.67,
        "match_source": "metadata",
        "filter_info": {
            "used_ner_query": True,
            "applied_negative_filter": False
        }
    }
}

ACCESS CONTROL:
- Integrates with AccessControlManager for user-based filtering
- Applies access filters at multiple stages (statistics, search, results)
- Supports hierarchical permission checking

ERROR HANDLING:
- Graceful degradation with default statistics when collections fail
- Batch processing with individual error isolation
- Comprehensive logging for debugging and monitoring

PERFORMANCE OPTIMIZATIONS:
- Parallel execution of multiple table searches using asyncio.gather()
- Batch processing for metadata completion
- Configurable search limits based on table importance
- Early filtering to reduce downstream processing

This design enables sophisticated semantic search with intelligent query understanding,
multi-faceted filtering, and robust ranking across heterogeneous metadata sources.
"""
import asyncio
import logging
from typing import Dict, Any, Tuple, Optional, List, Coroutine, Iterable

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

    async def _get_collection_distance_stats_for_query(
            self,
            query: str,
            collections: List[str],
            user_id: Optional[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Get min, max, median, and percentiles for each collection based on the actual user query.
        """
        self.logger.info(f"Getting min/max distances for all collections using query: {query}")

        # Build an access-control filter once (it does not change per collection)
        filters = {}
        if self.access_control_manager and user_id:
            filters = self.access_control_manager.create_access_filter(user_id)

        collection_stats: Dict[str, Dict[str, float]] = {}
        for collection_name in collections:
            stats = await self._get_stats_for_collection(collection_name, query, filters)
            collection_stats[collection_name] = stats

        return collection_stats

    async def _get_stats_for_collection(
            self,
            collection_name: str,
            query: str,
            filters: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        1) Runs the chroma_manager.search(...) call for up to 10,000 hits.
        2) Extracts distances (or defaults).
        3) Computes min/max/median/percentiles if any distances exist,
           otherwise returns default stats.
        4) Catches any exceptions and returns default stats on error.
        """
        try:
            result = await self.chroma_manager.search(
                collection_name=collection_name,
                query=query,
                where=filters,
                limit=10000,
                include=["distances"]
            )
            distances = self._extract_distances(result, collection_name)

            if not distances:
                self.logger.warning(
                    f"No distances found for collection {collection_name}, using defaults"
                )
                return self._default_stats()

            return self._compute_stats(distances)

        except Exception as e:
            self.logger.error(
                f"Error getting distance stats for collection {collection_name}: {e}"
            )
            return self._default_stats()

    def _extract_distances(
            self,
            search_result: Dict[str, Any],
            collection_name: str
    ) -> List[float]:
        """
        Walks through search_result['results'], calling distance_normalizer
        on each item. Returns a list of all non‐None distances.
        """
        distances: List[float] = []
        for idx, item in enumerate(search_result.get("results", [])):
            distance = self.distance_normalizer.extract_search_distance(
                search_result, idx, item, collection_name
            )
            if distance is not None:
                distances.append(distance)
        return distances

    def _compute_stats(self, distances: List[float]) -> Dict[str, float]:
        """
        Given a nonempty list of distances, sorts them and returns a dict with:
          - min, max, median
          - 10th percentile, 90th percentile
          - count
        """
        distances.sort()
        n = len(distances)

        min_distance = distances[0]
        max_distance = distances[-1]
        median = distances[int(n * 0.5)] if n > 2 else min_distance
        percentile_10 = distances[int(n * 0.1)] if n > 10 else min_distance
        percentile_90 = distances[int(n * 0.9)] if n > 10 else max_distance

        return {
            "min": min_distance,
            "max": max_distance,
            "median": median,
            "percentile_10": percentile_10,
            "percentile_90": percentile_90,
            "count": float(n)
        }

    def _default_stats(self) -> Dict[str, float]:
        """
        Returns a reasonable default stats dict when no distances or on error.
        """
        return {
            "min": 0.0,
            "max": 2.0,
            "median": 1.0,
            "percentile_10": 0.5,
            "percentile_90": 1.5,
            "count": 0.0
        }

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
            table_weights: Dict[str, float], user_id: Optional[str], ner_filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Search metadata tables based on NER analysis of a user query.

        This method handles four different cases based on NER entities:
        1. No NER entities: Search all tables with original query
        2. Only positive entities: Search only tables related to positive entities using entity values as queries
        3. Only negative entities: Search all tables with the original query, then filter out negative matches
        4. Both positive and negative entities: Search positive entity tables with entity values, filter out negative matches

        Args:
            query: The original user query
            chroma_filters: Database filters in Chroma format
            requested_limit: Number of results to return
            table_weights: Weights for each table
            user_id: User ID for access control
            ner_filters: Named entity filters extracted from a query

        Returns:
            Dictionary of search results
        """
        # Initialize result dictionary
        all_tables = list(table_weights.keys())

        # For debugging
        self.logger.info(f"Raw NER filters: {ner_filters}")

        # Step 1: Analyze NER filters and determine positive/negative entities and their query values
        entity_analysis_result = self._analyze_ner_filters(ner_filters, table_weights)

        has_positive_entities = entity_analysis_result["has_positive_entities"]
        has_negative_entities = entity_analysis_result["has_negative_entities"]
        positive_table_queries = entity_analysis_result["positive_table_queries"]
        entity_type_tables = entity_analysis_result["entity_type_tables"]
        negative_entities = entity_analysis_result["negative_entities"]

        # Step 2: Determine which tables to search based on positive entities
        tables_to_search = self._determine_tables_to_search(
            has_positive_entities, entity_type_tables, chroma_filters, all_tables
        )

        # Log the analysis results
        self.logger.info(f"NER analysis: positive={has_positive_entities}, negative={has_negative_entities}")
        self.logger.info(f"Tables to search: {tables_to_search}")
        self.logger.info(f"Table queries: {positive_table_queries}")
        self.logger.info(f"Negative entities to filter: {negative_entities}")

        # Step 3: Execute searches for positive NER fields
        positive_search_results = await self._execute_positive_searches(
            tables_to_search, has_positive_entities, positive_table_queries,
            query, requested_limit, chroma_filters
        )

        # Step 4: Execute searches for negative NER fields
        negative_results = {}
        if has_negative_entities:
            negative_results = await self._execute_negative_searches(
                negative_entities, requested_limit
            )

        print(f"Positive results: {positive_search_results}")
        print(f"Negative results: {negative_results}")

        # Step 5: Process search results and apply filters
        all_results = self._process_search_results(
            positive_search_results, tables_to_search, entity_type_tables,
            has_positive_entities, has_negative_entities, negative_results,
            user_id
        )

        # Log summary of results
        self.logger.info(f"Found {len(all_results)} models after NER filtering")
        return all_results

    def _analyze_ner_filters(
            self,
            ner_filters: Optional[Dict[str, Any]],
            table_weights: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Analyze NER filters to determine positive and negative entities.

        Args:
            ner_filters: Named entity filters extracted from query
            table_weights: Weights for each table

        Returns:
            Dictionary containing analysis results
        """
        # Initialize result
        result = {
            "has_positive_entities": False,
            "has_negative_entities": False,
            "positive_table_queries": {},  # {table_name: query_value}
            "entity_type_tables": {},  # {entity_type: set(tables)}
            "negative_entities": {}
        }

        # Table mappings for each entity type
        entity_table_mapping = {
            "architecture": ["model_architectures"],
            "dataset": ["model_datasets"],
            "training_config": ["model_training_configs"],
        }

        # Early return if no filters
        if not ner_filters:
            return result

        self.logger.info(f"Processing search with NER filters: {ner_filters}")

        # 1) Process single‐value entities (architecture and dataset)
        for entity_type in ("architecture", "dataset"):
            self._process_if_present(
                entity_type=entity_type,
                ner_filters=ner_filters,
                entity_table_mapping=entity_table_mapping,
                table_weights=table_weights,
                result=result,
            )

        # 2) Process training_config in its own helper
        self._process_training_config(
            training_config=ner_filters.get("training_config"),
            entity_table_mapping=entity_table_mapping,
            table_weights=table_weights,
            result=result,
        )

        return result

    def _process_if_present(
            self,
            entity_type: str,
            ner_filters: Dict[str, Any],
            entity_table_mapping: Dict[str, Any],
            table_weights: Dict[str, float],
            result: Dict[str, Any],
    ) -> None:
        """
        If `entity_type` exists in ner_filters, call the original
        _process_single_entity for that type.
        """
        if entity_type not in ner_filters:
            return

        # Delegate to the existing single‐entity handler
        self._process_single_entity(
            entity_type=entity_type,
            entity_data=ner_filters[entity_type],
            entity_table_mapping=entity_table_mapping,
            table_weights=table_weights,
            result=result,
        )

    def _process_training_config(
            self,
            training_config: Optional[Dict[str, Any]],
            entity_table_mapping: Dict[str, Any],
            table_weights: Dict[str, float],
            result: Dict[str, Any],
    ) -> None:
        """
        Handle any fields under 'training_config'. Sets positive vs.
        negative flags, builds positive_table_queries, and populates
        entity_type_tables["training_config"] if needed.
        """
        if not isinstance(training_config, dict):
            return

        seen_tables = set()
        for field_name, data in training_config.items():
            self._process_single_training_field(
                field_name, data, entity_table_mapping, table_weights, result, seen_tables
            )

        if seen_tables:
            result["entity_type_tables"]["training_config"] = seen_tables

    def _process_single_training_field(
            self,
            field_name: str,
            data: Any,
            entity_table_mapping: Dict[str, Any],
            table_weights: Dict[str, float],
            result: Dict[str, Any],
            seen_tables: set,
    ) -> None:
        """
        Process one field under training_config:
          - Skip if no valid value
          - If positive, mark flags and add to positive_table_queries
          - If negative, add to negative_entities
        """
        if not (isinstance(data, dict) and data.get("value") and data["value"] != "N/A"):
            return

        field_value = data["value"]
        is_positive = data.get("is_positive", True)
        self.logger.info(f"Training config {field_name}: {field_value}, positive: {is_positive}")

        if is_positive:
            result["has_positive_entities"] = True
            self._add_positive_training_tables(
                field_name, field_value, entity_table_mapping, table_weights, result, seen_tables
            )
        else:
            result["has_negative_entities"] = True
            result["negative_entities"].setdefault("training_config", {})[field_name] = field_value

    def _add_positive_training_tables(
            self,
            field_name: str,
            field_value: str,
            entity_table_mapping: Dict[str, Any],
            table_weights: Dict[str, float],
            result: Dict[str, Any],
            seen_tables: set,
    ) -> None:
        """
        For a positive training field, attach the query to each valid table
        exactly once, updating seen_tables and positive_table_queries.
        """
        for table in entity_table_mapping.get("training_config", []):
            if table in table_weights and table not in seen_tables:
                seen_tables.add(table)
                query_value = f"{field_name} {field_value}"
                result["positive_table_queries"][table] = query_value

    def _process_single_entity(
            self,
            entity_type: str,
            entity_data: Dict[str, Any],
            entity_table_mapping: Dict[str, List[str]],
            table_weights: Dict[str, float],
            result: Dict[str, Any],
    ) -> None:
        """
        Process a single-value entity type (architecture or dataset).

        Args:
            entity_type: Type of the entity (architecture or dataset)
            entity_data: Entity data from ner_filters
            entity_table_mapping: Mapping of entity types to tables
            table_weights: Weights for each table
            result: Result dictionary to update
        """
        # Early exit if no usable value
        if not (isinstance(entity_data, dict) and "value" in entity_data):
            return

        entity_value = entity_data["value"]
        if entity_value == "N/A":
            return

        is_positive = entity_data.get("is_positive", True)
        if is_positive:
            self._handle_positive_single_entity(
                entity_type, entity_value, entity_table_mapping, table_weights, result
            )
        else:
            result["has_negative_entities"] = True
            result["negative_entities"][entity_type] = entity_value

    def _handle_positive_single_entity(
            self,
            entity_type: str,
            entity_value: str,
            entity_table_mapping: Dict[str, List[str]],
            table_weights: Dict[str, float],
            result: Dict[str, Any],
    ) -> None:
        """
        Handles the positive‐entity case: logs, sets flags, and builds
        positive_table_queries + entity_type_tables entries.
        """
        self.logger.info(f"Found positive {entity_type} entity: {entity_value}")
        result["has_positive_entities"] = True

        # Filter only tables present in table_weights
        valid_tables = [
            tbl for tbl in entity_table_mapping.get(entity_type, [])
            if tbl in table_weights
        ]

        for tbl in valid_tables:
            result["positive_table_queries"][tbl] = entity_value

        if valid_tables:
            result["entity_type_tables"][entity_type] = set(valid_tables)

    def _determine_tables_to_search(
            self, has_positive_entities: bool, entity_type_tables: Dict[str, set],
            chroma_filters: Dict[str, Any], all_tables: List[str]
    ) -> List[str]:
        """Determine which tables to search based on positive entities.

        Args:
            has_positive_entities: Whether positive entities were found
            entity_type_tables: Mapping of entity types to their corresponding tables
            chroma_filters: Database filters in Chroma format
            all_tables: List of all available tables

        Returns:
            List of tables to search
        """
        tables_to_search = all_tables.copy()

        # Determine which tables to search based on positive entities
        if has_positive_entities and entity_type_tables:
            # Track which entity types have valid tables
            valid_entity_types = []
            tables_to_search = []

            # Log the tables matched for each entity type
            for entity_type, tables in entity_type_tables.items():
                self.logger.info(f"Entity type {entity_type} matches tables: {tables}")
                if tables:
                    valid_entity_types.append(entity_type)
                    # Add these tables to our search list
                    tables_to_search.extend(tables)

            # If there are date filters, add model_date to tables_to_search
            if 'created_month' in chroma_filters or 'created_year' in chroma_filters:
                tables_to_search.append('model_date')

            # Remove any duplicates
            tables_to_search = list(set(tables_to_search))
            self.logger.info(f"Tables to search based on positive entities: {tables_to_search}")

        # If no tables to search (could happen with incompatible positive entities),
        # use all tables as fallback
        if not tables_to_search:
            self.logger.warning("No tables matched positive entity criteria, using all tables")
            tables_to_search = all_tables

        return tables_to_search

    async def _execute_positive_searches(
            self, tables_to_search: List[str], has_positive_entities: bool,
            positive_table_queries: Dict[str, str], query: str,
            requested_limit: int, chroma_filters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Execute searches for positive NER fields.

        Args:
            tables_to_search: List of tables to search
            has_positive_entities: Whether positive entities were found
            positive_table_queries: Mapping of tables to their query values
            query: The original user query
            requested_limit: Number of results to return
            chroma_filters: Database filters in Chroma format

        Returns:
            List of search results
        """
        positive_search_tasks = []
        for table_name in tables_to_search:
            # Set base search limit
            search_limit = (
                requested_limit * 40
                if table_name == "model_descriptions"
                else requested_limit * 3
                if table_name == "model_date"
                else requested_limit
            )

            # Determine which query to use for this table
            if has_positive_entities and table_name in positive_table_queries:
                # Use the entity value as the query
                table_query = positive_table_queries[table_name]
                # Increase the limit by 5x for NER queries
                search_limit = search_limit * 3
                self.logger.info(f"Searching table {table_name} with NER query: {table_query} (limit: {search_limit})")
            else:
                # Use the original query if no positive entity for this table
                table_query = query
                self.logger.info(
                    f"Searching table {table_name} with original query: {table_query} (limit: {search_limit})")

            positive_search_tasks.append(self.chroma_manager.search(
                collection_name=table_name,
                query=table_query,
                where=chroma_filters,
                limit=search_limit,
                include=["metadatas", "documents", "distances"]
            ))

        # Execute all positive searches in parallel
        return await asyncio.gather(*positive_search_tasks)

    async def _execute_negative_searches(
            self,
            negative_entities: Dict[str, Any],
            requested_limit: int
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Execute searches for negative NER fields.

        Args:
            negative_entities: Mapping of negative entity types to their values
            requested_limit: Number of results to return

        Returns:
            Dictionary of negative search results, keyed by table (or table_field)
        """
        # 1) Build up all search tasks and keep track of which a query maps to which key
        negative_table_queries, negative_search_tasks = self._build_negative_search_tasks(
            negative_entities, requested_limit
        )

        # 2) If there are no tasks, just return an empty dict immediately
        if not negative_search_tasks:
            return {}

        # 3) Await all of them in parallel
        negative_search_results = await asyncio.gather(*negative_search_tasks)

        # 4) Process the raw returns into the final structure
        return self._process_negative_search_results(
            negative_search_results, negative_table_queries
        )

    def _build_negative_search_tasks(
            self,
            negative_entities: Dict[str, Any],
            requested_limit: int
    ) -> Tuple[Dict[str, str], List[Coroutine[Any, Any, Dict[str, Any]]]]:
        """
        Walk through `negative_entities` and, for each field, call
        `_prepare_negative_search(...)` to generate:
          - A new entry in negative_table_queries mapping “table” (or “table_field”) → query_value
          - A corresponding search‐task coroutine in negative_search_tasks

        Returns:
            (negative_table_queries, negative_search_tasks)
            - negative_table_queries: {table_or_table_field: query_value}
            - negative_search_tasks: [<coroutine for chroma_manager.search>, ...]
        """
        negative_table_queries: Dict[str, str] = {}
        negative_search_tasks: List[Coroutine[Any, Any, Dict[str, Any]]] = []

        entity_table_mapping = {
            "architecture": "model_architectures",
            "dataset": "model_datasets",
            "training_config": "model_training_configs",
        }

        for entity_type, value in negative_entities.items():
            if entity_type in ("architecture", "dataset"):
                table = entity_table_mapping[entity_type]
                # Single‐value entity: the `value` is the direct query string
                self._prepare_negative_search(
                    table=table,
                    query_value=value,
                    negative_table_queries=negative_table_queries,
                    negative_search_tasks=negative_search_tasks,
                    requested_limit=requested_limit
                )

            elif entity_type == "training_config":
                table = entity_table_mapping[entity_type]
                # Multi‐field: each field→value pair becomes its own query
                for field, field_value in value.items():
                    if field_value == "N/A":
                        continue
                    query_value = f"{field} {field_value}"
                    table_key = f"{table}_{field}"
                    self._prepare_negative_search(
                        table=table,
                        query_value=query_value,
                        negative_table_queries=negative_table_queries,
                        negative_search_tasks=negative_search_tasks,
                        requested_limit=requested_limit,
                        table_key=table_key
                    )

        return negative_table_queries, negative_search_tasks

    def _process_negative_search_results(
            self,
            negative_search_results: List[Dict[str, Any]],
            negative_table_queries: Dict[str, str]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Given the list of raw results (in the same order as negative_table_queries.items()),
        extract `model_id`, `distance`, and `metadata` for each hit—skipping "unknown" ID—and
        assemble a final dictionary keyed by table (or table_field).

        Returns:
            negative_results: {
                table_or_table_field: [
                    {'model_id': ..., 'distance': ..., 'metadata': {...}},
                    ...
                ],
                ...
            }
        """
        negative_results: Dict[str, List[Dict[str, Any]]] = {}

        # Iterate in parallel over (table_key, query_value) and its corresponding search result
        for idx, (table_key, query_value) in enumerate(negative_table_queries.items()):
            if idx >= len(negative_search_results):
                break  # safeguard if something mismatches

            result = negative_search_results[idx]
            negative_results[table_key] = []

            for hit_idx, item in enumerate(result.get("results", [])):
                metadata = item.get("metadata", {})
                model_id = metadata.get("model_id", "unknown")
                if model_id == "unknown":
                    continue

                # Determine which base table to pass into distance_normalizer
                base_table = table_key.split("_")[0] if "_" in table_key else table_key
                distance = self.distance_normalizer.extract_search_distance(
                    result, hit_idx, item, base_table
                )

                negative_results[table_key].append({
                    "model_id": model_id,
                    "distance": distance,
                    "metadata": metadata
                })

                self.logger.info(
                    f"Negative result for {table_key}: model_id={model_id}, distance={distance:.4f}"
                )

        return negative_results

    def _prepare_negative_search(
            self, table: str, query_value: str,
            negative_table_queries: Dict[str, str],
            negative_search_tasks: List[Coroutine],
            requested_limit: int,
            table_key: str = None
    ) -> None:
        """Prepare a negative search task.

        Args:
            table: Collection name to search
            query_value: Query value for search
            negative_table_queries: Dictionary to store table and query mapping
            negative_search_tasks: List to store search tasks
            requested_limit: Number of results to return
            table_key: Optional custom key for the table (used for training_config fields)
        """
        key = table_key or table
        negative_table_queries[key] = query_value
        negative_search_tasks.append(self.chroma_manager.search(
            collection_name=table,
            query=query_value,
            where={},
            limit=requested_limit * 3,  # Use higher limit for negative queries
            include=["metadatas", "documents", "distances"]
        ))

    def _process_search_results(
            self,
            positive_search_results: List[Dict[str, Any]],
            tables_to_search: List[str],
            entity_type_tables: Dict[str, set],
            has_positive_entities: bool,
            has_negative_entities: bool,
            negative_results: Dict[str, List[Dict[str, Any]]],
            user_id: Optional[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Process search results and apply filters.

        Args:
            positive_search_results: Results from positive entity searches
            tables_to_search: List of tables that were searched
            entity_type_tables: Mapping of entity types to their corresponding tables
            has_positive_entities: Whether positive entities were found
            has_negative_entities: Whether negative entities were found
            negative_results: Results from negative entity searches
            user_id: User ID for access control

        Returns:
            Dictionary of filtered search results
        """
        all_results: Dict[str, Dict[str, Any]] = {}

        # Initialize tracking for models by entity type (for intersection)
        models_by_entity_type = self._initialize_entity_type_tracking(
            has_positive_entities, entity_type_tables
        )

        # Process each table’s positive search hits via a helper
        for table_idx, result in enumerate(positive_search_results):
            table_name = tables_to_search[table_idx]
            matching_entity_type = self._find_matching_entity_type(
                table_name, entity_type_tables
            )

            self._process_positive_hits(
                result=result,
                table_name=table_name,
                entity_type=matching_entity_type,
                has_positive_entities=has_positive_entities,
                has_negative_entities=has_negative_entities,
                negative_results=negative_results,
                user_id=user_id,
                all_results=all_results,
                models_by_entity_type=models_by_entity_type,
                entity_type_tables=entity_type_tables
            )

        # If multiple positive entity types exist, enforce intersection
        if has_positive_entities and len(models_by_entity_type) > 1:
            all_results = self._apply_intersection_filtering(
                all_results, models_by_entity_type
            )

        return all_results

    def _process_positive_hits(
            self,
            result: Dict[str, Any],
            table_name: str,
            entity_type: Optional[str],
            has_positive_entities: bool,
            has_negative_entities: bool,
            negative_results: Dict[str, List[Dict[str, Any]]],
            user_id: Optional[str],
            all_results: Dict[str, Dict[str, Any]],
            models_by_entity_type: Dict[str, set],
            entity_type_tables: Dict[str, set]
    ) -> None:
        """
        Iterate over a single table’s search results. For each hit:
          - Skip if model_id is 'unknown'
          - Skip if access control fails
          - Track for intersection if needed
          - Filter out by negative entities if applicable
          - Add to all_results otherwise
        """
        for idx, item in enumerate(result.get("results", [])):
            metadata = item.get("metadata", {})
            model_id = metadata.get("model_id", "unknown")
            if model_id == "unknown":
                continue

            # Apply access control
            if not self._check_access_control(metadata, user_id):
                continue

            # Track model_id for intersection logic if positive entities exist
            self._track_model_for_entity_type(
                has_positive_entities,
                entity_type,
                models_by_entity_type,
                model_id,
                entity_type_tables
            )

            # Extract distance for this hit
            distance = self.distance_normalizer.extract_search_distance(
                result, idx, item, table_name
            )

            # If there are negative entities, check filtering
            if has_negative_entities and negative_results:
                skip, reason = self._should_filter_by_negative_entities(
                    model_id, negative_results
                )
                if skip:
                    self.logger.info(
                        f"Filtering out model {model_id} due to {reason}"
                    )
                    continue

            # If not filtered, add this hit into all_results
            self._add_result_to_all_results(
                all_results,
                model_id,
                table_name,
                distance,
                has_positive_entities,
                has_negative_entities
            )

    @staticmethod
    def _initialize_entity_type_tracking(
            has_positive_entities: bool, entity_type_tables: Dict[str, set]
    ) -> Dict[str, set]:
        """Initialize tracking for models by entity type.

        Args:
            has_positive_entities: Whether positive entities were found
            entity_type_tables: Mapping of entity types to their corresponding tables

        Returns:
            Dictionary to track models by entity type
        """
        models_by_entity_type = {}
        if has_positive_entities and len(entity_type_tables) > 1:
            # Only initialize tracking when we have multiple entity types
            for entity_type in entity_type_tables.keys():
                models_by_entity_type[entity_type] = set()
        return models_by_entity_type

    @staticmethod
    def _find_matching_entity_type(
            table_name: str, entity_type_tables: Dict[str, set]
    ) -> Optional[str]:
        """Find the entity type that matches the given table.

        Args:
            table_name: Name of the table
            entity_type_tables: Mapping of entity types to their corresponding tables

        Returns:
            Matching entity type or None
        """
        for entity_type, tables in entity_type_tables.items():
            if table_name in tables:
                return entity_type
        return None

    def _check_access_control(
            self, metadata: Dict[str, Any], user_id: Optional[str]
    ) -> bool:
        """Check if the user has access to the model.

        Args:
            metadata: Model metadata
            user_id: User ID for access control

        Returns:
            True if the user has access, False otherwise
        """
        if self.access_control_manager and user_id:
            return self.access_control_manager.check_access({'metadata': metadata}, user_id, "view")
        return True

    @staticmethod
    def _track_model_for_entity_type(
            has_positive_entities: bool, matching_entity_type: Optional[str],
            models_by_entity_type: Dict[str, set], model_id: str,
            entity_type_tables: Dict[str, set]
    ) -> None:
        """Track model for entity type intersection.

        Args:
            has_positive_entities: Whether positive entities were found
            matching_entity_type: Entity type that matches the current table
            models_by_entity_type: Dictionary tracking models by entity type
            model_id: ID of the current model
            entity_type_tables: Mapping of entity types to their corresponding tables
        """
        if has_positive_entities and matching_entity_type and len(entity_type_tables) > 1:
            if matching_entity_type in models_by_entity_type:  # Safety check
                models_by_entity_type[matching_entity_type].add(model_id)

    @staticmethod
    def _should_filter_by_negative_entities(
            model_id: str,
            negative_results: Dict[str, List[Dict[str, Any]]]
    ) -> Tuple[bool, str]:
        """
        Check if a model should be filtered out based on negative entities.

        Args:
            model_id: ID of the model to check
            negative_results: Results from negative entity searches

        Returns:
            Tuple of (should_filter, filter_reason)
        """
        negative_distance_threshold = 1.3  # Lower distance means better match

        for neg_table, neg_items in negative_results.items():
            # Gather any items matching this model_id
            matching = [item for item in neg_items if item["model_id"] == model_id]
            if not matching:
                continue

            # Find the best (smallest) distance among matches
            min_distance = min(item["distance"] for item in matching)
            if min_distance < negative_distance_threshold:
                reason = BaseSearchHandler._build_negative_filter_reason(neg_table, min_distance)
                return True, reason

        return False, ""

    @staticmethod
    def _build_negative_filter_reason(table_key: str, distance: float) -> str:
        """
        Construct a human-readable filter_reason based on which negative table_key
        matched and the corresponding distance.
        """
        base = table_key.split("_")[0] if "_" in table_key else table_key
        dist_str = f"{distance:.4f}"

        if base == "model_architectures":
            return f"negative architecture match (distance: {dist_str})"
        if base == "model_datasets":
            return f"negative dataset match (distance: {dist_str})"

        # Otherwise, assume a training_config field (table_key is like "model_training_configs_field")
        field_name = table_key.split("_")[-1]
        return f"negative training config match: {field_name} (distance: {dist_str})"

    @staticmethod
    def _add_result_to_all_results(
            all_results: Dict[str, Dict[str, Any]], model_id: str,
            table_name: str, distance: float, has_positive_entities: bool,
            has_negative_entities: bool
    ) -> None:
        """Add a result to the all_results dictionary.

        Args:
            all_results: Dictionary of all results
            model_id: ID of the model
            table_name: Name of the table
            distance: Distance score for this result
            has_positive_entities: Whether positive entities were found
            has_negative_entities: Whether negative entities were found
        """
        # If first time seeing this model, initialize its entry
        if model_id not in all_results:
            all_results[model_id] = {
                'model_id': model_id,
                'tables': [],
                'table_initial_distances': {},
                'match_source': 'metadata',
                'filter_info': {
                    'used_ner_query': has_positive_entities,
                    'applied_negative_filter': has_negative_entities
                }
            }

        # Add this table to the list of matching tables if not already there
        if table_name not in all_results[model_id]['tables']:
            all_results[model_id]['tables'].append(table_name)

        # Store distance in table_initial_distances
        if 'table_initial_distances' not in all_results[model_id]:
            all_results[model_id]['table_initial_distances'] = {}

        all_results[model_id]['table_initial_distances'][table_name] = distance

    def _apply_intersection_filtering(
            self, all_results: Dict[str, Dict[str, Any]],
            models_by_entity_type: Dict[str, set]
    ) -> Dict[str, Dict[str, Any]]:
        """Apply intersection filtering for models that match ALL entity types.

        Args:
            all_results: Dictionary of all results
            models_by_entity_type: Dictionary tracking models by entity type

        Returns:
            Filtered dictionary of results
        """
        # Check that all entity types have at least one model
        if all(len(models) > 0 for models in models_by_entity_type.values()):
            valid_models = set.intersection(*models_by_entity_type.values())
            self.logger.info(f"Models matching ALL positive entity criteria: {valid_models}")

            # Filter all_results to only include models in the intersection
            return {model_id: data for model_id, data in all_results.items()
                    if model_id in valid_models}

        return all_results

    async def _fetch_complete_model_metadata(
            self,
            query: str,
            all_results: Dict[str, Any],
            table_weights: Dict[str, float],
            user_id: Optional[str],
    ) -> Dict[str, Any]:
        """
        Fetch complete metadata for ALL found models with distances.
        """
        # 1) Collect all model IDs except 'unknown'
        all_model_ids = [mid for mid in all_results.keys() if mid != "unknown"]

        # 2) Early exit if no valid IDs
        if not all_model_ids:
            return all_results

        # 3) Process in small batches to reduce memory usage
        batch_size = 1
        for batch_ids in self._chunk_model_ids(all_model_ids, batch_size):
            await self._process_batch(batch_ids, query, table_weights, all_results)

        return all_results

    def _chunk_model_ids(
            self, all_model_ids: List[str], batch_size: int
    ) -> Iterable[List[str]]:
        """
        Yield consecutive slices of size `batch_size` from all_model_ids.
        """
        for start in range(0, len(all_model_ids), batch_size):
            yield all_model_ids[start: start + batch_size]

    async def _process_batch(
            self,
            batch_model_ids: List[str],
            query: str,
            table_weights: Dict[str, float],
            all_results: Dict[str, Any],
    ) -> None:
        """
        For a given batch of IDs, loop over each table (except 'model_descriptions')
        and fetch + process that table's metadata.
        """
        for table_name in table_weights:
            if table_name == "model_descriptions":
                continue

            try:
                await self._fetch_and_merge_table(
                    table_name, batch_model_ids, query, all_results
                )
            except Exception as exc:
                self.logger.error(
                    f"Error searching metadata from {table_name} for batch {batch_model_ids}: {exc}"
                )

    async def _fetch_and_merge_table(
            self,
            table_name: str,
            batch_model_ids: List[str],
            query: str,
            all_results: Dict[str, Any],
    ) -> None:
        """
        1) Call chroma_manager.search(...) on `table_name` for this batch.
        2) Loop over each returned item, extract distance + metadata, and merge
           into all_results[model_id].
        """
        # 1) Perform the actual search call
        result = await self.chroma_manager.search(
            collection_name=table_name,
            query=query,
            where={"model_id": {"$in": batch_model_ids}},
            include=["metadatas", "documents", "distances"],
        )

        # 2) If there are no 'results' or it's empty, do nothing
        for idx, item in enumerate(result.get("results", [])):
            self._merge_single_search_hit(
                table_name, idx, item, result, all_results
            )

    def _merge_single_search_hit(
            self,
            table_name: str,
            idx: int,
            item: Dict[str, Any],
            search_result: Dict[str, Any],
            all_results: Dict[str, Any],
    ) -> None:
        """
        Given one `item` from search_result['results'], extract
        - model_id
        - metadata dict
        - distance
        Then merge them into all_results[model_id].
        """
        metadata = item.get("metadata", {})
        model_id = metadata.get("model_id", "unknown")
        if model_id not in all_results:
            return

        # 1) Extract distance via distance_normalizer
        distance = self.distance_normalizer.extract_search_distance(
            search_result, idx, item, table_name
        )

        # 2) Ensure the sub‐dictionaries exist
        model_entry = all_results[model_id]
        model_entry.setdefault("metadata", {})
        model_entry.setdefault("tables", [])
        model_entry.setdefault("table_initial_distances", {})

        # 3) Merge the metadata from this table
        model_entry["metadata"].update(metadata)

        # 4) Add table_name to the 'tables' list if not already present
        if table_name not in model_entry["tables"]:
            model_entry["tables"].append(table_name)

        # 5) Finally, record the distance under `table_initial_distances`
        model_entry["table_initial_distances"][table_name] = distance

    async def _process_model_descriptions_text_search(
            self,
            query: str,
            all_results: Dict[str, Any],
            search_limit: int = 10
    ) -> Dict[str, Any]:
        """
        Process model descriptions using chunks.

        Args:
            query: The search query string
            all_results: Dictionary containing all model results
            search_limit: Number of records to search

        Returns:
            Updated all_results dictionary with description data

        Note: This method searches for 'search_limit' records but only uses
        the top 5 most similar records to calculate the distance.
        """
        top_chunks_for_distance = 5

        for model_id, model_data in all_results.items():
            # Skip invalid or placeholder IDs
            if not model_id or model_id == "unknown":
                continue

            # 1) Ensure required fields exist and initialize chunk‐related lists
            self._prepare_model_data(model_data)

            # 2) Perform the chunk search and merge into model_data
            await self._search_and_process_chunks(
                model_id=model_id,
                model_data=model_data,
                query=query,
                search_limit=search_limit,
                top_chunks_for_distance=top_chunks_for_distance
            )

        return all_results

    def _prepare_model_data(self, model_data: Dict[str, Any]) -> None:
        """
        Make sure 'metadata' and 'table_initial_distances' keys exist, and
        initialize chunk‐specific fields.
        """
        model_data.setdefault("metadata", {})
        model_data.setdefault("table_initial_distances", {})

        model_data["chunk_descriptions"] = []
        model_data["chunk_description_distances"] = []

    async def _search_and_process_chunks(
            self,
            model_id: str,
            model_data: Dict[str, Any],
            query: str,
            search_limit: int,
            top_chunks_for_distance: int
    ) -> None:
        """
        1) Call chroma_manager.search to retrieve up to 'search_limit' chunks
           for the given model_id.
        2) Extract distances and descriptions, sort by distance, pick top N.
        3) Merge resulting descriptions and distances into model_data.
        """
        try:
            model_chunks_search = await self.chroma_manager.search(
                collection_name="model_descriptions",
                query=query,
                where={"model_id": {"$eq": model_id}},
                limit=search_limit,
                include=["metadatas", "distances"]
            )
        except Exception as e:
            self.logger.error(f"Error in chunk description search for model {model_id}: {e}")
            # On error, set a moderate default distance and bail out
            model_data["table_initial_distances"]["model_descriptions"] = 2.0
            return

        # Extract raw chunk info (distance + description + offset)
        chunk_results, all_chunk_descriptions = self._extract_chunk_info(model_chunks_search, model_id)

        # Sort by distance and pick only the top chunks_for_distance
        chunk_results.sort(key=lambda x: x["distance"])
        top_chunks = chunk_results[:top_chunks_for_distance]
        chunk_distances = [chunk["distance"] for chunk in top_chunks]

        # Merge all descriptions by their offsets, then pick the merged string
        chunk_descriptions, merged_description = self._sort_and_merge_descriptions_by_offset(
            all_chunk_descriptions
        )

        # Write back into model_data
        model_data["chunk_descriptions"] = chunk_descriptions
        model_data["chunk_description_distances"] = chunk_distances
        model_data["merged_description"] = merged_description
        model_data["metadata"]["description"] = merged_description

        # Compute average distance (or default) for 'model_descriptions'
        if chunk_distances:
            avg_distance = sum(chunk_distances) / len(chunk_distances)
        else:
            avg_distance = 2.0

        model_data["table_initial_distances"]["model_descriptions"] = avg_distance

        # Ensure 'model_descriptions' is listed in the tables
        if "model_descriptions" not in model_data.get("tables", []):
            model_data.setdefault("tables", []).append("model_descriptions")

    def _extract_chunk_info(
            self,
            search_result: Dict[str, Any],
            model_id: str
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Walk through search_result['results'], and for each valid chunk dict:
          - Pull out its distance (or default to 2.0 if missing)
          - Pull out description & offset
          - Return two lists:
             1) chunk_results: [{'distance': float, 'description': str}, ...]
             2) all_chunk_descriptions: [{'description': str, 'offset': int}, ...]
        """
        chunk_results = []
        all_chunk_descriptions = []

        if not (isinstance(search_result, dict) and "results" in search_result):
            return chunk_results, all_chunk_descriptions

        for chunk_item in search_result.get("results", []):
            if not isinstance(chunk_item, dict):
                continue

            # 1) Extract distance (or default)
            raw_distance = chunk_item.get("distance")
            distance = raw_distance if raw_distance is not None else 2.0
            self.logger.debug(f"Distance for description chunk of model {model_id}: {distance}")

            # 2) Extract description & offset
            desc = None
            offset = 999999
            metadata = chunk_item.get("metadata")
            if isinstance(metadata, dict):
                desc = metadata.get("description")
                offset = metadata.get("offset", 999_999)

            if isinstance(desc, str) and desc:
                all_chunk_descriptions.append({"description": desc, "offset": offset})
                chunk_results.append({"distance": distance, "description": desc})

        return chunk_results, all_chunk_descriptions

    @staticmethod
    def _sort_and_merge_descriptions_by_offset(chunks: list[dict]) -> tuple[list[str], str]:
        """Sorts chunk descriptions by their offset and merges them into a single string.

        Args:
            chunks: List of dicts with keys 'description' and 'offset'

        Returns:
            A tuple of (sorted list of descriptions, merged string)
        """
        sorted_chunks = sorted(chunks, key=lambda x: x.get('offset', 999999))
        sorted_descriptions = [chunk['description'] for chunk in sorted_chunks]
        merged = " ".join(sorted_descriptions)
        return sorted_descriptions, merged

    def _calculate_model_distances(
            self,
            all_results: Dict[str, Any],
            table_weights: Dict[str, float],
            collection_stats: Dict[str, Dict[str, float]]
    ) -> Dict[str, Any]:
        """
        Calculate a weighted distance sum for all models using normalized distances.
        """
        self.logger.info("Calculating model distances using collection stats")

        for model_id, model_data in all_results.items():
            # Skip invalid entries
            if not isinstance(model_data, dict) or "model_id" not in model_data:
                continue

            # 1) Compute normalized distances for metadata tables
            table_normals, weighted_sum = self._calculate_metadata_components(
                model_data, table_weights, collection_stats
            )
            model_data["table_normalized_distances"] = table_normals

            metadata_distance = weighted_sum

            # 2) Compute final distance (including chunks if present)
            final_distance, has_chunks, _ = self._calculate_chunk_and_final_distance(
                model_data, metadata_distance, collection_stats
            )
            model_data["distance"] = final_distance
            model_data["metadata_distance"] = metadata_distance

            # 3) Build distance_stats for debugging
            model_data["distance_stats"] = {
                "weighted_sum": weighted_sum,
                "weight_sum": 1.0,  # Total weight is always 1.0
                "metadata_tables_count": len(table_normals),
                "has_chunks": has_chunks
            }

        return all_results

    def _calculate_metadata_components(
            self,
            model_data: Dict[str, Any],
            table_weights: Dict[str, float],
            collection_stats: Dict[str, Dict[str, float]]
    ) -> Tuple[Dict[str, float], float]:
        """
        For each table in table_weights:
          - If 'table_initial_distances' contains that table, normalize using collection_stats.
          - Otherwise treat as the worst possible match (1.0).
        Returns:
          (table_normalized_distances, weighted_sum_of_distances)
        """
        weighted_sum = 0.0
        table_normals: Dict[str, float] = {}

        for table_name, table_weight in table_weights.items():
            if table_name in model_data.get("table_initial_distances", {}):
                raw_dist = model_data["table_initial_distances"][table_name]
                default_stats = {
                    "min": 0.0,
                    "max": 2.0,
                    "percentile_10": 0.5,
                    "percentile_90": 1.5
                }
                stats = collection_stats.get(table_name, default_stats)
                normalized = self.distance_normalizer.normalize_distance(raw_dist, stats)
            else:
                normalized = 1.0

            table_normals[table_name] = normalized
            weighted_sum += normalized * table_weight

            self.logger.debug(
                f"Model {model_data.get('model_id')}, table {table_name}: "
                f"normalized={normalized}, weight={table_weight}"
            )

        return table_normals, weighted_sum

    def _calculate_chunk_and_final_distance(
            self,
            model_data: Dict[str, Any],
            metadata_distance: float,
            collection_stats: Dict[str, Dict[str, float]]
    ) -> Tuple[float, bool, Optional[float]]:
        """
        If 'chunk_initial_distance' exists, normalize it and combine with metadata_distance.
        Returns:
          (final_distance, has_chunks, normalized_chunk_distance_or_None)
        """
        if "chunk_initial_distance" in model_data:
            default_chunk_stats = {
                "min": 0.0,
                "max": 2.0,
                "percentile_10": 0.5,
                "percentile_90": 1.5
            }
            chunk_stats = collection_stats.get("model_scripts_chunks", default_chunk_stats)
            raw_chunk = model_data["chunk_initial_distance"]
            normalized_chunk = self.distance_normalizer.normalize_distance(raw_chunk, chunk_stats)
            model_data["chunk_normalized_distance"] = normalized_chunk

            if model_data.get("match_source") == "metadata+chunks":
                final = 0.9 * metadata_distance + 0.1 * normalized_chunk
            else:
                final = normalized_chunk

            return final, True, normalized_chunk

        return metadata_distance, False, None

    @staticmethod
    def _sort_and_limit_search_results(all_results: Dict[str, Any], requested_limit: int) -> List[
        Dict[str, Any]]:
        """Convert to list, sort by distance, and limit to requested number of results."""
        # Convert to list
        output_list = list(all_results.values())

        # Sort by distance (lower is better)
        output_list.sort(key=lambda x: x.get('distance', 2.0))

        # Limit to requested number of results
        return output_list[:requested_limit]

    @staticmethod
    def _prepare_text_search_items(output_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
