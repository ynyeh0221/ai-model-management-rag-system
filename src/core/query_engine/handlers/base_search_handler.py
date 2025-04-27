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
            table_weights: Dict[str, float], user_id: Optional[str], ner_filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Search metadata tables based on NER analysis of user query.

        This method handles four different cases based on NER entities:
        1. No NER entities: Search all tables with original query
        2. Only positive entities: Search only tables related to positive entities using entity values as queries
        3. Only negative entities: Search all tables with original query, then filter out negative matches
        4. Both positive and negative entities: Search positive entity tables with entity values, filter out negative matches

        Args:
            query: The original user query
            chroma_filters: Database filters in Chroma format
            requested_limit: Number of results to return
            table_weights: Weights for each table
            user_id: User ID for access control
            ner_filters: Named entity filters extracted from query

        Returns:
            Dictionary of search results
        """
        # Initialize result dictionary
        all_results = {}
        all_tables = list(table_weights.keys())
        tables_to_search = all_tables.copy()

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

    def _analyze_ner_filters(self, ner_filters: Optional[Dict[str, Any]], table_weights: Dict[str, float]) -> Dict[
        str, Any]:
        """Analyze NER filters to determine positive and negative entities.

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

        # Table mappings for each entity type using correct table names
        entity_table_mapping = {
            "architecture": ["model_architectures"],
            "dataset": ["model_datasets"],
            "training_config": ["model_training_configs"]
        }

        # If no NER filters, return default values
        if not ner_filters:
            return result

        self.logger.info(f"Processing search with NER filters: {ner_filters}")

        # Check architecture
        if 'architecture' in ner_filters:
            # Ensure it's a dict with value and is_positive fields
            if isinstance(ner_filters['architecture'], dict) and 'value' in ner_filters['architecture']:
                arch_value = ner_filters['architecture']['value']
                is_positive = ner_filters['architecture'].get('is_positive', True)

                # Only process if there's an actual value
                if arch_value != "N/A":
                    if is_positive:
                        self.logger.info(f"Found positive architecture entity: {arch_value}")
                        result["has_positive_entities"] = True
                        # Map to correct table
                        arch_tables = []
                        for table in entity_table_mapping.get("architecture", []):
                            if table in table_weights:
                                arch_tables.append(table)
                                # Use the architecture value as the query for this table
                                result["positive_table_queries"][table] = arch_value

                        # Store architecture tables for intersection
                        if arch_tables:
                            result["entity_type_tables"]["architecture"] = set(arch_tables)
                    else:
                        result["has_negative_entities"] = True
                        result["negative_entities"]['architecture'] = arch_value

        # Check dataset - note the nested structure
        if 'dataset' in ner_filters:
            # The dataset has a nested 'name' field
            if isinstance(ner_filters['dataset'], dict) and 'value' in ner_filters['dataset']:
                dataset_value = ner_filters['dataset'].get('value')
                is_positive = ner_filters['dataset'].get('is_positive', True)

                if dataset_value != "N/A":
                    if is_positive:
                        self.logger.info(f"Found positive dataset entity: {dataset_value}")
                        result["has_positive_entities"] = True
                        # Map to correct table
                        dataset_tables = []
                        for table in entity_table_mapping.get("dataset", []):
                            if table in table_weights:
                                dataset_tables.append(table)
                                result["positive_table_queries"][table] = dataset_value

                        # Store dataset tables for intersection
                        if dataset_tables:
                            result["entity_type_tables"]["dataset"] = set(dataset_tables)
                    else:
                        result["has_negative_entities"] = True
                        result["negative_entities"]['dataset'] = dataset_value

        # Check training config fields
        if 'training_config' in ner_filters:
            training_config = ner_filters['training_config']
            training_tables = []

            for field, data in training_config.items():
                if isinstance(data, dict) and 'value' in data and data['value'] != "N/A":
                    field_value = data['value']
                    is_positive = data.get('is_positive', True)

                    self.logger.info(f"Training config {field}: {field_value}, positive: {is_positive}")

                    if is_positive:
                        result["has_positive_entities"] = True
                        # Add training config tables with the field value as query
                        for table in entity_table_mapping.get("training_config", []):
                            if table in table_weights and table not in training_tables:
                                training_tables.append(table)
                                # Use the field name and value in the query
                                query_value = f"{field} {field_value}"
                                result["positive_table_queries"][table] = query_value
                    else:
                        result["has_negative_entities"] = True
                        result["negative_entities"].setdefault('training_config', {})[field] = field_value

            # Store training config tables for intersection
            if training_tables:
                result["entity_type_tables"]["training_config"] = set(training_tables)

        return result

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
                # Increase limit by 5x for NER queries
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
            self, negative_entities: Dict[str, Any], requested_limit: int
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Execute searches for negative NER fields.

        Args:
            negative_entities: Mapping of negative entity types to their values
            requested_limit: Number of results to return

        Returns:
            Dictionary of negative search results
        """
        negative_results = {}
        negative_search_tasks = []
        negative_table_queries = {}

        # Prepare negative entity searches
        for entity_type, value in negative_entities.items():
            if entity_type == 'architecture':
                table = "model_architectures"
                negative_table_queries[table] = value
                negative_search_tasks.append(self.chroma_manager.search(
                    collection_name=table,
                    query=value,
                    where={},
                    limit=requested_limit * 3,  # Use higher limit for negative queries too
                    include=["metadatas", "documents", "distances"]
                ))
            elif entity_type == 'dataset':
                table = "model_datasets"
                negative_table_queries[table] = value
                negative_search_tasks.append(self.chroma_manager.search(
                    collection_name=table,
                    query=value,
                    where={},
                    limit=requested_limit * 3,
                    include=["metadatas", "documents", "distances"]
                ))
            elif entity_type == 'training_config':
                table = "model_training_configs"
                # For training_config, we need to handle multiple fields
                for field, field_value in value.items():
                    if field_value != "N/A":
                        query_value = f"{field} {field_value}"
                        negative_table_queries[f"{table}_{field}"] = query_value
                        negative_search_tasks.append(self.chroma_manager.search(
                            collection_name=table,
                            query=query_value,
                            where={},
                            limit=requested_limit * 3,
                            include=["metadatas", "documents", "distances"]
                        ))

        # Execute all negative searches in parallel
        if negative_search_tasks:
            negative_search_results = await asyncio.gather(*negative_search_tasks)

            # Process negative search results
            neg_table_idx = 0
            for table, query_value in negative_table_queries.items():
                if neg_table_idx >= len(negative_search_results):
                    break

                result = negative_search_results[neg_table_idx]
                neg_table_idx += 1

                # Store results with their distances
                negative_results[table] = []

                for idx, item in enumerate(result.get('results', [])):
                    metadata = item.get('metadata', {})
                    model_id = metadata.get('model_id', 'unknown')

                    if model_id == 'unknown':
                        continue

                    # Get distance using the proper extraction method
                    distance = self.distance_normalizer.extract_search_distance(
                        result, idx, item, table.split('_')[0] if '_' in table else table
                    )

                    negative_results[table].append({
                        'model_id': model_id,
                        'distance': distance,
                        'metadata': metadata
                    })

                    self.logger.info(f"Negative result for {table}: model_id={model_id}, distance={distance:.4f}")

        return negative_results

    def _process_search_results(
            self, positive_search_results: List[Dict[str, Any]], tables_to_search: List[str],
            entity_type_tables: Dict[str, set], has_positive_entities: bool,
            has_negative_entities: bool, negative_results: Dict[str, List[Dict[str, Any]]],
            user_id: Optional[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Process search results and apply filters.

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
        all_results = {}

        # Track models that match each entity type (for intersection)
        models_by_entity_type = {}
        if has_positive_entities and len(entity_type_tables) > 1:
            # Only initialize this tracking when we have multiple entity types
            for entity_type in entity_type_tables.keys():
                models_by_entity_type[entity_type] = set()

        # Process positive search results
        for table_idx, result in enumerate(positive_search_results):
            table_name = tables_to_search[table_idx]

            # Determine which entity type this table belongs to (if any)
            matching_entity_type = None
            for entity_type, tables in entity_type_tables.items():
                if table_name in tables:
                    matching_entity_type = entity_type
                    break

            for idx, item in enumerate(result.get('results', [])):
                metadata = item.get('metadata', {})
                model_id = metadata.get('model_id', 'unknown')

                # Skip if no model_id
                if model_id == 'unknown':
                    continue

                # Apply access control
                if self.access_control_manager and user_id:
                    if not self.access_control_manager.check_access({'metadata': metadata}, user_id, "view"):
                        continue

                # Track this model for the matching entity type ONLY if we're tracking multiple entity types
                if has_positive_entities and matching_entity_type and len(entity_type_tables) > 1:
                    if matching_entity_type in models_by_entity_type:  # Safety check
                        models_by_entity_type[matching_entity_type].add(model_id)

                # Get distance for this positive result
                distance = self.distance_normalizer.extract_search_distance(
                    result, idx, item, table_name
                )

                # Check if this model should be filtered out based on negative entities
                if has_negative_entities and negative_results:
                    skip_result = False
                    filter_reason = ""
                    negative_distance_threshold = 1.3  # Lower distance means better match

                    # Check each negative entity result
                    for neg_table, neg_items in negative_results.items():
                        # Find this model in negative results
                        matching_neg_items = [item for item in neg_items if item['model_id'] == model_id]

                        if matching_neg_items:
                            # Get lowest distance (best match)
                            min_distance = min(item['distance'] for item in matching_neg_items)

                            # If distance is below threshold (better match), filter out this model
                            if min_distance < negative_distance_threshold:
                                table_type = neg_table.split('_')[0] if '_' in neg_table else neg_table
                                if table_type == "model_architectures":
                                    filter_reason = f"negative architecture match (distance: {min_distance:.4f})"
                                elif table_type == "model_datasets":
                                    filter_reason = f"negative dataset match (distance: {min_distance:.4f})"
                                else:
                                    # Extract field name for training_config
                                    field = neg_table.split('_')[-1] if '_' in neg_table else "unknown"
                                    filter_reason = f"negative training config match: {field} (distance: {min_distance:.4f})"

                                skip_result = True
                                break

                    # Skip this result if it matched negative criteria
                    if skip_result:
                        self.logger.info(f"Filtering out model {model_id} due to {filter_reason}")
                        continue

                # If we got here, the result passed all filters
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

        # If we have multiple positive entity types, find models that match ALL entity types
        if has_positive_entities and len(models_by_entity_type) > 1:
            # Check that all entity types have at least one model
            if all(len(models) > 0 for models in models_by_entity_type.values()):
                valid_models = set.intersection(*models_by_entity_type.values())
                self.logger.info(f"Models matching ALL positive entity criteria: {valid_models}")

                # Filter all_results to only include models in the intersection
                all_results = {model_id: data for model_id, data in all_results.items()
                               if model_id in valid_models}

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
                                                      search_limit: int = 12) -> Dict[str, Any]:
        """Process model descriptions using chunks.

        Args:
            query: The search query string
            all_results: Dictionary containing all model results
            search_limit: Number of records to search

        Returns:
            Updated all_results dictionary with description data

        Note: This method searches for search_limit records but only uses top 5
        most similar records to calculate the distance.
        """
        # Define the max number of chunks to use for distance calculation
        top_chunks_for_distance = 6
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
                            offset = 999999
                            if 'metadata' in chunk_result and isinstance(chunk_result['metadata'], dict):
                                description = chunk_result['metadata'].get('description')
                                offset = chunk_result['metadata'].get('offset', 999999)

                            # Store all valid descriptions regardless of distance
                            if description and isinstance(description, str):
                                all_chunk_descriptions.append({"description": description, "offset": offset})

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

                        chunk_descriptions, merged_description = self._sort_and_merge_descriptions_by_offset(
                            all_chunk_descriptions
                        )

                        model_data['chunk_descriptions'] = chunk_descriptions
                        model_data['chunk_description_distances'] = chunk_distances
                        model_data['merged_description'] = merged_description
                        model_data['metadata']['description'] = merged_description

                        if chunk_descriptions:
                            avg_distance = sum(chunk_distances) / len(chunk_distances)
                            model_data['table_initial_distances']['model_descriptions'] = avg_distance
                        else:
                            model_data['table_initial_distances']['model_descriptions'] = 2.0

                        if 'model_descriptions' not in model_data.get('tables', []):
                            model_data.setdefault('tables', []).append('model_descriptions')

                except Exception as e:
                    self.logger.error(f"Error in chunk description search for model {model_id}: {e}")
                    # Set a moderate default distance if error occurs
                    model_data['table_initial_distances']['model_descriptions'] = 2.0
            except Exception as e:
                self.logger.error(f"Error in model description handling for model: {e}")

        return all_results

    def _sort_and_merge_descriptions_by_offset(self, chunks: list[dict]) -> tuple[list[str], str]:
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