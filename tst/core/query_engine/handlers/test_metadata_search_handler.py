import unittest
from unittest.mock import AsyncMock, MagicMock

from src.core.query_engine.handlers.metadata_search_handler import MetadataSearchHandler
from src.core.query_engine.handlers.utils.distance_normalizer import DistanceNormalizer
from src.core.query_engine.handlers.utils.filter_translator import FilterTranslator
from src.core.query_engine.handlers.utils.metadata_table_manager import MetadataTableManager
from src.core.vector_db.access_control import AccessControlManager
from src.core.vector_db.chroma_manager import ChromaManager


class TestMetadataSearchHandler(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Create mock dependencies
        self.chroma_manager = MagicMock(spec=ChromaManager)
        self.access_control_manager = MagicMock(spec=AccessControlManager)
        self.filter_translator = MagicMock(spec=FilterTranslator)
        self.distance_normalizer = MagicMock(spec=DistanceNormalizer)

        # Make search method an AsyncMock
        self.chroma_manager.search = AsyncMock()

        # Create the instance we're testing
        self.handler = MetadataSearchHandler(
            chroma_manager=self.chroma_manager,
            filter_translator=self.filter_translator,
            distance_normalizer=self.distance_normalizer,
            access_control_manager=self.access_control_manager
        )

        # Mock the MetadataTableManager
        self.handler.metadata_table_manager = MagicMock(spec=MetadataTableManager)

        # Mock the parent class methods that we'll need to use
        # We'll set these in individual tests as needed
        self.handler._extract_text_search_parameters = MagicMock()
        self.handler._get_collection_distance_stats_for_query = AsyncMock()
        self.handler._search_all_metadata_tables = AsyncMock()
        self.handler._fetch_complete_model_metadata = AsyncMock()
        self.handler._process_model_descriptions_text_search = AsyncMock()
        self.handler._calculate_model_distances = MagicMock()
        self.handler._sort_and_limit_search_results = MagicMock()
        self.handler._prepare_text_search_items = MagicMock()

    async def test_handle_metadata_search_success(self):
        """Test successful metadata search with valid results"""
        # Setup mock return values
        query = "test query"
        parameters = {
            "user_id": "user123",
            "limit": 10,
            "filters": {"field": "value"}
        }

        # Mock _extract_text_search_parameters
        user_id = "user123"
        requested_limit = 10
        chroma_filters = {"translated_filter": "test"}
        self.handler._extract_text_search_parameters.return_value = (user_id, requested_limit, chroma_filters)

        # Mock get_metadata_table_weights
        table_weights = {
            "model_descriptions": 0.5,
            "model_params": 0.3,
            "model_tags": 0.2
        }
        self.handler.metadata_table_manager.get_metadata_table_weights.return_value = table_weights

        # Mock _get_collection_distance_stats_for_query
        collection_stats = {
            "model_descriptions": {"min": 0.1, "max": 0.9},
            "model_params": {"min": 0.2, "max": 0.8},
            "model_tags": {"min": 0.3, "max": 0.7},
            "model_scripts_chunks": {"min": 0.1, "max": 0.9}
        }
        self.handler._get_collection_distance_stats_for_query.return_value = collection_stats

        # Mock _search_all_metadata_tables
        all_results_1 = {
            "model1": {"model_id": "model1", "tables": ["model_descriptions"]},
            "model2": {"model_id": "model2", "tables": ["model_params"]}
        }
        self.handler._search_all_metadata_tables.return_value = all_results_1

        # Mock _fetch_complete_model_metadata
        all_results_2 = {
            "model1": {
                "model_id": "model1",
                "tables": ["model_descriptions", "model_params"],
                "metadata": {"description": "Test model 1"}
            },
            "model2": {
                "model_id": "model2",
                "tables": ["model_params", "model_tags"],
                "metadata": {"description": "Test model 2"}
            }
        }
        self.handler._fetch_complete_model_metadata.return_value = all_results_2

        # Mock _process_model_descriptions_text_search
        all_results_3 = {
            "model1": {
                "model_id": "model1",
                "tables": ["model_descriptions", "model_params"],
                "metadata": {"description": "Enhanced test model 1"},
                "merged_description": "Enhanced test model 1",
                "table_initial_distances": {"model_descriptions": 0.2}
            },
            "model2": {
                "model_id": "model2",
                "tables": ["model_params", "model_tags", "model_descriptions"],
                "metadata": {"description": "Enhanced test model 2"},
                "merged_description": "Enhanced test model 2",
                "table_initial_distances": {"model_descriptions": 0.3}
            }
        }
        self.handler._process_model_descriptions_text_search.return_value = all_results_3

        # Mock _calculate_model_distances
        all_results_4 = {
            "model1": {
                "model_id": "model1",
                "distance": 0.1,
                "metadata": {"description": "Enhanced test model 1"}
            },
            "model2": {
                "model_id": "model2",
                "distance": 0.2,
                "metadata": {"description": "Enhanced test model 2"}
            }
        }
        self.handler._calculate_model_distances.return_value = all_results_4

        # Mock _sort_and_limit_search_results
        output_list = [
            {
                "model_id": "model1",
                "distance": 0.1,
                "metadata": {"description": "Enhanced test model 1"}
            },
            {
                "model_id": "model2",
                "distance": 0.2,
                "metadata": {"description": "Enhanced test model 2"}
            }
        ]
        self.handler._sort_and_limit_search_results.return_value = output_list

        # Mock _prepare_text_search_items
        final_items = [
            {
                "id": "model_metadata_model1",
                "model_id": "model1",
                "metadata": {"description": "Enhanced test model 1"},
                "distance": 0.1,
                "rank": 1
            },
            {
                "id": "model_metadata_model2",
                "model_id": "model2",
                "metadata": {"description": "Enhanced test model 2"},
                "distance": 0.2,
                "rank": 2
            }
        ]
        self.handler._prepare_text_search_items.return_value = final_items

        # Call the method
        result = await self.handler.handle_metadata_search(query, parameters)

        # Assertions
        self.assertTrue(result["success"])
        self.assertEqual(result["type"], "metadata_search")
        self.assertEqual(result["items"], final_items)
        self.assertEqual(result["total_found"], 2)
        self.assertEqual(result["total_models"], 2)
        self.assertIn("performance", result)
        self.assertIn("metadata_search_time_ms", result["performance"])
        self.assertIn("total_time_ms", result["performance"])

        # Verify method calls with correct parameters
        self.handler._extract_text_search_parameters.assert_called_once_with(parameters)
        self.handler.metadata_table_manager.get_metadata_table_weights.assert_called_once()

        expected_collections = ["model_descriptions", "model_params", "model_tags", "model_scripts_chunks"]
        self.handler._get_collection_distance_stats_for_query.assert_called_once()
        call_args = self.handler._get_collection_distance_stats_for_query.call_args[0]
        self.assertEqual(call_args[0], query)
        # Check that all collections are included (order might be different)
        self.assertCountEqual(call_args[1], expected_collections)
        self.assertEqual(call_args[2], user_id)

        self.handler._search_all_metadata_tables.assert_called_once_with(
            query, chroma_filters, requested_limit, table_weights, user_id, None
        )

        self.handler._fetch_complete_model_metadata.assert_called_once_with(
            query, all_results_1, table_weights, user_id
        )

        self.handler._process_model_descriptions_text_search.assert_called_once_with(
            query, all_results_2
        )

        self.handler._calculate_model_distances.assert_called_once_with(
            all_results_3, table_weights, collection_stats
        )

        self.handler._sort_and_limit_search_results.assert_called_once_with(
            all_results_4, requested_limit
        )

        self.handler._prepare_text_search_items.assert_called_once_with(output_list)

    async def test_handle_metadata_search_empty_results(self):
        """Test metadata search when no results are found"""
        # Setup mock return values
        query = "test query"
        parameters = {
            "user_id": "user123",
            "limit": 10,
            "filters": {"field": "value"}
        }

        # Mock _extract_text_search_parameters
        user_id = "user123"
        requested_limit = 10
        chroma_filters = {"translated_filter": "test"}
        self.handler._extract_text_search_parameters.return_value = (user_id, requested_limit, chroma_filters)

        # Mock get_metadata_table_weights
        table_weights = {
            "model_descriptions": 0.5,
            "model_params": 0.3,
            "model_tags": 0.2
        }
        self.handler.metadata_table_manager.get_metadata_table_weights.return_value = table_weights

        # Mock _get_collection_distance_stats_for_query
        collection_stats = {
            "model_descriptions": {"min": 0.1, "max": 0.9},
            "model_params": {"min": 0.2, "max": 0.8},
            "model_tags": {"min": 0.3, "max": 0.7},
            "model_scripts_chunks": {"min": 0.1, "max": 0.9}
        }
        self.handler._get_collection_distance_stats_for_query.return_value = collection_stats

        # Mock _search_all_metadata_tables - empty results
        all_results_1 = {}
        self.handler._search_all_metadata_tables.return_value = all_results_1

        # Mock _fetch_complete_model_metadata - empty results
        all_results_2 = {}
        self.handler._fetch_complete_model_metadata.return_value = all_results_2

        # Mock _process_model_descriptions_text_search - empty results
        all_results_3 = {}
        self.handler._process_model_descriptions_text_search.return_value = all_results_3

        # Mock _calculate_model_distances - empty results
        all_results_4 = {}
        self.handler._calculate_model_distances.return_value = all_results_4

        # Mock _sort_and_limit_search_results - empty list
        output_list = []
        self.handler._sort_and_limit_search_results.return_value = output_list

        # Mock _prepare_text_search_items - empty list
        final_items = []
        self.handler._prepare_text_search_items.return_value = final_items

        # Call the method
        result = await self.handler.handle_metadata_search(query, parameters)

        # Assertions
        self.assertTrue(result["success"])
        self.assertEqual(result["type"], "metadata_search")
        self.assertEqual(result["items"], [])
        self.assertEqual(result["total_found"], 0)
        self.assertEqual(result["total_models"], 0)
        self.assertIn("performance", result)

    async def test_handle_metadata_search_error(self):
        """Test metadata search when an error occurs"""
        # Setup mock return values
        query = "test query"
        parameters = {
            "user_id": "user123",
            "limit": 10,
            "filters": {"field": "value"}
        }

        # Mock _extract_text_search_parameters to raise an exception
        self.handler._extract_text_search_parameters.side_effect = Exception("Test error")

        # Call the method and expect an exception
        with self.assertRaises(Exception) as context:
            await self.handler.handle_metadata_search(query, parameters)

        # Verify that the exception is propagated
        self.assertEqual(str(context.exception), "Test error")

    async def test_handle_metadata_search_integration(self):
        """Test metadata search with more realistic integration of parent class methods"""
        # For this test, we'll patch only some of the parent methods to make it more like an integration test
        # We'll let the actual implementation of some methods run

        # Restore the original methods we patched in setUp
        self.handler._extract_text_search_parameters = self.handler.__class__._extract_text_search_parameters.__get__(
            self.handler)
        self.handler._sort_and_limit_search_results = self.handler.__class__._sort_and_limit_search_results.__get__(
            self.handler)
        self.handler._prepare_text_search_items = self.handler.__class__._prepare_text_search_items.__get__(
            self.handler)

        # Setup test data
        query = "test query"
        parameters = {
            "user_id": "user123",
            "limit": 2,
            "filters": {"field": "value"}
        }

        # Mock the filter translator for _extract_text_search_parameters
        self.filter_translator.translate_to_chroma.return_value = {"translated_filter": "test"}

        # Mock get_metadata_table_weights
        table_weights = {
            "model_descriptions": 0.5,
            "model_params": 0.5
        }
        self.handler.metadata_table_manager.get_metadata_table_weights.return_value = table_weights

        # Mock _get_collection_distance_stats_for_query
        collection_stats = {
            "model_descriptions": {"min": 0.1, "max": 0.9, "percentile_10": 0.2, "percentile_90": 0.8},
            "model_params": {"min": 0.2, "max": 0.8, "percentile_10": 0.3, "percentile_90": 0.7},
            "model_scripts_chunks": {"min": 0.1, "max": 0.9, "percentile_10": 0.2, "percentile_90": 0.8}
        }
        self.handler._get_collection_distance_stats_for_query.return_value = collection_stats

        # Mock _search_all_metadata_tables
        all_results = {
            "model1": {
                "model_id": "model1",
                "tables": ["model_descriptions", "model_params"],
                "table_initial_distances": {"model_descriptions": 0.3, "model_params": 0.4},
                "match_source": "metadata",
                "metadata": {"name": "Model 1", "description": "Test model 1"}
            },
            "model2": {
                "model_id": "model2",
                "tables": ["model_descriptions", "model_params"],
                "table_initial_distances": {"model_descriptions": 0.2, "model_params": 0.3},
                "match_source": "metadata",
                "metadata": {"name": "Model 2", "description": "Test model 2"}
            },
            "model3": {
                "model_id": "model3",
                "tables": ["model_descriptions"],
                "table_initial_distances": {"model_descriptions": 0.5},
                "match_source": "metadata",
                "metadata": {"name": "Model 3", "description": "Test model 3"}
            }
        }

        # Set up the chain of method calls with the same mock data
        self.handler._search_all_metadata_tables.return_value = all_results
        self.handler._fetch_complete_model_metadata.return_value = all_results
        self.handler._process_model_descriptions_text_search.return_value = all_results

        # For _calculate_model_distances, we'll create model distance data
        # Since we restored the original _sort_and_limit_search_results method, we need to add distance values
        calculated_results = all_results.copy()
        calculated_results["model1"]["distance"] = 0.35  # (0.3 * 0.5 + 0.4 * 0.5)
        calculated_results["model2"]["distance"] = 0.25  # (0.2 * 0.5 + 0.3 * 0.5)
        calculated_results["model3"]["distance"] = 0.75  # (0.5 * 0.5 + 1.0 * 0.5) - missing table gets 1.0

        self.handler._calculate_model_distances.return_value = calculated_results

        # Call the method
        result = await self.handler.handle_metadata_search(query, parameters)

        # Assertions
        self.assertTrue(result["success"])
        self.assertEqual(result["type"], "metadata_search")
        self.assertEqual(len(result["items"]), 2)  # Limited to 2 as per parameters
        self.assertEqual(result["total_found"], 2)
        self.assertEqual(result["total_models"], 3)

        # The results should be ordered by distance (lower is better)
        self.assertEqual(result["items"][0]["model_id"], "model2")  # Lowest distance first
        self.assertEqual(result["items"][1]["model_id"], "model1")  # Second lowest distance

        # Model3 should not be included because limit is 2
        model_ids = [item["model_id"] for item in result["items"]]
        self.assertNotIn("model3", model_ids)

        # Check ranks
        self.assertEqual(result["items"][0]["rank"], 1)
        self.assertEqual(result["items"][1]["rank"], 2)

        # Check metadata
        self.assertEqual(result["items"][0]["metadata"]["name"], "Model 2")
        self.assertEqual(result["items"][1]["metadata"]["name"], "Model 1")


if __name__ == '__main__':
    unittest.main()