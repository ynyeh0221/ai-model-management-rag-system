import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.query_engine.handlers.text_search_handler import TextSearchHandler
from src.core.query_engine.handlers.utils.distance_normalizer import DistanceNormalizer
from src.core.query_engine.handlers.utils.filter_translator import FilterTranslator
from src.core.query_engine.handlers.utils.metadata_table_manager import MetadataTableManager
from src.core.vector_db.access_control import AccessControlManager
from src.core.vector_db.chroma_manager import ChromaManager


class TestTextSearchHandler(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Create mock dependencies
        self.chroma_manager = MagicMock(spec=ChromaManager)
        self.access_control_manager = MagicMock(spec=AccessControlManager)
        self.filter_translator = MagicMock(spec=FilterTranslator)
        self.distance_normalizer = MagicMock(spec=DistanceNormalizer)
        self.metadata_table_manager = MagicMock(spec=MetadataTableManager)

        # Make search method an AsyncMock
        self.chroma_manager.search = AsyncMock()

        # Create the instance we're testing
        self.handler = TextSearchHandler(
            metadata_table_manager=self.metadata_table_manager,
            access_control_manager=self.access_control_manager,
            filter_translator=self.filter_translator,
            chroma_manager=self.chroma_manager,
            distance_normalizer=self.distance_normalizer
        )

        # Mock performance metrics
        self.handler.performance_metrics = MagicMock()

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

        # We'll patch this method separately in tests where needed
        # to avoid patching it for tests that specifically test this method
        # self.handler._search_model_chunks_table = AsyncMock()

    async def test_search_model_chunks_table(self):
        """Test the _search_model_chunks_table method"""
        # Setup test data
        query = "test query"
        chroma_filters = {"field": "value"}
        requested_limit = 10
        user_id = "user123"
        chunks_search_start = time.time()

        # Initial results from metadata search
        all_results = {
            "model1": {
                "model_id": "model1",
                "tables": ["model_descriptions"],
                "match_source": "metadata"
            }
        }

        # Mock chunk search results
        self.chroma_manager.search.return_value = {
            "results": [
                {
                    "metadata": {"model_id": "model1", "chunk": "chunk1"},
                    "distance": 0.1
                },
                {
                    "metadata": {"model_id": "model2", "chunk": "chunk2"},
                    "distance": 0.2
                },
                {
                    "metadata": {"model_id": "model3", "chunk": "chunk3"},
                    "distance": 0.3
                },
                {
                    "metadata": {"model_id": "unknown", "chunk": "chunk4"},
                    "distance": 0.4
                }
            ]
        }

        # Mock access control to correctly handle the document structure
        def mock_check_access(document, user_id, permission_type):
            # Extract model_id from the document structure
            # Document is passed as {'metadata': metadata} where metadata contains model_id
            metadata = document.get('metadata', {})
            model_id = metadata.get('model_id')
            # Allow access to model1 and model2, deny for model3
            return model_id in ["model1", "model2"]

        self.access_control_manager.check_access = MagicMock(side_effect=mock_check_access)

        # Call the method
        result_all_results, chunks_search_time = await self.handler._search_model_chunks_table(
            query, chroma_filters, requested_limit, all_results, user_id, chunks_search_start
        )

        # Assertions
        # Should have called search with the right parameters
        self.chroma_manager.search.assert_called_once_with(
            collection_name="model_scripts_chunks",
            query=query,
            where=chroma_filters,
            limit=requested_limit * 100,
            include=["metadatas", "documents", "distances"]
        )

        # Should have processed the results correctly - we expect 2 models
        self.assertEqual(len(result_all_results), 2)  # model1 and model2, not model3 (access denied)

        # model1 should have been updated from metadata to metadata+chunks
        self.assertEqual(result_all_results["model1"]["match_source"], "metadata+chunks")
        self.assertEqual(result_all_results["model1"]["chunk_initial_distance"], 0.1)

        # model2 should have been added as chunks only
        self.assertEqual(result_all_results["model2"]["match_source"], "chunks")
        self.assertEqual(result_all_results["model2"]["chunk_initial_distance"], 0.2)

        # model3 should not be included due to access control
        self.assertNotIn("model3", result_all_results)

        # Check that access control was called for all models with valid IDs
        # It should be called 3 times (model1, model2, model3) but not for "unknown"
        self.assertEqual(self.access_control_manager.check_access.call_count, 3)

        # Should return a time measurement
        self.assertIsInstance(chunks_search_time, float)

    async def test_search_model_chunks_table_error(self):
        """Test the _search_model_chunks_table method when an error occurs in search"""
        # Setup test data
        query = "test query"
        chroma_filters = {"field": "value"}
        requested_limit = 10
        user_id = "user123"
        chunks_search_start = time.time()

        # Initial results from metadata search
        all_results = {
            "model1": {
                "model_id": "model1",
                "tables": ["model_descriptions"],
                "match_source": "metadata"
            }
        }

        # Mock chunk search to raise an exception
        self.chroma_manager.search.side_effect = Exception("Test search error")

        # Call the method
        result_all_results, chunks_search_time = await self.handler._search_model_chunks_table(
            query, chroma_filters, requested_limit, all_results, user_id, chunks_search_start
        )

        # Assertions
        # Should handle the error and return the original results
        self.assertEqual(result_all_results, all_results)
        self.assertEqual(len(result_all_results), 1)

        # Should return a time measurement
        self.assertIsInstance(chunks_search_time, float)

    async def test_handle_text_search(self):
        """Test the handle_text_search method"""
        # Setup test data
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
        self.metadata_table_manager.get_metadata_table_weights.return_value = table_weights

        # Mock _get_collection_distance_stats_for_query
        collection_stats = {
            "model_descriptions": {"min": 0.1, "max": 0.9},
            "model_params": {"min": 0.2, "max": 0.8},
            "model_tags": {"min": 0.3, "max": 0.7},
            "model_scripts_chunks": {"min": 0.1, "max": 0.9}
        }
        self.handler._get_collection_distance_stats_for_query.return_value = collection_stats

        # Mock _search_all_metadata_tables
        metadata_results = {
            "model1": {"model_id": "model1", "tables": ["model_descriptions"], "match_source": "metadata"},
            "model2": {"model_id": "model2", "tables": ["model_params"], "match_source": "metadata"}
        }
        self.handler._search_all_metadata_tables.return_value = metadata_results

        # Patch _search_model_chunks_table for this test
        with patch.object(self.handler, '_search_model_chunks_table', new_callable=AsyncMock) as mock_search_chunks:
            # Mock _search_model_chunks_table
            chunks_results = {
                "model1": {"model_id": "model1", "tables": ["model_descriptions"], "match_source": "metadata+chunks",
                           "chunk_initial_distance": 0.1},
                "model2": {"model_id": "model2", "tables": ["model_params"], "match_source": "metadata",
                           "chunk_initial_distance": None},
                "model3": {"model_id": "model3", "tables": ["chunks"], "match_source": "chunks",
                           "chunk_initial_distance": 0.3}
            }
            mock_search_chunks.return_value = (chunks_results, 50.0)  # Results and time in ms

            # Mock _fetch_complete_model_metadata
            metadata_complete_results = {
                "model1": {
                    "model_id": "model1",
                    "tables": ["model_descriptions", "model_params"],
                    "metadata": {"description": "Test model 1"},
                    "match_source": "metadata+chunks",
                    "chunk_initial_distance": 0.1
                },
                "model2": {
                    "model_id": "model2",
                    "tables": ["model_params", "model_tags"],
                    "metadata": {"description": "Test model 2"},
                    "match_source": "metadata"
                },
                "model3": {
                    "model_id": "model3",
                    "tables": ["chunks"],
                    "metadata": {"description": "Test model 3"},
                    "match_source": "chunks",
                    "chunk_initial_distance": 0.3
                }
            }
            self.handler._fetch_complete_model_metadata.return_value = metadata_complete_results

            # Mock _process_model_descriptions_text_search
            description_results = {
                "model1": {
                    "model_id": "model1",
                    "tables": ["model_descriptions", "model_params"],
                    "metadata": {"description": "Enhanced test model 1"},
                    "merged_description": "Enhanced test model 1",
                    "match_source": "metadata+chunks",
                    "chunk_initial_distance": 0.1,
                    "table_initial_distances": {"model_descriptions": 0.2}
                },
                "model2": {
                    "model_id": "model2",
                    "tables": ["model_params", "model_tags", "model_descriptions"],
                    "metadata": {"description": "Enhanced test model 2"},
                    "merged_description": "Enhanced test model 2",
                    "match_source": "metadata",
                    "table_initial_distances": {"model_descriptions": 0.3}
                },
                "model3": {
                    "model_id": "model3",
                    "tables": ["chunks", "model_descriptions"],
                    "metadata": {"description": "Enhanced test model 3"},
                    "merged_description": "Enhanced test model 3",
                    "match_source": "chunks",
                    "chunk_initial_distance": 0.3,
                    "table_initial_distances": {"model_descriptions": 0.4}
                }
            }
            self.handler._process_model_descriptions_text_search.return_value = description_results

            # Mock _calculate_model_distances
            calculated_results = {
                "model1": {
                    "model_id": "model1",
                    "distance": 0.15,
                    "metadata": {"description": "Enhanced test model 1"},
                    "match_source": "metadata+chunks"
                },
                "model2": {
                    "model_id": "model2",
                    "distance": 0.25,
                    "metadata": {"description": "Enhanced test model 2"},
                    "match_source": "metadata"
                },
                "model3": {
                    "model_id": "model3",
                    "distance": 0.35,
                    "metadata": {"description": "Enhanced test model 3"},
                    "match_source": "chunks"
                }
            }
            self.handler._calculate_model_distances.return_value = calculated_results

            # Mock _sort_and_limit_search_results
            sorted_results = [
                {
                    "model_id": "model1",
                    "distance": 0.15,
                    "metadata": {"description": "Enhanced test model 1"},
                    "match_source": "metadata+chunks"
                },
                {
                    "model_id": "model2",
                    "distance": 0.25,
                    "metadata": {"description": "Enhanced test model 2"},
                    "match_source": "metadata"
                }
            ]
            self.handler._sort_and_limit_search_results.return_value = sorted_results

            # Mock _prepare_text_search_items
            final_items = [
                {
                    "id": "model_metadata_model1",
                    "model_id": "model1",
                    "metadata": {"description": "Enhanced test model 1"},
                    "distance": 0.15,
                    "rank": 1,
                    "match_source": "metadata+chunks"
                },
                {
                    "id": "model_metadata_model2",
                    "model_id": "model2",
                    "metadata": {"description": "Enhanced test model 2"},
                    "distance": 0.25,
                    "rank": 2,
                    "match_source": "metadata"
                }
            ]
            self.handler._prepare_text_search_items.return_value = final_items

            # Mock performance metrics calculation
            self.handler.performance_metrics.calculate_text_search_performance.return_value = {
                "total_time_ms": 200.0,
                "metadata_search_time_ms": 100.0,
                "chunks_search_time_ms": 50.0
            }

            # Call the method
            result = await self.handler.handle_text_search(query, parameters)

            # Assertions
            self.assertTrue(result["success"])
            self.assertEqual(result["type"], "text_search")
            self.assertEqual(result["items"], final_items)
            self.assertEqual(result["total_found"], 2)
            self.assertEqual(result["total_models"], 3)
            self.assertIn("performance", result)

            # Verify method calls with correct parameters
            self.handler._extract_text_search_parameters.assert_called_once_with(parameters)
            self.metadata_table_manager.get_metadata_table_weights.assert_called_once()

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

            # Verify _search_model_chunks_table was called with correct parameters
            mock_search_chunks.assert_called_once()
            chunks_call_args = mock_search_chunks.call_args[0]
            self.assertEqual(chunks_call_args[0], query)
            self.assertEqual(chunks_call_args[1], chroma_filters)
            self.assertEqual(chunks_call_args[2], requested_limit)
            self.assertEqual(chunks_call_args[3], metadata_results)
            self.assertEqual(chunks_call_args[4], user_id)
            self.assertIsInstance(chunks_call_args[5], float)  # chunks_search_start time

            self.handler._fetch_complete_model_metadata.assert_called_once_with(
                query, chunks_results, table_weights, user_id
            )

            self.handler._process_model_descriptions_text_search.assert_called_once_with(
                query, metadata_complete_results
            )

            self.handler._calculate_model_distances.assert_called_once_with(
                description_results, table_weights, collection_stats
            )

            self.handler._sort_and_limit_search_results.assert_called_once_with(
                calculated_results, requested_limit
            )

            self.handler._prepare_text_search_items.assert_called_once_with(sorted_results)

            # Verify performance metrics were calculated
            self.handler.performance_metrics.calculate_text_search_performance.assert_called_once()
            perf_call_args = self.handler.performance_metrics.calculate_text_search_performance.call_args[0]
            self.assertIsInstance(perf_call_args[0], float)  # start_time
            self.assertIsInstance(perf_call_args[1], float)  # metadata_search_time
            self.assertEqual(perf_call_args[2], 50.0)  # chunks_search_time
            self.assertEqual(perf_call_args[3], parameters)  # parameters

    async def test_handle_text_search_empty_results(self):
        """Test handle_text_search when no results are found"""
        # Setup test data
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
        self.metadata_table_manager.get_metadata_table_weights.return_value = table_weights

        # Mock _get_collection_distance_stats_for_query
        collection_stats = {
            "model_descriptions": {"min": 0.1, "max": 0.9},
            "model_params": {"min": 0.2, "max": 0.8},
            "model_tags": {"min": 0.3, "max": 0.7},
            "model_scripts_chunks": {"min": 0.1, "max": 0.9}
        }
        self.handler._get_collection_distance_stats_for_query.return_value = collection_stats

        # Mock _search_all_metadata_tables - empty results
        metadata_results = {}
        self.handler._search_all_metadata_tables.return_value = metadata_results

        # Patch _search_model_chunks_table for this test
        with patch.object(self.handler, '_search_model_chunks_table', new_callable=AsyncMock) as mock_search_chunks:
            # Mock _search_model_chunks_table - still empty results
            mock_search_chunks.return_value = ({}, 50.0)

            # Mock remaining methods with empty results
            self.handler._fetch_complete_model_metadata.return_value = {}
            self.handler._process_model_descriptions_text_search.return_value = {}
            self.handler._calculate_model_distances.return_value = {}
            self.handler._sort_and_limit_search_results.return_value = []
            self.handler._prepare_text_search_items.return_value = []

            # Mock performance metrics calculation
            self.handler.performance_metrics.calculate_text_search_performance.return_value = {
                "total_time_ms": 200.0,
                "metadata_search_time_ms": 100.0,
                "chunks_search_time_ms": 50.0
            }

            # Call the method
            result = await self.handler.handle_text_search(query, parameters)

            # Assertions
            self.assertTrue(result["success"])
            self.assertEqual(result["type"], "text_search")
            self.assertEqual(result["items"], [])
            self.assertEqual(result["total_found"], 0)
            self.assertEqual(result["total_models"], 0)
            self.assertIn("performance", result)

    async def test_handle_text_search_error(self):
        """Test handle_text_search when an error occurs"""
        # Setup test data
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
            await self.handler.handle_text_search(query, parameters)

        # Verify that the exception is propagated
        self.assertEqual(str(context.exception), "Test error")

    async def test_handle_text_search_integration(self):
        """Test handle_text_search with more realistic integration of parent class methods"""
        # For this test, we'll patch fewer methods to make it more like an integration test

        # Restore some of the original methods we patched in setUp
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
        self.metadata_table_manager.get_metadata_table_weights.return_value = table_weights

        # Mock _get_collection_distance_stats_for_query
        collection_stats = {
            "model_descriptions": {"min": 0.1, "max": 0.9, "percentile_10": 0.2, "percentile_90": 0.8},
            "model_params": {"min": 0.2, "max": 0.8, "percentile_10": 0.3, "percentile_90": 0.7},
            "model_scripts_chunks": {"min": 0.1, "max": 0.9, "percentile_10": 0.2, "percentile_90": 0.8}
        }
        self.handler._get_collection_distance_stats_for_query.return_value = collection_stats

        # Set up the chain of method calls with test data
        # Mock _search_all_metadata_tables
        metadata_results = {
            "model1": {
                "model_id": "model1",
                "tables": ["model_descriptions"],
                "match_source": "metadata",
                "table_initial_distances": {"model_descriptions": 0.2}
            },
            "model2": {
                "model_id": "model2",
                "tables": ["model_params"],
                "match_source": "metadata",
                "table_initial_distances": {"model_params": 0.3}
            }
        }
        self.handler._search_all_metadata_tables.return_value = metadata_results

        # Patch _search_model_chunks_table for this test
        with patch.object(self.handler, '_search_model_chunks_table', new_callable=AsyncMock) as mock_search_chunks:
            # Mock _search_model_chunks_table to add a new model and update model1
            chunks_results = metadata_results.copy()
            chunks_results["model1"]["match_source"] = "metadata+chunks"
            chunks_results["model1"]["chunk_initial_distance"] = 0.15
            chunks_results["model3"] = {
                "model_id": "model3",
                "tables": ["chunks"],
                "match_source": "chunks",
                "chunk_initial_distance": 0.25
            }
            mock_search_chunks.return_value = (chunks_results, 50.0)

            # Mock _fetch_complete_model_metadata
            self.handler._fetch_complete_model_metadata.return_value = chunks_results

            # Mock _process_model_descriptions_text_search
            processed_results = chunks_results.copy()
            processed_results["model1"]["metadata"] = {"name": "Model 1", "description": "Description 1"}
            processed_results["model2"]["metadata"] = {"name": "Model 2", "description": "Description 2"}
            processed_results["model3"]["metadata"] = {"name": "Model 3", "description": "Description 3"}
            self.handler._process_model_descriptions_text_search.return_value = processed_results

            # Mock _calculate_model_distances
            # Set distances that will ensure specific sorting order
            calculated_results = processed_results.copy()
            calculated_results["model1"]["distance"] = 0.2  # Medium distance
            calculated_results["model2"]["distance"] = 0.3  # Highest distance
            calculated_results["model3"]["distance"] = 0.1  # Lowest distance
            self.handler._calculate_model_distances.return_value = calculated_results

            # Mock performance metrics calculation
            self.handler.performance_metrics.calculate_text_search_performance.return_value = {
                "total_time_ms": 200.0,
                "metadata_search_time_ms": 100.0,
                "chunks_search_time_ms": 50.0
            }

            # Call the method
            result = await self.handler.handle_text_search(query, parameters)

            # Assertions
            self.assertTrue(result["success"])
            self.assertEqual(result["type"], "text_search")
            self.assertEqual(len(result["items"]), 2)  # Limited to 2 as per parameters
            self.assertEqual(result["total_found"], 2)
            self.assertEqual(result["total_models"], 3)

            # The results should be ordered by distance (lower is better)
            self.assertEqual(result["items"][0]["model_id"], "model3")  # Lowest distance first
            self.assertEqual(result["items"][1]["model_id"], "model1")  # Second lowest distance

            # model2 should not be included because limit is 2 and it has the highest distance
            model_ids = [item["model_id"] for item in result["items"]]
            self.assertNotIn("model2", model_ids)

            # Check ranks
            self.assertEqual(result["items"][0]["rank"], 1)
            self.assertEqual(result["items"][1]["rank"], 2)

            # Check match sources
            self.assertEqual(result["items"][0]["match_source"], "chunks")
            self.assertEqual(result["items"][1]["match_source"], "metadata+chunks")


if __name__ == '__main__':
    unittest.main()