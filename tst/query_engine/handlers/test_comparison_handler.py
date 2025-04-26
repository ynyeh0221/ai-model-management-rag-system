import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from src.query_engine.handlers.comparison_handler import ComparisonHandler
from src.query_engine.handlers.metadata_search_handler import MetadataSearchHandler
from src.query_engine.handlers.utils.distance_normalizer import DistanceNormalizer
from src.query_engine.handlers.utils.filter_translator import FilterTranslator
from src.query_engine.handlers.utils.metadata_table_manager import MetadataTableManager
from src.vector_db_manager.access_control import AccessControlManager
from src.vector_db_manager.chroma_manager import ChromaManager


class TestComparisonHandler(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Create mock dependencies
        self.chroma_manager = MagicMock(spec=ChromaManager)
        self.access_control_manager = MagicMock(spec=AccessControlManager)
        self.filter_translator = MagicMock(spec=FilterTranslator)
        self.distance_normalizer = MagicMock(spec=DistanceNormalizer)
        self.metadata_table_manager = MagicMock(spec=MetadataTableManager)
        self.metadata_search_manager = MagicMock(spec=MetadataSearchHandler)

        # Make asynchronous methods
        self.metadata_search_manager.handle_metadata_search = AsyncMock()

        # Create the instance we're testing
        self.handler = ComparisonHandler(
            metadata_table_manager=self.metadata_table_manager,
            metadata_search_manager=self.metadata_search_manager,
            access_control_manager=self.access_control_manager,
            filter_translator=self.filter_translator,
            chroma_manager=self.chroma_manager,
            distance_normalizer=self.distance_normalizer
        )

        # Mock the parent class methods
        self.handler._extract_text_search_parameters = MagicMock()
        self.handler._search_all_metadata_tables = AsyncMock()
        self.handler._fetch_complete_model_metadata = AsyncMock()
        self.handler._process_model_descriptions_text_search = AsyncMock()

        # Set up test data that will be reused
        self.user_id = "user123"
        self.table_weights = {
            "model_descriptions": 0.5,
            "model_params": 0.3,
            "model_tags": 0.2
        }

        # Mock _extract_text_search_parameters
        self.handler._extract_text_search_parameters.return_value = (
            self.user_id, 10, {"filter": "test"}
        )

        # Mock get_metadata_table_weights
        self.metadata_table_manager.get_metadata_table_weights.return_value = self.table_weights

    async def test_handle_comparison_model_id_path(self):
        """Test handle_comparison calls handle_comparison_model_id when model_ids are provided."""
        # Setup
        query = "Compare models"
        parameters = {"filters": {"model_id": ["model1", "model2"]}}

        # Patch the specific methods to ensure they're called
        with patch.object(self.handler, 'handle_comparison_model_id', new_callable=AsyncMock) as mock_model_id, \
                patch.object(self.handler, 'handle_comparison_cohort', new_callable=AsyncMock) as mock_cohort:
            # Return values for the mocked methods
            mock_model_id.return_value = {"success": True, "type": "comparison"}

            # Call the method
            result = await self.handler.handle_comparison(query, parameters)

            # Assertions
            mock_model_id.assert_called_once_with(query, parameters)
            mock_cohort.assert_not_called()
            self.assertEqual(result, {"success": True, "type": "comparison"})

    async def test_handle_comparison_cohort_path(self):
        """Test handle_comparison calls handle_comparison_cohort when no model_ids are provided."""
        # Setup
        query = "Compare cohorts"
        parameters = {"filters": {}, "cohorts": ["cohort1", "cohort2"], "base_query": "base query"}

        # Patch the specific methods to ensure they're called
        with patch.object(self.handler, 'handle_comparison_model_id', new_callable=AsyncMock) as mock_model_id, \
                patch.object(self.handler, 'handle_comparison_cohort', new_callable=AsyncMock) as mock_cohort:
            # Return values for the mocked methods
            mock_cohort.return_value = {"success": True, "type": "comparison_cohort"}

            # Call the method
            result = await self.handler.handle_comparison(query, parameters)

            # Assertions
            mock_model_id.assert_not_called()
            mock_cohort.assert_called_once_with(query, parameters)
            self.assertEqual(result, {"success": True, "type": "comparison_cohort"})

    async def test_handle_comparison_model_id_success(self):
        """Test handle_comparison_model_id with valid model IDs."""
        # Setup
        query = "Compare models"
        parameters = {"filters": {"model_id": ["model1", "model2", "model3"]}}

        # Mock search results
        all_results = {
            "model1": {
                "model_id": "model1",
                "tables": ["model_descriptions"],
                "match_source": "metadata",
                "metadata": {"name": "Model 1", "description": "Description 1"},
                "merged_description": "Merged description 1",
                "distance": 0.1
            },
            "model2": {
                "model_id": "model2",
                "tables": ["model_params"],
                "match_source": "metadata",
                "metadata": {"name": "Model 2", "description": "Description 2"},
                "merged_description": "Merged description 2",
                "distance": 0.2
            },
            "model3": {
                "model_id": "model3",
                "tables": ["model_tags"],
                "match_source": "metadata",
                "metadata": {"name": "Model 3", "description": "Description 3"},
                "merged_description": "Merged description 3",
                "distance": 0.3
            }
        }

        # Set up mocks
        self.handler._search_all_metadata_tables.return_value = all_results
        self.handler._fetch_complete_model_metadata.return_value = all_results
        self.handler._process_model_descriptions_text_search.return_value = all_results

        # Call the method
        result = await self.handler.handle_comparison_model_id(query, parameters)

        # Assertions
        self.assertTrue(result['success'])
        self.assertEqual(result['type'], 'comparison')
        self.assertEqual(len(result['items']), 3)
        self.assertEqual(result['total_found'], 3)
        self.assertEqual(result['total_models'], 3)
        self.assertIn('performance', result)

        # Check that model IDs are in the correct order
        self.assertEqual(result['items'][0]['model_id'], 'model1')
        self.assertEqual(result['items'][1]['model_id'], 'model2')
        self.assertEqual(result['items'][2]['model_id'], 'model3')

        # Check that ranks are assigned correctly
        self.assertEqual(result['items'][0]['rank'], 1)
        self.assertEqual(result['items'][1]['rank'], 2)
        self.assertEqual(result['items'][2]['rank'], 3)

        # Check that the correct methods were called with correct parameters
        self.handler._extract_text_search_parameters.assert_called_once_with(parameters)
        self.metadata_table_manager.get_metadata_table_weights.assert_called_once()

        expected_chroma_filters = {"model_id": {"$in": ["model1", "model2", "model3"]}}
        self.handler._search_all_metadata_tables.assert_called_once_with(
            query="",
            chroma_filters=expected_chroma_filters,
            requested_limit=3,
            table_weights=self.table_weights,
            user_id=self.user_id
        )

        self.handler._fetch_complete_model_metadata.assert_called_once_with(
            query="",
            all_results=all_results,
            table_weights=self.table_weights,
            user_id=self.user_id
        )

        self.handler._process_model_descriptions_text_search.assert_called_once_with(
            query="",
            all_results=all_results
        )

    async def test_handle_comparison_model_id_error(self):
        """Test handle_comparison_model_id with insufficient model IDs."""
        # Setup
        query = "Compare models"
        parameters = {"filters": {"model_id": ["model1"]}}  # Only one model ID

        # Call the method and expect an exception
        with self.assertRaises(ValueError) as context:
            await self.handler.handle_comparison_model_id(query, parameters)

        # Assertions
        self.assertEqual(str(context.exception), "Comparison requires at least two model IDs")

        # Test with no model IDs at all
        parameters = {"filters": {}}
        with self.assertRaises(ValueError) as context:
            await self.handler.handle_comparison_model_id(query, parameters)

        self.assertEqual(str(context.exception), "Comparison requires at least two model IDs")

    async def test_handle_comparison_cohort_success(self):
        """Test handle_comparison_cohort with valid cohorts and base query."""
        # Setup
        query = "Compare cohorts"
        parameters = {
            "cohorts": ["cohort1", "cohort2"],
            "base_query": "Tell me about"
        }

        # Mock metadata search results for each cohort
        cohort1_results = {
            'success': True,
            'type': 'metadata_search',
            'items': [
                {
                    'id': 'model1',
                    'model_id': 'model1',
                    'metadata': {'description': 'Description 1'},
                    'distance': 0.1,
                    'rank': 1
                }
            ],
            'total_found': 1
        }

        cohort2_results = {
            'success': True,
            'type': 'metadata_search',
            'items': [
                {
                    'id': 'model2',
                    'model_id': 'model2',
                    'metadata': {'description': 'Description 2'},
                    'distance': 0.2,
                    'rank': 1
                }
            ],
            'total_found': 1
        }

        # Set up metadata_search_manager.handle_metadata_search to return different results
        self.metadata_search_manager.handle_metadata_search.side_effect = [
            cohort1_results,
            cohort2_results
        ]

        # Call the method
        result = await self.handler.handle_comparison_cohort(query, parameters)

        # Assertions
        self.assertTrue(result['success'])
        self.assertEqual(result['type'], 'comparison_cohort')
        self.assertEqual(len(result['items']), 2)
        self.assertEqual(result['total_found'], 2)
        self.assertIn('performance', result)

        # Check that the items are sorted by distance
        self.assertEqual(result['items'][0]['model_id'], 'model1')  # Lower distance (0.1)
        self.assertEqual(result['items'][1]['model_id'], 'model2')  # Higher distance (0.2)

        # Check that cohorts are tagged correctly
        self.assertEqual(result['items'][0]['cohort'], 'cohort1')
        self.assertEqual(result['items'][1]['cohort'], 'cohort2')

        # Check that the metadata search was called twice with correct queries
        self.assertEqual(self.metadata_search_manager.handle_metadata_search.call_count, 2)

        # First call should be for cohort1
        first_call_args = self.metadata_search_manager.handle_metadata_search.call_args_list[0][0]
        self.assertEqual(first_call_args[0], "Tell me about find models related to cohort1")
        self.assertEqual(first_call_args[1]['limit'], 2)

        # Second call should be for cohort2
        second_call_args = self.metadata_search_manager.handle_metadata_search.call_args_list[1][0]
        self.assertEqual(second_call_args[0], "Tell me about find models related to cohort2")
        self.assertEqual(second_call_args[1]['limit'], 2)

    async def test_handle_comparison_cohort_missing_cohorts(self):
        """Test handle_comparison_cohort with missing cohorts."""
        # Setup
        query = "Compare cohorts"
        parameters = {
            "base_query": "Tell me about"
            # Missing cohorts
        }

        # Call the method and expect an exception
        with self.assertRaises(ValueError) as context:
            await self.handler.handle_comparison_cohort(query, parameters)

        # Assertions
        self.assertEqual(str(context.exception), "More than one cohorts are required for comparison_cohort")

        # Test with invalid cohorts format
        parameters = {
            "cohorts": "not_a_list",
            "base_query": "Tell me about"
        }

        with self.assertRaises(ValueError) as context:
            await self.handler.handle_comparison_cohort(query, parameters)

        self.assertEqual(str(context.exception), "More than one cohorts are required for comparison_cohort")

    async def test_handle_comparison_cohort_missing_base_query(self):
        """Test handle_comparison_cohort with missing base query."""
        # Setup
        query = "Compare cohorts"
        parameters = {
            "cohorts": ["cohort1", "cohort2"]
            # Missing base_query
        }

        # Call the method and expect an exception
        with self.assertRaises(ValueError) as context:
            await self.handler.handle_comparison_cohort(query, parameters)

        # Assertions
        self.assertEqual(str(context.exception), "base_query is required for comparison_cohort")

    async def test_handle_comparison_cohort_metadata_search_error(self):
        """Test handle_comparison_cohort when metadata_search_manager raises an exception."""
        # Setup
        query = "Compare cohorts"
        parameters = {
            "cohorts": ["cohort1", "cohort2"],
            "base_query": "Tell me about"
        }

        # Make the first call to metadata_search_manager succeed and the second one fail
        cohort1_results = {
            'success': True,
            'type': 'metadata_search',
            'items': [
                {
                    'id': 'model1',
                    'model_id': 'model1',
                    'metadata': {'description': 'Description 1'},
                    'distance': 0.1,
                    'rank': 1
                }
            ],
            'total_found': 1
        }

        self.metadata_search_manager.handle_metadata_search.side_effect = [
            cohort1_results,
            Exception("Metadata search error")
        ]

        # Call the method and expect an exception
        with self.assertRaises(Exception) as context:
            await self.handler.handle_comparison_cohort(query, parameters)

        # Assertions
        self.assertEqual(str(context.exception), "Metadata search error")

        # Verify that metadata_search_manager.handle_metadata_search was called at least once
        self.metadata_search_manager.handle_metadata_search.assert_called()


if __name__ == '__main__':
    unittest.main()