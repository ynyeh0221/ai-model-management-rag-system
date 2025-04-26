import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.query_engine.handlers.notebook_request_handler import NotebookRequestHandler


class TestNotebookRequestHandler(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Create mock dependencies
        self.chroma_manager = MagicMock()
        self.model_data_fetcher = MagicMock()
        self.access_control_manager = MagicMock()
        self.analytics = MagicMock()

        # Make async methods
        self.model_data_fetcher.fetch_model_metadata = AsyncMock()

        # Create the instance we're testing
        self.handler = NotebookRequestHandler(
            chroma_manager=self.chroma_manager,
            model_data_fetcher=self.model_data_fetcher,
            access_control_manager=self.access_control_manager,
            analytics=self.analytics
        )

    async def test_handle_notebook_request_success(self):
        """Test handle_notebook_request with valid parameters and user access."""
        # Setup test data
        query = "Generate notebook"
        parameters = {
            "user_id": "user123",
            "model_ids": ["model1", "model2"],
            "analysis_types": ["basic", "advanced"],
            "dataset": "dataset1",
            "resources": "high-memory"
        }

        # Mock model_data_fetcher to return valid metadata
        model1_metadata = {"name": "Model 1", "version": "1.0"}
        model2_metadata = {"name": "Model 2", "version": "2.0"}
        self.model_data_fetcher.fetch_model_metadata.side_effect = [
            model1_metadata,
            model2_metadata
        ]

        # Mock access_control_manager to grant access to both models
        self.access_control_manager.check_access.return_value = True

        # Call the method
        result = await self.handler.handle_notebook_request(query, parameters)

        # Assertions
        self.assertTrue(result['success'])
        self.assertEqual(result['type'], 'notebook_request')

        # Check request data
        self.assertEqual(result['request']['model_ids'], ["model1", "model2"])
        self.assertEqual(result['request']['analysis_types'], ["basic", "advanced"])
        self.assertEqual(result['request']['dataset'], "dataset1")
        self.assertEqual(result['request']['resources'], "high-memory")
        self.assertEqual(result['request']['user_id'], "user123")

        # Check result data
        self.assertIn('notebook_id', result['result'])
        self.assertIn('title', result['result'])
        self.assertEqual(result['result']['status'], 'pending')
        self.assertIn('estimated_completion_time', result['result'])

        # Verify method calls
        self.model_data_fetcher.fetch_model_metadata.assert_any_call("model1")
        self.model_data_fetcher.fetch_model_metadata.assert_any_call("model2")
        self.access_control_manager.check_access.assert_any_call(
            {'metadata': model1_metadata}, "user123", "view"
        )
        self.access_control_manager.check_access.assert_any_call(
            {'metadata': model2_metadata}, "user123", "view"
        )

    async def test_handle_notebook_request_no_model_ids(self):
        """Test handle_notebook_request with missing model IDs."""
        # Setup test data
        query = "Generate notebook"
        parameters = {
            "user_id": "user123",
            # Missing model_ids
            "analysis_types": ["basic"]
        }

        # Call the method and expect an exception
        with self.assertRaises(ValueError) as context:
            await self.handler.handle_notebook_request(query, parameters)

        # Assertions
        self.assertEqual(str(context.exception), "Notebook generation requires at least one model ID")

    async def test_handle_notebook_request_no_access(self):
        """Test handle_notebook_request when user doesn't have access to any models."""
        # Setup test data
        query = "Generate notebook"
        parameters = {
            "user_id": "user123",
            "model_ids": ["model1", "model2"]
        }

        # Mock model_data_fetcher to return valid metadata
        model1_metadata = {"name": "Model 1", "version": "1.0"}
        model2_metadata = {"name": "Model 2", "version": "2.0"}
        self.model_data_fetcher.fetch_model_metadata.side_effect = [
            model1_metadata,
            model2_metadata
        ]

        # Mock access_control_manager to deny access to all models
        self.access_control_manager.check_access.return_value = False

        # Call the method and expect an exception
        with self.assertRaises(ValueError) as context:
            await self.handler.handle_notebook_request(query, parameters)

        # Assertions
        self.assertEqual(str(context.exception), "User does not have access to any of the requested models")

        # Verify method calls
        self.model_data_fetcher.fetch_model_metadata.assert_any_call("model1")
        self.model_data_fetcher.fetch_model_metadata.assert_any_call("model2")
        self.access_control_manager.check_access.assert_any_call(
            {'metadata': model1_metadata}, "user123", "view"
        )
        self.access_control_manager.check_access.assert_any_call(
            {'metadata': model2_metadata}, "user123", "view"
        )

    async def test_handle_notebook_request_partial_access(self):
        """Test handle_notebook_request when user has access to some but not all models."""
        # Setup test data
        query = "Generate notebook"
        parameters = {
            "user_id": "user123",
            "model_ids": ["model1", "model2", "model3"]
        }

        # Mock model_data_fetcher to return valid metadata
        model1_metadata = {"name": "Model 1", "version": "1.0"}
        model2_metadata = {"name": "Model 2", "version": "2.0"}
        model3_metadata = {"name": "Model 3", "version": "3.0"}
        self.model_data_fetcher.fetch_model_metadata.side_effect = [
            model1_metadata,
            model2_metadata,
            model3_metadata
        ]

        # Mock access_control_manager to grant access to only model1 and model3
        def check_access_side_effect(doc, user_id, permission):
            return doc['metadata'] in [model1_metadata, model3_metadata]

        self.access_control_manager.check_access.side_effect = check_access_side_effect

        # Call the method
        result = await self.handler.handle_notebook_request(query, parameters)

        # Assertions
        self.assertTrue(result['success'])
        # Should only include models with access
        self.assertEqual(result['request']['model_ids'], ["model1", "model3"])

        # Verify method calls
        self.model_data_fetcher.fetch_model_metadata.assert_any_call("model1")
        self.model_data_fetcher.fetch_model_metadata.assert_any_call("model2")
        self.model_data_fetcher.fetch_model_metadata.assert_any_call("model3")

    async def test_handle_notebook_request_default_values(self):
        """Test handle_notebook_request with minimal parameters, using default values."""
        # Setup test data
        query = "Generate notebook"
        parameters = {
            "user_id": "user123",
            "model_ids": ["model1"]
        }

        # Mock model_data_fetcher to return valid metadata
        model_metadata = {"name": "Model 1", "version": "1.0"}
        self.model_data_fetcher.fetch_model_metadata.return_value = model_metadata

        # Mock access_control_manager to grant access
        self.access_control_manager.check_access.return_value = True

        # Call the method
        result = await self.handler.handle_notebook_request(query, parameters)

        # Assertions
        self.assertTrue(result['success'])
        # Check default values were used
        self.assertEqual(result['request']['analysis_types'], ['basic'])
        self.assertEqual(result['request']['resources'], 'standard')
        self.assertIsNone(result['request']['dataset'])

    @patch('logging.getLogger')
    async def test_handle_notebook_request_logging(self, mock_get_logger):
        """Test that the handler logs messages appropriately."""
        # Setup mock logger
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        # Recreate handler with mocked logger
        handler = NotebookRequestHandler(
            chroma_manager=self.chroma_manager,
            model_data_fetcher=self.model_data_fetcher,
            access_control_manager=self.access_control_manager,
            analytics=self.analytics
        )

        # Setup test case
        query = "Generate notebook"
        parameters = {
            "user_id": "user123",
            "model_ids": ["model1"]
        }

        # Mock dependencies for success path
        self.model_data_fetcher.fetch_model_metadata.return_value = {"name": "Model 1"}
        self.access_control_manager.check_access.return_value = True

        # Call the method
        await handler.handle_notebook_request(query, parameters)

        # Verify logging
        mock_logger.debug.assert_called_once_with(f"Handling notebook request: {parameters}")
        mock_logger.error.assert_not_called()

        # Test error path
        self.model_data_fetcher.fetch_model_metadata.side_effect = Exception("Test error")

        # Call the method and expect an exception
        with self.assertRaises(Exception):
            await handler.handle_notebook_request(query, parameters)

        # Verify error logging
        mock_logger.error.assert_called_once()
        error_call_args = mock_logger.error.call_args[0]
        self.assertTrue("Error in notebook request" in error_call_args[0])


if __name__ == '__main__':
    unittest.main()