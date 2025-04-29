import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.query_engine.handlers.notebook_request_handler import NotebookRequestHandler


class TestNotebookRequestHandler(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Create mock dependencies
        self.chroma_manager = MagicMock()
        self.access_control_manager = MagicMock()
        self.analytics = MagicMock()

        # Setup chroma_manager.search to return appropriate model metadata
        async def mock_search(collection_name=None, query=None, where=None, include=None, **kwargs):
            # Simple implementation that returns metadata for requested models
            if query in ["model1", "model2", "model3"]:
                return [{"model_id": query, "metadata": {"name": f"Model {query[-1]}", "version": f"{query[-1]}.0"}}]

            # If using a filter, return matching models based on the test name
            if where is not None:
                if self._testMethodName == "test_handle_notebook_request_no_access":
                    return []  # No models accessible
                elif self._testMethodName == "test_handle_notebook_request_partial_access":
                    return [
                        {"model_id": "model1", "metadata": {"name": "Model 1", "version": "1.0"}},
                        {"model_id": "model3", "metadata": {"name": "Model 3", "version": "3.0"}}
                    ]  # Only model1 and model3 accessible
                else:
                    # Default case: return all models in the original model_ids parameter
                    models = []
                    for model_id in self.current_model_ids:
                        models.append({"model_id": model_id,
                                       "metadata": {"name": f"Model {model_id[-1]}", "version": f"{model_id[-1]}.0"}})
                    return models

            return []

        self.chroma_manager.search = AsyncMock(side_effect=mock_search)

        # Setup access_control_manager.check_access to control access based on test name
        def mock_check_access(doc, user_id, permission_type):
            model_id = doc.get("model_id")

            if self._testMethodName == "test_handle_notebook_request_no_access":
                return False  # No access for any model
            elif self._testMethodName == "test_handle_notebook_request_partial_access":
                return model_id in ["model1", "model3"]  # Only access to model1 and model3
            else:
                return True  # Default: access to all models

        self.access_control_manager.check_access = MagicMock(side_effect=mock_check_access)

        # Initialize an empty list to track the current model_ids in the test
        self.current_model_ids = []

        # Create the instance we're testing
        self.handler = NotebookRequestHandler(
            chroma_manager=self.chroma_manager,
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

        # Store model_ids for mock_search
        self.current_model_ids = parameters["model_ids"]

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
        self.chroma_manager.search.assert_any_call(
            collection_name="model_descriptions",
            query="model1",
            include=["metadatas"]
        )
        self.chroma_manager.search.assert_any_call(
            collection_name="model_descriptions",
            query="model2",
            include=["metadatas"]
        )
        self.access_control_manager.check_access.assert_called()

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

        # Store model_ids for mock_search
        self.current_model_ids = parameters["model_ids"]

        # Call the method and expect an exception
        with self.assertRaises(ValueError) as context:
            await self.handler.handle_notebook_request(query, parameters)

        # Assertions
        self.assertEqual(str(context.exception), "User does not have access to any of the requested models")

        # Verify method calls
        self.chroma_manager.search.assert_any_call(
            collection_name="model_descriptions",
            query="model1",
            include=["metadatas"]
        )
        self.chroma_manager.search.assert_any_call(
            collection_name="model_descriptions",
            query="model2",
            include=["metadatas"]
        )
        self.access_control_manager.check_access.assert_called()

    async def test_handle_notebook_request_partial_access(self):
        """Test handle_notebook_request when user has access to some but not all models."""
        # Setup test data
        query = "Generate notebook"
        parameters = {
            "user_id": "user123",
            "model_ids": ["model1", "model2", "model3"]
        }

        # Store model_ids for mock_search
        self.current_model_ids = parameters["model_ids"]

        # Call the method
        result = await self.handler.handle_notebook_request(query, parameters)

        # Assertions
        self.assertTrue(result['success'])
        # Should only include models with access
        self.assertEqual(result['request']['model_ids'], ["model1", "model3"])

        # Verify method calls
        self.chroma_manager.search.assert_any_call(
            collection_name="model_descriptions",
            query="model1",
            include=["metadatas"]
        )
        self.chroma_manager.search.assert_any_call(
            collection_name="model_descriptions",
            query="model2",
            include=["metadatas"]
        )
        self.chroma_manager.search.assert_any_call(
            collection_name="model_descriptions",
            query="model3",
            include=["metadatas"]
        )
        self.access_control_manager.check_access.assert_called()

    async def test_handle_notebook_request_default_values(self):
        """Test handle_notebook_request with minimal parameters, using default values."""
        # Setup test data
        query = "Generate notebook"
        parameters = {
            "user_id": "user123",
            "model_ids": ["model1"]
        }

        # Store model_ids for mock_search
        self.current_model_ids = parameters["model_ids"]

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
            access_control_manager=self.access_control_manager,
            analytics=self.analytics
        )

        # Setup test case
        query = "Generate notebook"
        parameters = {
            "user_id": "user123",
            "model_ids": ["model1"]
        }

        # Store model_ids for mock_search
        self.current_model_ids = parameters["model_ids"]

        # Call the method
        await handler.handle_notebook_request(query, parameters)

        # Verify logging
        mock_logger.debug.assert_called_once_with(f"Handling notebook request: {parameters}")
        mock_logger.error.assert_not_called()

        # Test error path by making search raise an exception
        self.chroma_manager.search.side_effect = Exception("Test error")

        # Call the method and expect an exception
        with self.assertRaises(Exception):
            await handler.handle_notebook_request(query, parameters)

        # Verify error logging
        mock_logger.error.assert_called_once()
        error_call_args = mock_logger.error.call_args[0]
        self.assertTrue("Error in notebook request" in error_call_args[0])


if __name__ == '__main__':
    unittest.main()