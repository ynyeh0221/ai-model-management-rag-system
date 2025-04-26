import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.query_engine.handlers.fallback_search_handler import FallbackSearchHandler
from src.core.query_engine.handlers.metadata_search_handler import MetadataSearchHandler


class TestFallbackSearchHandler(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Create mock for MetadataSearchHandler
        self.metadata_search_manager = MagicMock(spec=MetadataSearchHandler)
        self.metadata_search_manager.handle_metadata_search = AsyncMock()

        # Create the instance we're testing
        self.handler = FallbackSearchHandler(metadata_search_manager=self.metadata_search_manager)

        # Add a mock for handle_text_search since it's called but not implemented in the provided code
        self.handler.handle_text_search = AsyncMock()

    async def test_fallback_search_text_search_success(self):
        """Test the case where text search succeeds and returns results."""
        # Setup mock return values
        query = "test query"
        parameters = {"user_id": "user123", "limit": 10}

        # Mock text search results with success and some items
        text_search_results = {
            'success': True,
            'type': 'text_search',
            'items': [
                {'id': 'item1', 'metadata': {'description': 'Test item 1'}},
                {'id': 'item2', 'metadata': {'description': 'Test item 2'}}
            ],
            'total_found': 2
        }
        self.handler.handle_text_search.return_value = text_search_results

        # Call the method
        result = await self.handler.handle_fallback_search(query, parameters)

        # Assertions
        self.assertTrue(result['success'])
        self.assertEqual(result['type'], 'text_search')
        self.assertEqual(len(result['items']), 2)
        self.assertEqual(result['total_found'], 2)

        # Verify that text search was called but metadata search was not
        self.handler.handle_text_search.assert_called_once_with(query, parameters)
        self.metadata_search_manager.handle_metadata_search.assert_not_called()

    async def test_fallback_search_metadata_search_success(self):
        """Test the case where text search fails to find results but metadata search succeeds."""
        # Setup mock return values
        query = "test query"
        parameters = {"user_id": "user123", "limit": 10}

        # Mock text search results with success but no items
        text_search_results = {
            'success': True,
            'type': 'text_search',
            'items': [],
            'total_found': 0
        }
        self.handler.handle_text_search.return_value = text_search_results

        # Mock metadata search results with success and some items
        metadata_search_results = {
            'success': True,
            'type': 'metadata_search',
            'items': [
                {'id': 'item1', 'metadata': {'description': 'Test item 1'}},
                {'id': 'item2', 'metadata': {'description': 'Test item 2'}}
            ],
            'total_found': 2
        }
        self.metadata_search_manager.handle_metadata_search.return_value = metadata_search_results

        # Call the method
        result = await self.handler.handle_fallback_search(query, parameters)

        # Assertions
        self.assertTrue(result['success'])
        self.assertEqual(result['type'], 'metadata_search')
        self.assertEqual(len(result['items']), 2)
        self.assertEqual(result['total_found'], 2)

        # Verify that both searches were called
        self.handler.handle_text_search.assert_called_once_with(query, parameters)
        self.metadata_search_manager.handle_metadata_search.assert_called_once_with(query, parameters)

    async def test_fallback_search_no_results(self):
        """Test the case where both text search and metadata search fail to find results."""
        # Setup mock return values
        query = "test query"
        parameters = {"user_id": "user123", "limit": 10}

        # Mock text search results with success but no items
        text_search_results = {
            'success': True,
            'type': 'text_search',
            'items': [],
            'total_found': 0
        }
        self.handler.handle_text_search.return_value = text_search_results

        # Mock metadata search results with success but no items
        metadata_search_results = {
            'success': True,
            'type': 'metadata_search',
            'items': [],
            'total_found': 0
        }
        self.metadata_search_manager.handle_metadata_search.return_value = metadata_search_results

        # Call the method
        result = await self.handler.handle_fallback_search(query, parameters)

        # Assertions
        self.assertTrue(result['success'])
        self.assertEqual(result['type'], 'fallback_search')
        self.assertEqual(len(result['items']), 0)
        self.assertEqual(result['total_found'], 0)
        self.assertIn('message', result)
        self.assertEqual(result['message'], "No results found using various search strategies")

        # Verify that both searches were called
        self.handler.handle_text_search.assert_called_once_with(query, parameters)
        self.metadata_search_manager.handle_metadata_search.assert_called_once_with(query, parameters)

    async def test_fallback_search_text_search_exception(self):
        """Test the case where text search throws an exception."""
        # Setup mock return values
        query = "test query"
        parameters = {"user_id": "user123", "limit": 10}

        # Mock text search to raise exception
        self.handler.handle_text_search.side_effect = Exception("Text search error")

        # Call the method
        result = await self.handler.handle_fallback_search(query, parameters)

        # Assertions
        self.assertFalse(result['success'])
        self.assertEqual(result['type'], 'fallback_search')
        self.assertEqual(len(result['items']), 0)
        self.assertEqual(result['total_found'], 0)
        self.assertIn('error', result)
        self.assertEqual(result['error'], "An error occurred during the search")

        # Verify that text search was called but metadata search was not
        self.handler.handle_text_search.assert_called_once_with(query, parameters)
        self.metadata_search_manager.handle_metadata_search.assert_not_called()

    async def test_fallback_search_metadata_search_exception(self):
        """Test the case where text search finds no results and metadata search throws an exception."""
        # Setup mock return values
        query = "test query"
        parameters = {"user_id": "user123", "limit": 10}

        # Mock text search results with success but no items
        text_search_results = {
            'success': True,
            'type': 'text_search',
            'items': [],
            'total_found': 0
        }
        self.handler.handle_text_search.return_value = text_search_results

        # Mock metadata search to raise exception
        self.metadata_search_manager.handle_metadata_search.side_effect = Exception("Metadata search error")

        # Call the method
        result = await self.handler.handle_fallback_search(query, parameters)

        # Assertions
        self.assertTrue(result['success'])
        self.assertEqual(result['type'], 'fallback_search')
        self.assertEqual(len(result['items']), 0)
        self.assertEqual(result['total_found'], 0)
        self.assertIn('message', result)
        self.assertEqual(result['message'], "No results found using various search strategies")

        # Verify that both searches were called
        self.handler.handle_text_search.assert_called_once_with(query, parameters)
        self.metadata_search_manager.handle_metadata_search.assert_called_once_with(query, parameters)

    @patch('logging.getLogger')
    async def test_fallback_search_logging(self, mock_get_logger):
        """Test that the handler logs warnings and errors appropriately."""
        # Setup mock logger
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        # Recreate handler with mocked logger
        handler = FallbackSearchHandler(metadata_search_manager=self.metadata_search_manager)
        handler.handle_text_search = AsyncMock()

        # Setup test case
        query = "test query"
        parameters = {"user_id": "user123", "limit": 10}

        # Mock text search to raise exception
        handler.handle_text_search.side_effect = Exception("Text search error")

        # Call the method
        await handler.handle_fallback_search(query, parameters)

        # Verify logging
        mock_logger.warning.assert_called_once_with(f"Using fallback search for query: {query}")
        mock_logger.error.assert_called_once()
        # Check that "Error in fallback search" is in the error message
        error_call_args = mock_logger.error.call_args[0]
        self.assertTrue("Error in fallback search" in error_call_args[0])


if __name__ == '__main__':
    unittest.main()