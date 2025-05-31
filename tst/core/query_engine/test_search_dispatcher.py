import asyncio
import unittest
from unittest.mock import Mock, patch, AsyncMock

from src.core.query_engine.query_intent import QueryIntent
from src.core.query_engine.search_dispatcher import SearchDispatcher


class TestSearchDispatcher(unittest.TestCase):
    """Test class for SearchDispatcher"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create mock dependencies
        self.mock_chroma_manager = Mock()
        self.mock_text_embedder = Mock()
        self.mock_image_embedder = Mock()
        self.mock_access_control_manager = Mock()
        self.mock_analytics = Mock()
        self.mock_image_search_manager = Mock()

        # Mock the utility classes and handlers
        self.distance_normalizer_patcher = patch('src.core.query_engine.search_dispatcher.DistanceNormalizer')
        self.filter_translator_patcher = patch('src.core.query_engine.search_dispatcher.FilterTranslator')
        self.performance_metrics_patcher = patch('src.core.query_engine.search_dispatcher.PerformanceMetricsCalculator')
        self.metadata_table_manager_patcher = patch('src.core.query_engine.search_dispatcher.MetadataTableManager')
        self.metadata_search_handler_patcher = patch('src.core.query_engine.search_dispatcher.MetadataSearchHandler')
        self.image_search_handler_patcher = patch('src.core.query_engine.search_dispatcher.ImageSearchHandler')
        self.notebook_request_handler_patcher = patch('src.core.query_engine.search_dispatcher.NotebookRequestHandler')
        self.fallback_search_handler_patcher = patch('src.core.query_engine.search_dispatcher.FallbackSearchHandler')

        # Start all patches
        self.mock_distance_normalizer = self.distance_normalizer_patcher.start()
        self.mock_filter_translator = self.filter_translator_patcher.start()
        self.mock_performance_metrics = self.performance_metrics_patcher.start()
        self.mock_metadata_table_manager = self.metadata_table_manager_patcher.start()
        self.mock_metadata_search_handler = self.metadata_search_handler_patcher.start()
        self.mock_image_search_handler = self.image_search_handler_patcher.start()
        self.mock_notebook_request_handler = self.notebook_request_handler_patcher.start()
        self.mock_fallback_search_handler = self.fallback_search_handler_patcher.start()

        # Configure mock handlers to return AsyncMock for async methods
        self.mock_metadata_search_handler.return_value.handle_metadata_search = AsyncMock()
        self.mock_image_search_handler.return_value.handle_image_search = AsyncMock()
        self.mock_notebook_request_handler.return_value.handle_notebook_request = AsyncMock()
        self.mock_fallback_search_handler.return_value.handle_fallback_search = AsyncMock()

    def tearDown(self):
        """Clean up after each test method."""
        # Stop all patches
        self.distance_normalizer_patcher.stop()
        self.filter_translator_patcher.stop()
        self.performance_metrics_patcher.stop()
        self.metadata_table_manager_patcher.stop()
        self.metadata_search_handler_patcher.stop()
        self.image_search_handler_patcher.stop()
        self.notebook_request_handler_patcher.stop()
        self.fallback_search_handler_patcher.stop()

    def test_init_with_required_parameters(self):
        """Test SearchDispatcher initialization with required parameters only."""
        dispatcher = SearchDispatcher(
            chroma_manager=self.mock_chroma_manager,
            text_embedder=self.mock_text_embedder,
            image_embedder=self.mock_image_embedder
        )

        # Verify required attributes are set
        self.assertEqual(dispatcher.chroma_manager, self.mock_chroma_manager)
        self.assertEqual(dispatcher.text_embedder, self.mock_text_embedder)
        self.assertEqual(dispatcher.image_embedder, self.mock_image_embedder)
        self.assertIsNone(dispatcher.access_control_manager)
        self.assertIsNone(dispatcher.analytics)

        # Verify utility classes are initialized
        self.mock_distance_normalizer.assert_called_once()
        self.mock_filter_translator.assert_called_once()
        self.mock_performance_metrics.assert_called_once_with(None)
        self.mock_metadata_table_manager.assert_called_once_with(self.mock_chroma_manager, None)

    def test_init_with_all_parameters(self):
        """Test SearchDispatcher initialization with all parameters."""
        dispatcher = SearchDispatcher(
            chroma_manager=self.mock_chroma_manager,
            text_embedder=self.mock_text_embedder,
            image_embedder=self.mock_image_embedder,
            access_control_manager=self.mock_access_control_manager,
            analytics=self.mock_analytics,
            image_search_manager=self.mock_image_search_manager
        )

        # Verify all attributes are set
        self.assertEqual(dispatcher.chroma_manager, self.mock_chroma_manager)
        self.assertEqual(dispatcher.text_embedder, self.mock_text_embedder)
        self.assertEqual(dispatcher.image_embedder, self.mock_image_embedder)
        self.assertEqual(dispatcher.access_control_manager, self.mock_access_control_manager)
        self.assertEqual(dispatcher.analytics, self.mock_analytics)
        self.assertEqual(dispatcher.image_search_manager, self.mock_image_search_manager)

    def test_handlers_mapping_initialization(self):
        """Test that handler mapping is correctly initialized."""
        dispatcher = SearchDispatcher(
            chroma_manager=self.mock_chroma_manager,
            text_embedder=self.mock_text_embedder,
            image_embedder=self.mock_image_embedder
        )

        # Verify all intents are mapped
        expected_intents = [
            QueryIntent.RETRIEVAL,
            QueryIntent.NOTEBOOK,
            QueryIntent.IMAGE_SEARCH,
            QueryIntent.METADATA,
            QueryIntent.UNKNOWN
        ]

        for intent in expected_intents:
            self.assertIn(intent, dispatcher.handlers)

    async def test_dispatch_with_retrieval_intent(self):
        """Test dispatch method with RETRIEVAL intent."""
        dispatcher = SearchDispatcher(
            chroma_manager=self.mock_chroma_manager,
            text_embedder=self.mock_text_embedder,
            image_embedder=self.mock_image_embedder,
            analytics=self.mock_analytics
        )

        # Mock handler response
        mock_response = {'items': [{'id': 1, 'content': 'test'}]}
        dispatcher.metadata_search_manager.handle_metadata_search.return_value = mock_response

        # Mock performance metrics sanitize method
        dispatcher.performance_metrics.sanitize_parameters = Mock(return_value={'clean': 'params'})

        query = "test query"
        parameters = {'param1': 'value1', 'query_id': 'test123'}

        result = await dispatcher.dispatch(query, QueryIntent.RETRIEVAL, parameters)

        # Verify handler was called correctly
        dispatcher.metadata_search_manager.handle_metadata_search.assert_called_once_with(query, parameters)

        # Verify result structure
        self.assertIn('items', result)
        self.assertIn('metadata', result)
        self.assertEqual(result['metadata']['intent'], 'retrieval')  # Changed to lowercase
        self.assertIn('execution_time_ms', result['metadata'])
        self.assertEqual(result['metadata']['result_count'], 1)

        # Verify analytics logging
        self.mock_analytics.log_performance_metrics.assert_called_once()

    async def test_dispatch_with_string_intent(self):
        """Test dispatch method with string intent that gets converted to enum."""
        dispatcher = SearchDispatcher(
            chroma_manager=self.mock_chroma_manager,
            text_embedder=self.mock_text_embedder,
            image_embedder=self.mock_image_embedder
        )

        mock_response = {'items': []}
        dispatcher.notebook_manager.handle_notebook_request.return_value = mock_response
        dispatcher.performance_metrics.sanitize_parameters = Mock(return_value={})

        result = await dispatcher.dispatch("test", "notebook", {})  # Changed to lowercase

        # Verify the correct handler was called
        dispatcher.notebook_manager.handle_notebook_request.assert_called_once()
        self.assertEqual(result['metadata']['intent'], 'notebook')  # Changed to lowercase

    async def test_dispatch_with_invalid_string_intent(self):
        """Test dispatch method with invalid string intent."""
        dispatcher = SearchDispatcher(
            chroma_manager=self.mock_chroma_manager,
            text_embedder=self.mock_text_embedder,
            image_embedder=self.mock_image_embedder
        )

        mock_response = {'items': []}
        dispatcher.metadata_search_manager.handle_metadata_search.return_value = mock_response
        dispatcher.performance_metrics.sanitize_parameters = Mock(return_value={})

        # Mock the dispatcher's logger directly
        with patch.object(dispatcher, 'logger') as mock_logger:
            result = await dispatcher.dispatch("test", "INVALID_INTENT", {})

        # Verify warning was logged and fallback to RETRIEVAL occurred
        mock_logger.warning.assert_called_once()
        dispatcher.metadata_search_manager.handle_metadata_search.assert_called_once()
        self.assertEqual(result['metadata']['intent'], 'retrieval')  # Changed to lowercase

    async def test_dispatch_with_user_id(self):
        """Test dispatch method with user_id parameter."""
        dispatcher = SearchDispatcher(
            chroma_manager=self.mock_chroma_manager,
            text_embedder=self.mock_text_embedder,
            image_embedder=self.mock_image_embedder
        )

        mock_response = {'items': []}
        dispatcher.image_search_manager.handle_image_search.return_value = mock_response
        dispatcher.performance_metrics.sanitize_parameters = Mock(return_value={})

        parameters = {'param1': 'value1'}
        user_id = 'user123'

        await dispatcher.dispatch("test", QueryIntent.IMAGE_SEARCH, parameters, user_id)

        # Verify user_id was added to parameters
        expected_parameters = {'param1': 'value1', 'user_id': 'user123'}
        dispatcher.image_search_manager.handle_image_search.assert_called_once_with("test", expected_parameters)

    async def test_dispatch_exception_handling(self):
        """Test dispatch method exception handling."""
        dispatcher = SearchDispatcher(
            chroma_manager=self.mock_chroma_manager,
            text_embedder=self.mock_text_embedder,
            image_embedder=self.mock_image_embedder,
            analytics=self.mock_analytics
        )

        # Mock handler to raise exception
        dispatcher.metadata_search_manager.handle_metadata_search.side_effect = Exception("Test error")

        parameters = {'query_id': 'test123'}

        # Mock the dispatcher's logger directly
        with patch.object(dispatcher, 'logger') as mock_logger:
            result = await dispatcher.dispatch("test", QueryIntent.RETRIEVAL, parameters)

        # Verify error handling
        self.assertFalse(result['success'])
        self.assertEqual(result['error'], "Test error")
        self.assertIn('metadata', result)

        # Verify error logging
        mock_logger.error.assert_called_once()

        # Verify analytics error logging
        self.mock_analytics.update_query_status.assert_called_once_with(
            query_id='test123',
            status='failed'
        )

    async def test_dispatch_unknown_intent_fallback(self):
        """Test dispatch method with UNKNOWN intent uses fallback handler."""
        dispatcher = SearchDispatcher(
            chroma_manager=self.mock_chroma_manager,
            text_embedder=self.mock_text_embedder,
            image_embedder=self.mock_image_embedder
        )

        mock_response = {'items': []}
        dispatcher.fallback_search_manager.handle_fallback_search.return_value = mock_response
        dispatcher.performance_metrics.sanitize_parameters = Mock(return_value={})

        result = await dispatcher.dispatch("test", QueryIntent.UNKNOWN, {})

        # Verify fallback handler was called
        dispatcher.fallback_search_manager.handle_fallback_search.assert_called_once()
        self.assertEqual(result['metadata']['intent'], 'unknown')  # Changed to lowercase

    async def test_dispatch_without_analytics(self):
        """Test dispatch method when analytics is None."""
        dispatcher = SearchDispatcher(
            chroma_manager=self.mock_chroma_manager,
            text_embedder=self.mock_text_embedder,
            image_embedder=self.mock_image_embedder,
            analytics=None
        )

        mock_response = {'items': []}
        dispatcher.metadata_search_manager.handle_metadata_search.return_value = mock_response
        dispatcher.performance_metrics.sanitize_parameters = Mock(return_value={})

        # This should not raise an exception
        result = await dispatcher.dispatch("test", QueryIntent.RETRIEVAL, {})

        # Verify the result is still returned properly
        self.assertIn('metadata', result)
        self.assertIn('execution_time_ms', result['metadata'])

    async def test_dispatch_execution_time_measurement(self):
        """Test that execution time is properly measured and included in results."""
        dispatcher = SearchDispatcher(
            chroma_manager=self.mock_chroma_manager,
            text_embedder=self.mock_text_embedder,
            image_embedder=self.mock_image_embedder
        )

        mock_response = {'items': []}
        dispatcher.metadata_search_manager.handle_metadata_search.return_value = mock_response
        dispatcher.performance_metrics.sanitize_parameters = Mock(return_value={})

        result = await dispatcher.dispatch("test", QueryIntent.RETRIEVAL, {})

        # Verify execution time is measured and is a positive number
        execution_time = result['metadata']['execution_time_ms']
        self.assertIsInstance(execution_time, (int, float))
        self.assertGreaterEqual(execution_time, 0)  # Just verify it's non-negative

        # Verify the metadata structure is correct
        self.assertIn('execution_time_ms', result['metadata'])
        self.assertIn('intent', result['metadata'])
        self.assertIn('result_count', result['metadata'])

    def test_async_test_helper(self):
        """Helper method to run async tests in sync test methods."""
        # This is a utility method that can be used if needed
        pass


# Test runner for async tests
def async_test(coro):
    """Decorator to run async test methods."""

    def wrapper(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro(self))
        finally:
            loop.close()

    return wrapper


# Apply async_test decorator to async test methods
for attr_name in dir(TestSearchDispatcher):
    attr = getattr(TestSearchDispatcher, attr_name)
    if callable(attr) and attr_name.startswith('test_') and asyncio.iscoroutinefunction(attr):
        setattr(TestSearchDispatcher, attr_name, async_test(attr))

if __name__ == '__main__':
    unittest.main()