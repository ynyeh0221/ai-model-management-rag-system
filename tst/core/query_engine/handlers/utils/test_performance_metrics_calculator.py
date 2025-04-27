import unittest
from unittest.mock import MagicMock, patch
import time

from src.core.query_engine.handlers.utils.performance_metrics_calculator import PerformanceMetricsCalculator


class TestPerformanceMetricsCalculator(unittest.TestCase):
    def setUp(self):
        # Create a mock analytics object
        self.mock_analytics = MagicMock()

        # Create calculator instances for testing
        self.calculator_with_analytics = PerformanceMetricsCalculator(analytics=self.mock_analytics)
        self.calculator_no_analytics = PerformanceMetricsCalculator()

    def test_calculate_text_search_performance_basic(self):
        """Test basic calculation functionality with mock time values"""
        # Set up test parameters
        start_time = time.time() - 1  # 1 second ago
        metadata_search_time = 150.0  # 150 ms
        chunks_search_time = 250.0  # 250 ms
        parameters = {'query': 'test query'}

        # Call the method
        result = self.calculator_no_analytics.calculate_text_search_performance(
            start_time, metadata_search_time, chunks_search_time, parameters
        )

        # Assertions
        self.assertEqual(result['metadata_search_time_ms'], 150.0)
        self.assertEqual(result['chunks_search_time_ms'], 250.0)
        self.assertEqual(result['total_search_time_ms'], 400.0)
        self.assertGreaterEqual(result['total_time_ms'], 1000.0)  # At least 1000 ms

        # Verify analytics not called (no query_id)
        self.mock_analytics.log_performance_metrics.assert_not_called()

    def test_calculate_with_analytics_and_query_id(self):
        """Test logging behavior when analytics is available and query_id is provided"""
        # Set up test parameters
        start_time = time.time() - 0.5  # 0.5 seconds ago
        metadata_search_time = 100.0  # 100 ms
        chunks_search_time = 200.0  # 200 ms
        parameters = {'query': 'test query', 'query_id': 'test_id_123'}

        # Call the method
        result = self.calculator_with_analytics.calculate_text_search_performance(
            start_time, metadata_search_time, chunks_search_time, parameters
        )

        # Verify analytics was called
        self.mock_analytics.log_performance_metrics.assert_called_once()
        # Verify the correct parameters were passed
        call_args = self.mock_analytics.log_performance_metrics.call_args[1]
        self.assertEqual(call_args['query_id'], 'test_id_123')
        self.assertEqual(call_args['search_time_ms'], 300)  # 100 + 200 = 300
        self.assertGreaterEqual(call_args['total_time_ms'], 500)  # At least 500 ms

    def test_calculate_without_query_id(self):
        """Test that no logging occurs when query_id is not in parameters"""
        # Set up test parameters
        start_time = time.time() - 0.2  # 0.2 seconds ago
        metadata_search_time = 50.0  # 50 ms
        chunks_search_time = 150.0  # 150 ms
        parameters = {'query': 'test query'}  # No query_id

        # Call the method
        result = self.calculator_with_analytics.calculate_text_search_performance(
            start_time, metadata_search_time, chunks_search_time, parameters
        )

        # Verify analytics was not called
        self.mock_analytics.log_performance_metrics.assert_not_called()

    @patch('time.time')
    def test_calculate_with_fixed_time(self, mock_time):
        """Test calculation with mocked time.time() for consistent results"""
        # Set up mocked time values
        mock_time.return_value = 101.5  # Current time
        start_time = 100.0  # 1.5 seconds ago
        metadata_search_time = 300.0  # 300 ms
        chunks_search_time = 700.0  # 700 ms
        parameters = {'query_id': 'fixed_time_test'}

        # Call the method
        result = self.calculator_with_analytics.calculate_text_search_performance(
            start_time, metadata_search_time, chunks_search_time, parameters
        )

        # Assertions with precise expected values
        self.assertEqual(result['metadata_search_time_ms'], 300.0)
        self.assertEqual(result['chunks_search_time_ms'], 700.0)
        self.assertEqual(result['total_search_time_ms'], 1000.0)
        self.assertEqual(result['total_time_ms'], 1500.0)  # (101.5 - 100.0) * 1000 = 1500

    def test_sanitize_parameters_empty(self):
        """Test sanitize_parameters with empty input"""
        result = self.calculator_no_analytics.sanitize_parameters({})
        self.assertEqual(result, {})

        result = self.calculator_no_analytics.sanitize_parameters(None)
        self.assertEqual(result, {})

    def test_sanitize_parameters_with_sensitive_fields(self):
        """Test sanitize_parameters with sensitive fields"""
        parameters = {
            'user_id': 'user123',
            'access_token': 'secret_token',
            'auth_context': {'role': 'admin'},
            'raw_query': 'sensitive query',
            'query_id': 'query123',
            'search_term': 'public search term',
            'filter': {'category': 'books'}
        }

        result = self.calculator_no_analytics.sanitize_parameters(parameters)

        # Verify sensitive fields are removed
        self.assertNotIn('user_id', result)
        self.assertNotIn('access_token', result)
        self.assertNotIn('auth_context', result)
        self.assertNotIn('raw_query', result)
        self.assertNotIn('query_id', result)

        # Verify non-sensitive fields remain
        self.assertIn('search_term', result)
        self.assertIn('filter', result)
        self.assertEqual(result['search_term'], 'public search term')
        self.assertEqual(result['filter'], {'category': 'books'})

    def test_sanitize_parameters_with_image_data(self):
        """Test sanitize_parameters with image data"""
        parameters = {
            'search_term': 'test image',
            'image_data': b'binary_image_data_here'
        }

        result = self.calculator_no_analytics.sanitize_parameters(parameters)

        # Verify image data is replaced with placeholder
        self.assertIn('image_data', result)
        self.assertEqual(result['image_data'], "[binary data removed]")

        # Verify non-sensitive fields remain unchanged
        self.assertEqual(result['search_term'], 'test image')

    def test_sanitize_parameters_no_modification(self):
        """Test sanitize_parameters doesn't modify the original parameters"""
        original = {
            'user_id': 'user123',
            'search_term': 'test',
            'image_data': b'image'
        }

        # Make a copy to compare later
        original_copy = original.copy()

        # Call sanitize
        self.calculator_no_analytics.sanitize_parameters(original)

        # Verify original wasn't modified
        self.assertEqual(original, original_copy)


if __name__ == '__main__':
    unittest.main()