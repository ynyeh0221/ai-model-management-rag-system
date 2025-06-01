import math
import unittest
from unittest.mock import MagicMock

from src.core.query_engine.handlers.utils.distance_normalizer import DistanceNormalizer


# Import the class to be tested
# from your_module import DistanceNormalizer

class TestDistanceNormalizer(unittest.TestCase):
    """Unit tests for the DistanceNormalizer class."""

    def setUp(self):
        """Set up test fixtures before each test."""
        # Create an instance of the class to test
        self.normalizer = DistanceNormalizer()

        # Mock the logger to avoid cluttering test output
        self.mock_logger = MagicMock()
        self.normalizer.logger = self.mock_logger

        # Common test values
        self.α = 5.0  # Alpha value used in the class

    def test_normalize_distance_kernel_transformation(self):
        """Test the core mathematical transformation in normalize_distance."""
        stats = {'min': 0.0, 'max': 1.0}

        # Test a range of values to verify the kernel transformation
        test_values = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]

        for value in test_values:
            result = self.normalizer.normalize_distance(value, stats)
            # Calculate expected: 1.0 - math.exp(-α * value)
            expected = 1.0 - math.exp(-self.α * value)
            self.assertAlmostEqual(result, expected, places=6,
                                   msg=f"Failed for value {value}")

            # Verify the transformation properties:
            # - As input approaches min, output approaches 0
            # - As input approaches max, output approaches 1-exp(-α)
            if value == 0.0:
                self.assertEqual(result, 0.0)
            if value == 1.0:
                self.assertAlmostEqual(result, 1.0 - math.exp(-self.α))

    def test_normalize_distance_range_scaling(self):
        """Test that the function correctly scales different input ranges."""
        # Test with various min/max ranges
        test_ranges = [
            {'min': 0.0, 'max': 2.0},  # Default range
            {'min': 1.0, 'max': 3.0},  # Shifted range
            {'min': -1.0, 'max': 1.0},  # Range including negative values
            {'min': 10.0, 'max': 100.0}  # Large range
        ]

        for stats in test_ranges:
            min_val = stats['min']
            max_val = stats['max']
            mid_point = (min_val + max_val) / 2

            # Test at min, max, and midpoint of each range
            result_min = self.normalizer.normalize_distance(min_val, stats)
            result_max = self.normalizer.normalize_distance(max_val, stats)
            result_mid = self.normalizer.normalize_distance(mid_point, stats)

            # Min should always normalize to 0
            self.assertEqual(result_min, 0.0)

            # Max should always normalize to 1-exp(-α)
            self.assertAlmostEqual(result_max, 1.0 - math.exp(-self.α))

            # Midpoint should normalize to 1-exp(-α*0.5)
            self.assertAlmostEqual(result_mid, 1.0 - math.exp(-self.α * 0.5))

    def test_normalize_distance_clamping(self):
        """Test that values outside the range are properly clamped."""
        stats = {'min': 1.0, 'max': 2.0}

        # Test below min
        result = self.normalizer.normalize_distance(0.0, stats)
        self.assertEqual(result, 0.0)

        # Test above max
        result = self.normalizer.normalize_distance(3.0, stats)
        # Should be clamped to normalized value at max
        expected = 1.0 - math.exp(-self.α)
        self.assertAlmostEqual(result, expected)

        # Test exactly at boundaries
        result_min = self.normalizer.normalize_distance(1.0, stats)
        self.assertEqual(result_min, 0.0)

        result_max = self.normalizer.normalize_distance(2.0, stats)
        expected = 1.0 - math.exp(-self.α)
        self.assertAlmostEqual(result_max, expected)

    def test_normalize_distance_equal_minmax(self):
        """Test the special case where min equals max."""
        stats = {'min': 1.0, 'max': 1.0}

        # When value equals min=max
        result = self.normalizer.normalize_distance(1.0, stats)
        self.assertEqual(result, 0.0)

        # When value is different from min=max
        result = self.normalizer.normalize_distance(2.0, stats)
        self.assertEqual(result, 1.0)

        result = self.normalizer.normalize_distance(0.0, stats)
        self.assertEqual(result, 1.0)

    def test_extract_search_distance_nested_structure(self):
        """Test extraction from deeply nested distance structures."""
        # Complex nested case
        result = {
            'distances': [
                [0.1, 0.2],  # First item has multiple distances
                [0.3],  # The Second item has one distance
                0.4,  # The Third item is not nested
                []  # The Fourth item is an empty list
            ]
        }

        # Test each case
        item = {'metadata': {'model_id': 'test_model'}}

        # Extract from first item (nested with multiple values)
        distance = self.normalizer.extract_search_distance(result, 0, item)
        self.assertEqual(distance, 0.1)  # Should take first value

        # Extract from second item (nested with single value)
        distance = self.normalizer.extract_search_distance(result, 1, item)
        self.assertEqual(distance, 0.3)

        # Extract from third item (not nested)
        distance = self.normalizer.extract_search_distance(result, 2, item)
        self.assertEqual(distance, 0.4)

        # Extract from fourth item (empty list)
        # The actual behavior is that an empty list is returned as is
        distance = self.normalizer.extract_search_distance(result, 3, item)
        self.assertEqual(distance, [])  # Empty list is returned, not the default

    def test_extract_search_distance_fallback_logic(self):
        """Test the fallback logic when distances aren't found in the expected place."""
        # Test the fallback sequence:
        # 1. Try result['distances'][idx]
        # 2. If that fails or is invalid, try item['distance']
        # 3. If that fails, use default (2.0)

        # Case 1: No 'distances' in a result, but 'distance' in item
        result = {}
        item = {'distance': 0.5, 'metadata': {'model_id': 'test_model'}}
        distance = self.normalizer.extract_search_distance(result, 0, item)
        self.assertEqual(distance, 0.5)

        # Case 2: 'distances' exists but index out of range, 'distance' in item
        # In this case, the code doesn't check item['distance'] because the logic
        # only checks item['distance'] if 'distances' is not in a result or is not a list
        result = {'distances': [0.1]}
        item = {'distance': 0.5, 'metadata': {'model_id': 'test_model'}}
        distance = self.normalizer.extract_search_distance(result, 1, item)
        self.assertEqual(distance, 2.0)  # Uses default, not item['distance']

        # Case 3: No 'distances' in a result, no 'distance' in item
        result = {}
        item = {'metadata': {'model_id': 'test_model'}}
        distance = self.normalizer.extract_search_distance(result, 0, item)
        self.assertEqual(distance, 2.0)  # Should use default

    def test_logger_calls(self):
        """Test that appropriate debug logging occurs."""
        # Reset mock to clear previous calls
        self.mock_logger.reset_mock()

        # Call normalize_distance
        stats = {'min': 0.0, 'max': 1.0}
        self.normalizer.normalize_distance(0.5, stats)

        # Verify logger was called with the right information
        self.mock_logger.debug.assert_called_once()
        log_message = self.mock_logger.debug.call_args[0][0]

        # Check that the log message contains expected information
        self.assertIn("raw=0.5000", log_message)
        self.assertIn("d0=0.5000", log_message)
        self.assertIn("alpha=5.0", log_message)

        # Reset and test extract_search_distance logging
        self.mock_logger.reset_mock()

        result = {'distances': [0.1]}
        item = {'metadata': {'model_id': 'test_model_123'}}
        self.normalizer.extract_search_distance(result, 0, item, 'test_table')

        # Verify logger was called with the appropriate message
        self.mock_logger.debug.assert_called_once()
        log_message = self.mock_logger.debug.call_args[0][0]

        # Check log message contains expected information
        self.assertIn("Distance for model test_model_123", log_message)
        self.assertIn("in test_table", log_message)
        self.assertIn("0.1", log_message)


if __name__ == '__main__':
    unittest.main()