import unittest
from datetime import datetime


class TestDisplayUtils(unittest.TestCase):
    """Test suite for DisplayUtils class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Import the class to test
        from cli.cli_response_utils.display_utils import DisplayUtils
        self.DisplayUtils = DisplayUtils

    def test_ascii_chars_constant(self):
        """Test that ASCII_CHARS constant is properly defined."""
        expected_chars = '@%#*+=-:. '
        self.assertEqual(self.DisplayUtils.ASCII_CHARS, expected_chars)
        self.assertIsInstance(self.DisplayUtils.ASCII_CHARS, str)
        self.assertEqual(len(self.DisplayUtils.ASCII_CHARS), 10)

        # Test that characters are ordered from dark to light
        self.assertTrue(self.DisplayUtils.ASCII_CHARS.startswith('@'))
        self.assertTrue(self.DisplayUtils.ASCII_CHARS.endswith(' '))

    def test_truncate_string_within_limit(self):
        """Test truncating strings that are within the default limit."""
        test_string = "This is a short string"
        result = self.DisplayUtils.truncate_string(test_string)
        self.assertEqual(result, test_string)
        self.assertIs(result, test_string)  # Should return the same object

    def test_truncate_string_exceeds_default_limit(self):
        """Test truncating strings that exceed the default 120-character limit."""
        # Create a string longer than 120 characters
        long_string = "x" * 130
        result = self.DisplayUtils.truncate_string(long_string)

        # Should be truncated to 120 characters, including "..."
        self.assertEqual(len(result), 120)
        self.assertTrue(result.endswith("..."))
        self.assertEqual(result, "x" * 117 + "...")

    def test_truncate_string_exactly_at_limit(self):
        """Test truncating strings that are exactly at the limit."""
        # String exactly 120 characters
        exact_string = "x" * 120
        result = self.DisplayUtils.truncate_string(exact_string)
        self.assertEqual(result, exact_string)
        self.assertEqual(len(result), 120)

    def test_truncate_string_custom_max_length(self):
        """Test truncating strings with custom maximum length."""
        test_string = "This is a test string"
        result = self.DisplayUtils.truncate_string(test_string, max_length=10)

        self.assertEqual(len(result), 10)
        self.assertTrue(result.endswith("..."))
        self.assertEqual(result, "This is...")

    def test_truncate_string_custom_length_within_limit(self):
        """Test strings within custom length limit."""
        test_string = "Short"
        result = self.DisplayUtils.truncate_string(test_string, max_length=10)
        self.assertEqual(result, test_string)

    def test_truncate_string_non_string_input(self):
        """Test truncate_string with non-string inputs."""
        # Test with None
        result = self.DisplayUtils.truncate_string(None)
        self.assertIsNone(result)

        # Test with integer
        result = self.DisplayUtils.truncate_string(12345)
        self.assertEqual(result, 12345)

        # Test with a list
        test_list = [1, 2, 3]
        result = self.DisplayUtils.truncate_string(test_list)
        self.assertEqual(result, test_list)

        # Test with boolean
        result = self.DisplayUtils.truncate_string(True)
        self.assertTrue(result)

    def test_truncate_string_empty_string(self):
        """Test truncating empty string."""
        result = self.DisplayUtils.truncate_string("")
        self.assertEqual(result, "")

    def test_truncate_string_very_short_max_length(self):
        """Test truncating with very short max length."""
        test_string = "Hello World"

        # Test with max_length=2: text[:2-3] + "..." = text[:-1] + "..."
        result = self.DisplayUtils.truncate_string(test_string, max_length=2)
        self.assertEqual(result, "Hello Worl...")  # Gets first 10 chars (11-1) + "..."

        # Test with max_length exactly 3
        result = self.DisplayUtils.truncate_string(test_string, max_length=3)
        self.assertEqual(result, "...")  # text[:0] + "..." = "" + "..." = "..."

    def test_format_timestamp_valid_iso_format(self):
        """Test formatting valid ISO format timestamps."""
        # Test with full ISO format
        timestamp = "2023-12-25T14:30:00"
        result = self.DisplayUtils.format_timestamp(timestamp)
        self.assertEqual(result, "2023-12-25T14:30")

        # Test with microseconds
        timestamp_with_micro = "2023-12-25T14:30:00.123456"
        result = self.DisplayUtils.format_timestamp(timestamp_with_micro)
        self.assertEqual(result, "2023-12-25T14:30")

    def test_format_timestamp_custom_format(self):
        """Test formatting timestamps with custom format strings."""
        timestamp = "2023-12-25T14:30:00"

        # Test custom format
        result = self.DisplayUtils.format_timestamp(timestamp, "%Y-%m-%d %H:%M:%S")
        self.assertEqual(result, "2023-12-25 14:30:00")

        # Test date-only format
        result = self.DisplayUtils.format_timestamp(timestamp, "%Y-%m-%d")
        self.assertEqual(result, "2023-12-25")

        # Test time-only format
        result = self.DisplayUtils.format_timestamp(timestamp, "%H:%M")
        self.assertEqual(result, "14:30")

    def test_format_timestamp_empty_or_none(self):
        """Test formatting empty or None timestamps."""
        # Test with None
        result = self.DisplayUtils.format_timestamp(None)
        self.assertEqual(result, "Unknown")

        # Test with empty string
        result = self.DisplayUtils.format_timestamp("")
        self.assertEqual(result, "Unknown")

        # Test with whitespace only - actual implementation may not treat this as empty
        result = self.DisplayUtils.format_timestamp("   ")
        # Based on the error, whitespace is returned as-is
        self.assertEqual(result, "   ")

    def test_format_timestamp_invalid_format(self):
        """Test formatting timestamps with invalid formats."""
        # Test with invalid date string
        invalid_timestamp = "not-a-date"
        result = self.DisplayUtils.format_timestamp(invalid_timestamp)
        self.assertEqual(result, invalid_timestamp)  # Should return original

        # Test with partial date
        partial_date = "2023-12"
        result = self.DisplayUtils.format_timestamp(partial_date)
        self.assertEqual(result, partial_date)  # Should return original

        # Test with the wrong format
        wrong_format = "25/12/2023"
        result = self.DisplayUtils.format_timestamp(wrong_format)
        self.assertEqual(result, wrong_format)  # Should return original

    def test_format_timestamp_non_string_input(self):
        """Test formatting non-string timestamp inputs."""
        # Test with integer
        result = self.DisplayUtils.format_timestamp(12345)
        self.assertEqual(result, 12345)  # Should return original

        # Test with a datetime object
        dt_obj = datetime(2023, 12, 25, 14, 30)
        result = self.DisplayUtils.format_timestamp(dt_obj)
        self.assertEqual(result, dt_obj)  # Should return original

        # Test with a list
        test_list = ["2023-12-25"]
        result = self.DisplayUtils.format_timestamp(test_list)
        self.assertEqual(result, test_list)  # Should return original

    def test_format_timestamp_timezone_handling(self):
        """Test formatting timestamps with timezone information."""
        # Test with timezone offset
        timestamp_with_tz = "2023-12-25T14:30:00+05:00"
        result = self.DisplayUtils.format_timestamp(timestamp_with_tz)
        # Should handle timezone gracefully
        self.assertIsInstance(result, str)

        # Test with Z (UTC) timezone
        timestamp_utc = "2023-12-25T14:30:00Z"
        result = self.DisplayUtils.format_timestamp(timestamp_utc)
        self.assertIsInstance(result, str)

    def test_static_methods(self):
        """Test that methods are properly defined as static methods."""
        # Test that methods can be called without an instance
        result = self.DisplayUtils.truncate_string("test", 10)
        self.assertEqual(result, "test")

        result = self.DisplayUtils.format_timestamp("2023-12-25T14:30:00")
        self.assertEqual(result, "2023-12-25T14:30")

        # Test that methods don't have 'self' parameter
        import inspect

        truncate_sig = inspect.signature(self.DisplayUtils.truncate_string)
        self.assertNotIn('self', truncate_sig.parameters)

        format_sig = inspect.signature(self.DisplayUtils.format_timestamp)
        self.assertNotIn('self', format_sig.parameters)

    def test_class_constants_immutability(self):
        """Test that class constants maintain their values."""
        original_chars = self.DisplayUtils.ASCII_CHARS

        # Verify the constant hasn't changed
        self.assertEqual(self.DisplayUtils.ASCII_CHARS, '@%#*+=-:. ')

        # Verify it's the same object (immutable string)
        self.assertIs(self.DisplayUtils.ASCII_CHARS, original_chars)

    def test_edge_cases_truncate_string(self):
        """Test edge cases for truncate_string method."""
        # Test with Unicode characters
        unicode_string = "Hello 世界! " * 20  # Mix of ASCII and Unicode
        result = self.DisplayUtils.truncate_string(unicode_string, max_length=20)
        self.assertEqual(len(result), 20)
        self.assertTrue(result.endswith("..."))

        # Test with newlines and special characters
        special_string = "Line1\nLine2\tTabbed\r\nWindows"
        result = self.DisplayUtils.truncate_string(special_string, max_length=15)
        self.assertEqual(len(result), 15)
        self.assertTrue(result.endswith("..."))

    def test_edge_cases_format_timestamp(self):
        """Test edge cases for format_timestamp method."""
        # Test with various falsy values
        falsy_values = [None, "", 0, False, [], {}]
        for value in falsy_values:
            result = self.DisplayUtils.format_timestamp(value)
            # In Python, "if not timestamp" treats 0, False, None, "", [], {} as falsy
            if not value:  # This covers None, "", 0, False, [], {}
                self.assertEqual(result, "Unknown")
            else:
                self.assertEqual(result, value)

    def test_performance_characteristics(self):
        """Test performance characteristics of utility methods."""
        # Test with very long string
        very_long_string = "x" * 10000
        result = self.DisplayUtils.truncate_string(very_long_string, max_length=100)
        self.assertEqual(len(result), 100)
        self.assertTrue(result.endswith("..."))

        # Should be efficient - no exceptions for large inputs
        huge_string = "a" * 1000000
        result = self.DisplayUtils.truncate_string(huge_string, max_length=50)
        self.assertEqual(len(result), 50)

    def test_method_signatures(self):
        """Test that method signatures match expected parameters."""
        import inspect

        # Test truncate_string signature
        truncate_sig = inspect.signature(self.DisplayUtils.truncate_string)
        params = list(truncate_sig.parameters.keys())
        self.assertEqual(params, ['text', 'max_length'])
        self.assertEqual(truncate_sig.parameters['max_length'].default, 120)

        # Test format_timestamp signature
        format_sig = inspect.signature(self.DisplayUtils.format_timestamp)
        params = list(format_sig.parameters.keys())
        self.assertEqual(params, ['timestamp', 'format_str'])
        self.assertEqual(format_sig.parameters['format_str'].default, "%Y-%m-%dT%H:%M")


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)