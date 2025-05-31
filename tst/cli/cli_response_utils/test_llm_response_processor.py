import json
import unittest
from unittest.mock import Mock, patch

from src.cli.cli_response_utils.llm_response_processor import LLMResponseProcessor


class TestLLMResponseProcessor(unittest.TestCase):
    """Test suite for LLMResponseProcessor class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Import the class to test
        self.LLMResponseProcessor = LLMResponseProcessor

    def test_print_llm_content_dict_with_content(self):
        """Test processing dict response with 'content' field."""
        response = {"content": "This is the content"}

        with patch('builtins.print') as mock_print:
            self.LLMResponseProcessor.print_llm_content(response)
            mock_print.assert_called_once_with("This is the content")

    def test_print_llm_content_dict_with_text(self):
        """Test processing dict response with 'text' field."""
        response = {"text": "This is the text"}

        with patch('builtins.print') as mock_print:
            self.LLMResponseProcessor.print_llm_content(response)
            mock_print.assert_called_once_with("This is the text")

    def test_print_llm_content_dict_with_response(self):
        """Test processing dict response with 'response' field."""
        response = {"response": "This is the response"}

        with patch('builtins.print') as mock_print:
            self.LLMResponseProcessor.print_llm_content(response)
            mock_print.assert_called_once_with("This is the response")

    def test_print_llm_content_dict_with_answer(self):
        """Test processing dict response with 'answer' field."""
        response = {"answer": "This is the answer"}

        with patch('builtins.print') as mock_print:
            self.LLMResponseProcessor.print_llm_content(response)
            mock_print.assert_called_once_with("This is the answer")

    def test_print_llm_content_dict_with_message_string(self):
        """Test processing dict response with 'message' field as string."""
        response = {"message": "This is the message"}

        with patch('builtins.print') as mock_print:
            self.LLMResponseProcessor.print_llm_content(response)
            mock_print.assert_called_once_with("This is the message")

    def test_print_llm_content_dict_with_message_dict(self):
        """Test processing dict response with 'message' field as dict (OpenAI format)."""
        response = {"message": {"content": "OpenAI format content"}}

        with patch('builtins.print') as mock_print:
            self.LLMResponseProcessor.print_llm_content(response)
            mock_print.assert_called_once_with("OpenAI format content")

    def test_print_llm_content_dict_no_recognizable_field_small(self):
        """Test processing dict response with no recognizable fields (small response)."""
        response = {"unknown_field": "some value", "other_field": 123}

        with patch('builtins.print') as mock_print:
            self.LLMResponseProcessor.print_llm_content(response)

            # Should print error message, available fields, and response content
            calls = mock_print.call_args_list
            self.assertGreaterEqual(len(calls), 3)

            # Check for expected messages
            call_args = [str(call[0][0]) for call in calls]
            self.assertTrue(any("No recognizable content field found" in arg for arg in call_args))
            self.assertTrue(any("Available fields:" in arg for arg in call_args))
            self.assertTrue(any("Response content:" in arg for arg in call_args))

    def test_print_llm_content_dict_no_recognizable_field_large(self):
        """Test processing dict response with no recognizable fields (large response)."""
        # Create a large response (over 1000 characters)
        large_value = "x" * 1000
        response = {"unknown_field": large_value}

        with patch('builtins.print') as mock_print:
            self.LLMResponseProcessor.print_llm_content(response)

            calls = mock_print.call_args_list
            call_args = [str(call[0][0]) for call in calls]

            # Should not include "Response content": for large responses
            self.assertTrue(any("No recognizable content field found" in arg for arg in call_args))
            self.assertTrue(any("Available fields:" in arg for arg in call_args))
            self.assertFalse(any("Response content:" in arg for arg in call_args))

    def test_print_llm_content_string_plain_text(self):
        """Test processing plain string response."""
        response = "This is a plain text response"

        with patch('builtins.print') as mock_print:
            self.LLMResponseProcessor.print_llm_content(response)
            mock_print.assert_called_once_with("This is a plain text response")

    def test_print_llm_content_string_valid_json_dict(self):
        """Test processing string response that contains valid JSON dict."""
        json_dict = {"content": "JSON content"}
        response = json.dumps(json_dict)

        with patch('builtins.print') as mock_print:
            self.LLMResponseProcessor.print_llm_content(response)
            mock_print.assert_called_once_with("JSON content")

    def test_print_llm_content_string_valid_json_non_dict(self):
        """Test processing string response that contains valid JSON but not a dict."""
        response = json.dumps(["item1", "item2"])

        with patch('builtins.print') as mock_print:
            self.LLMResponseProcessor.print_llm_content(response)
            mock_print.assert_called_once_with(response)  # Should print original string

    def test_print_llm_content_string_invalid_json(self):
        """Test processing string response with invalid JSON."""
        response = '{"invalid": json content}'

        with patch('builtins.print') as mock_print:
            self.LLMResponseProcessor.print_llm_content(response)
            mock_print.assert_called_once_with(response)  # Should print as-is

    def test_print_llm_content_list_with_dict(self):
        """Test processing list response with dict as the first element."""
        response = [{"content": "First item content"}, {"other": "data"}]

        with patch('builtins.print') as mock_print:
            self.LLMResponseProcessor.print_llm_content(response)
            mock_print.assert_called_once_with("First item content")

    def test_print_llm_content_list_with_string(self):
        """Test processing list response with string as the first element."""
        response = ["First item string", "Second item"]

        with patch('builtins.print') as mock_print:
            self.LLMResponseProcessor.print_llm_content(response)
            mock_print.assert_called_once_with("First item string")

    def test_print_llm_content_empty_list(self):
        """Test processing empty list response."""
        response = []

        with patch('builtins.print') as mock_print:
            self.LLMResponseProcessor.print_llm_content(response)
            mock_print.assert_called_once_with("Empty list response")

    def test_print_llm_content_unsupported_type(self):
        """Test processing unsupported response type."""
        response = 12345  # Integer

        with patch('builtins.print') as mock_print:
            self.LLMResponseProcessor.print_llm_content(response)

            calls = mock_print.call_args_list
            self.assertEqual(len(calls), 2)

            # The First call should indicate unsupported type
            self.assertIn("Unsupported response type", str(calls[0][0][0]))
            self.assertIn("int", str(calls[0][0][0]))

            # The Second call should print the value
            self.assertEqual(calls[1][0][0], "12345")

    def test_print_llm_content_exception_in_raw_response(self):
        """Test exception handling when even raw response fails."""
        # Create a mock that raises exception on str() call
        response = Mock()
        response.__str__ = Mock(side_effect=Exception("String conversion failed"))
        response.side_effect = Exception("Test exception")

        with patch('builtins.print') as mock_print:
            self.LLMResponseProcessor.print_llm_content(response)

            calls = mock_print.call_args_list
            call_args = [str(call[0][0]) for call in calls]

            # Should print error and fallback message
            self.assertTrue(any("Error extracting content" in arg for arg in call_args))
            self.assertTrue(any("Could not print raw response" in arg for arg in call_args))

    def test_print_dict_response_priority_order(self):
        """Test that dict response fields are checked in the correct priority order."""
        # Test that 'content' takes priority over other fields
        response = {
            "content": "Content field",
            "text": "Text field",
            "response": "Response field"
        }

        with patch('builtins.print') as mock_print:
            self.LLMResponseProcessor._print_dict_response(response)
            mock_print.assert_called_once_with("Content field")

    def test_print_str_response_recursive_call(self):
        """Test that string response correctly calls dict response for JSON."""
        json_response = {"text": "Parsed JSON content"}
        string_response = json.dumps(json_response)

        with patch.object(self.LLMResponseProcessor, '_print_dict_response') as mock_dict_method:
            self.LLMResponseProcessor._print_str_response(string_response)
            mock_dict_method.assert_called_once_with(json_response)

    def test_print_list_response_recursive_call(self):
        """Test that list response correctly calls dict response for dict elements."""
        dict_element = {"answer": "List content"}
        list_response = [dict_element, "other", "items"]

        with patch.object(self.LLMResponseProcessor, '_print_dict_response') as mock_dict_method:
            self.LLMResponseProcessor._print_list_response(list_response)
            mock_dict_method.assert_called_once_with(dict_element)

    def test_static_methods(self):
        """Test that all methods are properly defined as static methods."""
        # Test that methods can be called without an instance
        response = {"content": "test"}

        with patch('builtins.print'):
            # These should not raise TypeError about missing 'self'
            self.LLMResponseProcessor.print_llm_content(response)
            self.LLMResponseProcessor._print_dict_response(response)
            self.LLMResponseProcessor._print_str_response("test")
            self.LLMResponseProcessor._print_list_response([response])

        # Check method signatures don't include 'self'
        import inspect

        for method_name in ['print_llm_content', '_print_dict_response', '_print_str_response', '_print_list_response']:
            method = getattr(self.LLMResponseProcessor, method_name)
            sig = inspect.signature(method)
            self.assertNotIn('self', sig.parameters)

    def test_edge_cases_empty_strings(self):
        """Test handling of empty strings and whitespace."""
        test_cases = ["", "   ", "\n", "\t"]

        for test_case in test_cases:
            with patch('builtins.print') as mock_print:
                self.LLMResponseProcessor.print_llm_content(test_case)
                mock_print.assert_called_once_with(test_case)

    def test_edge_cases_special_characters(self):
        """Test handling of special characters and Unicode."""
        test_cases = [
            "String with émojis 🎉",
            "String with\nnewlines\nand\ttabs",
            "String with \"quotes\" and 'apostrophes'",
            "String with {curly} and [square] brackets"
        ]

        for test_case in test_cases:
            with patch('builtins.print') as mock_print:
                self.LLMResponseProcessor.print_llm_content(test_case)
                mock_print.assert_called_once_with(test_case)

    def test_complex_nested_structures(self):
        """Test handling of complex nested data structures."""
        complex_response = {
            "message": {
                "content": "Nested content",
                "metadata": {
                    "timestamp": "2023-01-01",
                    "model": "test-model"
                }
            },
            "other_data": [1, 2, 3]
        }

        with patch('builtins.print') as mock_print:
            self.LLMResponseProcessor.print_llm_content(complex_response)
            mock_print.assert_called_once_with("Nested content")

    def test_json_parsing_edge_cases(self):
        """Test JSON parsing with various edge cases."""
        # Valid JSON but empty dict
        with patch('builtins.print') as mock_print:
            self.LLMResponseProcessor.print_llm_content("{}")
            calls = mock_print.call_args_list
            self.assertTrue(any("No recognizable content field found" in str(call[0][0]) for call in calls))

        # Valid JSON with null values
        with patch('builtins.print') as mock_print:
            self.LLMResponseProcessor.print_llm_content('{"content": null}')
            mock_print.assert_called_with(None)

    def test_raw_response_truncation(self):
        """Test that raw response is truncated to 500 characters."""
        long_response = "x" * 1000  # 1000 character string

        # Create a mock that raises an exception to trigger raw response printing
        with patch.object(self.LLMResponseProcessor, '_print_str_response', side_effect=Exception("Test error")):
            with patch('builtins.print') as mock_print:
                self.LLMResponseProcessor.print_llm_content(long_response)

                calls = mock_print.call_args_list
                call_args = [str(call[0][0]) for call in calls]

                # Find the raw response print
                raw_response_call = None
                for arg in call_args:
                    if len(arg) == 500 and "x" in arg:
                        raw_response_call = arg
                        break

                self.assertIsNotNone(raw_response_call)
                self.assertEqual(len(raw_response_call), 500)

    def test_method_coverage_completeness(self):
        """Test that all response types are handled by dedicated methods."""
        # Verify that each response type calls its corresponding method

        # Dict response
        with patch.object(self.LLMResponseProcessor, '_print_dict_response') as mock_dict:
            self.LLMResponseProcessor.print_llm_content({"test": "value"})
            mock_dict.assert_called_once()

        # String response
        with patch.object(self.LLMResponseProcessor, '_print_str_response') as mock_str:
            self.LLMResponseProcessor.print_llm_content("test string")
            mock_str.assert_called_once()

        # List response
        with patch.object(self.LLMResponseProcessor, '_print_list_response') as mock_list:
            self.LLMResponseProcessor.print_llm_content(["test", "list"])
            mock_list.assert_called_once()


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)