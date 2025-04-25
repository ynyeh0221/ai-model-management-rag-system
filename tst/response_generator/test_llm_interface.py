import json
import unittest
from unittest.mock import patch, MagicMock, call

import requests

from src.response_generator.llm_interface import LLMInterface


class TestLLMInterface(unittest.TestCase):

    @patch('requests.get')
    @patch('requests.post')
    def test_initialize_client_model_present(self, mock_post, mock_get):
        """
        Test that during initialization, if the model is already available
        (i.e. returned in the /api/tags response), no pull request is made.
        """
        # Simulate /api/tags response that includes the desired model "deepseek-r1:7b"
        mock_get_response = MagicMock()
        mock_get_response.json.return_value = {"models": [{"name": "deepseek-r1:7b"}]}
        mock_get_response.raise_for_status.return_value = None
        mock_get.return_value = mock_get_response

        # Instantiate LLMInterface. The _initialize_client method should
        # detect that the model is present.
        llm = LLMInterface(provider="ollama", model_name="deepseek-r1:7b")

        # Verify that no POST call (pull operation) was made.
        mock_post.assert_not_called()

    @patch('requests.get')
    @patch('requests.post')
    def test_initialize_client_model_not_present(self, mock_post, mock_get):
        """
        Test that if the model is not found in the /api/tags response,
        the initialization will attempt to pull the desired model.
        """
        # Simulate /api/tags response with a model list that does NOT include "deepseek-r1:7b"
        mock_get_response = MagicMock()
        mock_get_response.json.return_value = {"models": [{"name": "other_model"}]}
        mock_get_response.raise_for_status.return_value = None
        mock_get.return_value = mock_get_response

        # Simulate a successful POST call to /api/pull
        mock_post_response = MagicMock()
        mock_post_response.raise_for_status.return_value = None
        mock_post.return_value = mock_post_response

        # Initialize LLMInterface with use_gpu=False to avoid adding GPU options
        llm = LLMInterface(provider="ollama", model_name="deepseek-r1:7b", use_gpu=False)

        # The pull endpoint should be called because "deepseek-r1:7b" was not found.
        mock_post.assert_called_with(
            f"{llm.base_url}/api/pull",
            json={"name": llm.model_name},
            timeout=llm.timeout
        )

    @patch('requests.get')
    @patch('requests.post')
    def test_generate_response_non_streaming(self, mock_post, mock_get):
        """
        Test that generate_response (non-streaming) correctly posts to the generate endpoint
        and returns the expected text answer.
        """
        # Simulate initialization response for /api/tags indicating the model is available.
        mock_get_response = MagicMock()
        mock_get_response.json.return_value = {"models": [{"name": "deepseek-r1:7b"}]}
        mock_get_response.raise_for_status.return_value = None
        mock_get.return_value = mock_get_response

        llm = LLMInterface(provider="ollama", model_name="deepseek-r1:7b")

        # Set up the POST response for the generate endpoint.
        mock_post_response = MagicMock()
        mock_post_response.json.return_value = {"response": "Test answer"}
        mock_post_response.raise_for_status.return_value = None
        mock_post.return_value = mock_post_response

        # Update the test to match the implementation
        # The implementation returns a dictionary with content field, not just the raw string
        result = llm.generate_response(
            prompt="Hello?", temperature=0.7, max_tokens=50, streaming=False
        )
        self.assertEqual(result, {"id": "llm_response_0", "content": "Test answer", "metadata": {}})

    @patch('requests.get')
    @patch('requests.post')
    def test_generate_response_streaming(self, mock_post, mock_get):
        """
        Test that generate_response (streaming) correctly streams response chunks.
        """
        # Simulate initialization response indicating that the model is available.
        mock_get_response = MagicMock()
        mock_get_response.json.return_value = {"models": [{"name": "deepseek-r1:7b"}]}
        mock_get_response.raise_for_status.return_value = None
        mock_get.return_value = mock_get_response

        llm = LLMInterface(provider="ollama", model_name="deepseek-r1:7b")

        # Create dummy streaming response lines.
        dummy_lines = [
            json.dumps({"response": "Chunk1"}).encode('utf-8'),
            json.dumps({"response": "Chunk2", "done": True}).encode('utf-8')
        ]
        dummy_response = MagicMock()
        dummy_response.iter_lines.return_value = dummy_lines
        dummy_response.raise_for_status.return_value = None

        # Patch requests.post so that the streaming endpoint returns the dummy response.
        mock_post.return_value = dummy_response

        # Update the test to match the implementation
        # The implementation returns a dictionary with content field containing joined chunks
        result = llm.generate_response(
            prompt="Hello streaming", temperature=0.7, max_tokens=50, streaming=True
        )
        self.assertEqual(result, {"id": "llm_response_0", "content": "Chunk1Chunk2", "metadata": {}})

    @patch('requests.get')
    @patch('requests.post')
    def test_generate_structured_response(self, mock_post, mock_get):
        """
        Test that generate_structured_response posts to the chat endpoint and returns
        the structured answer.
        """
        # Set up initialization /api/tags response.
        mock_get_response = MagicMock()
        mock_get_response.json.return_value = {"models": [{"name": "deepseek-r1:7b"}]}
        mock_get_response.raise_for_status.return_value = None
        mock_get.return_value = mock_get_response

        llm = LLMInterface(provider="ollama", model_name="deepseek-r1:7b")

        # Set up POST response for the structured chat endpoint.
        structured_response = {"message": {"content": "Structured answer"}}
        mock_post_response = MagicMock()
        mock_post_response.json.return_value = structured_response
        mock_post_response.raise_for_status.return_value = None
        mock_post.return_value = mock_post_response

        # Update the test to match the implementation
        # The implementation returns a dictionary with content field, not just the raw string
        result = llm.generate_structured_response(
            system_prompt="System", user_prompt="User query",
            temperature=0.5, max_tokens=100
        )
        self.assertEqual(result, {"id": "llm_structured_response_0", "content": "Structured answer", "metadata": {}})

    @patch('time.sleep', return_value=None)
    def test_handle_rate_limiting_success(self, mock_sleep):
        """
        Test that handle_rate_limiting retries a function when rate-limited and
        eventually returns a successful result.
        """
        call_count = [0]  # mutable counter

        def flaky_function(x):
            if call_count[0] < 2:
                call_count[0] += 1
                raise requests.exceptions.RequestException("429 Too Many Requests")
            return "Success"

        # For initialization, patch requests.get so no actual HTTP call is made.
        with patch('requests.get') as mock_get:
            mock_get_response = MagicMock()
            mock_get_response.json.return_value = {"models": [{"name": "deepseek-r1:7b"}]}
            mock_get_response.raise_for_status.return_value = None
            mock_get.return_value = mock_get_response
            llm = LLMInterface(provider="ollama", model_name="deepseek-r1:7b")

        result = llm.handle_rate_limiting(flaky_function, "input")
        self.assertEqual(result, "Success")
        # Verify that sleep was called with exponential backoff delays (2 and then 4 seconds).
        expected_calls = [call(2), call(4)]
        mock_sleep.assert_has_calls(expected_calls, any_order=False)

    @patch('requests.get')
    @patch('requests.post')  # Add this to mock POST requests as well
    def test_get_model_info(self, mock_post, mock_get):
        """
        Test that get_model_info returns the expected model information.
        """
        # Mock for initialization GET to /api/tags
        mock_tags_response = MagicMock()
        mock_tags_response.json.return_value = {"models": [{"name": "deepseek-r1:7b"}]}
        mock_tags_response.raise_for_status.return_value = None

        # Mock for GET to /api/show in get_model_info
        dummy_info = {"name": "deepseek-r1:7b"}
        mock_show_response = MagicMock()
        mock_show_response.json.return_value = dummy_info
        mock_show_response.raise_for_status.return_value = None

        # Configure mock_get to return different responses based on URL
        def mock_get_side_effect(url, **kwargs):
            if "/api/tags" in url:
                return mock_tags_response
            elif "/api/show" in url:
                return mock_show_response
            return MagicMock()

        mock_get.side_effect = mock_get_side_effect

        # Mock POST response (shouldn't be called if tags include the model)
        mock_post_response = MagicMock()
        mock_post_response.raise_for_status.return_value = None
        mock_post.return_value = mock_post_response

        llm = LLMInterface(provider="ollama", model_name="deepseek-r1:7b")
        info = llm.get_model_info()
        self.assertEqual(info, dummy_info)

    @patch('requests.get')
    def test_list_available_models(self, mock_get):
        """
        Test that list_available_models returns a list of model names.
        """
        dummy_models = {"models": [{"name": "deepseek-r1:7b"}, {"name": "other_model"}]}
        mock_get_response = MagicMock()
        mock_get_response.json.return_value = dummy_models
        mock_get_response.raise_for_status.return_value = None
        mock_get.return_value = mock_get_response

        llm = LLMInterface(provider="ollama", model_name="deepseek-r1:7b")
        models = llm.list_available_models()
        self.assertEqual(models, ["deepseek-r1:7b", "other_model"])

if __name__ == "__main__":
    unittest.main()