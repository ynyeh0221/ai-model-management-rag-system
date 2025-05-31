import asyncio
import json
import unittest
from unittest.mock import Mock, AsyncMock, patch

from src.core.rag_system import RAGSystem


class TestRAGSystem(unittest.TestCase):
    """Test class for RAGSystem core functionality"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.rag_system = RAGSystem()

        # Mock components
        self.mock_components = {
            "query_engine": {
                "query_parser": Mock(),
                "search_dispatcher": AsyncMock(),
                "query_analytics": Mock(),
                "reranker": Mock()
            },
            "response_generator": {
                "llm_interface": Mock()
            }
        }

        # Set up mock return values
        self.mock_components["query_engine"]["query_parser"].parse_query.return_value = {
            "processed_query": "test query",
            "intent": "text_search",
            "parameters": {}
        }

        self.mock_components["query_engine"]["search_dispatcher"].dispatch.return_value = {
            "items": [
                {
                    "model_id": "test_model_1",
                    "merged_description": "Test model description",
                    "metadata": {
                        "architecture": "transformer",
                        "dataset": "test_dataset",  # Provide string instead of dict
                        "file": '{"size_bytes": 1000, "creation_date": "2023-01-01"}',
                        "framework": '{"name": "PyTorch", "version": "1.0"}',
                        "training_config": '{"batch_size": 32, "learning_rate": 0.001}'
                    }
                }
            ]
        }

        self.mock_components["query_engine"]["reranker"].rerank.return_value = [
            {
                "model_id": "test_model_1",
                "merged_description": "Test model description",
                "metadata": {
                    "architecture": "transformer",
                    "dataset": "test_dataset"  # Provide string instead of dict
                },
                "rerank_score": 0.95
            }
        ]

        # Mock LLM interface responses
        self.mock_llm_response = {
            "content": "This is a test response from the LLM."
        }
        self.mock_components["response_generator"][
            "llm_interface"].generate_structured_response.return_value = self.mock_llm_response

        # Initialize callbacks for testing
        self.callback_calls = {
            "on_log": [],
            "on_result": [],
            "on_error": [],
            "on_status": []
        }

        # Register test callbacks
        self.rag_system.register_callback("on_log", lambda msg: self.callback_calls["on_log"].append(msg))
        self.rag_system.register_callback("on_result", lambda result: self.callback_calls["on_result"].append(result))
        self.rag_system.register_callback("on_error", lambda error: self.callback_calls["on_error"].append(error))
        self.rag_system.register_callback("on_status", lambda status: self.callback_calls["on_status"].append(status))

    def tearDown(self):
        """Clean up after each test method."""
        self.rag_system = None
        self.mock_components = None
        self.callback_calls = None

    def test_initialization_success(self):
        """Test successful initialization of RAG system."""
        result = self.rag_system.initialize(self.mock_components, "test_user")

        self.assertTrue(result)
        self.assertEqual(self.rag_system.user_id, "test_user")
        self.assertEqual(self.rag_system.components, self.mock_components)

        # Check if callbacks were called
        self.assertGreater(len(self.callback_calls["on_log"]), 0)
        self.assertGreater(len(self.callback_calls["on_status"]), 0)

        # Check status callback
        status_calls = self.callback_calls["on_status"]
        self.assertIn("ready", status_calls)

    def test_initialization_failure(self):
        """Test initialization failure handling."""
        # Test with components that cause an exception during initialization
        # Mock _update_status to raise an exception instead of _log
        with patch.object(self.rag_system, '_update_status') as mock_update_status:
            mock_update_status.side_effect = Exception("Status update failed")

            result = self.rag_system.initialize(self.mock_components, "test_user")

            self.assertFalse(result)
            mock_update_status.assert_called_with("ready")

    def test_register_callback_valid(self):
        """Test registering valid callbacks."""
        test_callback = Mock()

        self.rag_system.register_callback("on_log", test_callback)
        self.assertEqual(self.rag_system.callbacks["on_log"], test_callback)

    def test_register_callback_invalid(self):
        """Test registering an invalid callback type."""
        test_callback = Mock()

        with patch.object(self.rag_system, '_log') as mock_log:
            self.rag_system.register_callback("invalid_event", test_callback)
            mock_log.assert_called_with("Unknown event type: invalid_event", level="warning")

    async def test_process_query_regular_success(self):
        """Test successful processing of a regular query."""
        # Initialize the system
        self.rag_system.initialize(self.mock_components, "test_user")

        result = await self.rag_system.process_query("test query", enable_clarity_check=False,
                                                     enable_comparison_detection=False)

        self.assertEqual(result["type"], "text_search")
        self.assertEqual(result["query"], "test query")
        self.assertIn("final_response", result)

        # Verify component calls
        self.mock_components["query_engine"]["query_parser"].parse_query.assert_called_with("test query")
        self.mock_components["query_engine"]["search_dispatcher"].dispatch.assert_called_once()

    async def test_process_query_comparison_detection(self):
        """Test comparison query detection and processing."""
        # Initialize the system
        self.rag_system.initialize(self.mock_components, "test_user")

        # Mock comparison detection to return True
        comparison_response = {
            "content": json.dumps({
                "is_comparison": True,
                "retrieval_queries": ["query about A", "query about B"]
            })
        }
        self.mock_components["response_generator"][
            "llm_interface"].generate_structured_response.return_value = comparison_response

        result = await self.rag_system.process_query("compare A and B", enable_comparison_detection=True)

        self.assertEqual(result["type"], "comparison_search")
        self.assertEqual(len(result["sub_queries"]), 2)
        self.assertIn("final_response", result)

    async def test_process_query_clarity_check_unclear(self):
        """Test query clarity check when a query is unclear."""
        # Initialize the system
        self.rag_system.initialize(self.mock_components, "test_user")

        # Mock clarity check to return unclear
        clarity_response = {
            "content": json.dumps({
                "is_clear": False,
                "reason": "Query is too vague",
                "improved_query": "improved query",
                "suggestions": ["Be more specific"]
            })
        }
        self.mock_components["response_generator"][
            "llm_interface"].generate_structured_response.return_value = clarity_response

        result = await self.rag_system.process_query("unclear query", enable_clarity_check=True)

        self.assertEqual(result["type"], "needs_clarification")
        self.assertEqual(result["query"], "unclear query")
        self.assertIn("clarity_result", result)

    async def test_process_query_clarity_check_clear(self):
        """Test query clarity check when a query is clear."""
        # Initialize the system
        self.rag_system.initialize(self.mock_components, "test_user")

        # Mock clarity check to return clear with improvement
        clarity_response = {
            "content": json.dumps({
                "is_clear": True,
                "improved_query": "improved clear query",
                "suggestions": [],
                "reason": ""
            })
        }

        # Mock comparison detection to return False
        comparison_response = {
            "content": json.dumps({
                "is_comparison": False,
                "retrieval_queries": []
            })
        }

        # Set up side_effect for multiple calls to LLM interface
        self.mock_components["response_generator"]["llm_interface"].generate_structured_response.side_effect = [
            clarity_response,  # First call for clarity check
            comparison_response,  # Second call for comparison detection
            self.mock_llm_response  # Third call for final response generation
        ]

        result = await self.rag_system.process_query("clear query", enable_clarity_check=True,
                                                     enable_comparison_detection=True)

        # Should process as a regular query with improved query text
        self.assertEqual(result["type"], "text_search")

        # Reset side_effect for other tests
        self.mock_components["response_generator"]["llm_interface"].generate_structured_response.side_effect = None
        self.mock_components["response_generator"][
            "llm_interface"].generate_structured_response.return_value = self.mock_llm_response

    async def test_process_query_image_search(self):
        """Test processing image search query."""
        # Initialize the system
        self.rag_system.initialize(self.mock_components, "test_user")

        # Mock parser to return image search intent
        self.mock_components["query_engine"]["query_parser"].parse_query.return_value = {
            "processed_query": "test image query",
            "intent": "image_search",
            "parameters": {}
        }

        result = await self.rag_system.process_query("test image query", enable_comparison_detection=False)

        self.assertEqual(result["type"], "image_search")
        self.assertEqual(result["query"], "test image query")

    async def test_process_query_error_handling(self):
        """Test error handling in query processing."""
        # Initialize the system
        self.rag_system.initialize(self.mock_components, "test_user")

        # Mock an exception in search dispatcher
        self.mock_components["query_engine"]["search_dispatcher"].dispatch.side_effect = Exception("Search failed")

        result = await self.rag_system.process_query("test query", enable_comparison_detection=False)

        self.assertEqual(result["type"], "error")
        self.assertEqual(result["query"], "test query")
        self.assertIn("error", result)

        # Clean upside effect for other tests
        self.mock_components["query_engine"]["search_dispatcher"].dispatch.side_effect = None

    async def test_process_query_uninitialized_components(self):
        """Test query processing with uninitialized components."""
        # Create a fresh RAGSystem instance that hasn't been initialized
        fresh_rag_system = RAGSystem()

        result = await fresh_rag_system.process_query("test query")

        self.assertEqual(result["type"], "error")
        self.assertIn("System components not initialized", result["error"])

    def test_execute_command_exit(self):
        """Test executing exit command."""
        result = self.rag_system.execute_command("exit")

        self.assertEqual(result["type"], "command")
        self.assertEqual(result["command"], "exit")
        self.assertEqual(result["result"], "exit")

    def test_execute_command_help(self):
        """Test executing help command."""
        result = self.rag_system.execute_command("help")

        self.assertEqual(result["type"], "command")
        self.assertEqual(result["command"], "help")
        self.assertIn("available_commands", result["result"])

    def test_execute_command_unknown(self):
        """Test executing unknown command."""
        result = self.rag_system.execute_command("/unknown")

        self.assertEqual(result["type"], "command")
        self.assertEqual(result["command"], "/unknown")
        self.assertIn("Unknown command", result["result"])

    def test_execute_command_query(self):
        """Test executing a query through command interface."""
        # Initialize the system
        self.rag_system.initialize(self.mock_components, "test_user")

        # Mock the event loop for async operation
        with patch('asyncio.get_event_loop') as mock_loop:
            mock_loop.return_value.run_until_complete.return_value = {
                "type": "text_search",
                "query": "test query"
            }

            result = self.rag_system.execute_command("test query")

            self.assertEqual(result["type"], "text_search")

    def test_process_search_results_with_reranker(self):
        """Test processing search results with reranker."""
        search_results = {
            "items": [
                {
                    "model_id": "test_model",
                    "merged_description": "Test description",
                    "metadata": {
                        "architecture": "transformer",
                        "dataset": "test_dataset"  # Provide string instead of dict
                    }
                }
            ]
        }

        parsed_query = {"processed_query": "test"}
        query_text = "test"

        result = self.rag_system._process_search_results(
            search_results,
            self.mock_components["query_engine"]["reranker"],
            parsed_query,
            query_text
        )

        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)

        # Verify reranker was called
        self.mock_components["query_engine"]["reranker"].rerank.assert_called_once()

    def test_process_search_results_without_reranker(self):
        """Test processing search results without reranker."""
        search_results = {
            "items": [{"model_id": "test_model"}]
        }

        result = self.rag_system._process_search_results(
            search_results, None, {}, "test"
        )

        self.assertEqual(result, search_results["items"])

    def test_process_search_results_invalid_format(self):
        """Test processing invalid search results format."""
        invalid_search_results = "invalid format"

        result = self.rag_system._process_search_results(
            invalid_search_results, None, {}, "test"
        )

        self.assertEqual(result, [])

    def test_build_results_text(self):
        """Test building results text from search results."""
        reranked_results = [
            {
                "model_id": "test_model_1",
                "merged_description": "Test description",
                "metadata": {
                    "file": '{"size_bytes": 1000, "creation_date": "2023-01-01"}',
                    "framework": '{"name": "PyTorch", "version": "1.0"}',
                    "architecture": '{"type": "transformer", "reason": "Good for NLP"}',
                    "dataset": '{"name": "CommonCrawl"}',
                    "training_config": '{"batch_size": 32, "learning_rate": 0.001}'
                },
                "rerank_score": 0.95
            }
        ]

        result = self.rag_system._build_results_text(reranked_results, True)

        self.assertIn("Model #1", result)
        self.assertIn("test_model_1", result)
        self.assertIn("Test description", result)
        self.assertIn("Has More Models: True", result)

    def test_remove_field(self):
        """Test removing field from a dictionary list."""
        dict_list = [
            {"field1": "value1", "field2": "value2"},
            {"field1": "value3", "field2": "value4"}
        ]

        result = self.rag_system._remove_field(dict_list, "field1")

        for item in result:
            self.assertNotIn("field1", item)
            self.assertIn("field2", item)

    def test_remove_field_empty_list(self):
        """Test removing field from an empty list."""
        result = self.rag_system._remove_field([], "field1")
        self.assertEqual(result, [])

    def test_remove_thinking_sections(self):
        """Test removing thinking sections from the text."""
        text_with_thinking = """
        This is regular text.
        <thinking>
        This is thinking content that should be removed.
        Multiple lines of thinking.
        </thinking>
        This is more regular text.
        <thinking>Another thinking section</thinking>
        Final text.
        """

        result = self.rag_system.remove_thinking_sections(text_with_thinking)

        self.assertNotIn("<thinking>", result)
        self.assertNotIn("</thinking>", result)
        self.assertIn("This is regular text.", result)
        self.assertIn("Final text.", result)

    def test_log_callback(self):
        """Test logging functionality and callbacks."""
        self.rag_system._log("Test message", "info")

        self.assertGreater(len(self.callback_calls["on_log"]), 0)
        last_log = self.callback_calls["on_log"][-1]
        self.assertEqual(last_log["level"], "info")
        self.assertEqual(last_log["message"], "Test message")

    def test_update_status_callback(self):
        """Test status update functionality and callbacks."""
        self.rag_system._update_status("testing")

        self.assertIn("testing", self.callback_calls["on_status"])

    def test_handle_result_callback(self):
        """Test result handling functionality and callbacks."""
        test_result = {"type": "test", "data": "test_data"}
        self.rag_system._handle_result(test_result)

        self.assertIn(test_result, self.callback_calls["on_result"])

    def test_handle_error_callback(self):
        """Test error handling functionality and callbacks."""
        test_error = Exception("Test error")
        self.rag_system._handle_error(test_error)

        self.assertIn(test_error, self.callback_calls["on_error"])

    async def test_detect_comparison_query_true(self):
        """Test comparison query detection returning True."""
        # Initialize the system
        self.rag_system.initialize(self.mock_components, "test_user")

        comparison_response = {
            "content": json.dumps({
                "is_comparison": True,
                "retrieval_queries": ["query1", "query2"]
            })
        }

        llm_interface = self.mock_components["response_generator"]["llm_interface"]
        llm_interface.generate_structured_response.return_value = comparison_response

        is_comparison, queries = await self.rag_system._detect_comparison_query("compare A and B", llm_interface)

        self.assertTrue(is_comparison)
        self.assertEqual(queries, ["query1", "query2"])

    async def test_detect_comparison_query_false(self):
        """Test comparison query detection returning False."""
        # Initialize the system
        self.rag_system.initialize(self.mock_components, "test_user")

        comparison_response = {
            "content": json.dumps({
                "is_comparison": False,
                "retrieval_queries": []
            })
        }

        llm_interface = self.mock_components["response_generator"]["llm_interface"]
        llm_interface.generate_structured_response.return_value = comparison_response

        is_comparison, queries = await self.rag_system._detect_comparison_query("regular query", llm_interface)

        self.assertFalse(is_comparison)
        self.assertEqual(queries, [])

    async def test_detect_comparison_query_error(self):
        """Test comparison query detection with parsing error."""
        # Initialize the system
        self.rag_system.initialize(self.mock_components, "test_user")

        # Return invalid JSON
        comparison_response = {"content": "invalid json"}

        llm_interface = self.mock_components["response_generator"]["llm_interface"]
        llm_interface.generate_structured_response.return_value = comparison_response

        is_comparison, queries = await self.rag_system._detect_comparison_query("test query", llm_interface)

        self.assertFalse(is_comparison)
        self.assertEqual(queries, [])

    async def test_generate_comparison_response(self):
        """Test generating comparison response."""
        # Initialize the system
        self.rag_system.initialize(self.mock_components, "test_user")

        comparison_results = [
            {"query": "query1", "type": "text_search", "results_text": "Results for query 1"},
            {"query": "query2", "type": "text_search", "results_text": "Results for query 2"}
        ]

        llm_interface = self.mock_components["response_generator"]["llm_interface"]
        llm_interface.generate_structured_response.return_value = {"content": "Comparison response"}

        result = await self.rag_system._generate_comparison_response(
            "compare A and B", comparison_results, llm_interface
        )

        self.assertEqual(result, "Comparison response")
        llm_interface.generate_structured_response.assert_called_once()

    async def test_check_query_clarity_clear(self):
        """Test query clarity check for a clear query."""
        # Initialize the system
        self.rag_system.initialize(self.mock_components, "test_user")

        clarity_response = {
            "content": json.dumps({
                "is_clear": True,
                "improved_query": "improved query",
                "suggestions": [],
                "reason": ""
            })
        }

        llm_interface = self.mock_components["response_generator"]["llm_interface"]
        llm_interface.generate_structured_response.return_value = clarity_response

        result = await self.rag_system._check_query_clarity("clear query", llm_interface)

        self.assertTrue(result["is_clear"])
        self.assertEqual(result["improved_query"], "improved query")

    async def test_check_query_clarity_unclear(self):
        """Test query clarity check for an unclear query."""
        # Initialize the system
        self.rag_system.initialize(self.mock_components, "test_user")

        clarity_response = {
            "content": json.dumps({
                "is_clear": False,
                "improved_query": "unclear query",
                "suggestions": ["Be more specific"],
                "reason": "Query is too vague"
            })
        }

        llm_interface = self.mock_components["response_generator"]["llm_interface"]
        llm_interface.generate_structured_response.return_value = clarity_response

        result = await self.rag_system._check_query_clarity("unclear query", llm_interface)

        self.assertFalse(result["is_clear"])
        self.assertEqual(result["reason"], "Query is too vague")

    def test_async_test_runner(self):
        """Helper method to run async tests."""

        # This is a helper for running async tests in the unittest framework
        def run_async_test(async_test_func):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(async_test_func())
            finally:
                loop.close()

        # Test various async methods individually to avoid side effects
        # Each test should be independent

        # Reset mocks before each async test
        def reset_mocks():
            # Reset all mock calls and side effects
            for component_group in self.mock_components.values():
                for component in component_group.values():
                    if hasattr(component, 'reset_mock'):
                        component.reset_mock()
                    # Clear any side effects
                    if hasattr(component, 'side_effect'):
                        component.side_effect = None

            # Reset LLM interface to default behavior
            self.mock_components["response_generator"]["llm_interface"].generate_structured_response.side_effect = None
            self.mock_components["response_generator"][
                "llm_interface"].generate_structured_response.return_value = self.mock_llm_response

            # Reset search dispatcher side effect specifically
            self.mock_components["query_engine"]["search_dispatcher"].dispatch.side_effect = None
            self.mock_components["query_engine"]["search_dispatcher"].dispatch.return_value = {
                "items": [
                    {
                        "model_id": "test_model_1",
                        "merged_description": "Test model description",
                        "metadata": {
                            "architecture": "transformer",
                            "dataset": "test_dataset",
                            "file": '{"size_bytes": 1000, "creation_date": "2023-01-01"}',
                            "framework": '{"name": "PyTorch", "version": "1.0"}',
                            "training_config": '{"batch_size": 32, "learning_rate": 0.001}'
                        }
                    }
                ]
            }

        try:
            reset_mocks()
            run_async_test(self.test_process_query_regular_success)

            reset_mocks()
            run_async_test(self.test_process_query_comparison_detection)

            reset_mocks()
            run_async_test(self.test_process_query_clarity_check_unclear)

            reset_mocks()
            run_async_test(self.test_process_query_clarity_check_clear)

            reset_mocks()
            run_async_test(self.test_process_query_image_search)

            reset_mocks()
            run_async_test(self.test_process_query_error_handling)

            # Don't reset mocks for the uninitialized components test
            # as it uses a fresh RAGSystem instance
            run_async_test(self.test_process_query_uninitialized_components)

            reset_mocks()
            run_async_test(self.test_detect_comparison_query_true)

            reset_mocks()
            run_async_test(self.test_detect_comparison_query_false)

            reset_mocks()
            run_async_test(self.test_detect_comparison_query_error)

            reset_mocks()
            run_async_test(self.test_generate_comparison_response)

            reset_mocks()
            run_async_test(self.test_check_query_clarity_clear)

            reset_mocks()
            run_async_test(self.test_check_query_clarity_unclear)

        except AssertionError as e:
            # If any async test fails, provide more context
            self.fail(f"Async test failed: {str(e)}")


if __name__ == '__main__':
    # Custom test runner for async tests

    class AsyncTestSuite(unittest.TestSuite):
        def run(self, result):
            for test in self:
                if hasattr(test._testMethodName, '__name__') and 'async' in test._testMethodName:
                    # Handle async tests
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        loop.run_until_complete(test._testMethodDoc)
                    finally:
                        loop.close()
                else:
                    test.run(result)
            return result


    # Run the tests
    unittest.main(verbosity=2)