import html
import json
import unittest
from unittest.mock import Mock, MagicMock

# Import the class to test
from src.response_generator.prompt_visualizer import PromptVisualizer


class TestPromptVisualizer(unittest.TestCase):
    """Test cases for the PromptVisualizer class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a mock template manager
        self.template_manager_mock = MagicMock()

        # Configure default return values
        self.template_manager_mock.get_template.return_value = Mock()
        self.template_manager_mock.get_template_metadata.return_value = {
            "created_at": "2023-01-01",
            "author": "Test Author",
            "message": "Test message"
        }
        self.template_manager_mock.get_template_content.return_value = "Test template content"
        self.template_manager_mock.get_template_versions.return_value = ["v1", "v2", "v3"]
        self.template_manager_mock.get_template_usage_stats.return_value = {"total_uses": 100}
        self.template_manager_mock.get_template_performance_metrics.return_value = {"avg_response_time": 0.5}

        # Create the visualizer
        self.visualizer = PromptVisualizer(self.template_manager_mock)

    # Test render_preview method
    def test_render_preview_success(self):
        # Configure mock
        template_mock = Mock()
        template_mock.render.return_value = "Rendered content"
        self.template_manager_mock.get_template.return_value = template_mock

        # Call method
        context = {"var1": "value1", "var2": "value2"}
        result = self.visualizer.render_preview("test_template", context)

        # Check results
        self.assertTrue(result["success"])
        self.assertEqual(result["preview"], "Rendered content")
        self.assertIn("metadata", result)
        self.assertIn("rendered_at", result)
        self.assertIn("context_sample", result)
        self.assertIn("html_preview", result)

        # Verify calls
        self.template_manager_mock.get_template.assert_called_once_with("test_template", None)
        self.template_manager_mock.get_template_metadata.assert_called_once()
        template_mock.render.assert_called_once_with(var1="value1", var2="value2")

    def test_render_preview_template_not_found(self):
        # Configure mock
        self.template_manager_mock.get_template.return_value = None

        # Call method
        result = self.visualizer.render_preview("nonexistent_template", {})

        # Check results
        self.assertFalse(result["success"])
        self.assertIn("error", result)
        self.assertIn("Template", result["error"])
        self.assertIn("not found", result["error"])
        self.assertIsNone(result["preview"])
        self.assertIsNone(result["metadata"])

    def test_render_preview_exception(self):
        # Configure mock to raise an exception
        template_mock = Mock()
        template_mock.render.side_effect = Exception("Test exception")
        self.template_manager_mock.get_template.return_value = template_mock

        # Call method
        result = self.visualizer.render_preview("test_template", {})

        # Check results
        self.assertFalse(result["success"])
        self.assertIn("error", result)
        self.assertIn("Test exception", result["error"])
        self.assertIsNone(result["preview"])
        self.assertIsNone(result["metadata"])

    # Test generate_diff method
    def test_generate_diff_success(self):
        # Configure mock
        self.template_manager_mock.get_template_content.side_effect = [
            "Template version A content",
            "Template version B content"
        ]

        # Call method
        result = self.visualizer.generate_diff("test_template", "v1", "v2")

        # Check results
        self.assertTrue(result["success"])
        self.assertEqual(result["template_id"], "test_template")
        self.assertIn("version_a", result)
        self.assertIn("version_b", result)
        self.assertIn("diff", result)
        self.assertIn("stats", result)
        self.assertIn("generated_at", result)

        # Verify diff content
        self.assertIn("text", result["diff"])
        self.assertIn("html", result["diff"])
        self.assertIn("unified", result["diff"])

        # Verify stats
        self.assertIn("similarity_ratio", result["stats"])
        self.assertIn("added_chars", result["stats"])
        self.assertIn("deleted_chars", result["stats"])

    def test_generate_diff_template_not_found(self):
        # Configure mock
        self.template_manager_mock.get_template_content.side_effect = [None, "Content"]

        # Call method
        result = self.visualizer.generate_diff("test_template", "v1", "v2")

        # Check results
        self.assertFalse(result["success"])
        self.assertIn("error", result)
        self.assertIn("not found", result["error"])
        self.assertIsNone(result["diff"])

    def test_generate_diff_exception(self):
        # Configure mock to raise an exception
        self.template_manager_mock.get_template_content.side_effect = Exception("Test exception")

        # Call method
        result = self.visualizer.generate_diff("test_template", "v1", "v2")

        # Check results
        self.assertFalse(result["success"])
        self.assertIn("error", result)
        self.assertIn("Test exception", result["error"])
        self.assertIsNone(result["diff"])

    # Test create_html_preview method
    def test_create_html_preview_json(self):
        # Test with JSON content
        json_content = json.dumps({"key": "value", "nested": {"inner": "value"}})
        result = self.visualizer.create_html_preview(json_content)

        # Check results
        self.assertIn("JSON Preview", result)
        self.assertIn("language-json", result)
        self.assertIn(html.escape('"key": "value"'), result)

    def test_create_html_preview_markdown(self):
        # Test with markdown content
        md_content = "# Heading\n\n* Item 1\n* Item 2\n\n**Bold text**"
        result = self.visualizer.create_html_preview(md_content)

        # Check results
        self.assertIn("Markdown Preview", result)
        self.assertIn("markdown-rendered", result)
        self.assertIn("View Source", result)

    def test_create_html_preview_plain_text(self):
        # Test with plain text
        text_content = "This is plain text without any special formatting."
        result = self.visualizer.create_html_preview(text_content)

        # Check results
        self.assertIn("Text Preview", result)
        self.assertTrue(text_content in result or html.escape(text_content) in result)

    # Test visualize_template_history method
    def test_visualize_template_history_success(self):
        # Configure mock
        self.template_manager_mock.get_template_versions.return_value = ["v1", "v2", "v3"]
        self.template_manager_mock.get_template_metadata.side_effect = [
            {"created_at": "2023-01-03", "author": "Author3", "message": "Message3"},
            {"created_at": "2023-01-02", "author": "Author2", "message": "Message2"},
            {"created_at": "2023-01-01", "author": "Author1", "message": "Message1"}
        ]

        # Call method
        result = self.visualizer.visualize_template_history("test_template")

        # Check results
        self.assertTrue(result["success"])
        self.assertEqual(result["template_id"], "test_template")
        self.assertIn("versions", result)
        self.assertIn("timeline", result)
        self.assertIn("total_versions", result)
        self.assertIn("generated_at", result)

        # Check versions
        self.assertEqual(len(result["versions"]), 3)

        # Check timeline
        self.assertIn("points", result["timeline"])
        self.assertIn("connections", result["timeline"])
        self.assertEqual(len(result["timeline"]["points"]), 3)
        self.assertEqual(len(result["timeline"]["connections"]), 2)  # Should be (n-1) connections for n points

    def test_visualize_template_history_no_versions(self):
        # Configure mock
        self.template_manager_mock.get_template_versions.return_value = []

        # Call method
        result = self.visualizer.visualize_template_history("test_template")

        # Check results
        self.assertFalse(result["success"])
        self.assertIn("error", result)
        self.assertIn("No versions found", result["error"])
        self.assertIsNone(result["history"])

    def test_visualize_template_history_exception(self):
        # Configure mock to raise an exception
        self.template_manager_mock.get_template_versions.side_effect = Exception("Test exception")

        # Call method
        result = self.visualizer.visualize_template_history("test_template")

        # Check results
        self.assertFalse(result["success"])
        self.assertIn("error", result)
        self.assertIn("Test exception", result["error"])
        self.assertIsNone(result["history"])

    # Test create_template_report method
    def test_create_template_report_success(self):
        # Configure mock
        template_mock = Mock()
        self.template_manager_mock.get_template.return_value = template_mock

        # For the history visualization
        self.template_manager_mock.get_template_versions.return_value = ["v1", "v2"]
        self.template_manager_mock.get_template_metadata.side_effect = [
            {"created_at": "2023-01-02", "author": "Author2", "message": "Message2"},
            {"created_at": "2023-01-01", "author": "Author1", "message": "Message1"},
            {"created_at": "2023-01-02", "author": "Author2", "message": "Message2"}
            # Called again for the first version
        ]

        # Call method
        result = self.visualizer.create_template_report("test_template")

        # Check results
        self.assertTrue(result["success"])
        self.assertEqual(result["template_id"], "test_template")
        self.assertIn("metadata", result)
        self.assertIn("content_sample", result)
        self.assertIn("usage_stats", result)
        self.assertIn("history", result)
        self.assertIn("performance_metrics", result)
        self.assertIn("generated_at", result)

    def test_create_template_report_template_not_found(self):
        # Configure mock
        self.template_manager_mock.get_template.return_value = None

        # Call method
        result = self.visualizer.create_template_report("nonexistent_template")

        # Check results
        self.assertFalse(result["success"])
        self.assertIn("error", result)
        self.assertIn("Template", result["error"])
        self.assertIn("not found", result["error"])
        self.assertIsNone(result["report"])

    def test_create_template_report_exception(self):
        # Configure mock to raise an exception
        self.template_manager_mock.get_template.side_effect = Exception("Test exception")

        # Call method
        result = self.visualizer.create_template_report("test_template")

        # Check results
        self.assertFalse(result["success"])
        self.assertIn("error", result)
        self.assertIn("Test exception", result["error"])
        self.assertIsNone(result["report"])

    # Test private methods
    def test_generate_text_diff(self):
        text_a = "Original text"
        text_b = "Modified text"

        result = self.visualizer._generate_text_diff(text_a, text_b)

        # Should have different parts for equal, delete, and insert operations
        self.assertTrue(len(result) > 0)
        self.assertTrue(any(part["type"] == "equal" for part in result))
        self.assertTrue(any(part["type"] == "delete" for part in result))
        self.assertTrue(any(part["type"] == "insert" for part in result))

    def test_calculate_diff_stats(self):
        text_a = "Original text"
        text_b = "Modified text"

        result = self.visualizer._calculate_diff_stats(text_a, text_b)

        self.assertIn("total_chars_a", result)
        self.assertIn("total_chars_b", result)
        self.assertIn("total_lines_a", result)
        self.assertIn("total_lines_b", result)
        self.assertIn("added_chars", result)
        self.assertIn("deleted_chars", result)
        self.assertIn("unchanged_chars", result)
        self.assertIn("similarity_ratio", result)

        self.assertEqual(result["total_chars_a"], len(text_a))
        self.assertEqual(result["total_chars_b"], len(text_b))
        self.assertTrue(0 <= result["similarity_ratio"] <= 1)

    def test_truncate_context(self):
        # Test with a complex context
        context = {
            "short_string": "Short",
            "long_string": "A" * 200,
            "nested": {
                "inner_short": "Inner short",
                "inner_long": "B" * 200
            },
            "list": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        }

        result = self.visualizer._truncate_context(context)

        self.assertEqual(result["short_string"], "Short")
        self.assertTrue(result["long_string"].endswith("..."))
        self.assertTrue(len(result["long_string"]) <= 103)  # 100 chars + 3 for ellipsis
        self.assertEqual(result["nested"]["inner_short"], "Inner short")
        self.assertTrue(result["nested"]["inner_long"].endswith("..."))
        self.assertEqual(len(result["list"]), 6)  # 5 items + '...'
        self.assertEqual(result["list"][5], "...")

    def test_generate_timeline_data(self):
        version_data = [
            {"version": "v3", "created_at": "2023-01-03", "author": "Author3", "message": "Message3"},
            {"version": "v2", "created_at": "2023-01-02", "author": "Author2", "message": "Message2"},
            {"version": "v1", "created_at": "2023-01-01", "author": "Author1", "message": "Message1"}
        ]

        result = self.visualizer._generate_timeline_data(version_data)

        self.assertIn("points", result)
        self.assertIn("connections", result)
        self.assertEqual(len(result["points"]), 3)
        self.assertEqual(len(result["connections"]), 2)  # Should be (n-1) connections for n points

        # Check point structure
        point = result["points"][0]
        self.assertIn("id", point)
        self.assertIn("version", point)
        self.assertIn("date", point)
        self.assertIn("author", point)
        self.assertIn("message", point)

        # Check connection structure
        connection = result["connections"][0]
        self.assertIn("from", connection)
        self.assertIn("to", connection)
        self.assertIn("type", connection)


if __name__ == "__main__":
    unittest.main()