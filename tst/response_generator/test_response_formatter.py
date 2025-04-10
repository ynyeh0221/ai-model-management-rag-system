import unittest
from unittest.mock import MagicMock, patch
from jinja2 import Template
import sys
import os

from src.response_generator.response_formatter import ResponseFormatter

class TestResponseFormatter(unittest.TestCase):
    """Test cases for the ResponseFormatter class."""

    def setUp(self):
        """Set up test fixtures."""
        self.template_manager = MagicMock()
        self.template_mock = MagicMock(spec=Template)
        self.template_mock.render.return_value = "Rendered content"
        self.template_manager.get_template.return_value = self.template_mock

        # Create ResponseFormatter instance
        self.formatter = ResponseFormatter(self.template_manager)

        # Mock the Jinja Environment
        self.env_mock = MagicMock()
        self.env_mock.from_string.return_value = self.template_mock
        self.formatter.env = self.env_mock

        # Sample test data
        self.sample_results = [
            {
                "id": "result1",
                "content": "Sample content 1",
                "metadata": {
                    "model_id": "model_a",
                    "version": "1.0",
                    "framework": {"name": "PyTorch"},
                    "filepath": "/path/to/file1.txt",
                    "type": "model_script"
                }
            },
            {
                "id": "result2",
                "content": "Sample content 2",
                "metadata": {
                    "model_id": "model_b",
                    "version": "2.0",
                    "framework": {"name": "TensorFlow"},
                    "filepath": "/path/to/file2.txt",
                    "type": "documentation"
                }
            }
        ]

        self.sample_query_dict = {
            "intent": "retrieval",
            "type": "text"
        }

        self.sample_query_str = "sample query"

    def test_init(self):
        """Test initialization of ResponseFormatter."""
        self.assertEqual(self.formatter.template_manager, self.template_manager)
        self.assertIsNotNone(self.formatter.logger)
        self.assertIsNotNone(self.formatter.env)

    def test_get_default_template(self):
        """Test getting default templates."""
        # Test with template manager returning a template
        self.template_manager.get_template.return_value = self.template_mock
        template = self.formatter._get_default_template("text")
        self.assertEqual(template, self.template_mock)
        self.template_manager.get_template.assert_called_with("default_text")

        # Test fallback when template manager returns None
        self.template_manager.get_template.return_value = None

        # For text type
        template = self.formatter._get_default_template("text")
        self.env_mock.from_string.assert_called()
        self.assertEqual(template, self.template_mock)

        # For markdown type
        template = self.formatter._get_default_template("markdown")
        self.env_mock.from_string.assert_called()
        self.assertEqual(template, self.template_mock)

        # For HTML type
        template = self.formatter._get_default_template("html")
        self.env_mock.from_string.assert_called()
        self.assertEqual(template, self.template_mock)

    def test_format_response_with_string_query(self):
        """Test format_response with a string query."""
        response = self.formatter.format_response(
            self.sample_results, self.sample_query_str, "text"
        )

        self.assertEqual(response["content"], "Rendered content")
        self.assertIn("citations", response)
        self.assertIn("metadata", response)
        self.assertIn("timestamp", response["metadata"])
        self.assertEqual(response["metadata"]["result_count"], 2)

    def test_format_response_with_dict_query(self):
        """Test format_response with a dictionary query."""
        response = self.formatter.format_response(
            self.sample_results, self.sample_query_dict
        )

        self.assertEqual(response["content"], "Rendered content")
        self.assertIn("citations", response)
        self.assertIn("metadata", response)
        self.assertIn("timestamp", response["metadata"])
        self.assertEqual(response["metadata"]["result_count"], 2)

    def test_format_response_no_results(self):
        """Test format_response with no results."""
        response = self.formatter.format_response([], self.sample_query_dict)

        self.assertEqual(response["content"], "No results found.")
        self.assertEqual(response["citations"], [])
        self.assertIn("metadata", response)
        self.assertIn("timestamp", response["metadata"])
        self.assertEqual(response["metadata"]["result_count"], 0)

    def test_format_response_filter_llm_fallbacks(self):
        """Test filtering of fallback LLM responses."""
        results_with_llm = self.sample_results + [
            {"id": "llm_response_1", "content": "LLM fallback content"}
        ]

        response = self.formatter.format_response(
            results_with_llm, self.sample_query_dict
        )

        # Should have filtered out the llm_response
        self.template_mock.render.assert_called()
        context = self.template_mock.render.call_args[1]
        self.assertEqual(len(context.get("results", [])), 2)

    def test_format_response_template_selection(self):
        """Test template selection logic in format_response."""
        # Test specific intent_response_type combination
        self.formatter.format_response(self.sample_results, self.sample_query_dict)
        self.template_manager.get_template.assert_any_call("retrieval_text")

        # Test falling back to intent
        self.template_manager.get_template.reset_mock()
        self.template_manager.get_template.side_effect = [None, self.template_mock]
        self.formatter.format_response(self.sample_results, self.sample_query_dict)
        self.template_manager.get_template.assert_any_call("retrieval")

        # Test falling back to information_retrieval for retrieval intent
        self.template_manager.get_template.reset_mock()
        self.template_manager.get_template.side_effect = [None, None, self.template_mock]
        self.formatter.format_response(self.sample_results, self.sample_query_dict)
        self.template_manager.get_template.assert_any_call("information_retrieval")

        # Test falling back to default
        self.template_manager.get_template.reset_mock()
        self.template_manager.get_template.side_effect = [None, None, None, None]
        with patch.object(self.formatter, '_get_default_template') as mock_default:
            mock_default.return_value = self.template_mock
            self.formatter.format_response(self.sample_results, self.sample_query_dict)
            mock_default.assert_called_with("text")

    def test_format_model_info(self):
        """Test formatting model information response."""
        query = {"intent": "model_info", "type": "markdown"}

        response = self.formatter.format_response(self.sample_results, query)

        self.assertEqual(response["content"], "Rendered content")
        self.assertIn("citations", response)
        self.assertIn("metadata", response)
        self.assertIn("timestamp", response["metadata"])

    def test_format_model_info_no_results(self):
        """Test formatting model information with no results."""
        query = {"intent": "model_info", "type": "markdown"}

        response = self.formatter.format_response([], query)
        self.assertEqual(response["content"], "No results found.")
        self.assertEqual(response["citations"], [])

    def test_format_model_comparison(self):
        """Test formatting model comparison response."""
        query = {
            "intent": "model_comparison",
            "type": "markdown",
            "parameters": {
                "model_ids": ["model_a", "model_b"],
                "comparison_points": ["architecture", "performance"]
            }
        }

        # Add performance data to sample results
        for idx, result in enumerate(self.sample_results):
            result["metadata"]["performance"] = {
                "accuracy": {"value": 0.85 + idx * 0.05},
                "f1_score": {"value": 0.80 + idx * 0.05}
            }
            result["metadata"]["architecture_type"] = {"value": f"Architecture{idx + 1}"}
            result["metadata"]["model_dimensions"] = {
                "hidden_size": 768 + idx * 256,
                "num_layers": 12 + idx * 4
            }

        response = self.formatter.format_response(self.sample_results, query)

        self.assertEqual(response["content"], "Rendered content")
        self.assertIn("citations", response)
        self.assertIn("metadata", response)
        self.assertIn("timestamp", response["metadata"])

    def test_format_image_gallery(self):
        """Test formatting image gallery response."""
        query = {"intent": "image_search", "type": "html"}

        # Create sample image results
        image_results = [
            {
                "id": "image1",
                "metadata": {
                    "thumbnail_path": "/path/to/thumb1.jpg",
                    "image_path": "/path/to/image1.jpg",
                    "prompt": {"value": "A beautiful landscape"},
                    "source_model_id": "stable_diffusion",
                    "resolution": {"width": 512, "height": 512},
                    "style_tags": {"value": ["landscape", "photorealistic"]},
                    "creation_date": "2023-01-01",
                    "clip_score": {"value": 0.78},
                    "guidance_scale": {"value": 7.5},
                    "num_inference_steps": {"value": 50},
                    "seed": {"value": 42}
                }
            },
            {
                "id": "image2",
                "metadata": {
                    "thumbnail_path": "/path/to/thumb2.jpg",
                    "image_path": "/path/to/image2.jpg",
                    "prompt": {"value": "A futuristic city"},
                    "source_model_id": "midjourney",
                    "resolution": {"width": 1024, "height": 1024},
                    "style_tags": {"value": ["futuristic", "cityscape"]},
                    "creation_date": "2023-01-02",
                    "clip_score": {"value": 0.82},
                    "guidance_scale": {"value": 8.0},
                    "num_inference_steps": {"value": 75},
                    "seed": {"value": 123}
                }
            }
        ]

        response = self.formatter.format_response(image_results, query)

        self.assertEqual(response["content"], "Rendered content")
        self.assertIn("citations", response)
        self.assertIn("metadata", response)
        self.assertIn("timestamp", response["metadata"])
        self.assertEqual(response["metadata"]["image_count"], 2)

    def test_extract_citations(self):
        """Test extraction of citations from results."""
        citations = self.formatter._extract_citations(self.sample_results)

        self.assertEqual(len(citations), 2)
        self.assertEqual(citations[0]["id"], "result1")
        self.assertEqual(citations[0]["type"], "model_script")
        self.assertEqual(citations[0]["filepath"], "/path/to/file1.txt")
        self.assertEqual(citations[0]["model_id"], "model_a")

        self.assertEqual(citations[1]["id"], "result2")
        self.assertEqual(citations[1]["type"], "documentation")
        self.assertEqual(citations[1]["filepath"], "/path/to/file2.txt")
        self.assertEqual(citations[1]["model_id"], "model_b")

    def test_generate_description(self):
        """Test description generation for different result types."""
        # Test for model script
        description = self.formatter._generate_description(self.sample_results[0])
        self.assertEqual(description, "Model script for model_a (v1.0)")

        # Test for generated image
        image_result = {
            "metadata": {
                "type": "generated_image",
                "source_model_id": "stable_diffusion",
                "prompt": {"value": "A sunset over mountains"}
            }
        }
        description = self.formatter._generate_description(image_result)
        self.assertEqual(description, "Image generated by stable_diffusion with prompt: A sunset over mountains")

        # Test for generic document
        generic_result = {
            "metadata": {
                "filepath": "/path/to/document.txt"
            }
        }
        description = self.formatter._generate_description(generic_result)
        self.assertEqual(description, "Document: /path/to/document.txt")

    def test_create_performance_comparison_table(self):
        """Test creating performance comparison table."""
        model_data = {
            "model_a": {
                "performance": {
                    "accuracy": {"value": 0.85},
                    "f1_score": {"value": 0.80}
                },
                "citations": []
            },
            "model_b": {
                "performance": {
                    "accuracy": {"value": 0.90},
                    "precision": {"value": 0.88}
                },
                "citations": []
            }
        }

        table_data = self.formatter._create_performance_comparison_table(model_data, "performance")

        self.assertIn("headers", table_data)
        self.assertIn("rows", table_data)
        self.assertEqual(table_data["headers"][0], "Metric")
        self.assertTrue(any(row["metric"] == "accuracy" for row in table_data["rows"]))
        self.assertTrue(any(row["metric"] == "f1_score" for row in table_data["rows"]))
        self.assertTrue(any(row["metric"] == "precision" for row in table_data["rows"]))

    def test_create_architecture_comparison_table(self):
        """Test creating architecture comparison table."""
        model_data = {
            "model_a": {
                "architecture": {
                    "type": "Transformer",
                    "dimensions": {
                        "hidden_size": 768,
                        "num_layers": 12
                    }
                },
                "citations": []
            },
            "model_b": {
                "architecture": {
                    "type": "CNN",
                    "dimensions": {
                        "hidden_size": 1024,
                        "filters": 64
                    }
                },
                "citations": []
            }
        }

        table_data = self.formatter._create_architecture_comparison_table(model_data, "architecture")

        self.assertIn("type", table_data)
        self.assertIn("dimensions", table_data)
        self.assertIn("table", table_data["type"])
        self.assertIn("rows", table_data["type"])
        self.assertEqual(table_data["type"]["table"]["headers"][0], "Model")
        self.assertEqual(table_data["type"]["table"]["headers"][1], "Architecture Type")
        self.assertEqual(table_data["dimensions"]["table"]["headers"][0], "Dimension")

    def test_create_training_comparison_table(self):
        """Test creating training comparison table."""
        model_data = {
            "model_a": {
                "training": {
                    "config": {
                        "batch_size": 32,
                        "learning_rate": 5e-5
                    },
                    "dataset": {
                        "name": "MNIST",
                        "size": 60000
                    }
                },
                "citations": []
            },
            "model_b": {
                "training": {
                    "config": {
                        "batch_size": 64,
                        "learning_rate": 3e-5
                    },
                    "dataset": {
                        "name": "CIFAR-10",
                        "size": 50000
                    }
                },
                "citations": []
            }
        }

        table_data = self.formatter._create_training_comparison_table(model_data, "training")

        self.assertIn("config", table_data)
        self.assertIn("dataset", table_data)
        self.assertIn("table", table_data["config"])
        self.assertIn("rows", table_data["config"])
        self.assertEqual(table_data["config"]["table"]["headers"][0], "Parameter")
        self.assertEqual(table_data["dataset"]["table"]["headers"][0], "Parameter")

    def test_create_generic_comparison_table(self):
        """Test creating generic comparison table."""
        model_data = {
            "model_a": {
                "framework": {
                    "name": "PyTorch",
                    "version": "1.9.0"
                },
                "citations": []
            },
            "model_b": {
                "framework": {
                    "name": "TensorFlow",
                    "version": "2.6.0"
                },
                "citations": []
            }
        }

        table_data = self.formatter._create_generic_comparison_table(model_data, "framework")

        self.assertIn("headers", table_data)
        self.assertIn("rows", table_data)
        self.assertEqual(table_data["headers"][0], "Model")
        self.assertEqual(table_data["headers"][1], "Framework")
        self.assertEqual(len(table_data["rows"]), 2)

    def test_format_markdown(self):
        """Test formatting content as markdown."""
        content = "# Test Markdown\n\nThis is a test."
        formatted = self.formatter._format_markdown(content)
        self.assertEqual(formatted, content)

    @patch('markdown.markdown')
    def test_format_html(self, mock_markdown):
        """Test formatting content as HTML."""
        # Test with already HTML content
        html_content = "<div>Test HTML</div>"
        formatted = self.formatter._format_html(html_content)
        self.assertEqual(formatted, html_content)
        mock_markdown.assert_not_called()

        # Test with markdown content
        mock_markdown.reset_mock()
        mock_markdown.return_value = "<h1>Test Markdown</h1><p>This is a test.</p>"
        markdown_content = "# Test Markdown\n\nThis is a test."
        formatted = self.formatter._format_html(markdown_content)
        self.assertEqual(formatted, mock_markdown.return_value)
        mock_markdown.assert_called_once_with(markdown_content)


if __name__ == '__main__':
    unittest.main()