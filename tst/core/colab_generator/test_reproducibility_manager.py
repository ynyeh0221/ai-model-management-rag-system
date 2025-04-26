import os
import unittest
from unittest.mock import patch

import nbformat

from src.core.colab_generator.reproducibility_manager import ReproducibilityManager


class TestReproducibilityManager(unittest.TestCase):

    def setUp(self):
        self.manager = ReproducibilityManager()
        self.sample_notebook = nbformat.v4.new_notebook()
        self.sample_notebook.cells.append(nbformat.v4.new_markdown_cell("## Hello world"))
        self.sample_parameters = {
            "learning_rate": 0.001,
            "batch_size": 16
        }

    def test_calculate_hash_digest(self):
        digest = self.manager.calculate_hash_digest(self.sample_notebook)
        self.assertIsInstance(digest, str)
        self.assertEqual(len(digest), 64)  # SHA256 hash length

    def test_generate_execution_log(self):
        log = self.manager.generate_execution_log(self.sample_notebook, self.sample_parameters)
        self.assertIn("execution_id", log)
        self.assertIn("notebook_hash", log)
        self.assertIn("parameters", log)
        self.assertEqual(log["parameters"], self.sample_parameters)

    def test_record_environment(self):
        env = self.manager.record_environment("exec_123")
        self.assertIn("execution_id", env)
        self.assertIn("system_info", env)
        self.assertIn("packages", env)

    def test_export_to_html(self):
        output_path = self.manager.export_to_html(self.sample_notebook)
        self.assertTrue(os.path.exists(output_path))
        with open(output_path) as f:
            contents = f.read()
            self.assertIn("Hello world", contents)
        os.remove(output_path)

    @patch("src.core.colab_generator.reproducibility_manager.PDFExporter.from_notebook_node")
    def test_export_to_pdf(self, mock_pdf_export):
        mock_pdf_export.return_value = (b"%PDF-1.4 fake pdf data", {})
        output_path = self.manager.export_to_pdf(self.sample_notebook)
        self.assertTrue(os.path.exists(output_path))
        with open(output_path, "rb") as f:
            contents = f.read()
            self.assertTrue(contents.startswith(b"%PDF-1.4"))
        os.remove(output_path)


if __name__ == "__main__":
    unittest.main()
