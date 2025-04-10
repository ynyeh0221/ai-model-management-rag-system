import unittest
import tempfile
import os
import nbformat

from src.colab_generator.template_engine import NotebookTemplateEngine


SAMPLE_TEMPLATE = """{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "# Model: {{ model_name }}\\n",
        "Framework: {{ framework }}"
      ]
    }
  ],
  "metadata": {
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 5
}
"""


class TestNotebookTemplateEngine(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.template_path = os.path.join(self.temp_dir.name, "test_template.ipynb.j2")
        with open(self.template_path, "w") as f:
            f.write(SAMPLE_TEMPLATE)

        self.engine = NotebookTemplateEngine(templates_dir=self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_list_templates(self):
        templates = self.engine.list_templates()
        self.assertIn("test_template", templates)

    def test_get_template_success(self):
        template = self.engine.get_template("test_template")
        self.assertTrue(callable(template.render))

    def test_get_template_not_found(self):
        with self.assertRaises(FileNotFoundError):
            self.engine.get_template("non_existent_template")

    def test_render_template(self):
        context = {
            "model_name": "TransformerX",
            "framework": "PyTorch"
        }
        rendered_nb = self.engine.render_template("test_template", context)
        self.assertIsInstance(rendered_nb, nbformat.NotebookNode)
        self.assertEqual(rendered_nb.cells[0].cell_type, "markdown")
        content = "".join(rendered_nb.cells[0].source)
        self.assertIn("TransformerX", content)
        self.assertIn("PyTorch", content)


if __name__ == "__main__":
    unittest.main()
