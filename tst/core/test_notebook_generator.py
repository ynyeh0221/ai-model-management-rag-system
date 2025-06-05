import unittest
from unittest.mock import MagicMock, patch
import tempfile
import os
import sys
from types import ModuleType

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Mock nbformat to avoid dependency during import
nbformat_mock = ModuleType("nbformat")
nbformat_v4 = ModuleType("nbformat.v4")
nbformat_v4.new_notebook = lambda cells=None: {}
nbformat_v4.new_code_cell = lambda text: {}
nbformat_mock.v4 = nbformat_v4
nbformat_mock.write = lambda nb, f: None
sys.modules.setdefault('nbformat', nbformat_mock)
sys.modules.setdefault('nbformat.v4', nbformat_v4)

from src.core.notebook_generator import NotebookGenerator

class TestNotebookGenerator(unittest.TestCase):
    def test_prepare_chunk_contents_various_inputs(self):
        """_prepare_chunk_contents handles different chunk formats."""
        chunks = [
            {"document": "print('a')", "metadata": {"offset": 1}},
            {"document": {"content": "print('b')"}, "metadata": {"offset": 2}},
            {"document": 123, "metadata": {"offset": 3}},
        ]
        processed = NotebookGenerator._prepare_chunk_contents(chunks)
        self.assertEqual(len(processed), 3)
        self.assertEqual(processed[0], {"text": "print('a')", "offset": 1})
        self.assertEqual(processed[1], {"text": "print('b')", "offset": 2})
        self.assertEqual(processed[2], {"text": "123", "offset": 3})

    @patch("src.core.notebook_generator.nbformat.write")
    @patch("os.makedirs")
    @patch("asyncio.run")
    def test_generate_notebook_success(self, mock_run, mock_makedirs, mock_write):
        """generate_notebook creates a notebook file with reconstructed code."""
        mock_run.side_effect = [
            {"results": [{"metadata": {"model_id": "id", "name": "demo"}}]},
            {"results": [{"document": "print('hello')", "metadata": {"chunk_id": 1, "offset": 0}}]}
        ]
        mock_code_generator = MagicMock()
        mock_code_generator.generate_full_script.return_value = "full code"
        mock_repro_manager = MagicMock()
        mock_repro_manager.add_reproducibility_info.side_effect = lambda nb, mid: nb
        mock_chroma = MagicMock()
        comps = {
            "colab_generator": {
                "code_generator": mock_code_generator,
                "reproducibility_manager": mock_repro_manager,
            },
            "vector_db": {"chroma_manager": mock_chroma},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "out.ipynb")
            path = NotebookGenerator.generate_notebook(comps, "id", out_path)
            self.assertEqual(path, out_path)
            mock_write.assert_called_once()
            mock_makedirs.assert_called()
            mock_code_generator.generate_full_script.assert_called_once()


