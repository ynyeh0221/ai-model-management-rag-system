from typing import List, Dict

import nbformat


class CodeGenerator:
    """
    Generates Python code or Jupyter notebooks based on reconstructed content
    from document chunks without using templates.
    """

    def __init__(self):
        # (Optional) Framework-specific imports if needed elsewhere
        self.framework_imports = {
            "pytorch": ["import torch", "import torch.nn as nn", "import torch.optim as optim"],
            "tensorflow": ["import tensorflow as tf", "from tensorflow import keras"],
            "jax": ["import jax", "import jax.numpy as jnp", "import flax"]
        }

    def generate_full_script(self, chunks: List[Dict], use_offset=True, overlap: int = 100) -> str:
        """
        Reconstruct full Python script from structured chunks (optionally using offset metadata).

        Args:
            chunks (List[Dict]): A list of chunk dictionaries from split_ast_and_subsplit_chunks
            use_offset (bool): Whether to sort and merge based on 'offset' field
            overlap (int): Max overlapping chars to deduplicate between chunks

        Returns:
            str: Full reconstructed code.
        """
        if not chunks:
            return ""

        if use_offset:
            # Sort chunks by recorded position in original source
            chunks = sorted(chunks, key=lambda c: c.get("offset", 0))

        result = chunks[0]["text"]

        for i in range(1, len(chunks)):
            prev_tail = result[-overlap:]
            curr = chunks[i]["text"]

            # Try to find the overlap (deduplication)
            match_len = 0
            for j in range(min(len(prev_tail), len(curr), overlap), 0, -1):
                if prev_tail[-j:] == curr[:j]:
                    match_len = j
                    break

            result += curr[match_len:]

        return result

    def generate_notebook_from_chunks(self, chunks: List[str]) -> nbformat.NotebookNode:
        """
        Generate a minimal Jupyter notebook from code chunks.

        Args:
            chunks (List[str]): List of code strings (chunks).

        Returns:
            nbformat.NotebookNode: A notebook object containing a single code cell.
        """
        code = self.generate_full_script(chunks)

        nb = nbformat.v4.new_notebook()
        nb.cells = [nbformat.v4.new_code_cell(code)]

        return nb
