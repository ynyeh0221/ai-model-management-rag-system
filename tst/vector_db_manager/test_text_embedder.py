import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.vector_db_manager.text_embedder import TextEmbedder


# Create a dummy model to replace SentenceTransformer
class DummyModel:
    def __init__(self, embedding_dim=10):
        self._dim = embedding_dim

    def encode(self, text, **kwargs):
        # If given a list of texts, return an array of ones per text.
        if isinstance(text, list):
            return np.array([np.full(self._dim, 1) for _ in text])
        # For a single text, return a vector of ones.
        return np.full(self._dim, 1)

    def get_sentence_embedding_dimension(self):
        return self._dim


class TestTextEmbedder(unittest.TestCase):
    def setUp(self):
        # Skip the _initialize_model method to avoid importing SentenceTransformer
        with patch.object(TextEmbedder, '_initialize_model'):
            self.embedder = TextEmbedder(model_name="dummy-model", device="cpu")

        # Set up the embedder with our dummy model directly
        self.embedder.model = DummyModel(embedding_dim=10)
        self.embedder.embedding_dim = self.embedder.model.get_sentence_embedding_dimension()

    def test_embed_text_empty(self):
        """An empty string should return a zero vector."""
        embedding = self.embedder.embed_text("")
        np.testing.assert_array_equal(embedding, np.zeros(self.embedder.embedding_dim))

    def test_embed_text_normal(self):
        """Embedding a normal text should return the dummy model's vector."""
        text = "Hello world"
        embedding = self.embedder.embed_text(text)
        expected = np.full(self.embedder.embedding_dim, 1)
        np.testing.assert_array_equal(embedding, expected)

    def test_embed_batch_empty_list(self):
        """An empty list should return an empty numpy array."""
        embeddings = self.embedder.embed_batch([])
        self.assertEqual(embeddings.size, 0)

    def test_embed_batch_normal(self):
        """Batch embedding should process each text (including empty ones) correctly."""
        texts = ["Hello", "world", ""]
        embeddings = self.embedder.embed_batch(texts)
        # All three texts should return vectors of ones because of our DummyModel
        # which returns ones for any input, including a single space (which is what
        # empty strings are replaced with)
        expected = np.array([np.full(self.embedder.embedding_dim, 1) for _ in range(len(texts))])
        np.testing.assert_array_equal(embeddings, expected)

    def test_embed_mixed_content_empty(self):
        """
        Providing empty content (i.e. an empty dict) should result in a zero vector.
        """
        embedding = self.embedder.embed_mixed_content({})
        np.testing.assert_array_equal(embedding, np.zeros(self.embedder.embedding_dim))

    def test_embed_mixed_content_normal(self):
        """
        Test that a dictionary with various text sections is combined properly.
        We mock the embed_text method to verify the combined text.
        """
        content = {
            "title": "Test Title",
            "description": "Description here.",
            "code": "print('hello')",
            "comments": "Some comments."
        }
        # The expected combined text with title repeated for higher weight
        expected_combined_text = "Test Title Test Title Description here. print('hello') Some comments."

        # Mock the embed_text method to verify the combined text
        with patch.object(self.embedder, 'embed_text') as mock_embed_text:
            mock_embed_text.return_value = np.full(self.embedder.embedding_dim, 1)
            self.embedder.embed_mixed_content(content)

            # Verify that embed_text was called with the expected combined text
            mock_embed_text.assert_called_once_with(expected_combined_text, True)

    def test_compute_similarity(self):
        """
        Test the compute_similarity method with:
          - Identical vectors (expect similarity 1)
          - Orthogonal vectors (expect similarity 0)
        """
        # Test with identical vectors.
        vector = np.array([1, 2, 3], dtype=float)
        sim = self.embedder.compute_similarity(vector, vector)
        self.assertAlmostEqual(sim, 1.0)

        # Test with orthogonal vectors.
        v1 = np.array([1, 0, 0], dtype=float)
        v2 = np.array([0, 1, 0], dtype=float)
        sim = self.embedder.compute_similarity(v1, v2)
        self.assertAlmostEqual(sim, 0.0)

    def test_find_most_similar(self):
        """
        Test finding the most similar embeddings.
        Verifies that the returned indices correspond to expected positions
        and that scores are calculated correctly.
        """
        embeddings = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1]
        ], dtype=float)
        query_embedding = np.array([1, 0, 0], dtype=float)
        results = self.embedder.find_most_similar(query_embedding, embeddings, top_k=2)

        # Verify we got exactly 2 results
        self.assertEqual(len(results), 2)

        # Verify the first result is index 0 (exact match) with score 1.0
        self.assertEqual(results[0]['index'], 0)
        self.assertAlmostEqual(results[0]['score'], 1.0)

        # Verify the second result is index 2 ([1, 1, 0]) with score 0.7071...
        self.assertEqual(results[1]['index'], 2)
        self.assertAlmostEqual(results[1]['score'], 1 / np.sqrt(2))

    def test_save_and_load_embeddings(self):
        """
        Test that embeddings can be successfully saved to and loaded from a file.
        """
        embeddings = np.array([[1, 2, 3], [4, 5, 6]])
        with tempfile.TemporaryDirectory() as tmpdirname:
            filepath = os.path.join(tmpdirname, "embeddings.npy")
            success = self.embedder.save_embeddings(embeddings, filepath)
            self.assertTrue(success)
            loaded = self.embedder.load_embeddings(filepath)
            np.testing.assert_array_equal(loaded, embeddings)

    def test_load_nonexistent_file(self):
        """
        Attempting to load embeddings from a non-existent file should return None.
        """
        result = self.embedder.load_embeddings("non_existent_file.npy")
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()