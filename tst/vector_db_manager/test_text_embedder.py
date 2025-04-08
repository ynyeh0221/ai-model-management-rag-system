import os
import unittest
import tempfile
import numpy as np
from unittest.mock import patch
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
        # Patch SentenceTransformer to avoid downloading a real model.
        patcher = patch("src.vector_db_manager.text_embedder.SentenceTransformer", return_value=DummyModel(embedding_dim=10))
        self.addCleanup(patcher.stop)
        self.mock_sentence_transformer = patcher.start()
        # Create an instance of TextEmbedder. This will use the DummyModel.
        self.embedder = TextEmbedder(model_name="dummy-model", device="cpu")

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
        expected = np.array([np.full(self.embedder.embedding_dim, 1) for _ in texts])
        np.testing.assert_array_equal(embeddings, expected)

    def test_reduce_dimensions_no_reduction(self):
        """
        If the target dimension is equal to or higher than the embedding dimension,
        the original embeddings should be returned.
        """
        embeddings = np.random.rand(5, self.embedder.embedding_dim)
        reduced = self.embedder.reduce_dimensions(embeddings, dim=self.embedder.embedding_dim)
        np.testing.assert_array_almost_equal(reduced, embeddings)

    def test_reduce_dimensions_with_reduction(self):
        """
        Test reducing dimensions when the target dimension is lower than the original.
        A PCA should fit and transform the embeddings accordingly.
        """
        embeddings = np.random.rand(10, self.embedder.embedding_dim)
        target_dim = 5
        reduced = self.embedder.reduce_dimensions(embeddings, dim=target_dim, fit=True)
        self.assertEqual(reduced.shape[1], target_dim)

    def test_embed_code(self):
        """
        The embed_code method should produce the same result as embed_text.
        """
        code = "print('hello')"
        embedding_text = self.embedder.embed_text(code)
        embedding_code = self.embedder.embed_code(code)
        np.testing.assert_array_equal(embedding_text, embedding_code)

    def test_embed_mixed_content_empty(self):
        """
        Providing empty content (i.e. an empty dict) should result in a zero vector.
        """
        embedding = self.embedder.embed_mixed_content({})
        np.testing.assert_array_equal(embedding, np.zeros(self.embedder.embedding_dim))

    def test_embed_mixed_content_normal(self):
        """
        Test that a dictionary with various text sections is combined properly.
        In the dummy model, this still returns a vector of ones.
        """
        content = {
            "title": "Test Title",
            "description": "Description here.",
            "code": "print('hello')",
            "comments": "Some comments."
        }
        # The embed_mixed_content method repeats title for higher weight.
        combined_text = "Test Title Test Title Description here. print('hello') Some comments."
        # DummyModel.encode always returns ones.
        expected = np.full(self.embedder.embedding_dim, 1)
        embedding = self.embedder.embed_mixed_content(content)
        np.testing.assert_array_equal(embedding, expected)

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
        Verifies that the returned indices correspond to expected positions.
        """
        embeddings = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1]
        ], dtype=float)
        query_embedding = np.array([1, 0, 0], dtype=float)
        results = self.embedder.find_most_similar(query_embedding, embeddings, top_k=2)
        # Since the query is [1,0,0], the first result should be the index with a vector [1,0,0]
        indices = [result['index'] for result in results]
        self.assertIn(0, indices)
        self.assertLessEqual(len(results), 2)

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
