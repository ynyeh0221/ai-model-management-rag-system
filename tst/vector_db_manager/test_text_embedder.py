import os
import tempfile
import unittest
import numpy as np
from sklearn.decomposition import PCA

# Import the class to test
from src.vector_db_manager.text_embedder import TextEmbedder


# Define a dummy SentenceTransformer so we don't actually download models.
class DummySentenceTransformer:
    def __init__(self, *args, **kwargs):
        self.embedding_dimension = 10

    def get_sentence_embedding_dimension(self):
        return self.embedding_dimension

    def encode(self, data, **kwargs):
        """
        For simplicity, if a list is provided, return an array of ones
        with shape (n, embedding_dimension). If a string is provided,
        return a one-dimensional array of ones.
        """
        if isinstance(data, list):
            return np.ones((len(data), self.embedding_dimension))
        else:
            return np.ones(self.embedding_dimension)


# Subclass TextEmbedder to override the model initialization.
class DummyTextEmbedder(TextEmbedder):
    def _initialize_model(self):
        # Instead of loading a real model, just use the dummy transformer.
        self.model = DummySentenceTransformer()
        self.embedding_dim = self.model.get_sentence_embedding_dimension()


class TestTextEmbedder(unittest.TestCase):
    def setUp(self):
        # Instantiate our dummy text embedder.
        self.embedder = DummyTextEmbedder()

    def test_embed_text_non_empty(self):
        text = "This is a test."
        result = self.embedder.embed_text(text)
        # Expect a vector with shape (embedding_dim,) filled with ones
        self.assertEqual(result.shape, (self.embedder.embedding_dim,))
        self.assertTrue(np.all(result == 1))

    def test_embed_text_empty(self):
        result = self.embedder.embed_text("")
        # Expect a zero vector when text is empty
        self.assertEqual(result.shape, (self.embedder.embedding_dim,))
        self.assertTrue(np.all(result == 0))

    def test_embed_batch_non_empty(self):
        texts = ["test one", "test two", "test three"]
        result = self.embedder.embed_batch(texts)
        # Should return an array of shape (number of texts, embedding_dim)
        self.assertEqual(result.shape, (3, self.embedder.embedding_dim))
        self.assertTrue(np.all(result == 1))
         
    def test_embed_batch_empty_list(self):
        result = self.embedder.embed_batch([])
        # Expect an empty array when no texts are provided
        self.assertEqual(result.shape, (0,))

    def test_reduce_dimensions_no_reduction(self):
        # If target dimension is not lower than current dimension, the embeddings remain unchanged.
        embeddings = np.random.rand(5, self.embedder.embedding_dim)
        reduced = self.embedder.reduce_dimensions(embeddings, dim=self.embedder.embedding_dim)
        np.testing.assert_array_almost_equal(reduced, embeddings)

    def test_reduce_dimensions(self):
        # Create dummy embeddings and reduce from embedding_dim (10) to 5 dimensions.
        embeddings = np.random.rand(5, self.embedder.embedding_dim)
        reduced = self.embedder.reduce_dimensions(embeddings, dim=5)
        self.assertEqual(reduced.shape, (5, 5))

    def test_embed_code(self):
        code = "def foo(): pass"
        # embed_code is defined as calling embed_text directly.
        result_text = self.embedder.embed_text(code)
        result_code = self.embedder.embed_code(code)
        np.testing.assert_array_equal(result_text, result_code)

    def test_embed_mixed_content_empty(self):
        # If content is empty, should return a zero vector.
        result = self.embedder.embed_mixed_content({})
        self.assertEqual(result.shape, (self.embedder.embedding_dim,))
        self.assertTrue(np.all(result == 0))

    def test_embed_mixed_content(self):
        # Provide content with several keys. Since our dummy always returns ones,
        # the combined text leads to the same result as a non-empty string.
        content = {
            "title": "Title",
            "description": "Desc",
            "code": "print('Hello')",
            "comments": "Nice code"
        }
        result = self.embedder.embed_mixed_content(content)
        self.assertEqual(result.shape, (self.embedder.embedding_dim,))
        self.assertTrue(np.all(result == 1))

    def test_compute_similarity(self):
        # Test with two identical vectors.
        vec1 = np.array([1, 0])
        vec2 = np.array([1, 0])
        similarity = self.embedder.compute_similarity(vec1, vec2)
        self.assertAlmostEqual(similarity, 1.0)
        
        # Test with a zero vector.
        vec_zero = np.array([0, 0])
        similarity_zero = self.embedder.compute_similarity(vec1, vec_zero)
        self.assertEqual(similarity_zero, 0.0)
         
    def test_find_most_similar(self):
        # For this test, we work with low-dimensional vectors for clarity.
        # Temporarily override embedding_dim for test purposes.
        self.embedder.embedding_dim = 2
        # Create three dummy embeddings.
        #
