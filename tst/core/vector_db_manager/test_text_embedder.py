import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.core.vector_db.text_embedder import TextEmbedder


# Create a fake model to replace SentenceTransformer
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

        # Set up the embedder with our fake model directly
        self.embedder.model = DummyModel(embedding_dim=10)
        self.embedder.embedding_dim = self.embedder.model.get_sentence_embedding_dimension()

    def test_embed_text_empty(self):
        """An empty string should return a zero vector."""
        embedding = self.embedder.embed_text("")
        np.testing.assert_array_equal(embedding, np.zeros(self.embedder.embedding_dim))

    def test_embed_text_normal(self):
        """Embedding a normal text should return the fake model's vector."""
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
        # All three texts should return vectors of ones because of our DummyModel,
        # which returns ones for any input, including a single space (which is what
        # empty strings are replaced with)
        expected = np.array([np.full(self.embedder.embedding_dim, 1) for _ in range(len(texts))])
        np.testing.assert_array_equal(embeddings, expected)

    def test_embed_mixed_content_empty(self):
        """
        Providing empty content (i.e., an empty dict) should result in a zero vector.
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

    def test_call_method_empty_input(self):
        """Test __call__ method with an empty input list."""
        result = self.embedder([])
        self.assertEqual(result, [])

    def test_call_method_normal_input(self):
        """Test __call__ method with normal input (ChromaDB compatibility)."""
        texts = ["Hello", "world"]
        result = self.embedder(texts)
        # Should return a list of lists (embeddings as lists, not numpy arrays)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], list)
        self.assertEqual(len(result[0]), self.embedder.embedding_dim)

    def test_call_method_error_handling(self):
        """Test __call__ method error handling."""
        # Mock embed_batch to raise an exception
        with patch.object(self.embedder, 'embed_batch', side_effect=Exception("Test error")):
            result = self.embedder(["test"])
            # Should return list of zero vectors when error occurs
            expected = [[0.0] * self.embedder.embedding_dim]
            self.assertEqual(result, expected)

    def test_name_method(self):
        """Test the name method returns the correct format."""
        expected_name = f"TextEmbedder::{self.embedder.model_name}"
        self.assertEqual(self.embedder.name(), expected_name)

    def test_embed_text_error_handling(self):
        """Test embed_text error handling when model.encode fails."""
        with patch.object(self.embedder.model, 'encode', side_effect=Exception("Model error")):
            embedding = self.embedder.embed_text("test text")
            # Should return zero vector when error occurs
            np.testing.assert_array_equal(embedding, np.zeros(self.embedder.embedding_dim))

    def test_embed_batch_error_handling(self):
        """Test embed_batch error handling when model.encode fails."""
        texts = ["text1", "text2"]
        with patch.object(self.embedder.model, 'encode', side_effect=Exception("Model error")):
            embeddings = self.embedder.embed_batch(texts)
            # Should return zero matrix when error occurs
            expected = np.zeros((len(texts), self.embedder.embedding_dim))
            np.testing.assert_array_equal(embeddings, expected)

    def test_compute_similarity_zero_vectors(self):
        """Test compute_similarity with zero vectors."""
        zero_vector = np.zeros(3)
        normal_vector = np.array([1, 2, 3])

        # Zero vectors vs. normal vector should return 0
        sim = self.embedder.compute_similarity(zero_vector, normal_vector)
        self.assertEqual(sim, 0.0)

        # Zero vector vs zero vector should return 0
        sim = self.embedder.compute_similarity(zero_vector, zero_vector)
        self.assertEqual(sim, 0.0)

    def test_compute_similarity_error_handling(self):
        """Test compute_similarity error handling."""
        # Test with incompatible shapes or other errors
        with patch('numpy.dot', side_effect=Exception("Computation error")):
            sim = self.embedder.compute_similarity(np.array([1, 2]), np.array([3, 4]))
            self.assertEqual(sim, 0.0)

    def test_find_most_similar_all_zero_embeddings(self):
        """Test find_most_similar when all embeddings are zero vectors."""
        embeddings = np.zeros((3, 5))  # 3 zero vectors of dimension 5
        query_embedding = np.array([1, 2, 3, 4, 5])

        results = self.embedder.find_most_similar(query_embedding, embeddings, top_k=2)
        # Should return an empty list when all embeddings are zero
        self.assertEqual(results, [])

    def test_find_most_similar_fewer_embeddings_than_top_k(self):
        """Test find_most_similar when there are fewer valid embeddings than top_k."""
        embeddings = np.array([
            [1, 0, 0],
            [0, 1, 0]
        ])
        query_embedding = np.array([1, 0, 0])

        results = self.embedder.find_most_similar(query_embedding, embeddings, top_k=5)
        # Should return all available embeddings (2) even though top_k=5
        self.assertEqual(len(results), 2)

    def test_find_most_similar_error_handling(self):
        """Test find_most_similar error handling."""
        embeddings = np.array([[1, 2, 3]])
        query_embedding = np.array([1, 2, 3])

        with patch('numpy.dot', side_effect=Exception("Computation error")):
            results = self.embedder.find_most_similar(query_embedding, embeddings, top_k=1)
            self.assertEqual(results, [])

    def test_save_embeddings_directory_creation(self):
        """Test that save_embeddings creates directories if they don't exist."""
        embeddings = np.array([[1, 2, 3]])

        with tempfile.TemporaryDirectory() as tmpdirname:
            # Create a nested path that doesn't exist
            filepath = os.path.join(tmpdirname, "nested", "folder", "embeddings.npy")
            success = self.embedder.save_embeddings(embeddings, filepath)

            self.assertTrue(success)
            self.assertTrue(os.path.exists(filepath))

    def test_save_embeddings_error_handling(self):
        """Test save_embeddings error handling."""
        embeddings = np.array([[1, 2, 3]])

        # Try to save it to an invalid path
        with patch('numpy.save', side_effect=Exception("Write error")):
            success = self.embedder.save_embeddings(embeddings, "test.npy")
            self.assertFalse(success)

    def test_load_embeddings_error_handling(self):
        """Test load_embeddings error handling."""
        # Mock np.load to raise an exception
        with patch('numpy.load', side_effect=Exception("Load error")):
            # Create a temporary file that exists but will fail to load
            with tempfile.NamedTemporaryFile(suffix=".npy") as temp_file:
                result = self.embedder.load_embeddings(temp_file.name)
                self.assertIsNone(result)

    def test_embed_mixed_content_partial_content(self):
        """Test embed_mixed_content with only some fields present."""
        # Test with only title
        content = {"title": "Only Title"}
        with patch.object(self.embedder, 'embed_text') as mock_embed_text:
            mock_embed_text.return_value = np.full(self.embedder.embedding_dim, 1)
            self.embedder.embed_mixed_content(content)
            # Title should be repeated for higher weight
            mock_embed_text.assert_called_once_with("Only Title Only Title", True)

    def test_embed_mixed_content_error_handling(self):
        """Test embed_mixed_content error handling."""
        content = {"title": "Test"}

        with patch.object(self.embedder, 'embed_text', side_effect=Exception("Embed error")):
            embedding = self.embedder.embed_mixed_content(content)
            np.testing.assert_array_equal(embedding, np.zeros(self.embedder.embedding_dim))

    def test_embed_batch_with_whitespace_only_texts(self):
        """Test embed_batch handling of whitespace-only texts."""
        texts = ["  ", "\t\n", "normal text", ""]
        embeddings = self.embedder.embed_batch(texts)

        # All texts should be processed (empty ones replaced with single space)
        self.assertEqual(len(embeddings), len(texts))
        for embedding in embeddings:
            self.assertEqual(len(embedding), self.embedder.embedding_dim)

    def test_compute_similarity_range_clamping(self):
        """Test that compute_similarity clamps results to [0, 1] ranges."""
        # This is more of a safety test since cosine similarity should naturally be in [-1, 1]
        # but the code explicitly clamps to [0, 1]
        v1 = np.array([1, 0, 0])
        v2 = np.array([-1, 0, 0])  # Opposite direction, should give negative similarity

        sim = self.embedder.compute_similarity(v1, v2)
        # Result should be clamped to 0.0 (minimum)
        self.assertEqual(sim, 0.0)

    def test_initialize_model_successful(self):
        """Test successful model initialization."""
        mock_model = DummyModel(embedding_dim=384)

        with patch('sentence_transformers.SentenceTransformer') as mock_st:
            mock_st.return_value = mock_model

            embedder = TextEmbedder(model_name="test-model", device="cpu")

            # Verify model was loaded correctly
            self.assertEqual(embedder.model, mock_model)
            self.assertEqual(embedder.embedding_dim, 384)
            self.assertEqual(embedder.device, "cpu")
            mock_st.assert_called_once_with("test-model", device="cpu")

    def test_initialize_model_import_error(self):
        """Test initialization when sentence-transformers are not installed."""
        # We need to patch the import itself, not the class
        import builtins
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == 'sentence_transformers':
                raise ImportError("No module named 'sentence_transformers'")
            return original_import(name, *args, **kwargs)

        with patch('builtins.__import__', side_effect=mock_import):
            with self.assertRaises(ImportError):
                TextEmbedder(model_name="test-model")

    def test_initialize_model_device_detection_mps(self):
        """Test automatic device detection - MPS available."""
        mock_model = DummyModel()

        with patch('sentence_transformers.SentenceTransformer') as mock_st, \
                patch('torch.backends.mps.is_available', return_value=True), \
                patch('torch.cuda.is_available', return_value=False):
            mock_st.return_value = mock_model

            embedder = TextEmbedder(model_name="test-model", device=None)

            self.assertEqual(embedder.device, "mps")
            mock_st.assert_called_once_with("test-model", device="mps")

    def test_initialize_model_device_detection_cuda(self):
        """Test automatic device detection - CUDA available, MPS not available."""
        mock_model = DummyModel()

        with patch('sentence_transformers.SentenceTransformer') as mock_st, \
                patch('torch.backends.mps.is_available', return_value=False), \
                patch('torch.cuda.is_available', return_value=True):
            mock_st.return_value = mock_model

            embedder = TextEmbedder(model_name="test-model", device=None)

            self.assertEqual(embedder.device, "cuda")
            mock_st.assert_called_once_with("test-model", device="cuda")

    def test_initialize_model_device_detection_cpu_fallback(self):
        """Test automatic device detection - CPU fallback when neither MPS nor CUDA available."""
        mock_model = DummyModel()

        with patch('sentence_transformers.SentenceTransformer') as mock_st, \
                patch('torch.backends.mps.is_available', return_value=False), \
                patch('torch.cuda.is_available', return_value=False):
            mock_st.return_value = mock_model

            embedder = TextEmbedder(model_name="test-model", device=None)

            self.assertEqual(embedder.device, "cpu")
            mock_st.assert_called_once_with("test-model", device="cpu")

    def test_initialize_model_with_cache_folder(self):
        """Test initialization with cache folder - should create directory and set environment variable."""
        mock_model = DummyModel()

        with patch('sentence_transformers.SentenceTransformer') as mock_st, \
                patch('os.makedirs') as mock_makedirs, \
                patch.dict('os.environ', {}, clear=True):
            mock_st.return_value = mock_model
            cache_folder = "/path/to/cache"

            embedder = TextEmbedder(model_name="test-model", cache_folder=cache_folder)

            # Verify cache folder was created and environment variable was set
            mock_makedirs.assert_called_once_with(cache_folder, exist_ok=True)
            self.assertEqual(os.environ.get('TRANSFORMERS_CACHE'), cache_folder)
            self.assertEqual(embedder.cache_folder, cache_folder)

    def test_initialize_model_general_exception(self):
        """Test initialization when model loading fails with general exception."""
        with patch('sentence_transformers.SentenceTransformer', side_effect=RuntimeError("Model loading failed")):
            with self.assertRaises(RuntimeError):
                TextEmbedder(model_name="test-model")

    def test_initialize_model_logs_info_on_success(self):
        """Test that successful initialization logs appropriate info message."""
        mock_model = DummyModel(embedding_dim=768)

        with patch('sentence_transformers.SentenceTransformer') as mock_st, \
                patch('logging.getLogger') as mock_get_logger:
            mock_st.return_value = mock_model
            mock_logger = mock_get_logger.return_value

            embedder = TextEmbedder(model_name="test-model", device="cpu")

            # Verify info message was logged
            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args[0][0]
            self.assertIn("Initialized TextEmbedder with model test-model", call_args)
            self.assertIn("dimension: 768", call_args)
            self.assertIn("on cpu", call_args)

    def test_initialize_model_logs_error_on_import_failure(self):
        """Test that import error is properly logged."""
        import builtins
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == 'sentence_transformers':
                raise ImportError("No module named 'sentence_transformers'")
            return original_import(name, *args, **kwargs)

        with patch('builtins.__import__', side_effect=mock_import), \
                patch('logging.getLogger') as mock_get_logger:
            mock_logger = mock_get_logger.return_value

            with self.assertRaises(ImportError):
                TextEmbedder(model_name="test-model")

            # Verify error message was logged
            mock_logger.error.assert_called_once()
            call_args = mock_logger.error.call_args[0][0]
            self.assertIn("sentence-transformers package not found", call_args)

    def test_initialize_model_logs_error_on_general_exception(self):
        """Test that general exceptions during initialization are properly logged."""
        with patch('sentence_transformers.SentenceTransformer', side_effect=RuntimeError("Generic error")), \
                patch('logging.getLogger') as mock_get_logger:
            mock_logger = mock_get_logger.return_value

            with self.assertRaises(RuntimeError):
                TextEmbedder(model_name="test-model")

            # Verify error message was logged with exc_info
            mock_logger.error.assert_called_once()
            call_args = mock_logger.error.call_args
            self.assertIn("Error initializing embedding model", call_args[0][0])
            self.assertTrue(call_args[1]['exc_info'])  # exc_info=True was passed

    def test_initialize_model_no_cache_folder_no_env_change(self):
        """Test that environment variable is not changed when no cache folder is provided."""
        mock_model = DummyModel()
        original_env = os.environ.get('TRANSFORMERS_CACHE', 'NOT_SET')

        with patch('sentence_transformers.SentenceTransformer') as mock_st:
            mock_st.return_value = mock_model

            embedder = TextEmbedder(model_name="test-model", cache_folder=None)

            # Environment variable should not have been modified
            current_env = os.environ.get('TRANSFORMERS_CACHE', 'NOT_SET')
            self.assertEqual(current_env, original_env)

if __name__ == '__main__':
    unittest.main()