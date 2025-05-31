import logging
import os
from typing import List, Optional, Dict, Any

import numpy as np
import torch


class TextEmbedder:
    """
    Text embedding generator that creates vector representations
    of text and code for semantic search and comparison.
    """

    def __init__(self, model_name="BAAI/bge-m3", device=None, cache_folder=None):
        """
        Initialize the TextEmbedder with a specific model.

        Args:
            model_name: Name of the SentenceTransformer model to use
            device: Device to run the model on ('cpu', 'cuda', or None for auto-detection)
            cache_folder: Optional folder to cache downloaded models
        """
        self.model_name = model_name
        self.device = device
        self.cache_folder = cache_folder
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.embedding_dim = None
        self._initialize_model()

    def _initialize_model(self):
        """
        Initialize the embedding model.
        Downloads and loads the model from Hugging Face.
        """
        try:
            # Import here for better error handling
            from sentence_transformers import SentenceTransformer

            # Set cache folder if provided
            if self.cache_folder:
                os.makedirs(self.cache_folder, exist_ok=True)
                os.environ['TRANSFORMERS_CACHE'] = self.cache_folder

            # Detect device if not specified
            if self.device is None:
                # Extract nested conditional into separate statements
                if torch.backends.mps.is_available():
                    self.device = "mps"
                elif torch.cuda.is_available():
                    self.device = "cuda"
                else:
                    self.device = "cpu"

            # Load the model
            self.model = SentenceTransformer(self.model_name, device=self.device)

            # Store embedding dimension
            self.embedding_dim = self.model.get_sentence_embedding_dimension()

            self.logger.info(
                f"Initialized TextEmbedder with model {self.model_name} "
                f"(dimension: {self.embedding_dim}) on {self.device}"
            )

        except ImportError:
            self.logger.error(
                "sentence-transformers package not found. Install with: pip install sentence-transformers"
            )
            raise
        except Exception as e:
            self.logger.error(f"Error initializing embedding model: {e}", exc_info=True)
            raise

    def embed_text(self, text: str, normalize: bool = True) -> np.ndarray:
        """
        Generate embeddings for a text string.

        Args:
            text: Text to embed
            normalize: Whether to L2-normalize the embedding

        Returns:
            numpy.ndarray: Embedding vector
        """
        if not text:
            self.logger.warning("Received empty text for embedding")
            # Return zero vector with correct dimension
            return np.zeros(self.embedding_dim)

        try:
            # Generate embedding
            embedding = self.model.encode(text, normalize_embeddings=normalize)
            return embedding

        except Exception as e:
            self.logger.error(f"Error embedding text: {e}", exc_info=True)
            # Return zero vector in case of error
            return np.zeros(self.embedding_dim)

    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        normalize: bool = True,
        show_progress_bar: bool = False
    ) -> np.ndarray:
        """
        Generate embeddings for a batch of texts.

        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            normalize: Whether to L2-normalize the embeddings
            show_progress_bar: Whether to show a progress bar

        Returns:
            numpy.ndarray: Array of embedding vectors
        """
        if not texts:
            self.logger.warning("Received empty list for batch embedding")
            return np.array([])

        try:
            # Filter out empty texts and replace with single space
            processed_texts = [text if text.strip() else " " for text in texts]

            # Generate embeddings in batch
            embeddings = self.model.encode(
                processed_texts,
                batch_size=batch_size,
                normalize_embeddings=normalize,
                show_progress_bar=show_progress_bar
            )

            return embeddings

        except Exception as e:
            self.logger.error(
                f"Error embedding batch of {len(texts)} texts: {e}", exc_info=True
            )
            # Return zero matrix in case of error
            return np.zeros((len(texts), self.embedding_dim))

    def embed_mixed_content(
        self, content: Dict[str, Any], normalize: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for mixed content with different sections.

        Args:
            content: Dictionary with text sections
            normalize: Whether to L2-normalize the embedding

        Returns:
            numpy.ndarray: Embedding vector
        """
        if not content:
            self.logger.warning("Received empty content for embedding")
            return np.zeros(self.embedding_dim)

        try:
            # Combine different content sections with weights
            sections = []

            # Add title with higher weight
            if 'title' in content and content['title']:
                sections.append(content['title'] + " " + content['title'])  # Repeat for higher weight

            # Add description
            if 'description' in content and content['description']:
                sections.append(content['description'])

            # Add code with special handling
            if 'code' in content and content['code']:
                sections.append(content['code'])

            # Add comments or docstrings
            if 'comments' in content and content['comments']:
                sections.append(content['comments'])

            # Combine all sections
            combined_text = " ".join(sections)

            # Generate embedding
            return self.embed_text(combined_text, normalize)

        except Exception as e:
            self.logger.error(f"Error embedding mixed content: {e}", exc_info=True)
            # Return zero vector in case of error
            return np.zeros(self.embedding_dim)

    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
            float: Cosine similarity score (0-1)
        """
        try:
            # Ensure embeddings are normalized
            if embedding1.ndim == 1:
                embedding1 = embedding1.reshape(1, -1)
            if embedding2.ndim == 1:
                embedding2 = embedding2.reshape(1, -1)

            # Normalize if needed
            norm1 = np.linalg.norm(embedding1, axis=1, keepdims=True)
            norm2 = np.linalg.norm(embedding2, axis=1, keepdims=True)

            if np.any(norm1 == 0) or np.any(norm2 == 0):
                return 0.0

            embedding1_normalized = embedding1 / norm1
            embedding2_normalized = embedding2 / norm2

            # Compute cosine similarity
            similarity = np.dot(embedding1_normalized, embedding2_normalized.T)[0, 0]

            # Ensure the result is in the range [0, 1]
            return max(0.0, min(1.0, float(similarity)))

        except Exception as e:
            self.logger.error(f"Error computing similarity: {e}", exc_info=True)
            return 0.0

    def find_most_similar(
        self,
        query_embedding: np.ndarray,
        embeddings: np.ndarray,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find the most similar embeddings to a query embedding.

        Args:
            query_embedding: Query embedding
            embeddings: Array of embeddings to search
            top_k: Number of top matches to return

        Returns:
            List of dicts with indices and similarity scores
        """
        try:
            # Ensure query embedding is normalized
            query_norm = np.linalg.norm(query_embedding)
            if query_norm > 0:
                query_embedding = query_embedding / query_norm

            # Ensure embeddings are normalized
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            # Use np.nonzero when only condition parameter is provided
            valid_mask = (norms.flatten() > 0)
            valid_indices = np.nonzero(valid_mask)[0]

            if len(valid_indices) == 0:
                return []

            valid_embeddings = embeddings[valid_indices] / norms[valid_indices]

            # Compute similarities
            similarities = np.dot(valid_embeddings, query_embedding)

            # Get top k indices
            if len(similarities) <= top_k:
                top_indices = np.argsort(-similarities)
            else:
                top_indices = np.argpartition(-similarities, top_k)[:top_k]
                top_indices = top_indices[np.argsort(-similarities[top_indices])]

            # Map back to original indices
            original_indices = valid_indices[top_indices]

            # Create results
            results = [
                {
                    'index': int(original_indices[i]),
                    'score': float(similarities[top_indices[i]])
                }
                for i in range(len(top_indices))
            ]

            return results

        except Exception as e:
            self.logger.error(f"Error finding most similar embeddings: {e}", exc_info=True)
            return []

    def save_embeddings(self, embeddings: np.ndarray, filepath: str) -> bool:
        """
        Save embeddings to a file.

        Args:
            embeddings: Embeddings to save
            filepath: Path to save to

        Returns:
            bool: Success status
        """
        try:
            directory = os.path.dirname(filepath)
            if directory:
                os.makedirs(directory, exist_ok=True)

            np.save(filepath, embeddings)

            self.logger.debug(f"Saved embeddings to {filepath}")

            return True

        except Exception as e:
            self.logger.error(f"Error saving embeddings to {filepath}: {e}", exc_info=True)
            return False

    def load_embeddings(self, filepath: str) -> Optional[np.ndarray]:
        """
        Load embeddings from a file.

        Args:
            filepath: Path to load from

        Returns:
            numpy.ndarray or None: Loaded embeddings or None if error
        """
        try:
            if not os.path.exists(filepath):
                self.logger.warning(f"Embeddings file {filepath} not found")
                return None

            embeddings = np.load(filepath)

            self.logger.debug(f"Loaded embeddings from {filepath}")

            return embeddings

        except Exception as e:
            self.logger.error(f"Error loading embeddings from {filepath}: {e}", exc_info=True)
            return None

    def __call__(self, input: List[str]) -> List[List[float]]:
        """
        Make TextEmbedder compatible with ChromaDB's embedding_function interface.

        Args:
            input: List of strings to embed

        Returns:
            List of embedding vectors (List[List[float]])
        """
        if not input:
            return []

        try:
            embeddings = self.embed_batch(input)
            return embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings
        except Exception as e:
            self.logger.error(f"__call__ failed in TextEmbedder: {e}", exc_info=True)
            return [[0.0] * self.embedding_dim for _ in input]

    def name(self) -> str:
        return f"TextEmbedder::{self.model_name}"
