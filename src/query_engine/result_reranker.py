# src/query_engine/result_reranker.py
import logging
from typing import List, Dict, Any, Union, Optional


class CrossEncoderReranker:
    """
    A reranking model that uses a cross-encoder architecture to
    rerank search results based on query-document relevance.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: str = "cpu",
                 batch_size: int = 16):
        """
        Initialize the reranker with a cross-encoder model.

        Args:
            model_name: Name of the pretrained cross-encoder model to use
            device: Device to run inference on ('cpu' or 'cuda')
            batch_size: Batch size for inference
        """
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.model = None

        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(model_name, device=device)
            self.logger.info(f"Loaded cross-encoder model: {model_name}")
        except ImportError:
            self.logger.warning(
                "Could not import sentence-transformers. Please install with: pip install sentence-transformers")
            self.logger.warning("Falling back to simple BM25-style reranking")
        except Exception as e:
            self.logger.error(f"Error loading cross-encoder model: {str(e)}")
            self.logger.warning("Falling back to simple BM25-style reranking")

    def rerank(self, query: str, results: List[Dict[str, Any]],
               top_k: Optional[int] = None,
               threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Rerank the search results based on query-document relevance.

        Args:
            query: The original query string
            results: List of search results with at least 'id' and 'content' fields
            top_k: Number of top results to return (None for all)
            threshold: Minimum score threshold (None for no threshold)

        Returns:
            Reranked list of search results with scores
        """
        if not results:
            return []

        if self.model is None:
            # Fallback to simple term-matching reranking
            return self._fallback_rerank(query, results, top_k, threshold)

        # Prepare cross-encoder inputs
        query_doc_pairs = [(query, result.get("content", "")) for result in results]

        # Get cross-encoder scores
        try:
            scores = self.model.predict(query_doc_pairs)

            import numpy as np
            def softmax_normalize(scores, temperature=10.0):
                exp_scores = np.exp(np.array(scores) / temperature)
                return exp_scores / np.sum(exp_scores)

            scores = np.array(softmax_normalize(scores))

            # Add scores to results
            for i, result in enumerate(results):
                result["rerank_score"] = float(scores[i])

            # Sort by score in descending order
            reranked_results = sorted(results, key=lambda x: x.get("rerank_score", 0), reverse=True)

            # Apply threshold filter if specified
            if threshold is not None:
                reranked_results = [r for r in reranked_results if r.get("rerank_score", 0) >= threshold]

            # Apply top_k filter if specified
            if top_k is not None and top_k > 0:
                reranked_results = reranked_results[:top_k]

            return reranked_results

        except Exception as e:
            self.logger.error(f"Error during cross-encoder reranking: {str(e)}")
            return self._fallback_rerank(query, results, top_k, threshold)

    def _fallback_rerank(self, query: str, results: List[Dict[str, Any]],
                         top_k: Optional[int] = None,
                         threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Simple fallback reranking method based on term matching.

        Args:
            query: The original query string
            results: List of search results
            top_k: Number of top results to return
            threshold: Minimum score threshold

        Returns:
            Reranked list of search results
        """
        self.logger.info("Using fallback reranking method")

        # Simple term matching reranking
        query_terms = set(query.lower().split())

        for result in results:
            content = result.get("content", "")
            if not content:
                result["rerank_score"] = 0.0
                continue

            content_lower = content.lower()

            # Calculate term match score
            term_matches = sum(1 for term in query_terms if term in content_lower)
            term_match_ratio = term_matches / len(query_terms) if query_terms else 0

            # Calculate weighted score using both the original similarity and term matching
            original_score = result.get("score", 0.5)  # Default to 0.5 if no score

            # Weighted combination
            result["rerank_score"] = 0.7 * original_score + 0.3 * term_match_ratio

        # Sort by rerank_score
        reranked_results = sorted(results, key=lambda x: x.get("rerank_score", 0), reverse=True)

        # Apply top_k filter
        if top_k is not None and top_k > 0:
            reranked_results = reranked_results[:top_k]

        # Apply threshold filter
        if threshold is not None:
            reranked_results = [r for r in reranked_results if r.get("rerank_score", 0) >= threshold]

        return reranked_results


class DenseReranker:
    """
    A reranking model that uses a dense retrieval model to rerank search results.
    Can be used with models like SBERT, DPR, or other bi-encoders.
    """

    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "cpu"):
        """
        Initialize the dense reranker with a bi-encoder model.

        Args:
            model_name: Name of the pretrained sentence-transformer model to use
            device: Device to run inference on ('cpu' or 'cuda')
        """
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        self.device = device
        self.model = None

        try:
            from sentence_transformers import SentenceTransformer
            import torch

            self.model = SentenceTransformer(model_name, device=device)
            self.logger.info(f"Loaded SentenceTransformer model: {model_name}")
        except ImportError:
            self.logger.warning(
                "Could not import sentence-transformers. Please install with: pip install sentence-transformers")
            self.logger.warning("Falling back to original ranking order")
        except Exception as e:
            self.logger.error(f"Error loading SentenceTransformer model: {str(e)}")
            self.logger.warning("Falling back to original ranking order")

    def rerank(self, query: str, results: List[Dict[str, Any]],
               top_k: Optional[int] = None,
               threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Rerank the search results based on query-document similarity using dense embeddings.

        Args:
            query: The original query string
            results: List of search results with at least 'id' and 'content' fields
            top_k: Number of top results to return (None for all)
            threshold: Minimum score threshold (None for no threshold)

        Returns:
            Reranked list of search results with scores
        """
        if not results:
            return []

        if self.model is None:
            # Simply return original results if model is not available
            for i, result in enumerate(results):
                # Preserve original scores
                result["rerank_score"] = result.get("score", 1.0 - i * 0.01)

            # Apply top_k filter if specified
            if top_k is not None and top_k > 0:
                results = results[:top_k]

            return results

        try:
            # Encode query
            query_embedding = self.model.encode(query, convert_to_tensor=True)

            # Encode documents
            documents = [result.get("content", "") for result in results]
            doc_embeddings = self.model.encode(documents, convert_to_tensor=True)

            # Calculate cosine similarity
            import torch
            import torch.nn.functional as F

            # Normalize embeddings
            query_embedding = F.normalize(query_embedding, p=2, dim=0)
            doc_embeddings = F.normalize(doc_embeddings, p=2, dim=1)

            # Calculate similarities
            similarities = torch.matmul(doc_embeddings, query_embedding).cpu().numpy()

            # Add scores to results
            for i, result in enumerate(results):
                result["rerank_score"] = float(similarities[i])

            # Sort by score in descending order
            reranked_results = sorted(results, key=lambda x: x.get("rerank_score", 0), reverse=True)

            # Apply top_k filter if specified
            if top_k is not None and top_k > 0:
                reranked_results = reranked_results[:top_k]

            # Apply threshold filter if specified
            if threshold is not None:
                reranked_results = [r for r in reranked_results if r.get("rerank_score", 0) >= threshold]

            return reranked_results

        except Exception as e:
            self.logger.error(f"Error during dense reranking: {str(e)}")
            # Return original results on error
            return results


class RerankerFactory:
    """Factory class to create different types of rerankers."""

    @staticmethod
    def create_reranker(reranker_type: str, **kwargs) -> Union[CrossEncoderReranker, DenseReranker]:
        """
        Create a reranker based on the specified type.

        Args:
            reranker_type: Type of reranker to create ('cross-encoder' or 'dense')
            **kwargs: Additional arguments to pass to the reranker constructor

        Returns:
            An instance of the specified reranker
        """
        if reranker_type.lower() == "cross-encoder":
            return CrossEncoderReranker(**kwargs)
        elif reranker_type.lower() == "dense":
            return DenseReranker(**kwargs)
        else:
            raise ValueError(f"Unknown reranker type: {reranker_type}")