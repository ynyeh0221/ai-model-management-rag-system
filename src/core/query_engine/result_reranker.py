"""
SEARCH RESULT RERANKING SYSTEM
==============================

Overall System Architecture:
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RERANKING PIPELINE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────┐    ┌──────────────────┐    ┌─────────────────────────────────┐ │
│  │ Query + │───▶│  RerankerFactory │───▶│    Choose Reranker Type         │ │
│  │Results  │    │                  │    │                                 │ │
│  └─────────┘    └──────────────────┘    └─────────────────────────────────┘ │
│       │                                                │                    │
│       │         ┌───────────────────────┐              │                    │
│       │         │                       │              ▼                    │
│       │    ┌────┴───────┐         ┌─────┴─────┐  ┌──────────┐               │
│       │    │CrossEncoder│         │Dense      │  │ Choose   │               │
│       │    │Reranker    │         │Reranker   │  │ Method   │               │
│       │    └────┬───────┘         └─────┬─────┘  └──────────┘               │
│       │         │                       │                                   │
│       │         ▼                       ▼                                   │
│       │  ┌─────────────┐         ┌─────────────┐                            │
│       └─▶│Query-Doc    │         │Separate     │                            │
│          │Pair Scoring │         │Embeddings   │                            │
│          └─────────────┘         └─────────────┘                            │
│                 │                       │                                   │
│                 ▼                       ▼                                   │
│          ┌─────────────┐         ┌─────────────┐                            │
│          │Softmax      │         │Cosine       │                            │
│          │Normalization│         │Similarity   │                            │
│          └─────────────┘         └─────────────┘                            │
│                 │                       │                                   │
│                 └───────────┬───────────┘                                   │
│                             ▼                                               │
│                     ┌─────────────────┐                                     │
│                     │Sort by Score +  │                                     │
│                     │Apply Filters    │                                     │
│                     └─────────────────┘                                     │
│                             │                                               │
│                             ▼                                               │
│                     ┌─────────────────┐                                     │
│                     │Reranked Results │                                     │
│                     └─────────────────┘                                     │
└─────────────────────────────────────────────────────────────────────────────┘
"""
import logging
from typing import List, Dict, Any, Union, Optional


class CrossEncoderReranker:
    """
    A reranking model that uses a cross-encoder architecture to
    rerank search results based on query-document relevance.
    
    CROSS-ENCODER WORKFLOW:
    ┌──────────────────────────────────────────────────────────────────────────┐
    │                      CROSS-ENCODER RERANKING                             │
    ├──────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  Input: Query + Documents                                                │
    │  ┌─────────┐                                                             │
    │  │"python" │  ┌─────────────────────────────────────────────────────────┐│
    │  │         │  │Document 1: "Python programming tutorial..."             ││
    │  │         │  │Document 2: "Java programming concepts..."               ││
    │  │         │  │Document 3: "Python data structures guide..."            ││
    │  └─────────┘  └─────────────────────────────────────────────────────────┘│
    │       │                                │                                 │
    │       └────────────┬───────────────────┘                                 │
    │                    ▼                                                     │
    │  ┌──────────────────────────────────────────────────────────────────────┐│
    │  │               CREATE QUERY-DOCUMENT PAIRS                            ││
    │  │  [("python", "Python programming tutorial..."),                      ││
    │  │   ("python", "Java programming concepts..."),                        ││
    │  │   ("python", "Python data structures guide...")]                     ││
    │  └──────────────────────────────────────────────────────────────────────┘│
    │                                   │                                      │
    │                                   ▼                                      │
    │  ┌──────────────────────────────────────────────────────────────────────┐│
    │  │                    CROSS-ENCODER MODEL                               ││
    │  │                                                                      ││
    │  │    [CLS] query [SEP] document [SEP]                                  ││
    │  │           │                                                          ││
    │  │           ▼                                                          ││
    │  │    ┌─────────────┐                                                   ││
    │  │    │Transformer  │                                                   ││
    │  │    │Layers       │                                                   ││
    │  │    └─────────────┘                                                   ││
    │  │           │                                                          ││
    │  │           ▼                                                          ││
    │  │    ┌─────────────┐                                                   ││
    │  │    │Relevance    │                                                   ││
    │  │    │Score        │                                                   ││
    │  │    └─────────────┘                                                   ││
    │  └──────────────────────────────────────────────────────────────────────┘│
    │                                   │                                      │
    │                                   ▼                                      │
    │  ┌──────────────────────────────────────────────────────────────────────┐│
    │  │                    RAW SCORES                                        ││
    │  │    Document 1: 0.85                                                  ││
    │  │    Document 2: 0.32                                                  ││
    │  │    Document 3: 0.91                                                  ││
    │  └──────────────────────────────────────────────────────────────────────┘│
    │                                   │                                      │
    │                                   ▼                                      │
    │  ┌──────────────────────────────────────────────────────────────────────┐│
    │  │                 SOFTMAX NORMALIZATION                                ││
    │  │    exp(score/temperature) / sum(exp(all_scores/temperature))         ││
    │  │                                                                      ││
    │  │    Document 1: 0.42                                                  ││
    │  │    Document 2: 0.18                                                  ││
    │  │    Document 3: 0.40                                                  ││
    │  └──────────────────────────────────────────────────────────────────────┘│
    │                                   │                                      │
    │                                   ▼                                      │
    │  ┌──────────────────────────────────────────────────────────────────────┐│
    │  │                    FINAL RANKING                                     ││
    │  │    1. Document 1: "Python programming tutorial..." (0.42)            ││
    │  │    2. Document 3: "Python data structures guide..." (0.40)           ││
    │  │    3. Document 2: "Java programming concepts..." (0.18)              ││
    │  └──────────────────────────────────────────────────────────────────────┘│
    │                                                                          │
    └──────────────────────────────────────────────────────────────────────────┘
    
    FALLBACK MECHANISM (When model fails):
    ┌──────────────────────────────────────────────────────────────────────────┐
    │                        FALLBACK RERANKING                                │
    ├──────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  Query: "python tutorial"                                                │
    │  ├─ Query Terms: ["python", "tutorial"]                                  │
    │  │                                                                       │
    │  ▼                                                                       │
    │  ┌──────────────────────────────────────────────────────────────────────┐│
    │  │                    TERM MATCHING ANALYSIS                            ││
    │  │                                                                      ││
    │  │  Doc1: "Python programming tutorial guide"                           ││
    │  │    ├─ Contains: "python" ✓, "tutorial" ✓                             ││
    │  │    ├─ Term Match Ratio: 2/2 = 1.0                                    ││
    │  │    └─ Original Score: 0.7                                            ││
    │  │                                                                      ││
    │  │  Doc2: "Java programming concepts"                                   ││
    │  │    ├─ Contains: "python" ✗, "tutorial" ✗                             ││
    │  │    ├─ Term Match Ratio: 0/2 = 0.0                                    ││
    │  │    └─ Original Score: 0.6                                            ││
    │  └──────────────────────────────────────────────────────────────────────┘│
    │                                   │                                      │
    │                                   ▼                                      │
    │  ┌──────────────────────────────────────────────────────────────────────┐│
    │  │                   WEIGHTED COMBINATION                               ││
    │  │    Final Score = 0.7 × Original + 0.3 × Term Match                   ││
    │  │                                                                      ││
    │  │    Doc1: 0.7 × 0.7 + 0.3 × 1.0 = 0.79                                ││
    │  │    Doc2: 0.7 × 0.6 + 0.3 × 0.0 = 0.42                                ││
    │  └──────────────────────────────────────────────────────────────────────┘│
    └──────────────────────────────────────────────────────────────────────────┘
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2", device: str = "cpu",
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
    
    DENSE RERANKING WORKFLOW:
    ┌──────────────────────────────────────────────────────────────────────────┐
    │                        DENSE RERANKING PIPELINE                          │
    ├──────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  Input: Query + Documents                                                │
    │  ┌─────────────┐    ┌─────────────────────────────────────────────────┐  │
    │  │"machine     │    │Doc1: "Machine learning with Python..."          │  │
    │  │ learning"   │    │Doc2: "Deep learning fundamentals..."            │  │
    │  │             │    │Doc3: "Natural language processing..."           │  │
    │  └─────────────┘    └─────────────────────────────────────────────────┘  │
    │        │                                    │                            │
    │        ▼                                    ▼                            │
    │  ┌─────────────┐                    ┌─────────────┐                      │
    │  │   ENCODER   │                    │   ENCODER   │                      │
    │  │  (Bi-LSTM/  │                    │  (Bi-LSTM/  │                      │
    │  │Transformer) │                    │Transformer) │                      │
    │  └─────────────┘                    └─────────────┘                      │
    │        │                                    │                            │
    │        ▼                                    ▼                            │
    │  ┌─────────────┐                    ┌───────────────┐                    │
    │  │   Query     │                    │  Document     │                    │
    │  │ Embedding   │                    │ Embeddings    │                    │
    │  │   Vector    │                    │   Vectors     │                    │
    │  │[0.2,0.8,...]│                    │[[0.1,0.7,...] │                    │
    │  └─────────────┘                    │ [0.3,0.5,...] │                    │
    │        │                            │ [0.6,0.2,...]]│                    │
    │        │                            └───────────────┘                    │
    │        │                                    │                            │
    │        └─────────────────┬──────────────────┘                            │
    │                          ▼                                               │
    │  ┌──────────────────────────────────────────────────────────────────────┐│
    │  │                    COSINE SIMILARITY                                 ││
    │  │                                                                      ││
    │  │    similarity = (query · doc) / (||query|| × ||doc||)                ││
    │  │                                                                      ││
    │  │    Query: [0.2, 0.8, 0.1]                                            ││
    │  │    Doc1:  [0.1, 0.7, 0.2]  →  sim = 0.89                             ││
    │  │    Doc2:  [0.3, 0.5, 0.1]  →  sim = 0.75                             ││
    │  │    Doc3:  [0.6, 0.2, 0.3]  →  sim = 0.61                             ││
    │  └──────────────────────────────────────────────────────────────────────┘│
    │                                   │                                      │
    │                                   ▼                                      │
    │  ┌──────────────────────────────────────────────────────────────────────┐│
    │  │                      RANKING                                         ││
    │  │    1. Doc1: "Machine learning with Python..." (sim: 0.89)            ││
    │  │    2. Doc2: "Deep learning fundamentals..." (sim: 0.75)              ││
    │  │    3. Doc3: "Natural language processing..." (sim: 0.61)             ││
    │  └──────────────────────────────────────────────────────────────────────┘│
    │                                                                          │
    └──────────────────────────────────────────────────────────────────────────┘
    
    EMBEDDING VISUALIZATION:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                       VECTOR SPACE REPRESENTATION                       │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │     High-dimensional space (e.g., 768 dimensions)                       │
    │                                                                         │
    │         Doc2 ●                                                          │
    │                \                                                        │
    │                 \                                                       │
    │                  \                                                      │
    │                   ● Query                                               │
    │                  /                                                      │
    │                 /                                                       │
    │                /                                                        │
    │         Doc1 ●                                                          │
    │                                                                         │
    │                                                                         │
    │                        ● Doc3                                           │
    │                                                                         │
    │   Closer vectors = Higher similarity = Better relevance                 │
    └─────────────────────────────────────────────────────────────────────────┘
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
    """
    Factory class to create different types of rerankers.
    
    FACTORY PATTERN WORKFLOW:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                          RERANKER FACTORY                               │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │  Input: reranker_type + configuration                                   │
    │  ┌─────────────────┐                                                    │
    │  │"cross-encoder"  │                                                    │
    │  │  + kwargs       │                                                    │
    │  └─────────────────┘                                                    │
    │           │                                                             │
    │           ▼                                                             │
    │  ┌─────────────────┐                                                    │
    │  │   Type Check    │                                                    │
    │  │                 │                                                    │
    │  │ if type == "cross-encoder":                                          │
    │  │     return CrossEncoderReranker(**kwargs)                            │
    │  │ elif type == "dense":                                                │
    │  │     return DenseReranker(**kwargs)                                   │
    │  │ else:                                                                │
    │  │     raise ValueError(...)                                            │
    │  └─────────────────┘                                                    │
    │           │                                                             │
    │           ▼                                                             │
    │  ┌─────────────────┐                                                    │
    │  │   Instantiate   │                                                    │
    │  │   Appropriate   │                                                    │
    │  │   Reranker      │                                                    │
    │  └─────────────────┘                                                    │
    │           │                                                             │
    │           ▼                                                             │
    │  ┌─────────────────┐                                                    │
    │  │Return Configured│                                                    │
    │  │   Reranker      │                                                    │
    │  │   Instance      │                                                    │
    │  └─────────────────┘                                                    │
    │                                                                         │
    │  Usage Example:                                                         │
    │  reranker = RerankerFactory.create_reranker(                            │
    │      "cross-encoder",                                                   │
    │      model_name="ms-marco-MiniLM-L-12-v2",                              │
    │      device="cuda"                                                      │
    │  )                                                                      │
    └─────────────────────────────────────────────────────────────────────────┘
    """

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
