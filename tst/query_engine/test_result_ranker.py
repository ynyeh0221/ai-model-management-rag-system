import unittest
from datetime import datetime
from src.query_engine.result_ranker import ResultRanker


class TestResultRanker(unittest.TestCase):

    def setUp(self):
        # Create a sample list of result dicts
        self.results = [
            {
                "id": "r1",
                "score": 0.90,
                "metadata": {
                    "creation_date": "2023-01-15T10:30:00Z",
                    "model_id": "transformer-v1",
                    "version": "1.0.0",
                    "model_family": "transformer",
                    "framework": {"name": "pytorch"}
                }
            },
            {
                "id": "r2",
                "score": 0.85,
                "metadata": {
                    "creation_date": "2023-05-20T14:45:00Z",
                    "model_id": "transformer-v2",
                    "version": "2.0.0",
                    "model_family": "transformer",
                    "framework": {"name": "pytorch"}
                }
            },
            {
                "id": "r3",
                "score": 0.80,
                "metadata": {
                    "creation_date": "2023-03-10T09:15:00Z",
                    "model_id": "cnn-v1",
                    "version": "1.0.0",
                    "model_family": "cnn",
                    "framework": {"name": "tensorflow"}
                }
            }
        ]
        self.ranker = ResultRanker()

    def test_similarity_ranking(self):
        # In similarity ranking, higher score should come first.
        ranked = self.ranker.similarity_ranking(self.results)
        # Expected order: r1 (0.90), r2 (0.85), r3 (0.80)
        expected_order = ["r1", "r2", "r3"]
        self.assertEqual([r["id"] for r in ranked], expected_order)

    def test_similarity_ranking_missing_score(self):
        # Remove score from one result.
        results = [
            {"id": "a", "metadata": {}},
            {"id": "b", "score": 0.5, "metadata": {}},
            {"id": "c", "score": 0.8, "metadata": {}}
        ]
        ranked = self.ranker.similarity_ranking(results)
        # Result "c" (0.8) should be first, then "b" (0.5),
        # then "a" (score defaulted to 0.0).
        expected_order = ["c", "b", "a"]
        self.assertEqual([r["id"] for r in ranked], expected_order)

    def test_recency_ranking(self):
        # Recency ranking sorts by "metadata.creation_date". Newest first.
        ranked = self.ranker.recency_ranking(self.results)
        # Our dates: r2 (May), r3 (March), r1 (January)
        expected_order = ["r2", "r3", "r1"]
        self.assertEqual([r["id"] for r in ranked], expected_order)

    def test_recency_ranking_invalid_date(self):
        # A result with an invalid or missing date should default to the epoch start.
        results = [
            {"id": "a", "score": 0.7, "metadata": {"creation_date": "invalid-date"}},
            {"id": "b", "score": 0.8, "metadata": {"creation_date": "2023-04-01T12:00:00"}}
        ]
        ranked = self.ranker.recency_ranking(results)
        # "b" should come first, "a" falls back to epoch.
        expected_order = ["b", "a"]
        self.assertEqual([r["id"] for r in ranked], expected_order)

    def test_version_aware_ranking(self):
        # Version-aware ranking groups by base model and sorts by version (higher is better).
        # Our sample: Two transformer results (r1: v1.0.0, r2: v2.0.0) and one cnn (r3).
        # r2 (transformer-v2) should be ranked higher than r1.
        ranked = self.ranker.version_aware_ranking(self.results)
        # The implementation interleaves groups, with highest version first in each group
        # First from transformer group with highest version: r2
        # First from cnn group: r3
        # Remaining transformer: r1
        expected_order = ["r2", "r3", "r1"]
        self.assertEqual([r["id"] for r in ranked], expected_order)

    def test_hybrid_ranking(self):
        # Hybrid ranking uses a weighted combination of strategies.
        # With default weights (similarity: 0.6, recency: 0.3, version: 0.1)
        # This test needs adjustment based on actual implementation behavior
        weights = {"similarity": 0.6, "recency": 0.3, "version": 0.1}
        ranked = self.ranker.hybrid_ranking(self.results, weights=weights)

        # Based on the implementation, r2 should be first because it ranks well in
        # recency and version, followed by r1 (high similarity) and then r3
        expected_order = ["r2", "r1", "r3"]
        self.assertEqual([r["id"] for r in ranked], expected_order)

    def test_rank_results_with_custom_ranker(self):
        # Test that providing a custom ranker overrides the built-in strategies.
        def custom_ranker(results, **kwargs):
            return list(reversed(results))

        ranked = self.ranker.rank_results(self.results, custom_ranker=custom_ranker)
        expected_order = ["r3", "r2", "r1"]
        self.assertEqual([r["id"] for r in ranked], expected_order)

    def test_rank_results_empty_input(self):
        # When passed an empty list, it should return an empty list.
        ranked = self.ranker.rank_results([])
        self.assertEqual(ranked, [])

    def test_diversity_reranking(self):
        # Test diversity re-ranking on a similarity-sorted result list.
        # Our sample: r1 and r2 share the same model_family and framework,
        # while r3 is from a different family and framework.
        similarity_sorted = self.ranker.similarity_ranking(self.results)
        # similarity_sorted order is: [r1, r2, r3] (by score)

        # Set diversity_weight explicitly for predictable results
        diversity_reranked = self.ranker.diversity_reranking(
            similarity_sorted,
            diversity_fields=["metadata.model_family", "metadata.framework.name"],
            diversity_weight=0.7  # Higher weight on diversity
        )

        # First item stays the same (r1), then r3 (different family) comes before r2
        expected_order = ["r1", "r3", "r2"]
        self.assertEqual([r["id"] for r in diversity_reranked], expected_order)

    def test_rank_results_unknown_strategy(self):
        # When an unknown strategy is given, rank_results should fall back to similarity ranking.
        ranked = self.ranker.rank_results(self.results, strategy="nonexistent")
        # Similarity ranking order from our sample: r1, r2, r3.
        expected_order = ["r1", "r2", "r3"]
        self.assertEqual([r["id"] for r in ranked], expected_order)


if __name__ == '__main__':
    unittest.main()