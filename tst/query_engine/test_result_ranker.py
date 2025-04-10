import unittest
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
            {"id": "b", "score": 0.8, "metadata": {"creation_date": "2023-04-01T12:00:00Z"}}
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
        # Since the ranking interleaves groups, expect:
        # First from transformer group: r2, then from cnn: r3, then the remaining transformer: r1.
        expected_order = ["r2", "r3", "r1"]
        self.assertEqual([r["id"] for r in ranked], expected_order)

    def test_hybrid_ranking(self):
        # Hybrid ranking uses a weighted combination of similarity, recency, and version.
        # We can compute approximate combined scores:
        # For our sample:
        #   Similarity ranking order: r1 (pos 0) => normalized score 1,
        #                             r2 (pos 1) => 1 - 1/2 = 0.5,
        #                             r3 (pos 2) => 0.
        #   Recency ranking order: r2 (pos 0) => 1,
        #                          r3 (pos 1) => 0.5,
        #                          r1 (pos 2) => 0.
        #   Version ranking order: r2 (pos 0) => 1,
        #                          r3 (pos 0 for its own group) => 1,
        #                          r1 (pos 1 in transformer group) => 0.
        # The default weights are: similarity: 0.6, recency: 0.3, version: 0.1.
        # So combined scores: r1: 1*0.6 + 0*0.3 + 0*0.1 = 0.6,
        #                      r2: 0.5*0.6 + 1*0.3 + 1*0.1 = 0.7,
        #                      r3: 0*0.6 + 0.5*0.3 + 1*0.1 = 0.25.
        # Expect hybrid order: r2, then r1, then r3.
        ranked = self.ranker.hybrid_ranking(self.results)
        expected_order = ["r2", "r1", "r3"]
        self.assertEqual([r["id"] for r in ranked], expected_order)

    def test_rank_results_with_custom_ranker(self):
        # Test that providing a custom ranker overrides the built-in strategies.
        # For example, a custom ranker that simply reverses the list.
        custom_ranker = lambda results, **kwargs: list(reversed(results))
        ranked = self.ranker.rank_results(self.results, custom_ranker=custom_ranker)
        expected_order = [r["id"] for r in reversed(self.results)]
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
        diversity_reranked = self.ranker.diversity_reranking(similarity_sorted)
        # Since diversity_reranking always keeps the top result (r1),
        # and then should prefer a candidate with a different 'metadata.model_family'
        # among the remaining results, we expect r3 to be placed before r2.
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
