import unittest
from datetime import datetime
from src.query_engine.query_parser import QueryParser, QueryIntent


class TestQueryParser(unittest.TestCase):

    def setUp(self):
        # Initialize QueryParser with use_langchain disabled for predictable behavior.
        self.parser = QueryParser(nlp_model="en_core_web_sm", use_langchain=False)

    def test_preprocess_query(self):
        query = "   The quick, brown foxes jumped over the lazy dogs!   "
        processed = self.parser.preprocess_query(query)
        # Check that the result is lowercased and stopwords/punctuation removed.
        self.assertNotIn("the", processed)
        self.assertNotIn(",", processed)
        self.assertNotIn("!", processed)
        # Check that some words are preserved (lemmatized form)
        self.assertIn("quick", processed)
        self.assertIn("fox", processed)
        self.assertIn("jump", processed)
        self.assertIn("lazy", processed)

    def test_classify_intent_retrieval(self):
        # Query that should default to retrieval because it asks for information about a model.
        query = "What is transformer-v1 and how does it work?"
        intent = self.parser.classify_intent(query)
        # Rule-based check: presence of "what is" and mention of model names should lead to retrieval.
        self.assertEqual(intent, QueryIntent.RETRIEVAL)

    def test_classify_intent_comparison(self):
        # Query that indicates a comparison.
        query = "Compare transformer-v1 and transformer-v2 for performance differences."
        intent = self.parser.classify_intent(query)
        self.assertEqual(intent, QueryIntent.COMPARISON)

    def test_classify_intent_notebook(self):
        # Query to generate a notebook.
        query = "Generate a notebook to analyze model performance."
        intent = self.parser.classify_intent(query)
        self.assertEqual(intent, QueryIntent.NOTEBOOK)

    def test_classify_intent_image_search(self):
        # Query to search for images.
        query = "Show me images of generated art."
        intent = self.parser.classify_intent(query)
        self.assertEqual(intent, QueryIntent.IMAGE_SEARCH)

    def test_classify_intent_metadata(self):
        # Query to fetch metadata details.
        query = "What metadata fields does transformer-v1 have?"
        intent = self.parser.classify_intent(query)
        self.assertEqual(intent, QueryIntent.METADATA)

    def test_extract_parameters_limit_and_sort(self):
        query = ("Show model_id: transformer-v1 details with accuracy metric, "
                 "limit 10, sort by score descending")
        params = self.parser.extract_parameters(query, intent=QueryIntent.RETRIEVAL)
        # Check that explicit model id extraction works.
        self.assertIn("model_ids", params)
        self.assertIn("transformer-v1", [mid.lower() for mid in params["model_ids"]])
        # Check metric extraction.
        self.assertIn("metrics", params)
        self.assertIn("accuracy", params["metrics"])
        # Check limit extraction.
        self.assertIn("limit", params)
        self.assertEqual(params["limit"], 10)
        # Check sort extraction.
        self.assertIn("sort_by", params)
        self.assertEqual(params["sort_by"]["field"], "score")
        self.assertEqual(params["sort_by"]["order"], "descending")

    def test_extract_comparison_parameters(self):
        # Directly test the internal comparison parameter extraction.
        query = "Comparing by accuracy, speed and latency"
        comparison_params = self.parser._extract_comparison_parameters(query)
        # Expect the dimensions to be extracted and split.
        self.assertIn("comparison_dimensions", comparison_params)
        dimensions = comparison_params["comparison_dimensions"]
        self.assertIsInstance(dimensions, list)
        self.assertIn("accuracy", dimensions)
        self.assertIn("speed", dimensions)
        self.assertIn("latency", dimensions)
        # If query contains a visualization keyword, it should extract that too.
        query2 = "Compare performance and also show charts"
        comparison_params2 = self.parser._extract_comparison_parameters(query2)
        self.assertIn("visualize", comparison_params2)
        self.assertTrue(comparison_params2["visualize"])

    def test_extract_notebook_parameters(self):
        # Test extraction for notebook generation queries.
        query = "Generate a notebook to analyze performance, dataset: sample_data, using high memory resources"
        nb_params = self.parser._extract_notebook_parameters(query)
        # Expect analysis_types from "analyze performance"
        self.assertIn("analysis_types", nb_params)
        self.assertIn("performance", nb_params["analysis_types"])
        # Check dataset extraction.
        self.assertIn("dataset", nb_params)
        self.assertEqual(nb_params["dataset"], "sample_data")
        # Check resources extraction.
        self.assertIn("resources", nb_params)
        self.assertEqual(nb_params["resources"], "high memory")

    def test_extract_image_parameters(self):
        # Test extraction for image search queries.
        query = "Find image with prompt: sunset, style: vintage, resolution: 1920x1080"
        img_params = self.parser._extract_image_parameters(query)
        # Expect prompt terms extraction.
        self.assertIn("prompt_terms", img_params)
        self.assertEqual(img_params["prompt_terms"], "sunset")
        # Expect style tags.
        self.assertIn("style_tags", img_params)
        self.assertIn("vintage", img_params["style_tags"])
        # Expect resolution extraction.
        self.assertIn("resolution", img_params)
        self.assertEqual(img_params["resolution"]["width"], 1920)
        self.assertEqual(img_params["resolution"]["height"], 1080)

    def test_parse_query_complete(self):
        # Test the complete parse_query method.
        query = ("Compare model_id: transformer-v1 and model_id: transformer-v2 "
                 "for accuracy and speed, limit 5, sort by score descending")
        parsed = self.parser.parse_query(query)
        # Check returned dictionary has required keys.
        self.assertIn("intent", parsed)
        self.assertIn("parameters", parsed)
        self.assertIn("processed_query", parsed)
        # For this query we expect a comparison intent.
        self.assertEqual(parsed["intent"], QueryIntent.COMPARISON)
        # Check that parameters include model_ids from the query.
        self.assertIn("model_ids", parsed["parameters"])
        # There should be at least one metric ("accuracy" or "speed")
        self.assertTrue("metrics" in parsed["parameters"] and len(parsed["parameters"]["metrics"]) > 0)

    def test_nlp_based_intent_classification_fallback(self):
        # Test _nlp_based_intent_classification as a fallback.
        # Use a query that likely does not match any rule-based patterns.
        query = "I wonder about the weather today."
        intent = self.parser._nlp_based_intent_classification(query)
        # Since the query does not contain model-specific clues, expect default to UNKNOWN.
        self.assertEqual(intent, QueryIntent.UNKNOWN)

    def test_extract_model_mentions(self):
        # Test explicit extraction of model mentions.
        query = "Find details for model: GPT-3 and model_id: bert_base"
        model_ids = self.parser._extract_model_mentions(query)
        # Expected to find both explicit model mentions.
        # The regex lowercases the captured group.
        self.assertIn("gpt-3", [mid.lower() for mid in model_ids])
        self.assertIn("bert_base", [mid.lower() for mid in model_ids])

if __name__ == '__main__':
    unittest.main()
