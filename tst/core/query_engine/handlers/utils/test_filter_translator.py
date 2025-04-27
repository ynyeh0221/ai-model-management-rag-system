import unittest
from unittest.mock import patch, MagicMock

from src.core.query_engine.handlers.utils.filter_translator import FilterTranslator


class TestFilterTranslator(unittest.TestCase):

    def setUp(self):
        self.translator = FilterTranslator()

    def test_initialization(self):
        """Test that the FilterTranslator initializes correctly."""
        self.assertIsInstance(self.translator, FilterTranslator)
        self.assertIsNotNone(self.translator.logger)

    def test_none_input(self):
        """Test handling of None input."""
        result = self.translator.translate_to_chroma(None)
        self.assertEqual(result, {})

    @patch('src.core.query_engine.handlers.utils.filter_translator.logging.getLogger')
    def test_list_input(self, mock_get_logger):
        """Test handling of list input."""
        # Set up mock logger
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        # Create a new translator to use the mocked logger
        translator = FilterTranslator()

        # Call method with list input
        result = translator.translate_to_chroma([])

        # Assert warning was logged
        mock_logger.warning.assert_called_once_with(
            "Filters received as list instead of dictionary. Converting to empty dict."
        )

        # Assert empty dict is returned
        self.assertEqual(result, {})

    def test_already_structured_filter(self):
        """Test handling of already properly structured filter."""
        filters = {"$and": [{"field1": {"$eq": "value1"}}, {"field2": {"$eq": "value2"}}]}
        result = self.translator.translate_to_chroma(filters)
        self.assertEqual(result, filters)

    def test_multiple_top_level_conditions(self):
        """Test handling of multiple top-level filter conditions."""
        filters = {
            "field1": "value1",
            "field2": "value2"
        }
        expected = {
            "$and": [
                {"field1": {"$eq": "value1"}},
                {"field2": {"$eq": "value2"}}
            ]
        }
        result = self.translator.translate_to_chroma(filters)
        self.assertEqual(result, expected)

    def test_multiple_top_level_with_nested_operators(self):
        """Test handling of multiple top-level conditions with nested operators."""
        filters = {
            "field1": {"$eq": "value1"},
            "field2": "value2"
        }
        expected = {
            "$and": [
                {"field1": {"$eq": "value1"}},
                {"field2": {"$eq": "value2"}}
            ]
        }
        result = self.translator.translate_to_chroma(filters)
        self.assertEqual(result, expected)

    def test_multiple_top_level_with_list_values(self):
        """Test handling of multiple top-level conditions with list values."""
        filters = {
            "field1": ["value1", "value2"],
            "field2": "value3"
        }
        expected = {
            "$and": [
                {"field1": {"$in": ["value1", "value2"]}},
                {"field2": {"$eq": "value3"}}
            ]
        }
        result = self.translator.translate_to_chroma(filters)
        self.assertEqual(result, expected)

    def test_single_condition_with_nested_operator(self):
        """Test handling of a single condition with a nested operator."""
        filters = {"field1": {"$gt": 10}}
        result = self.translator.translate_to_chroma(filters)
        self.assertEqual(result, filters)

    def test_single_condition_with_list_value(self):
        """Test handling of a single condition with a list value."""
        filters = {"field1": ["value1", "value2"]}
        expected = {"field1": {"$in": ["value1", "value2"]}}
        result = self.translator.translate_to_chroma(filters)
        self.assertEqual(result, expected)

    def test_single_condition_with_simple_value(self):
        """Test handling of a single condition with a simple value."""
        filters = {"field1": "value1"}
        expected = {"field1": {"$eq": "value1"}}
        result = self.translator.translate_to_chroma(filters)
        self.assertEqual(result, expected)

    def test_empty_dict(self):
        """Test handling of an empty dictionary."""
        filters = {}
        result = self.translator.translate_to_chroma(filters)
        self.assertEqual(result, {})


if __name__ == '__main__':
    unittest.main()