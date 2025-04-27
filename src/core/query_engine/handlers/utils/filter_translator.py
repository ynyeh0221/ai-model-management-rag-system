import logging
from typing import Dict, Any


class FilterTranslator:
    """
    Utility class for translating filter dictionaries to Chroma format.

    This class provides methods to convert various filter formats to the specific
    format expected by Chroma, which is a vector database designed for embeddings
    and similarity search. It handles different types of input filters and
    converts them to a consistent structure that Chroma can process.

    Attributes:
        logger: A logger instance for logging warnings and errors.

    Examples:
        >>> translator = FilterTranslator()
        >>> filters = {"field": "value"}
        >>> chroma_filters = translator.translate_to_chroma(filters)
        >>> print(chroma_filters)
        {'field': {'$eq': 'value'}}

        >>> filters = {"field1": "value1", "field2": "value2"}
        >>> chroma_filters = translator.translate_to_chroma(filters)
        >>> print(chroma_filters)
        {'$and': [{'field1': {'$eq': 'value1'}}, {'field2': {'$eq': 'value2'}}]}
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def translate_to_chroma(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Translate query filters to Chroma's filter format.

        Args:
            filters: Filters in the query format

        Returns:
            Filters in Chroma's format
        """
        # Add defensive checks at the beginning
        if filters is None:
            return {}

        if isinstance(filters, list):
            self.logger.warning("Filters received as list instead of dictionary. Converting to empty dict.")
            return {}

        # Handle case where we already have a properly structured filter (with $and, $or, etc.)
        if len(filters) == 1 and list(filters.keys())[0].startswith('$'):
            return filters

        # Handle multiple top-level filter conditions
        # ChromaDB expects them to be wrapped with an operator like $and
        if len(filters) > 1:
            conditions = []
            for key, value in filters.items():
                # Skip nested operators for separate handling
                if isinstance(value, dict) and any(op.startswith('$') for op in value.keys()):
                    conditions.append({key: value})
                # Handle list values
                elif isinstance(value, list):
                    conditions.append({key: {"$in": value}})
                # Handle simple values
                else:
                    conditions.append({key: {"$eq": value}})

            return {"$and": conditions}

        # Handle single filter condition
        if len(filters) == 1:
            key, value = next(iter(filters.items()))

            # Handle nested operator filters
            if isinstance(value, dict) and any(op.startswith('$') for op in value.keys()):
                return filters

            # Handle list values
            if isinstance(value, list):
                return {key: {"$in": value}}

            # Handle simple values
            return {key: {"$eq": value}}

        return {}