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
        """
        # 1) Early defensive check
        if filters is None:
            return {}
        if isinstance(filters, list):
            self.logger.warning("Filters received as list instead of dictionary. Converting to empty dict.")
            return {}

        # 2) If already wrapped (e.g., {"$and": [...]}) just return
        if self._is_already_wrapped(filters):
            return filters

        # 3) Multiple top-level keys → combine with $and
        if len(filters) > 1:
            return self._build_combined_filters(filters)

        # 4) Exactly one top-level key → translate it
        if len(filters) == 1:
            return self._translate_single_filter(filters)

        return {}

    @staticmethod
    def _is_already_wrapped(filters: Dict[str, Any]) -> bool:
        """
        Return True if `filters` is a dict with exactly one key starting with '$'.
        """
        if len(filters) == 1:
            key = next(iter(filters))
            return key.startswith("$")
        return False

    def _build_combined_filters(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a dict with multiple top-level conditions into {"$and": [ ... ]}.
        Each condition is built via `_translate_single_condition`.
        """
        conditions = []
        for key, value in filters.items():
            conditions.append(self._translate_single_condition(key, value))
        return {"$and": conditions}

    def _translate_single_filter(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Translate a single top-level filter. If it's already using
        a nested operator (e.g. {"field": {"$in": [...]}}), return as-is.
        Otherwise, wrap with $in or $eq.
        """
        key, value = next(iter(filters.items()))
        # If nested operator dict, just return original dict
        if isinstance(value, dict) and any(op.startswith("$") for op in value):
            return filters
        # Otherwise, delegate to the same helper used in the combined case
        return self._translate_single_condition(key, value)

    def _translate_single_condition(self, key: str, value: Any) -> Dict[str, Any]:
        """
        For one key/value pair:
          - If value is a dict with a $-operator inside, keep it as {key: value}.
          - If value is a list, translate to {key: {"$in": value}}.
          - Otherwise, translate to {key: {"$eq": value}}.
        """
        if isinstance(value, dict) and any(op.startswith("$") for op in value):
            return {key: value}
        if isinstance(value, list):
            return {key: {"$in": value}}
        return {key: {"$eq": value}}
