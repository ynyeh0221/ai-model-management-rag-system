import ast
import re


class DatasetClassVisitor(ast.NodeVisitor):
    """
    AST visitor that identifies custom dataset classes through inheritance patterns.

    This visitor specializes in finding dataset implementations by analyzing class
    definitions, inheritance hierarchies, and docstrings. It uses sophisticated
    filtering to avoid false positives while detecting project-specific dataset classes.

    Key capabilities:
      - Recognizes classes that inherit from Dataset or related base classes
      - Analyzes class names to extract dataset identifiers
      - Examines docstrings for dataset references
      - Applies filtering to avoid common words being detected as datasets
      - Maps relationships between custom dataset implementations

    Attributes:
        detected_datasets (Set[str]): Set of dataset names detected from class analysis
        dataset_classes (Set[str]): Set of class names identified as dataset implementations
        COMMON_WORDS (Set[str]): Set of common words that should not be considered dataset names
    """

    # Common words that should not be considered dataset names
    COMMON_WORDS = {
        # Common English words
        'that', 'each', 'every', 'returns', 'sample', 'with', 'from', 'this',
        'these', 'those', 'which', 'what', 'where', 'when', 'who', 'how',
        'will', 'would', 'should', 'could', 'may', 'might', 'must', 'can',
        'other', 'another', 'some', 'any', 'all', 'both', 'either', 'neither',
        'none', 'much', 'many', 'more', 'most', 'few', 'less', 'least',
        'several', 'various', 'different', 'similar', 'same', 'such',

        # Common programming terms
        'class', 'method', 'function', 'variable', 'object', 'instance',
        'value', 'type', 'parameter', 'argument', 'return', 'result',
        'input', 'output', 'wrapper', 'base', 'super', 'sub', 'parent',
        'child', 'file', 'directory', 'path', 'name', 'label', 'source',
        'target', 'model', 'array', 'list', 'dict', 'set', 'tuple',
        'string', 'integer', 'float', 'boolean', 'true', 'false',

        # Dataset-related terms that aren't dataset names
        'dataset', 'data', 'loader', 'batch', 'transform', 'augmentation',
        'train', 'test', 'validation', 'eval', 'image', 'label', 'target',
        'feature', 'class', 'category', 'split', 'subset', 'sampling'
    }

    def __init__(self):
        super().__init__()
        self.detected_datasets = set()
        self.dataset_classes = set()

    def visit_ClassDef(self, node: ast.ClassDef):
        class_name = node.name

        # If any base indicates a dataset, record it and do follow‐up steps
        if self._inherits_dataset_class(node.bases):
            self.dataset_classes.add(class_name)

            # Extract a normalized dataset name from the class name
            dataset_name = self._extract_dataset_name_from_class(class_name)
            if dataset_name and not self._is_common_word(dataset_name):
                self.detected_datasets.add(dataset_name.upper())

            # If there’s a docstring, look for dataset patterns there
            doc = ast.get_docstring(node)
            if doc:
                self._extract_dataset_from_docstring(doc)

        # Continue traversing into child nodes
        self.generic_visit(node)

    def _inherits_dataset_class(self, bases: list[ast.expr]) -> bool:
        """
        Return True if any base class name/attribute contains
        'Dataset', 'DataModule', or ends with 'Data'.
        """
        for base in bases:
            base_name = None
            if isinstance(base, ast.Name):
                base_name = base.id
            elif isinstance(base, ast.Attribute):
                base_name = base.attr

            if base_name is not None and (
                "Dataset" in base_name
                or "DataModule" in base_name
                or base_name.endswith("Data")
            ):
                return True

        return False

    def _is_common_word(self, word: str) -> bool:
        """Check if a word is too common to be a dataset name."""
        word_lower = word.lower()
        return (word_lower in self.COMMON_WORDS or
                len(word_lower) <= 3 or  # Too short to be meaningful
                word_lower.isdigit())  # Just a number

    def _extract_dataset_from_docstring(self, docstring: str) -> None:
        """Extract dataset names from docstring using more targeted patterns."""
        # Common patterns that might indicate a dataset name
        dataset_patterns = [
            r'(?:based on|uses|from) the (\w+) dataset',
            r'(?:wrapper for|based on) (?:the )?(\w+)',
            r'dataset (?:for|of) (\w+)',
            r'(\w+)(?:Dataset|Data)',
        ]

        docstring_lower = docstring.lower()

        # Apply each pattern
        for pattern in dataset_patterns:
            matches = re.findall(pattern, docstring_lower)
            for match in matches:
                if match and not self._is_common_word(match):
                    self.detected_datasets.add(match.upper())

        # For custom dataset classes, also look for capitalized multi-word names
        # that might indicate dataset names (e.g., "Flowers102", "MNIST", "ImageNet")
        capitalized_words = re.findall(r'\b([A-Z][a-z]*\d*)\b', docstring)
        for word in capitalized_words:
            if not self._is_common_word(word):
                self.detected_datasets.add(word.upper())

    def _extract_dataset_name_from_class(self, class_name: str) -> str:
        """Extract dataset name from a class name."""
        # Remove common suffixes
        name = re.sub(r'(Dataset|DataModule|Data)$', '', class_name)
        # Remove common prefixes
        name = re.sub(r'^(Custom|My|Base)', '', name)

        if name:
            return name
        return class_name