import ast
from typing import Set


class KeywordBasedDatasetVisitor(ast.NodeVisitor):
    """
    AST visitor that collects references to well-known datasets across various code elements.

    This primary dataset detection visitor scans Python code for references to known
    datasets by analyzing variable names, imports, function calls, class definitions,
    and string literals. It uses a predefined set of dataset keywords to identify
    datasets with high precision.

    Key capabilities:
      - Recognizes a comprehensive list of standard dataset names (MNIST, CIFAR, etc.)
      - Detects dataset references in variable names and identifiers
      - Identifies dataset imports from common libraries
      - Finds dataset usage in function calls and arguments
      - Extracts dataset mentions from docstrings and string literals
      - Analyzes class definitions that might represent datasets

    This visitor provides the first line of detection in a two-tiered approach.
    It offers high precision but may miss custom or non-standard datasets that
    aren't in its predefined keyword list.

    Attributes:
        datasets (Set[str]): Set of dataset names collected during tree traversal
        DATASET_KEYWORDS (Set[str]): Predefined set of known dataset names to look for

    Example:
        ```python
        # For code like:
        mnist_data = load_dataset('mnist')

        # This visitor would detect:
        # datasets = {"MNIST"}
        ```
    """

    DATASET_KEYWORDS = {
        "mnist", "cifar", "imagenet", "coco", "kitti", "voc", "pascal",
        "celeba", "clevr", "shapenet", "kinetics", "audio", "audioset",
        "librispeech", "voxceleb", "lfw", "ucf101", "hmdb51", "mini-imagenet",
        "omniglot", "wikitext", "squad", "glue", "snli", "conll", "penn",
        "ptb", "sst", "imdb", "yelp", "amazon", "20newsgroups", "enwik8"
    }

    def __init__(self) -> None:
        super().__init__()
        self.datasets: Set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        """
        Detect dataset keywords in variable names.
        """
        name_lower = node.id.lower()
        for kw in self._keywords_in_text(name_lower):
            self.datasets.add(kw.upper())
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """
        Detect imports from dataset modules and torchvision/tensorflow dataset imports.
        """
        module = node.module.lower() if node.module else ''

        # Imports from modules matching dataset keywords
        for kw in self._keywords_in_text(module):
            self.datasets.add(kw.upper())

        # Explicit torchvision or tensorflow dataset imports
        if any(pkg in module for pkg in ("torchvision.datasets", "tensorflow.keras.datasets", "tensorflow_datasets")):
            for alias in node.names:
                alias_name = alias.name.lower()
                if alias_name in self.DATASET_KEYWORDS:
                    self.datasets.add(alias_name.upper())

        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """
        Detect dataset loading calls and string arguments referring to datasets.
        """
        # Function name cases
        if isinstance(node.func, ast.Name):
            func_lower = node.func.id.lower()
            for kw in self._keywords_in_text(func_lower):
                self.datasets.add(kw.upper())

        elif isinstance(node.func, ast.Attribute):
            attr_lower = node.func.attr.lower()
            for kw in self._keywords_in_text(attr_lower):
                self.datasets.add(kw.upper())

        # String arguments like load_dataset("mnist")
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                for kw in self._keywords_in_text(arg.value.lower(), require_context=True):
                    self.datasets.add(kw.upper())

        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """
        Detect custom dataset classes via class name, docstring, or base classes.
        """
        name_lower = node.name.lower()

        # Class name contains keyword or 'dataset'
        for kw in self._keywords_in_text(name_lower):
            self.datasets.add(kw.upper())

        # Docstring analysis
        doc = ast.get_docstring(node)
        if doc:
            for kw in self._keywords_in_text(doc.lower()):
                self.datasets.add(kw.upper())

        # Base classes with 'dataset'
        for base in node.bases:
            if isinstance(base, ast.Name) and 'dataset' in base.id.lower():
                if 'dataset' not in name_lower:
                    for kw in self._keywords_in_text(name_lower):
                        self.datasets.add(kw.upper())

        self.generic_visit(node)

    def visit_Str(self, node: ast.Str) -> None:
        """
        (Python <3.8) Detect dataset references in string literals.
        """
        self._handle_literal(node.s)

    def visit_Constant(self, node: ast.Constant) -> None:
        """
        (Python >=3.8) Detect dataset references in constant strings.
        """
        if isinstance(node.value, str):
            self._handle_literal(node.value)

    def _handle_literal(self, text: str) -> None:
        """
        Check a string for dataset keywords when context suggests a dataset.
        """
        for kw in self._keywords_in_text(text.lower(), require_context=True):
            self.datasets.add(kw.upper())

    def _keywords_in_text(self, text: str, require_context: bool = False) -> Set[str]:
        """
        Return dataset keywords found in text.
        If you require_context is True, only matches when a text also contains
        context words like 'dataset', 'data', 'load', or 'download'.
        """
        found = {kw for kw in self.DATASET_KEYWORDS if kw in text}
        if require_context:
            contexts = ('dataset', 'data', 'load', 'download')
            if not any(ctx in text for ctx in contexts):
                return set()
        return found
