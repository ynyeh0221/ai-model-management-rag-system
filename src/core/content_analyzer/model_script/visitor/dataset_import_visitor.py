import ast


class DatasetImportVisitor(ast.NodeVisitor):
    """
    AST visitor that detects dataset imports from common libraries.

    This visitor specializes in identifying datasets by analyzing import statements.
    It recognizes imports from common dataset libraries and modules,
    extracting dataset names when they are explicitly imported.

    Key capabilities:
      - Detects imports from dataset-specific modules
      - Identifies individual dataset classes being imported
      - Recognizes common dataset libraries and frameworks
      - Detects both direct imports and from-imports

    Attributes:
        detected_datasets (Set[str]): Set of dataset names detected from import analysis
        COMMON_DATASET_MODULES (Set[str]): Set of module paths known to contain datasets

    Example:
        ```python
        # For code like:
        from torchvision.datasets import Flowers102

        # This visitor would detect:
        # detected_datasets = {"FLOWERS102"}
        ```
    """

    COMMON_DATASET_MODULES = {
        'torchvision.datasets',
        'tensorflow.keras.datasets',
        'tensorflow_datasets',
        'torch.utils.data',
        'keras.datasets',
        'sklearn.datasets'
    }

    def __init__(self):
        super().__init__()
        self.detected_datasets = set()

    def visit_ImportFrom(self, node: ast.ImportFrom):
        module = node.module if node.module else ''

        # Check if importing from a dataset module
        if any(module.startswith(prefix) for prefix in self.COMMON_DATASET_MODULES):
            for alias in node.names:
                # Skip utility classes/functions
                if alias.name not in ('Dataset', 'DataLoader', 'TensorDataset', 'random_split'):
                    self.detected_datasets.add(alias.name.upper())

        self.generic_visit(node)

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            module_name = alias.name
            if any(module_name.startswith(prefix) for prefix in self.COMMON_DATASET_MODULES):
                self.detected_datasets.add('IMPORTED_DATASET')

        self.generic_visit(node)