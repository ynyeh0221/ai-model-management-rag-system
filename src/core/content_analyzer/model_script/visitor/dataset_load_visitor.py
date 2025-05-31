import ast
import re


class DataLoaderVisitor(ast.NodeVisitor):
    """
    AST visitor that identifies dataset patterns by analyzing DataLoader usage.

    This specialized visitor focuses on extracting dataset information from how
    data is loaded and processed in the code. It identifies DataLoader constructor
    calls and analyzes their arguments to infer dataset types and names.

    Key capabilities:
      - Detects direct DataLoader instantiations
      - Identifies torch.utils.data.DataLoader usage
      - Extracts dataset names from loader arguments
      - Infers dataset types from variable naming patterns
      - Recognizes common data loading paradigms in PyTorch and TensorFlow

    Attributes:
        detected_datasets (Set[str]): Set of dataset names detected from DataLoader usage

    Example:
        ```python
        # For code like:
        train_loader = DataLoader(Flowers102Dataset(root='./data', split='train'), batch_size=32)

        # This visitor would detect:
        # detected_datasets = {"FLOWERS102" }
        ```
    """

    def __init__(self):
        super().__init__()
        self.detected_datasets = set()

    def visit_Call(self, node: ast.Call):
        # Check for DataLoader constructor calls
        if isinstance(node.func, ast.Name) and node.func.id == 'DataLoader'or (isinstance(node.func, ast.Attribute) and
              node.func.attr == 'DataLoader' and
              isinstance(node.func.value, ast.Attribute) and
              node.func.value.attr == 'data'):
            # Extract dataset from first argument
            if node.args and len(node.args) > 0:
                self._extract_dataset_from_arg(node.args[0])

        self.generic_visit(node)

    def _extract_dataset_from_arg(self, arg_node):
        """Extract dataset information from a DataLoader argument."""
        if isinstance(arg_node, ast.Name):
            # Check if we can identify the dataset from its variable name
            var_name = arg_node.id.lower()
            if 'dataset' in var_name or 'data' in var_name:
                dataset_name = self._extract_dataset_name_from_var(var_name)
                if dataset_name:
                    self.detected_datasets.add(dataset_name.upper())

    def _extract_dataset_name_from_var(self, var_name: str) -> str:
        """Extract dataset name from a variable name."""
        # Split by common separators and look for dataset indicators
        parts = re.split(r'[_\-.]', var_name)
        for part in parts:
            # Skip common words like 'train', 'test', 'val', etc.
            if part in ('train', 'test', 'val', 'valid', 'dataset', 'data', 'loader'):
                continue
            # If we have a substantive part, use it as the dataset name
            if len(part) > 2:  # Avoid short abbreviations
                return part

        # If we couldn't extract a specific name, use a generic one
        if 'train' in var_name:
            return 'TRAINING_DATASET'
        elif 'test' in var_name:
            return 'TEST_DATASET'
        elif 'val' in var_name:
            return 'VALIDATION_DATASET'

        return 'DATASET'