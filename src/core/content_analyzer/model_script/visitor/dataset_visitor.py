import ast


class DatasetVisitor(ast.NodeVisitor):
    def __init__(self):
        self.datasets = set()
        # Common dataset names and libraries
        self.dataset_keywords = {
            "mnist", "cifar", "imagenet", "coco", "kitti", "voc", "pascal",
            "celeba", "clevr", "shapenet", "kinetics", "audio", "audioset",
            "librispeech", "voxceleb", "lfw", "ucf101", "hmdb51", "mini-imagenet",
            "omniglot", "wikitext", "squad", "glue", "snli", "conll", "penn",
            "ptb", "sst", "imdb", "yelp", "amazon", "20newsgroups", "enwik8"
        }

    def visit_Name(self, node):
        # Check for variable names that indicate datasets
        if any(keyword in node.id.lower() for keyword in self.dataset_keywords):
            # Extract the dataset name from the variable
            for keyword in self.dataset_keywords:
                if keyword in node.id.lower():
                    self.datasets.add(keyword.upper())
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        # Check for imports from dataset modules
        if node.module and any(keyword in node.module.lower() for keyword in self.dataset_keywords):
            for keyword in self.dataset_keywords:
                if node.module and keyword in node.module.lower():
                    self.datasets.add(keyword.upper())

        # Check torchvision.datasets or tensorflow.keras.datasets imports
        if node.module and ("torchvision.datasets" in node.module or
                            "tensorflow.keras.datasets" in node.module or
                            "tensorflow_datasets" in node.module):
            for alias in node.names:
                if alias.name.lower() in self.dataset_keywords:
                    self.datasets.add(alias.name.upper())
        self.generic_visit(node)

    def visit_Call(self, node):
        # Check for dataset loading calls like "load_dataset", "CIFAR10" etc.
        if isinstance(node.func, ast.Name):
            if "dataset" in node.func.id.lower() or any(
                    keyword in node.func.id.lower() for keyword in self.dataset_keywords):
                for keyword in self.dataset_keywords:
                    if keyword in node.func.id.lower():
                        self.datasets.add(keyword.upper())

        # Check for calls like torchvision.datasets.MNIST()
        elif isinstance(node.func, ast.Attribute):
            attr_name = node.func.attr.lower()
            if any(keyword in attr_name for keyword in self.dataset_keywords):
                for keyword in self.dataset_keywords:
                    if keyword in attr_name:
                        self.datasets.add(keyword.upper())

            # Check string arguments to functions like load_dataset("mnist")
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    str_val = arg.value.lower()
                    for keyword in self.dataset_keywords:
                        if keyword in str_val:
                            self.datasets.add(keyword.upper())

        self.generic_visit(node)

    def visit_ClassDef(self, node):
        # Check for custom dataset classes
        if "dataset" in node.name.lower() or any(
                keyword in node.name.lower() for keyword in self.dataset_keywords):
            for keyword in self.dataset_keywords:
                if keyword in node.name.lower():
                    self.datasets.add(keyword.upper())

        # Look for specific dataset name in docstrings
        doc = ast.get_docstring(node)
        if doc:
            for keyword in self.dataset_keywords:
                if keyword.lower() in doc.lower():
                    self.datasets.add(keyword.upper())

        # Check for dataset-related base classes
        for base in node.bases:
            if isinstance(base, ast.Name) and "dataset" in base.id.lower():
                # This is likely a dataset class
                if "dataset" not in node.name.lower():  # Avoid double-counting
                    # Try to infer the dataset from the class name
                    class_name = node.name.lower()
                    for keyword in self.dataset_keywords:
                        if keyword in class_name:
                            self.datasets.add(keyword.upper())

        self.generic_visit(node)

    def visit_Str(self, node):
        # Check for dataset names in string literals
        for keyword in self.dataset_keywords:
            if keyword in node.s.lower():
                # Only add if it looks like a dataset reference, not just any mention
                context = node.s.lower()
                if ("dataset" in context or "data" in context or
                        "load" in context or "download" in context):
                    self.datasets.add(keyword.upper())

    def visit_Constant(self, node):
        # For Python 3.8+ compatibility
        if isinstance(node.value, str):
            for keyword in self.dataset_keywords:
                if keyword in node.value.lower():
                    # Only add if it looks like a dataset reference
                    context = node.value.lower()
                    if ("dataset" in context or "data" in context or
                            "load" in context or "download" in context):
                        self.datasets.add(keyword.upper())