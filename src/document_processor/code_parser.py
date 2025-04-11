import ast
import datetime
import os

from git import Repo


class CodeParser:
    def __init__(self, schema_validator=None):
        self.schema_validator = schema_validator

    def parse(self, file_path):
        """
        Determine whether to process the file.
        Process Python files (.py and optionally .ipynb) using parse_file;
        skip others.
        """
        ext = os.path.splitext(file_path)[1].lower()
        if ext in ['.py', '.ipynb']:
            return self.parse_file(file_path)
        else:
            # If the file is not a Python code file, skip parsing and return None.
            return None

    def parse_file(self, file_path):
        """Parse a Python file and extract model information."""
        with open(file_path, "r") as f:
            file_content = f.read()

        try:
            tree = ast.parse(file_content, filename=file_path)
        except SyntaxError as e:
            raise ValueError(f"Syntax error when parsing {file_path}: {e}")

        # Initialize model_info with file-level metadata.
        model_info = {
            "creation_date": self._get_creation_date(file_path),
            "last_modified_date": self._get_last_modified_date(file_path)
        }

        # Extract basic model information from the AST.
        extracted_info = self._extract_model_info(tree)
        model_info.update(extracted_info)

        # Detect the ML framework from import statements.
        framework = self._detect_framework(tree)
        model_info["framework"] = framework

        # Extract architecture details.
        architecture = self._extract_architecture(tree, model_info)
        model_info["architecture"] = architecture

        # Extract dataset information.
        dataset_info = self._extract_dataset_info(tree, model_info)
        model_info["dataset"] = dataset_info

        # Extract training configuration.
        training_config = self._extract_training_config(tree, model_info)
        model_info["training_config"] = training_config

        # Extract performance metrics.
        performance_metrics = self._extract_performance_metrics(tree, model_info)
        model_info["performance"] = performance_metrics

        # Mark this file as a model script unconditionally.
        model_info["is_model_script"] = True

        # This is needed later for splitting into chunks.
        model_info["content"] = file_content

        # Optionally validate extracted metadata
        # if self.schema_validator:
        #     self.schema_validator.validate(model_info, "model_script_schema")

        return model_info

    def split_into_chunks(self, content, chunk_size=1000, overlap=200):
        """
        Split content into chunks for processing.

        Args:
            content (str): Text to split into chunks.
            chunk_size (int): Maximum number of characters per chunk.
            overlap (int): Number of overlapping characters between successive chunks.

        Returns:
            list of str: A list of text chunks.
        """
        chunks = []
        start = 0
        content_length = len(content)
        while start < content_length:
            end = start + chunk_size
            chunk = content[start:end]
            chunks.append(chunk)
            # Advance start position with overlap in mind.
            start += (chunk_size - overlap)
        return chunks

    def _get_creation_date(self, file_path):
        """Get file creation date, preferring git history if available."""
        try:
            repo = Repo(os.path.dirname(file_path), search_parent_directories=True)
            # Get the earliest commit for the file.
            commits = list(repo.iter_commits(paths=file_path, max_count=1, reverse=True))
            if commits:
                return datetime.datetime.fromtimestamp(commits[0].committed_date).isoformat()
        except Exception:
            pass

        # Fallback: use filesystem creation time
        try:
            stat = os.stat(file_path)
            return datetime.datetime.fromtimestamp(stat.st_ctime).isoformat()
        except Exception:
            return None

    def _get_last_modified_date(self, file_path):
        """Get file last modified date, preferring git history if available."""
        try:
            repo = Repo(os.path.dirname(file_path), search_parent_directories=True)
            commit = next(repo.iter_commits(paths=file_path, max_count=1))
            return datetime.datetime.fromtimestamp(commit.committed_date).isoformat()
        except Exception:
            pass

        # Fallback: use filesystem modification time
        try:
            stat = os.stat(file_path)
            return datetime.datetime.fromtimestamp(stat.st_mtime).isoformat()
        except Exception:
            return None

    def _extract_model_info(self, tree):
        """Extract model information from AST."""
        model_info = {}

        class ModelVisitor(ast.NodeVisitor):
            def __init__(self):
                self.model_name = None

            def visit_ClassDef(self, node):
                # Rudimentary check: if a class inherits from common ML module bases.
                for base in node.bases:
                    if hasattr(base, 'id') and base.id in ['Module', 'nn.Module']:
                        self.model_name = node.name
                        break  # Stop visiting once a model is found.

        visitor = ModelVisitor()
        visitor.visit(tree)

        if visitor.model_name:
            model_info['model_id'] = visitor.model_name.lower()
            model_info['model_family'] = 'custom'
            model_info['version'] = "1.0"
        else:
            model_info['model_id'] = "unknown"
            model_info['model_family'] = "unknown"
            model_info['version'] = "unknown"

        return model_info

    def _detect_framework(self, tree):
        """Detect ML framework from imports."""
        frameworks = ['torch', 'tensorflow', 'keras', 'jax']
        detected = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in frameworks:
                        detected['name'] = alias.name
                        detected['version'] = "unknown"
                        return detected
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    for fw in frameworks:
                        if fw in node.module:
                            detected['name'] = fw
                            detected['version'] = "unknown"
                            return detected

        return {"name": "unknown", "version": "unknown"}

    def _extract_architecture(self, tree, model_info):
        """Extract model architecture and dimensions."""
        architecture = {
            "type": "unknown",
            "dimensions": {}
        }

        class ArchitectureVisitor(ast.NodeVisitor):
            def __init__(self, model_id):
                self.model_id = model_id
                self.architecture_type = "unknown"
                self.dimensions = {}

            def visit_ClassDef(self, node):
                if node.name.lower() == self.model_id:
                    for body_item in node.body:
                        if isinstance(body_item, ast.FunctionDef) and body_item.name == '__init__':
                            for stmt in body_item.body:
                                if isinstance(stmt, ast.Assign):
                                    for target in stmt.targets:
                                        if (isinstance(target, ast.Attribute) and
                                            isinstance(target.value, ast.Name) and
                                            target.value.id == 'self'):
                                            attr_name = target.attr
                                            if attr_name in ['num_layers', 'hidden_size', 'num_attention_heads', 'total_parameters']:
                                                try:
                                                    value = ast.literal_eval(stmt.value)
                                                except Exception:
                                                    value = None
                                                self.dimensions[attr_name] = value

            def visit_Call(self, node):
                if hasattr(node.func, 'attr') and node.func.attr.lower() in ['transformer', 'cnn', 'rnn', 'mlp']:
                    self.architecture_type = node.func.attr.lower()

        arch_visitor = ArchitectureVisitor(model_info.get('model_id', ''))
        arch_visitor.visit(tree)
        architecture['type'] = arch_visitor.architecture_type
        architecture['dimensions'] = arch_visitor.dimensions
        return architecture

    def _extract_dataset_info(self, tree, model_info):
        """Extract dataset information."""
        dataset_info = {
            "name": "unknown",
            "version": "unknown",
            "num_samples": 0,
            "split": "unknown"
        }

        class DatasetVisitor(ast.NodeVisitor):
            def __init__(self):
                self.dataset_info = {}

            def visit_Assign(self, node):
                if isinstance(node.targets[0], ast.Name):
                    var_name = node.targets[0].id.lower()
                    if var_name in ['dataset', 'train_data']:
                        try:
                            value = ast.literal_eval(node.value)
                            if isinstance(value, str):
                                self.dataset_info['name'] = value
                            elif isinstance(value, dict):
                                self.dataset_info.update(value)
                        except Exception:
                            pass

        ds_visitor = DatasetVisitor()
        ds_visitor.visit(tree)
        if ds_visitor.dataset_info:
            dataset_info.update(ds_visitor.dataset_info)

        return dataset_info

    def _extract_training_config(self, tree, model_info):
        """Extract training configuration."""
        training_config = {
            "batch_size": None,
            "learning_rate": None,
            "optimizer": "unknown",
            "epochs": None,
            "training_time_hours": None,
            "hardware_used": "unknown"
        }

        class TrainingVisitor(ast.NodeVisitor):
            def __init__(self):
                self.config = {}

            def visit_Assign(self, node):
                if isinstance(node.targets[0], ast.Name):
                    var_name = node.targets[0].id.lower()
                    if var_name in training_config:
                        try:
                            value = ast.literal_eval(node.value)
                            self.config[var_name] = value
                        except Exception:
                            pass

        train_visitor = TrainingVisitor()
        train_visitor.visit(tree)
        training_config.update(train_visitor.config)
        return training_config

    def _extract_performance_metrics(self, tree, model_info):
        """Extract performance metrics."""
        performance = {
            "accuracy": None,
            "loss": None,
            "perplexity": None,
            "eval_dataset": "unknown"
        }

        class PerformanceVisitor(ast.NodeVisitor):
            def __init__(self):
                self.metrics = {}

            def visit_Assign(self, node):
                if isinstance(node.targets[0], ast.Name):
                    var_name = node.targets[0].id.lower()
                    if var_name in ['accuracy', 'loss', 'perplexity', 'eval_dataset']:
                        try:
                            value = ast.literal_eval(node.value)
                            self.metrics[var_name] = value
                        except Exception:
                            pass

        perf_visitor = PerformanceVisitor()
        perf_visitor.visit(tree)
        performance.update(perf_visitor.metrics)
        return performance
