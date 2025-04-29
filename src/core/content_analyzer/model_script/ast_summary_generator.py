import ast
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from src.core.content_analyzer.model_script.model_diagram_generator import (
    draw_model_architecture,
)
from src.core.content_analyzer.model_script.visitor.code_summary_visitor import (
    CodeSummaryVisitor,
)
from src.core.content_analyzer.model_script.visitor.dataset_class_visitor import DatasetClassVisitor
from src.core.content_analyzer.model_script.visitor.dataset_import_visitor import DatasetImportVisitor
from src.core.content_analyzer.model_script.visitor.dataset_load_visitor import DataLoaderVisitor
from src.core.content_analyzer.model_script.visitor.keyword_based_dataset_visitor import KeywordBasedDatasetVisitor


class ASTSummaryGenerator:
    """
    Generates a human-readable summary of model code via AST parsing
    and produces a diagram-ready architecture overview.
    """

    def __init__(self) -> None:
        # File and directory context
        self.current_file_path: str = "<unknown>"
        self.base_dir: str = ""

        # AST-derived model info
        self.model_layers: List[Dict] = []
        self.model_connections: Dict = {}
        self.layer_order: List[str] = []

        # Class and function tracking
        self.current_class: Optional[str] = None
        self.class_layers: Dict[str, List[str]] = {}
        self.class_hierarchy: Dict[str, List[str]] = {}
        self.function_calls: Dict[str, Any] = {}
        self.literal_vars: Dict[str, Any] = {}
        self.class_vars: Dict[str, Dict[str, Any]] = {}

        # Component usage tracking
        self.component_instantiations: Dict[str, Any] = {}
        self.used_components: Set[str] = set()

        # Dataset tracking - Add these two lines to fix the attribute error
        self.detected_datasets: Set[str] = set()
        self.dataset_classes: Set[str] = set()

        # Diagram settings
        self.important_layer_types: Set[str] = {
            "AdaptiveAvgPool2d", "Attention", "Conv2d", "ConvTranspose2d",
            "Embedding", "GRU", "Linear", "LSTM", "MultiheadAttention",
            "Sequential", "Transformer",
        }
        self.max_dims_to_show: int = 2 # Keep it low to make llm more focus on important information when parsing

    def generate_summary(self, code_str: str, file_path: str = "<unknown>") -> str:
        """
        Parse code and build a summary of datasets, components, and layers.
        """
        try:
            tree = ast.parse(code_str, filename=file_path)
        except SyntaxError as e:
            return f"# Failed to parse AST: {e}"

        # Initialize context for this file
        self._reset_state(file_path)

        lines: List[str] = []
        literal_vars: Dict[str, Any] = {}
        image_folders: Set[str] = set()
        default_paths: Set[str] = set()

        # Collect code summary entries
        visitor = CodeSummaryVisitor(self, lines, literal_vars, image_folders, default_paths)
        visitor.visit(tree)

        # Resolve actual argument values in function definitions
        self._process_function_calls(lines)

        # Detect dataset usage
        dataset_info = self._detect_datasets(tree)
        lines.append(f"Dataset: {dataset_info or 'unknown'}")

        # Build model architecture section
        lines.append("\nModel Architecture:")
        self._identify_used_components()
        components = self._filter_unused_components()

        # Append component and layer details
        for class_name in sorted(components):
            lines.append(f"Component: {class_name}")
            layer_names = self.class_layers.get(class_name, [])
            ordered = self._get_ordered_layers(layer_names)
            self._add_component_layers(lines, ordered, class_name)

        # Append dependencies
        lines.append("\nComponent Dependencies:")
        for class_name in sorted(components):
            deps = [
                layer["layer_type"] for layer in self.model_layers
                if layer["class"] == class_name and layer["layer_type"] in components
            ]
            if deps:
                unique = sorted(set(deps))
                lines.append(f"{class_name} depends on: {', '.join(unique)}")

        return "\n".join(lines)

    @staticmethod
    def analyze_and_visualize_model(
        python_file_path: str,
        output_diagram_path: str = "model_architecture.png",
        show_dimensions: bool = True,
    ) -> Tuple[str, str]:
        """
        Load a Python model file, generate AST summary, and draw the architecture.
        Returns (summary_text, diagram_message).
        """
        with open(python_file_path, 'r') as f:
            code = f.read()

        generator = ASTSummaryGenerator()
        summary = generator.generate_summary(code, python_file_path)

        tree, dims, root = generator._build_component_tree()
        diagram_msg = draw_model_architecture(
            summary,
            output_diagram_path,
            show_dimensions=show_dimensions,
            component_tree=tree,
            root_component=root,
        )
        return summary, diagram_msg

    def _reset_state(self, file_path: str) -> None:
        """Reset internal tracking state before processing a new file."""
        self.current_file_path = file_path
        self.base_dir = os.path.dirname(os.path.abspath(file_path))
        self.model_layers.clear()
        self.model_connections.clear()
        self.layer_order.clear()
        self.current_class = None
        self.class_layers.clear()
        self.class_hierarchy.clear()
        self.function_calls.clear()
        self.component_instantiations.clear()
        self.used_components.clear()

    def _process_function_calls(self, lines: List[str]) -> None:
        """
        Update function signature lines in summary with actual argument values
        extracted from recorded calls.
        """
        func_defs: Dict[str, int] = {}
        for idx, line in enumerate(lines):
            stripped = line.lstrip()  # strip leading whitespace
            if stripped.startswith("Function: "):
                match = re.match(r"Function: (\w+)\(.*\)", stripped)
                if match:
                    func_defs[match.group(1)] = idx

        for fname, def_idx in func_defs.items():
            if fname not in self.function_calls:
                continue
            sig_info = self.function_calls[fname]
            calls = [info for info in self.function_calls.values()
                     if isinstance(info, dict) and info.get('function') == fname]
            if not calls:
                continue
            last_call = calls[-1]
            params = sig_info.get('params', [])
            defaults = sig_info.get('defaults', {})
            actual = last_call.get('args', {})
            omitted = last_call.get('omitted_args', [])

            parts: List[str] = []
            for p in params:
                if p in actual:
                    parts.append(f"{p}={actual[p]}")
                elif p in defaults and p in omitted:
                    parts.append(f"{p}={defaults[p]}")
                else:
                    parts.append(p)

            prefix = lines[def_idx][:lines[def_idx].find("Function:")]
            lines[def_idx] = f"{prefix}Function: {fname}({', '.join(parts)})"

            insert_pos = def_idx + 1
            for p in params:
                if p in actual:
                    val = actual[p]
                    lines.insert(insert_pos, f"Variable: {p} = {val} (from call `{fname}`)")
                    insert_pos += 1
                elif p in defaults and p in omitted:
                    val = defaults[p]
                    lines.insert(insert_pos, f"Variable: {p} = {val} (default for `{fname}`)")
                    insert_pos += 1

    def _detect_datasets(self, tree: ast.AST) -> str:
        """
        Return a comma-separated list of datasets found in the AST.

        This method implements a two-tiered detection strategy:

        1. PRIMARY DETECTION (DatasetVisitor):
           - Faster and more direct approach
           - Looks for known dataset keywords and explicit dataset mentions
           - Scans imports, variable names, and string literals
           - Highly precise but may miss custom or non-standard datasets

        2. SECONDARY DETECTION (Fallback Methods):
           - Only triggered if primary detection finds nothing
           - More thorough but potentially more computationally expensive
           - Uses specialized visitors to detect datasets through:
             a) DataLoader usage patterns
             b) Dataset class inheritance hierarchies
             c) Common dataset import patterns from various libraries
           - Better at finding custom datasets and implicit references

        WHY TWO APPROACHES:
        - Primary detection is efficient but may miss complex cases
        - Secondary detection is more comprehensive but could be overkill for simple cases
        - Two-tiered approach balances speed and thoroughness
        - Ensures both explicit and implicit dataset references are found

        Args:
            tree: The AST tree of the Python code to analyze

        Returns:
            A comma-separated string of dataset names found in the code.
            Returns an empty string if no datasets are detected.
        """
        visitor = KeywordBasedDatasetVisitor()
        visitor.visit(tree)

        # Store detected datasets for other methods to use
        self.detected_datasets.update(visitor.datasets)

        # If primary detection found datasets, return them
        if visitor.datasets:
            return ", ".join(sorted(visitor.datasets))

        # If nothing found, try secondary detection methods
        dataset_names = self._detect_datasets_from_fallback_methods(tree)
        if dataset_names:
            self.detected_datasets.update(dataset_names)
            return ", ".join(sorted(dataset_names))

        return ""

    def _detect_datasets_from_fallback_methods(self, tree: ast.AST) -> Set[str]:
        """
        Apply secondary (fallback) strategies to detect datasets when primary detection fails.

        This method implements three specialized approaches that go beyond basic keyword
        matching to find datasets through structural patterns in the code:

        1. DATALOADER DETECTION (DataLoaderVisitor):
           - Identifies DataLoader constructor calls
           - Extracts dataset information from the loader's arguments
           - Analyzes variable names for context clues
           - Effective for finding datasets used with standard data loading patterns

        2. DATASET CLASS DETECTION (DatasetClassVisitor):
           - Recognizes custom dataset classes through inheritance patterns
           - Analyzes class names and docstrings for dataset references
           - Filters out common words to avoid false positives
           - Excellent for identifying project-specific dataset implementations

        3. IMPORT DETECTION (DatasetImportVisitor):
           - Scans for imports from common dataset libraries
           - Identifies dataset-specific import patterns
           - Detects imports of custom dataset modules
           - Useful for finding datasets referenced through standard libraries

        These methods are more thorough than primary detection but may take more
        processing time, which is why they're only used as a fallback.

        Args:
            tree: The AST tree of the Python code to analyze

        Returns:
            A set of dataset names detected through fallback methods
        """
        dataset_names = set()

        # Method 1: Scan for DataLoader and similar constructs
        loader_visitor = DataLoaderVisitor()
        loader_visitor.visit(tree)
        dataset_names.update(loader_visitor.detected_datasets)

        # Method 2: Look for common dataset class patterns and inheritance
        class_visitor = DatasetClassVisitor()
        class_visitor.visit(tree)
        dataset_names.update(class_visitor.detected_datasets)
        self.dataset_classes.update(class_visitor.dataset_classes)

        # Method 3: Check imports of common datasets
        import_visitor = DatasetImportVisitor()
        import_visitor.visit(tree)
        dataset_names.update(import_visitor.detected_datasets)

        return dataset_names

    def _get_ordered_layers(self, names: List[str]) -> List[str]:
        """Return layer names in definition order, preserving any extras."""
        ordered: List[str] = []
        seen: Set[str] = set()
        for n in self.layer_order:
            if n in names and n not in seen:
                ordered.append(n)
                seen.add(n)
        for n in names:
            if n not in seen:
                ordered.append(n)
                seen.add(n)
        return ordered

    def _add_component_layers(
        self,
        lines: List[str],
        ordered_layers: List[str],
        class_name: str,
    ) -> None:
        """Append formatted layer entries for a given component."""
        seen: Set[str] = set()
        for lname in ordered_layers:
            if lname in seen:
                continue
            seen.add(lname)
            layer = next((l for l in self.model_layers if l['name'] == lname), None)
            if not layer:
                continue
            ltype = layer.get('layer_type', 'Unknown')
            show = (
                any(t in ltype for t in self.important_layer_types)
                or ltype in self.class_layers
            )
            if not show:
                continue
            dims = self._extract_important_dimensions(layer)
            if dims:
                lines.append(f"  {lname}: {ltype}({dims})")
            else:
                lines.append(f"  {lname}: {ltype}")

    def _identify_used_components(self) -> None:
        """Mark components that appear in layers or forward calls as used."""
        # Instantiated within layers
        for layer in self.model_layers:
            lt = layer.get('layer_type', '')
            if lt in self.class_layers:
                self.used_components.add(lt)
        # Referenced in forward
        for fname, info in self.function_calls.items():
            if fname == 'forward' and isinstance(info, dict):
                cls = info.get('class')
                if cls and cls in self.class_layers:
                    self.used_components.add(cls)
                    for ref in info.get('references', []):
                        if ref in self.class_layers:
                            self.used_components.add(ref)

    def _filter_unused_components(self) -> Set[str]:
        """Return only those components deemed used, or fallback to main class."""
        if not self.used_components:
            # Pick class with most layers
            main, count = None, 0
            for cname, layers in self.class_layers.items():
                if len(layers) > count:
                    main, count = cname, len(layers)
            if main:
                self.used_components.add(main)
                for layer in self.model_layers:
                    if layer['class'] == main and layer['layer_type'] in self.class_layers:
                        self.used_components.add(layer['layer_type'])
        return (
            self.used_components
            if self.used_components
            else set(self.class_layers.keys())
        )

    def _build_component_tree(self) -> Tuple[Dict[str, List[str]], Dict[str, str], Optional[str]]:
        """
        Build a tree of components and layers, plus dependency mapping, and identify root.
        """
        tree: Dict[str, List[str]] = {c: [] for c in self.class_layers}
        deps: Dict[str, Set[str]] = {c: set() for c in self.class_layers}
        dims: Dict[str, str] = {}

        for layer in self.model_layers:
            name = layer['name']
            ltype = layer['layer_type']
            cls = layer['class']
            if ltype in tree:
                tree[cls].append(ltype)
                deps[cls].add(ltype)
            else:
                tree[cls].append(name)
                dims[name] = self._extract_important_dimensions(layer)

        # Function-call dependencies omitted for brevity

        roots = [c for c in tree if c not in {d for subs in tree.values() for d in subs}]
        root = roots[0] if roots else None
        return tree, dims, root

    def _extract_important_dimensions(self, layer: Dict) -> str:
        """Trim down layer args to the most meaningful dimension info."""
        args = layer.get('args', [])
        ltype = layer.get('layer_type', '')
        # Sequential shortcut
        if ltype == 'Sequential':
            seq = self._get_sequential_dimensions(args)
            return seq or '...'
        if ltype in self.class_hierarchy:
            return ltype
        dims: List[str] = []
        for arg in args:
            if isinstance(arg, (int, float)):
                dims.append(str(arg))
            elif isinstance(arg, str) and re.search(r'\d', arg):
                dims.append(arg)
        return ', '.join(dims[:self.max_dims_to_show])

    def _get_sequential_dimensions(self, args: List[Any]) -> str:
        input_dim = output_dim = None
        for arg in args:
            s = str(arg)
            m = re.search(r'(?:Conv\w*|Linear)\s*\(\s*(\d+)[^,]*,\s*(\d+)', s)
            if m:
                input_dim, _ = m.groups()
                break
        for arg in reversed(args):
            s = str(arg)
            m = re.search(r'(?:Conv\w*|Linear)\s*\(\s*\d+[^,]*,\s*(\d+)', s)
            if m:
                output_dim = m.group(1)
                break
        if input_dim and output_dim:
            return f"{input_dim} → {output_dim}"
        return input_dim or output_dim or ''

    def _eval_constant(self, node: ast.AST) -> Optional[Any]:
        if isinstance(node, ast.Constant) and isinstance(node.value, (str, int, float, bool)):
            return node.value
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left = self._eval_constant(node.left)
            right = self._eval_constant(node.right)
            if isinstance(left, str) and isinstance(right, str):
                return left + right
        if isinstance(node, ast.JoinedStr):
            parts: List[str] = []
            for v in node.values:
                if isinstance(v, ast.Constant) and isinstance(v.value, str):
                    parts.append(v.value)
                else:
                    return None
            return ''.join(parts)
        return None

    def _determine_folder(self, path_str: str, base_dir: str) -> str:
        abs_path = os.path.abspath(path_str if os.path.isabs(path_str) else os.path.join(base_dir, path_str))
        base, ext = os.path.splitext(abs_path)
        return os.path.dirname(abs_path) if ext else abs_path
