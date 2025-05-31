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

        # Dataset tracking
        self.detected_datasets: Set[str] = set()
        self.dataset_classes: Set[str] = set()

        # Diagram settings
        self.important_layer_types: Set[str] = {
            "AdaptiveAvgPool2d",
            "Attention",
            "Conv2d",
            "ConvTranspose2d",
            "Embedding",
            "GRU",
            "Linear",
            "LSTM",
            "MultiheadAttention",
            "Sequential",
            "Transformer",
        }
        self.max_dims_to_show: int = 2  # Keep it low to make llm more focus on important information

    def generate_summary(self, code_str: str, file_path: str = "<unknown>") -> str:
        """
        Parse code and build a summary of datasets, components, and layers.
        """
        try:
            tree = ast.parse(code_str, filename=file_path)
        except SyntaxError as e:
            return f"# Failed to parse AST: {e}"

        self._reset_state(file_path)

        # Phase 1: Collect raw summary data
        lines: List[str] = []
        literal_vars: Dict[str, Any] = {}
        image_folders: Set[str] = set()
        default_paths: Set[str] = set()

        visitor = CodeSummaryVisitor(self, lines, literal_vars, image_folders, default_paths)
        visitor.visit(tree)

        # Phase 2: Update function signatures with actual call values
        self._process_function_calls(lines)

        # Phase 3: Detect datasets and append to summary
        dataset_info = self._detect_datasets(tree)
        lines.append(f"Dataset: {dataset_info or 'unknown'}")

        # Phase 4: Build model architecture and dependencies
        self._append_model_architecture(lines)
        self._append_component_dependencies(lines)

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

        tree, _, root = generator._build_component_tree()
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
        func_defs = self._collect_function_definitions(lines)
        for fname, def_idx in func_defs.items():
            call_info = self.function_calls.get(fname)
            if not call_info or not isinstance(call_info, dict):
                continue
            params, defaults = call_info.get('params', []), call_info.get('defaults', {})
            calls = [
                info for info in self.function_calls.values()
                if isinstance(info, dict) and info.get('function') == fname
            ]
            if not calls:
                continue
            last_call = calls[-1]
            parts, annotations = self._format_signature_parts(params, defaults, last_call)
            prefix = lines[def_idx][: lines[def_idx].find("Function:")]
            lines[def_idx] = f"{prefix}Function: {fname}({', '.join(parts)})"
            self._insert_signature_annotations(lines, annotations, def_idx + 1)

    def _collect_function_definitions(self, lines: List[str]) -> Dict[str, int]:
        """Collect indices of lines that define a function signature."""
        func_defs: Dict[str, int] = {}
        pattern = re.compile(r"Function:\s*(\w+)\(.*\)")
        for idx, line in enumerate(lines):
            stripped = line.lstrip()
            match = pattern.match(stripped)
            if match:
                func_defs[match.group(1)] = idx
        return func_defs

    def _format_signature_parts(
        self,
        params: List[str],
        defaults: Dict[str, Any],
        call_info: Dict[str, Any],
    ) -> Tuple[List[str], List[Tuple[str, Any, bool]]]:
        """
        Return formatted signature parts and annotations given parameters, defaults,
        and the last call's info.
        """
        actual = call_info.get('args', {})
        omitted = call_info.get('omitted_args', [])
        parts: List[str] = []
        annotations: List[Tuple[str, Any, bool]] = []  # (param, value, is_default)

        for p in params:
            if p in actual:
                parts.append(f"{p}={actual[p]}")
                annotations.append((p, actual[p], False))
            elif p in defaults and p in omitted:
                parts.append(f"{p}={defaults[p]}")
                annotations.append((p, defaults[p], True))
            else:
                parts.append(p)
        return parts, annotations

    def _insert_signature_annotations(
        self, lines: List[str], annotations: List[Tuple[str, Any, bool]], insert_pos: int
    ) -> None:
        """Insert lines describing variable assignments under a function definition."""
        for param, value, is_default in annotations:
            if is_default:
                lines.insert(insert_pos, f"Variable: {param} = {value} (default)")
            else:
                lines.insert(insert_pos, f"Variable: {param} = {value} (from call)")
            insert_pos += 1

    def _detect_datasets(self, tree: ast.AST) -> str:
        """
        Return a comma-separated list of datasets found in the AST.
        Implements primary and fallback detection.
        """
        primary = self._primary_dataset_detection(tree)
        if primary:
            return ", ".join(sorted(primary))

        fallback = self._detect_datasets_from_fallback_methods(tree)
        return ", ".join(sorted(fallback)) if fallback else ""

    def _primary_dataset_detection(self, tree: ast.AST) -> Set[str]:
        """Use keyword-based visitor to quickly detect common dataset names."""
        visitor = KeywordBasedDatasetVisitor()
        visitor.visit(tree)
        self.detected_datasets.update(visitor.datasets)
        return visitor.datasets

    def _detect_datasets_from_fallback_methods(self, tree: ast.AST) -> Set[str]:
        """
        Apply secondary strategies to detect datasets:
          1. DataLoader usage
          2. Custom Dataset classes
          3. Imports of known dataset libraries
        """
        dataset_names: Set[str] = set()

        # (1) DataLoader usage
        loader_visitor = DataLoaderVisitor()
        loader_visitor.visit(tree)
        dataset_names.update(loader_visitor.detected_datasets)

        # (2) Custom Dataset classes
        class_visitor = DatasetClassVisitor()
        class_visitor.visit(tree)
        dataset_names.update(class_visitor.detected_datasets)
        self.dataset_classes.update(class_visitor.dataset_classes)

        # (3) Imports of datasets
        import_visitor = DatasetImportVisitor()
        import_visitor.visit(tree)
        dataset_names.update(import_visitor.detected_datasets)

        self.detected_datasets.update(dataset_names)
        return dataset_names

    def _append_model_architecture(self, lines: List[str]) -> None:
        """
        Append the "Model Architecture" section to lines. Identifies used components,
        filters unused, and adds layer details.
        """
        lines.append("\nModel Architecture:")
        self._identify_used_components()
        components = self._filter_unused_components()

        for class_name in sorted(components):
            lines.append(f"Component: {class_name}")
            layer_names = self.class_layers.get(class_name, [])
            ordered = self._get_ordered_layers(layer_names)
            self._add_component_layers(lines, ordered)

    def _append_component_dependencies(self, lines: List[str]) -> None:
        """
        Append the "Component Dependencies" section to lines, listing dependencies
        for each used component.
        """
        lines.append("\nComponent Dependencies:")
        components = self._filter_unused_components()
        for class_name in sorted(components):
            deps = [
                layer["layer_type"]
                for layer in self.model_layers
                if layer["class"] == class_name and layer["layer_type"] in components
            ]
            if deps:
                unique = sorted(set(deps))
                lines.append(f"{class_name} depends on: {', '.join(unique)}")

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
    ) -> None:
        """Append formatted layer entries for a given component."""
        seen: Set[str] = set()
        for lname in ordered_layers:
            if lname in seen:
                continue
            seen.add(lname)
            layer = next((l for l in self.model_layers if l["name"] == lname), None)
            if not layer:
                continue
            ltype = layer.get("layer_type", "Unknown")
            if not self._should_show_layer(ltype):
                continue
            dims = self._extract_important_dimensions(layer)
            if dims:
                lines.append(f"  {lname}: {ltype}({dims})")
            else:
                lines.append(f"  {lname}: {ltype}")

    def _should_show_layer(self, layer_type: str) -> bool:
        """Determine if a layer should be shown based on type or component membership."""
        if any(t in layer_type for t in self.important_layer_types):
            return True
        return layer_type in self.class_layers

    def _identify_used_components(self) -> None:
        """Mark components that appear in layers or forward calls as used."""
        for layer in self.model_layers:
            lt = layer.get("layer_type", "")
            if lt in self.class_layers:
                self.used_components.add(lt)

        forward_info = self.function_calls.get("forward")
        if isinstance(forward_info, dict):
            cls = forward_info.get("class")
            if cls and cls in self.class_layers:
                self.used_components.add(cls)
                for ref in forward_info.get("references", []):
                    if ref in self.class_layers:
                        self.used_components.add(ref)

    def _filter_unused_components(self) -> Set[str]:
        """
        Return only those components deemed used, or fallback to the main class.
        If no used components are found, select the class with the most layers.
        """
        if not self.used_components:
            main_class = self._find_main_class()
            if main_class:
                self.used_components.add(main_class)
                self._add_main_class_layers(main_class)

        return self.used_components or set(self.class_layers.keys())

    def _find_main_class(self) -> Optional[str]:
        """
        Find the class with the most layers. Return None if there are no classes.
        """
        main_class = None
        max_layers = 0
        for cname, layers in self.class_layers.items():
            layer_count = len(layers)
            if layer_count > max_layers:
                main_class = cname
                max_layers = layer_count
        return main_class

    def _add_main_class_layers(self, main_class: str) -> None:
        """
        For a given main_class, scan through self.model_layers and add any
        layer["layer_type"] to used_components if it belongs to that class
        and also appears as a component (i.e., is a key in self.class_layers).
        """
        for layer in self.model_layers:
            if (
                    layer.get("class") == main_class
                    and layer.get("layer_type") in self.class_layers
            ):
                self.used_components.add(layer["layer_type"])

    def _build_component_tree(self) -> Tuple[Dict[str, List[str]], Dict[str, str], Optional[str]]:
        """
        Build a tree of components and layers, plus a mapping of layer dimensions,
        and identify the root component.
        """
        tree: Dict[str, List[str]] = {c: [] for c in self.class_layers}
        deps: Dict[str, Set[str]] = {c: set() for c in self.class_layers}
        dims: Dict[str, str] = {}

        for layer in self.model_layers:
            name = layer["name"]
            ltype = layer["layer_type"]
            cls = layer["class"]
            if ltype in tree:
                tree[cls].append(ltype)
                deps[cls].add(ltype)
            else:
                tree[cls].append(name)
                dims[name] = self._extract_important_dimensions(layer)

        roots = [c for c in tree if c not in {dep for subs in tree.values() for dep in subs}]
        root = roots[0] if roots else None
        return tree, dims, root

    def _extract_important_dimensions(self, layer: Dict) -> str:
        """Trim down layer args to the most meaningful dimension info."""
        args = layer.get("args", [])
        ltype = layer.get("layer_type", "")
        if ltype == "Sequential":
            seq = self._get_sequential_dimensions(args)
            return seq or "..."
        if ltype in self.class_hierarchy:
            return ltype

        dims: List[str] = []
        for arg in args:
            if isinstance(arg, (int, float)):
                dims.append(str(arg))
            elif isinstance(arg, str) and re.search(r"\d", arg):
                dims.append(arg)
            if len(dims) >= self.max_dims_to_show:
                break
        return ", ".join(dims)

    def _get_sequential_dimensions(self, args: List[Any]) -> str:
        input_dim = output_dim = None
        for arg in args:
            s = str(arg)
            m = re.search(r"(?:Conv\w*|Linear)\s*\(\s*(\d+)[^,]*,\s*(\d+)", s)
            if m:
                input_dim, _ = m.groups()
                break
        for arg in reversed(args):
            s = str(arg)
            m = re.search(r"(?:Conv\w*|Linear)\s*\(\s*\d+[^,]*,\s*(\d+)", s)
            if m:
                output_dim = m.group(1)
                break
        if input_dim and output_dim:
            return f"{input_dim} → {output_dim}"
        return input_dim or output_dim or ""

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
            return "".join(parts)
        return None

    def _determine_folder(self, path_str: str, base_dir: str) -> str:
        abs_path = os.path.abspath(path_str if os.path.isabs(path_str) else os.path.join(base_dir, path_str))
        _, ext = os.path.splitext(abs_path)
        return os.path.dirname(abs_path) if ext else abs_path