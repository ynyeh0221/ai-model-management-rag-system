import ast
import os
import re
from typing import Any, Optional, List, Dict, Set

from src.core.content_analyzer.model_script.visitor.code_summary_visitor import CodeSummaryVisitor
from src.core.content_analyzer.model_script.visitor.dataset_visitor import DatasetVisitor
from src.core.content_analyzer.model_script.model_diagram_generator import draw_model_architecture


class ASTSummaryGenerator:
    def __init__(self):
        self.current_file_path = "<unknown>"
        self.base_dir = ""
        self.model_layers: List[Dict] = []
        self.model_connections = {}  # Track model connection flow
        self.layer_order = []  # Track order of layer definitions
        self.current_class = None  # Track the current class being processed
        self.class_layers = {}  # Map component classes to their layers
        self.class_hierarchy = {}  # Map of class name to its parent classes
        self.function_calls = {}  # Track function calls and their arguments
        self.literal_vars = {}
        self.class_vars: Dict[str, Dict[str, Any]] = {}  # vars per component

        # Settings for diagram-ready output
        self.important_layer_types = {
            "Conv2d", "ConvTranspose2d", "Linear", "LSTM", "GRU", "Transformer",
            "MultiheadAttention", "Embedding", "Sequential", "Attention", "AdaptiveAvgPool2d"
        }
        self.max_dims_to_show = 2  # Max number of dimensions to show

    def generate_summary(self, code_str: str, file_path: str = "<unknown>") -> str:
        """Parse code using AST and generate a human-readable digest."""
        try:
            tree = ast.parse(code_str, filename=file_path)
        except SyntaxError as e:
            return f"# Failed to parse AST: {e}"

        self.current_file_path = file_path
        self.base_dir = os.path.dirname(os.path.abspath(file_path))
        self.model_layers = []  # Reset
        self.model_connections = {}
        self.layer_order = []
        self.current_class = None
        self.class_layers = {}
        self.class_hierarchy = {}
        self.function_calls = {}  # Reset function calls tracking

        # Add tracking for component instantiations and usage
        self.component_instantiations = {}  # Track where components are instantiated
        self.used_components = set()  # Track which components are actually used

        lines = []
        literal_vars = {}
        image_folders = set()
        default_paths = set()

        visitor = self._create_code_summary_visitor(lines, literal_vars, image_folders, default_paths)
        visitor.visit(tree)

        # Process function calls to update the lines with actual argument values
        self._process_function_calls(lines)

        # Detect datasets used in the code
        dataset_info = self._detect_datasets(tree)
        if dataset_info:
            lines.append(f"Dataset: {dataset_info}")
        else:
            lines.append("Dataset: unknown")

        # [other code remains the same]

        # Generate diagram-ready Model Architecture output
        lines.append("\nModel Architecture:")

        # Identify which components are actually used or instantiated
        self._identify_used_components()

        # Filter to only include components that are actually used
        components_to_show = self._filter_unused_components()

        # Process and output components
        processed_components = set()

        # First output container components
        for class_name in sorted(components_to_show):
            if class_name in processed_components:
                continue

            processed_components.add(class_name)

            # Add component header
            lines.append(f"Component: {class_name}")

            # Get all layers for this class
            class_layer_names = self.class_layers.get(class_name, [])

            # Process layers in order of definition when possible
            ordered_layers = self._get_ordered_layers(class_layer_names)

            # Output each layer in the component (avoiding duplicates)
            self._add_component_layers(lines, ordered_layers, class_name)

        # Add Component Dependencies section
        lines.append("\nComponent Dependencies:")

        # Add dependency information for used components
        for class_name in sorted(components_to_show):
            dependencies = []
            for layer in self.model_layers:
                if layer["class"] == class_name and layer["layer_type"] in components_to_show:
                    if layer["layer_type"] not in dependencies:
                        dependencies.append(layer["layer_type"])

            if dependencies:
                lines.append(f"{class_name} depends on: {', '.join(dependencies)}")

        return '\n'.join(lines)

    def _get_ordered_layers(self, class_layer_names):
        """Get layers in order of definition, avoiding duplicates."""
        # Track processed layers to avoid duplicates
        ordered_layers = []
        processed_layers = set()

        # First try to use layer_order
        for layer_name in self.layer_order:
            if layer_name in class_layer_names and layer_name not in processed_layers:
                ordered_layers.append(layer_name)
                processed_layers.add(layer_name)

        # Add any remaining layers not in layer_order
        for layer_name in class_layer_names:
            if layer_name not in processed_layers:
                ordered_layers.append(layer_name)
                processed_layers.add(layer_name)

        return ordered_layers

    def _add_component_layers(self, lines, ordered_layers, class_name):
        """Add layers from a component to the output lines."""
        # Track processed layers to avoid duplicates
        processed_layers = set()

        for layer_name in ordered_layers:
            if layer_name in processed_layers:
                continue

            processed_layers.add(layer_name)

            # Find the layer definition
            layer = next((l for l in self.model_layers if l["name"] == layer_name), None)
            if layer:
                layer_type = layer.get("layer_type", "Unknown")

                # Only show important layers or layers that reference other components
                if any(important in layer_type for important in
                       self.important_layer_types) or layer_type in self.class_layers:
                    # Format arguments focusing on dimensions
                    dimensions = self._extract_important_dimensions(layer)
                    if dimensions:
                        lines.append(f"  {layer_name}: {layer_type}({dimensions})")
                    else:
                        lines.append(f"  {layer_name}: {layer_type}")

    def _identify_used_components(self):
        """
        Analyze the model layers to determine which components are actually used.
        Components are considered used if:
        1. They are instantiated in another component
        2. They are referenced in a forward method
        3. They are the top-level model class
        """
        # Components instantiated in other components
        for layer in self.model_layers:
            layer_type = layer.get("layer_type", "")
            if layer_type in self.class_layers:
                self.used_components.add(layer_type)

        # Components referenced in forward methods (via function calls)
        for func_name, info in self.function_calls.items():
            if isinstance(info, dict) and func_name == "forward":
                # Check forward method arguments and body for component references
                func_class = None

                # Try to determine which class this forward method belongs to
                for call_id, call_info in self.function_calls.items():
                    if isinstance(call_info, dict) and call_info.get('function') == func_name:
                        # This is a call to the forward method
                        if 'class' in call_info:
                            func_class = call_info['class']
                            break

                if func_class and func_class in self.class_layers:
                    # This forward method belongs to a component
                    self.used_components.add(func_class)

                    # Check if it references other components
                    if 'references' in info:
                        for ref in info['references']:
                            if ref in self.class_layers:
                                self.used_components.add(ref)

        # Look for component instantiations in model creation code
        # This would require more comprehensive analysis of the AST

    def _filter_unused_components(self):
        """
        Filter components to only include those that are actually used.
        Returns the set of components to show in the diagram.
        """
        # If no components are identified as used, show all components
        if not self.used_components:
            # Try to identify the "main" model class as a fallback
            # This is often a class that inherits from nn.Module and has the most layers
            main_component = None
            max_layers = 0

            for class_name, layers in self.class_layers.items():
                if len(layers) > max_layers:
                    max_layers = len(layers)
                    main_component = class_name

            if main_component:
                self.used_components.add(main_component)

                # Also add any components used by this main component
                for layer in self.model_layers:
                    if layer["class"] == main_component and layer["layer_type"] in self.class_layers:
                        self.used_components.add(layer["layer_type"])

        # If still no used components, show all components
        if not self.used_components:
            return set(self.class_layers.keys())

        # Log which components are being filtered out
        unused_components = set(self.class_layers.keys()) - self.used_components
        if unused_components:
            print(f"Filtering out unused components: {', '.join(unused_components)}")

        return self.used_components

    def _build_component_tree(self):
        # component → list of its children (either other components or layer-names)
        tree: Dict[str, List[str]] = {}
        # dims map: layer name → short dimension string
        dims: Dict[str, str] = {}
        # dependency map: component name → list of components it depends on
        dependencies: Dict[str, Set[str]] = {}

        # first, every class is a potential node
        for comp in self.class_layers:
            tree[comp] = []
            dependencies[comp] = set()

        # for each layer we recorded…
        for layer in self.model_layers:
            name = layer["name"]
            ctype = layer["layer_type"]
            cls = layer["class"]

            # 1) if the layer_type is itself a registered component, it's a subcomponent
            if ctype in tree:
                tree[cls].append(ctype)
                # Add dependency relationship
                if cls in dependencies:
                    dependencies[cls].add(ctype)
            else:
                # 2) else it's an atomic leaf
                tree[cls].append(name)
                # pull a dim summary
                dims[name] = self._extract_important_dimensions(layer)

        # Analyze function calls to identify additional dependencies
        for func_name, info in self.function_calls.items():
            if isinstance(info, dict) and info.get('function') and 'args' in info:
                # Look for component names in arguments
                for arg_name, arg_value in info['args'].items():
                    if isinstance(arg_value, str) and arg_value in tree:
                        # Found a component as an argument - likely a dependency
                        for comp_class in tree:
                            if func_name.startswith(f"{comp_class}_"):
                                dependencies[comp_class].add(arg_value)

        # Update tree with all dependencies
        for comp, deps in dependencies.items():
            for dep in deps:
                if dep not in tree[comp]:
                    tree[comp].append(dep)

        # find the "root" model class by heuristic (e.g. the one never used as a subcomponent)
        all_comps = set(tree)
        children = {c for subs in tree.values() for c in subs if c in all_comps}
        roots = list(all_comps - children)
        root = roots[0] if roots else None

        return tree, dims, root

    def _process_function_calls(self, lines):
        """Process function calls and update the function signature lines with actual argument values."""
        # First, find function definition lines in the output (allow leading newlines/spaces)
        function_def_indices = {}
        for i, line in enumerate(lines):
            stripped = line.lstrip('\n').lstrip(' ')
            if stripped.startswith("Function: "):
                m = re.match(r"Function: (\w+)\(.*\)", stripped)
                if m:
                    func_name = m.group(1)
                    function_def_indices[func_name] = i

        # Now update each function definition line with actual argument values
        for func_name, line_idx in function_def_indices.items():
            if func_name not in self.function_calls:
                continue
            function_info = self.function_calls[func_name]

            # Gather all call records for this function
            calls = [
                info for info in self.function_calls.values()
                if isinstance(info, dict) and info.get('function') == func_name
            ]
            if not calls:
                continue

            latest_call = calls[-1]
            actual_args = latest_call.get('args', {})
            default_args = function_info.get('defaults', {})
            omitted_args = latest_call.get('omitted_args', [])

            # Build the new signature
            formatted = []
            for param in function_info.get('params', []):
                if param in actual_args:
                    formatted.append(f"{param}={actual_args[param]}")
                elif param in default_args and param in omitted_args:
                    formatted.append(f"{param}={default_args[param]}")
                else:
                    formatted.append(param)

            # Preserve whatever whitespace/newline prefix was on the original line
            original = lines[line_idx]
            prefix = original[:original.find("Function: ")]
            lines[line_idx] = f"{prefix}Function: {func_name}({', '.join(formatted)})"

            # --- NEW: emit Variable lines with function and call context ---
            # Identify specific call entries for this function
            call_entries = [
                (cid, info)
                for cid, info in self.function_calls.items()
                if isinstance(info, dict) and info.get('function') == func_name
            ]
            if call_entries:
                latest_call_id, latest_call = call_entries[-1]
            else:
                latest_call_id = func_name

            insert_pos = line_idx + 1
            for param in function_info.get('params', []):
                if param in actual_args:
                    val = actual_args[param]
                    lines.insert(
                        insert_pos,
                        f"Variable: {param} = {val} (from call `{latest_call_id}`)"
                    )
                    insert_pos += 1
                elif param in default_args and param in omitted_args:
                    val = default_args[param]
                    lines.insert(
                        insert_pos,
                        f"Variable: {param} = {val} (default for `{func_name}`)"
                    )
                    insert_pos += 1

    def _detect_datasets(self, tree: ast.AST) -> str:
        """Detect datasets used in the code."""
        dataset_visitor = self._create_dataset_visitor()
        dataset_visitor.visit(tree)

        if dataset_visitor.datasets:
            return ", ".join(sorted(dataset_visitor.datasets))
        return ""

    def _create_dataset_visitor(self):
        """Create an AST visitor that looks for dataset usage."""
        return DatasetVisitor()

    def _extract_important_dimensions(self, layer: Dict) -> str:
        """Extract only the important dimensional information from layer arguments."""
        args = layer.get("args", [])
        layer_type = layer.get("layer_type", "")

        # For Sequential layers, try to get first and last layer dimensions
        if layer_type == "Sequential":
            # Check if we have actual layer information in the args
            sequential_dims = self._get_sequential_dimensions(args)
            if sequential_dims:
                return sequential_dims
            return "..."

        # For component types that reference other classes, show that class name
        if layer_type in self.class_hierarchy:
            return layer_type

        # For other layer types, extract dimensions
        dimension_args = []
        for arg in args:
            # Keep numeric values and expressions with numbers
            if isinstance(arg, (int, float)):
                dimension_args.append(str(arg))
            elif isinstance(arg, str) and re.search(r'[0-9]', str(arg)):
                # Clean up expressions
                arg_str = str(arg)
                # Keep typical dimension expressions like "1024 * 4 * 4"
                if "*" in arg_str and re.search(r'[0-9]', arg_str):
                    # Try to extract just the dimension part
                    dimension_args.append(arg_str)
                # For variable names with dimensions, keep them as-is
                elif re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', arg_str) and any(c.isdigit() for c in arg_str):
                    dimension_args.append(arg_str)
                # Extract meaningful dimension variables like latent_dim, embedding_dim, etc.
                elif any(dim_name in arg_str.lower() for dim_name in ["dim", "channel", "feature", "hidden", "unit"]):
                    dimension_args.append(arg_str)
                # Extract numeric parts from other expressions
                else:
                    numeric_parts = re.findall(r'[0-9]+', arg_str)
                    if numeric_parts:
                        dimension_args.append(arg_str)

        # Limit to the most important dimensions
        return ", ".join(dimension_args[:self.max_dims_to_show])

    def _get_sequential_dimensions(self, args) -> str:
        """Try to extract the first and last layer dimensions from a Sequential container."""
        if not args:
            return ""

        input_dim = None
        output_dim = None

        # Look for first Conv or Linear in the sequence to get input dimension
        for arg in args:
            arg_str = str(arg)
            # Find layers like nn.Conv2d(3, 64, ...) or nn.Linear(784, 128, ...)
            match = re.search(r'(?:Conv\w*|Linear)\s*\(\s*(\d+)[^\)]*?(\d+)', arg_str)
            if match:
                input_dim = match.group(1)
                intermediate_dim = match.group(2)
                break

        # Look for last Conv or Linear to get output dimension
        for arg in reversed(args):
            arg_str = str(arg)
            match = re.search(r'(?:Conv\w*|Linear)\s*\(\s*\d+[^\)]*?(\d+)', arg_str)
            if match:
                output_dim = match.group(1)
                break

        # Return dimensions in form "in → out"
        if input_dim and output_dim:
            return f"{input_dim} → {output_dim}"
        elif input_dim:
            return input_dim
        elif output_dim:
            return output_dim

        return ""

    def _eval_constant(self, node: ast.AST) -> Optional[Any]:
        """Resolve simple constants."""
        if isinstance(node, ast.Constant) and isinstance(node.value, (str, int, float, bool)):
            return node.value
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left = self._eval_constant(node.left)
            right = self._eval_constant(node.right)
            if isinstance(left, str) and isinstance(right, str):
                return left + right
        if isinstance(node, ast.JoinedStr):
            parts = []
            for v in node.values:
                if isinstance(v, ast.Constant) and isinstance(v.value, str):
                    parts.append(v.value)
                else:
                    return None
            return "".join(parts)
        return None

    def _determine_folder(self, path_str: str, base_dir: str) -> str:
        """Resolve to an absolute folder."""
        if not os.path.isabs(path_str):
            full_path = os.path.abspath(os.path.join(base_dir, path_str))
        else:
            full_path = os.path.abspath(path_str)
        base, ext = os.path.splitext(full_path)
        if ext:
            return os.path.dirname(full_path) or full_path
        return full_path

    def _create_code_summary_visitor(self, lines: list, literal_vars: dict,
                                     image_folders: set, default_paths: set) -> ast.NodeVisitor:
        return CodeSummaryVisitor(self, lines, literal_vars, image_folders, default_paths)

    @staticmethod
    def analyze_and_visualize_model(python_file_path, output_diagram_path="model_architecture.png",
                                    show_dimensions=True):
        """
        Process a Python file containing neural network code and generate both AST summary and architecture diagram.

        Args:
            python_file_path: Path to the Python file containing the model code
            output_diagram_path: Where to save the generated diagram
            show_dimensions: Whether to show dimension information in the diagram

        Returns:
            Tuple of (AST summary text, diagram output message)
        """
        # Read the Python file
        with open(python_file_path, 'r') as f:
            code = f.read()

        # Generate AST summary
        ast_generator = ASTSummaryGenerator()
        summary = ast_generator.generate_summary(code, python_file_path)

        # Get dependency information for diagram
        tree, dims, root = ast_generator._build_component_tree()

        # Create and save the diagram with dependency information
        diagram_message = draw_model_architecture(
            summary,
            output_diagram_path,
            show_dimensions=show_dimensions,
            component_tree=tree,
            root_component=root
        )

        return summary, diagram_message