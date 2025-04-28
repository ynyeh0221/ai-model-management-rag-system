import ast
from typing import List, Set, Dict, Any


class CodeSummaryVisitor(ast.NodeVisitor):
    def __init__(
            self,
            parent_generator: "ASTSummaryGenerator",
            lines: List[str],
            literal_vars: Dict[str, Any],
            image_folders: Set[str],
            default_paths: Set[str],
    ):
        # store a reference back to the generator
        self.parent_generator = parent_generator
        self.lines = lines
        self.literal_vars = literal_vars
        self.image_folders = image_folders
        self.default_paths = default_paths

        # Dictionary to track function calls and their arguments
        self.call_counter = {}  # Counter for generating unique call IDs
        # Track forward method references to components
        self.current_function = None
        self.function_references = {}

    def visit_Import(self, node):
        names = [alias.name for alias in node.names]
        self.lines.append(f"Import: {', '.join(names)}")

    def visit_ImportFrom(self, node):
        module = node.module or ""
        names = [alias.name for alias in node.names]
        self.lines.append(f"From {module} import {', '.join(names)}")

    def visit_ClassDef(self, node):
        # Extract base classes
        bases = [getattr(b, 'id', getattr(b, 'attr', 'object')) for b in node.bases]
        self.lines.append(f"\nClass: {node.name} (inherits from {', '.join(bases)})")
        doc = ast.get_docstring(node)
        if doc:
            self.lines.append(f"  Docstring: {doc.strip()}")

        # Store class hierarchy information
        self.parent_generator.class_hierarchy[node.name] = bases

        # Process class body for model architecture
        if 'nn.Module' in ' '.join(bases) or 'Module' in ' '.join(bases):
            # This is likely a neural network module
            self.lines.append(f"  Neural Network Module detected")

        # Save previous class and set current class
        previous_class = self.parent_generator.current_class
        self.parent_generator.current_class = node.name

        # Initialize class layers if not exist
        if node.name not in self.parent_generator.class_layers:
            self.parent_generator.class_layers[node.name] = []

        # Visit the class body
        self.generic_visit(node)

        # Restore previous class context
        self.parent_generator.current_class = previous_class

    def visit_FunctionDef(self, node):
        # Save previous function and set current function
        previous_function = self.current_function
        self.current_function = node.name

        # Initialize function references tracking
        if node.name not in self.function_references:
            self.function_references[node.name] = set()

        # Create mapping of parameter names to their default values
        param_defaults = {}
        params = []

        if node.args.args:
            params = [arg.arg for arg in node.args.args]

        # Store default values if they exist
        if node.args.defaults:
            default_offset = len(params) - len(node.args.defaults)
            for i, default_node in enumerate(node.args.defaults):
                param_idx = default_offset + i
                param_name = params[param_idx]
                default_value = self.parent_generator._eval_constant(default_node)
                param_defaults[param_name] = default_value

        # Store function signature information in the parent's function_calls dict
        self.parent_generator.function_calls[node.name] = {
            'params': params,
            'defaults': param_defaults
        }

        # Format args for display (without showing default values)
        self.lines.append(f"\nFunction: {node.name}({', '.join(params)})")

        # Check for default paths in arguments
        if node.args.defaults:
            for name_node, default_node in zip(params[-len(node.args.defaults):], node.args.defaults):
                if isinstance(name_node, str):
                    lname = name_node.lower()
                else:
                    lname = name_node.lower() if hasattr(name_node, 'lower') else str(name_node).lower()

                if any(k in lname for k in ('save_dir', 'results_dir', 'save_path', 'output_dir')):
                    val = self.parent_generator._eval_constant(default_node)
                    if isinstance(val, str):
                        self.default_paths.add(self.parent_generator._determine_folder(val, self.parent_generator.base_dir))

        # Check if this is forward() method to understand connections and track component usage
        if node.name == 'forward':
            # Store the class context for this forward method
            if self.parent_generator.current_class:
                if 'class' not in self.parent_generator.function_calls[node.name]:
                    self.parent_generator.function_calls[node.name]['class'] = self.parent_generator.current_class

            # Traditional forward method analysis
            self._analyze_forward_method(node)

            # Add the component that contains this forward method to used components
            if self.parent_generator.current_class:
                self.parent_generator.used_components.add(self.parent_generator.current_class)

        doc = ast.get_docstring(node)
        if doc:
            self.lines.append(f"  Docstring: {doc.strip()}")

        # Visit the function body
        self.generic_visit(node)

        # Restore previous function context
        self.current_function = previous_function

    def _analyze_forward_method(self, node):
        """Extract model flow from forward method and track component references"""
        references = set()
        connections = {}

        prev_output_var = None

        # Process each statement in the forward method
        for stmt in node.body:
            # Handle x = self.layer(x) pattern
            if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
                call_node = stmt.value
                if isinstance(call_node.func, ast.Attribute) and \
                        isinstance(call_node.func.value, ast.Name) and \
                        call_node.func.value.id == 'self':
                    # This is a call to self.layer
                    layer_name = call_node.func.attr

                    # Track connections for future
                    if isinstance(stmt.targets[0], ast.Name):
                        output_var = stmt.targets[0].id

                        # Track connection from previous layer if available
                        if prev_output_var:
                            connections[prev_output_var] = layer_name

                        prev_output_var = output_var

                    # Check if this layer corresponds to a component
                    for layer in self.parent_generator.model_layers:
                        if layer['name'] == layer_name and layer['class'] == self.parent_generator.current_class:
                            layer_type = layer['layer_type']
                            if layer_type in self.parent_generator.class_layers:
                                # This layer is a component instance
                                references.add(layer_type)
                                self.parent_generator.used_components.add(layer_type)

            # Also look for return statements that use components
            elif isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Call):
                self._process_call_for_references(stmt.value, references)

        # Store the references found in the forward method
        if references:
            if 'references' not in self.parent_generator.function_calls[self.current_function]:
                self.parent_generator.function_calls[self.current_function]['references'] = set()
            self.parent_generator.function_calls[self.current_function]['references'].update(references)

        # Store the connections for model flow visualization
        if connections:
            self.parent_generator.model_connections[self.parent_generator.current_class] = connections

    def _process_call_for_references(self, call_node, references):
        """Process a call node to find component references"""
        if isinstance(call_node.func, ast.Attribute) and \
                isinstance(call_node.func.value, ast.Name) and \
                call_node.func.value.id == 'self':
            # This is a call to self.layer
            layer_name = call_node.func.attr

            # Check if this layer corresponds to a component
            for layer in self.parent_generator.model_layers:
                if layer['name'] == layer_name and layer['class'] == self.parent_generator.current_class:
                    layer_type = layer['layer_type']
                    if layer_type in self.parent_generator.class_layers:
                        # This layer is a component instance
                        references.add(layer_type)
                        self.parent_generator.used_components.add(layer_type)

        # Process arguments recursively
        for arg in call_node.args:
            if isinstance(arg, ast.Call):
                self._process_call_for_references(arg, references)

        for keyword in call_node.keywords:
            if isinstance(keyword.value, ast.Call):
                self._process_call_for_references(keyword.value, references)

    def visit_Call(self, node):
        """Track function calls, extract args, and also pull out keyword literals."""
        # Determine function name
        func_name = None
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr

        # If we're tracking this function, capture the call
        if func_name and func_name in self.parent_generator.function_calls:
            func_info = self.parent_generator.function_calls[func_name]
            # Unique ID
            self.call_counter.setdefault(func_name, 0)
            self.call_counter[func_name] += 1
            call_id = f"{func_name}_call_{self.call_counter[func_name]}"

            # Positional args
            actual_args = {}
            provided = set()
            for i, arg in enumerate(node.args):
                if i < len(func_info['params']):
                    p = func_info['params'][i]
                    provided.add(p)
                    val = self.parent_generator._eval_constant(arg)
                    if val is not None:
                        actual_args[p] = val

            # Keyword args
            for kw in node.keywords:
                if kw.arg:
                    provided.add(kw.arg)
                    val = self.parent_generator._eval_constant(kw.value)
                    if val is not None:
                        actual_args[kw.arg] = val
                        # *** Record as a literal var ***
                        self.literal_vars[kw.arg] = val

            # Which defaults were actually used?
            omitted = [
                p for p in func_info.get('params', [])
                if p not in provided and p in func_info.get('defaults', {})
            ]

            self.parent_generator.function_calls[call_id] = {
                'function': func_name,
                'args': actual_args,
                'omitted_args': omitted
            }

            # Track call hierarchy
            if self.current_function:
                if 'called_by' not in self.parent_generator.function_calls[func_name]:
                    self.parent_generator.function_calls[func_name]['called_by'] = set()
                self.parent_generator.function_calls[func_name]['called_by'].add(self.current_function)

        # Check for component instantiations
        class_name = None
        if isinstance(node.func, ast.Name):
            class_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            class_name = node.func.attr

        if class_name and class_name in self.parent_generator.class_layers:
            self.parent_generator.used_components.add(class_name)
            if self.current_function:
                self.function_references.setdefault(self.current_function, set()).add(class_name)
                if self.parent_generator.current_class:
                    self.parent_generator.used_components.add(self.parent_generator.current_class)

        # Always continue walking
        self.generic_visit(node)

    def visit_Assign(self, node):
        """Handle assignments, including DataLoader batch_size capture."""
        # Detect DataLoader instantiation
        if isinstance(node.value, ast.Call):
            call_func = node.value.func
            func_name = None
            if isinstance(call_func, ast.Name):
                func_name = call_func.id
            elif isinstance(call_func, ast.Attribute):
                func_name = call_func.attr

            if func_name == 'DataLoader':
                # Extract batch_size
                batch_size = None
                for kw in node.value.keywords:
                    if kw.arg == 'batch_size':
                        val = self.parent_generator._eval_constant(kw.value)
                        if val is not None:
                            batch_size = val
                if batch_size is None and len(node.value.args) >= 2:
                    val = self.parent_generator._eval_constant(node.value.args[1])
                    if isinstance(val, int):
                        batch_size = val

                # Get the loader variable name
                loader_name = None
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name):
                        loader_name = tgt.id
                        break

                if batch_size is not None and loader_name:
                    self.literal_vars['batch_size'] = batch_size
                    self.lines.append(
                        f"Variable: batch_size = {batch_size} (from DataLoader {loader_name})"
                    )

        # Handle simple variables
        val = self.parent_generator._eval_constant(node.value)
        for tgt in node.targets:
            if isinstance(tgt, ast.Name):
                name = tgt.id
                key = name.lower()
                if val is not None:
                    if any(k in key for k in ('dir', 'path', 'folder')):
                        self.literal_vars[name] = val
                    if any(k in key for k in ('batch', 'lr', 'epoch', 'device', 'seed')):
                        self.literal_vars[name] = val

        # Extract model layers (self.xxx = Layer(args))
        if isinstance(node.value, ast.Call):
            call_func = node.value.func

            for tgt in node.targets:
                if isinstance(tgt, ast.Attribute) and \
                        isinstance(tgt.value, ast.Name) and \
                        tgt.value.id == "self":
                    # It's a model layer assignment
                    layer_name = tgt.attr

                    # Track layer order
                    if layer_name not in self.parent_generator.layer_order:
                        self.parent_generator.layer_order.append(layer_name)

                    # Get layer type
                    try:
                        if isinstance(call_func, ast.Name):
                            layer_type = call_func.id
                        elif isinstance(call_func, ast.Attribute):
                            # Handle cases like nn.Linear, torch.nn.Linear
                            layer_type = call_func.attr
                        else:
                            layer_type = ast.unparse(call_func).split('.')[-1]
                    except:
                        layer_type = "Unknown"

                    # Extract arguments
                    args_list = []
                    for arg in node.value.args:
                        try:
                            # Better argument extraction
                            if isinstance(arg, ast.Constant):
                                args_list.append(arg.value)
                            elif isinstance(arg, ast.Name):
                                args_list.append(arg.id)  # Variable name
                            else:
                                # More complex expressions
                                args_list.append(ast.unparse(arg))
                        except:
                            args_list.append("?")

                    # Handle keyword arguments too
                    kwargs = {}
                    for kwarg in node.value.keywords:
                        try:
                            kwargs[kwarg.arg] = ast.unparse(kwarg.value)
                        except:
                            kwargs[kwarg.arg] = "?"

                    # Store the layer info
                    layer_info = {
                        "name": layer_name,
                        "layer_type": layer_type,
                        "args": args_list,
                        "kwargs": kwargs,
                        "class": self.parent_generator.current_class
                    }
                    self.parent_generator.model_layers.append(layer_info)

                    # Add to class-specific layer list
                    if self.parent_generator.current_class:
                        if self.parent_generator.current_class not in self.parent_generator.class_layers:
                            self.parent_generator.class_layers[self.parent_generator.current_class] = []
                        if layer_name not in self.parent_generator.class_layers[self.parent_generator.current_class]:
                            self.parent_generator.class_layers[self.parent_generator.current_class].append(layer_name)

                    # Check if the layer type is a component we're tracking
                    if layer_type in self.parent_generator.class_layers:
                        # This indicates that the component is being used
                        self.parent_generator.used_components.add(layer_type)
                        # The class containing this layer also gets marked as used
                        if self.parent_generator.current_class:
                            self.parent_generator.used_components.add(self.parent_generator.current_class)

        # Look for Sequential container definitions
        if isinstance(node.value, ast.Call) and \
                ((isinstance(node.value.func, ast.Name) and node.value.func.id == 'Sequential') or \
                 (isinstance(node.value.func, ast.Attribute) and node.value.func.attr == 'Sequential')):
            for tgt in node.targets:
                if isinstance(tgt, ast.Attribute) and \
                        isinstance(tgt.value, ast.Name) and \
                        tgt.value.id == "self":
                    # It's a Sequential container
                    seq_name = tgt.attr

                    # Track layer order
                    if seq_name not in self.parent_generator.layer_order:
                        self.parent_generator.layer_order.append(seq_name)

                    # Extract Sequential contents
                    seq_layers = []
                    for arg in node.value.args:
                        if isinstance(arg, ast.List) or isinstance(arg, ast.Tuple):
                            for elt in arg.elts:
                                seq_layer_str = ast.unparse(elt)
                                seq_layers.append(seq_layer_str)

                                # Check if any sequential element is a component
                                for comp_name in self.parent_generator.class_layers:
                                    if comp_name in seq_layer_str:
                                        # This component is used in a Sequential
                                        self.parent_generator.used_components.add(comp_name)
                                        # The class containing this Sequential is also used
                                        if self.parent_generator.current_class:
                                            self.parent_generator.used_components.add(self.parent_generator.current_class)

                    # Store the layer info
                    layer_info = {
                        "name": seq_name,
                        "layer_type": "Sequential",
                        "args": seq_layers,
                        "kwargs": {},
                        "class": self.parent_generator.current_class
                    }
                    self.parent_generator.model_layers.append(layer_info)

                    # Add to class-specific layer list
                    if self.parent_generator.current_class:
                        if self.parent_generator.current_class not in self.parent_generator.class_layers:
                            self.parent_generator.class_layers[self.parent_generator.current_class] = []
                        if seq_name not in self.parent_generator.class_layers[self.parent_generator.current_class]:
                            self.parent_generator.class_layers[self.parent_generator.current_class].append(seq_name)

        try:
            expr = ast.unparse(node.value) if hasattr(ast, 'unparse') else '<expr>'
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and any(k in tgt.id.lower() for k in
                                                     ('optimizer', 'model', 'train')):
                    self.lines.append(f"Variable: {tgt.id} = {expr}")

                    # Check if this is a model instantiation
                    if isinstance(node.value, ast.Call):
                        class_name = None
                        if isinstance(node.value.func, ast.Name):
                            class_name = node.value.func.id
                        elif isinstance(node.value.func, ast.Attribute):
                            class_name = node.value.func.attr

                        # If this is a tracked component class, mark as used
                        if class_name and class_name in self.parent_generator.class_layers:
                            self.parent_generator.used_components.add(class_name)
                            print(f"Found top-level model instantiation: {tgt.id} = {class_name}")
        except Exception:
            pass

        self.generic_visit(node)

    def visit_For(self, node):
        """Detect training loop epochs like 'for epoch in range(num_epochs):'"""
        # Check for range-based loops
        if isinstance(node.iter, ast.Call) and isinstance(node.iter.func,
                                                          ast.Name) and node.iter.func.id == 'range':
            args = node.iter.args
            epoch_count = None
            # Single-argument range: range(N) or range(var)
            if len(args) == 1:
                arg0 = args[0]
                # Constant literal
                epoch_count = self.parent_generator._eval_constant(arg0)
                # Named literal from literal_vars
                if epoch_count is None and isinstance(arg0, ast.Name):
                    epoch_count = self.literal_vars.get(arg0.id)
            # Two-argument range: range(start, stop)
            elif len(args) >= 2:
                arg1 = args[1]
                epoch_count = self.parent_generator._eval_constant(arg1)
                if epoch_count is None and isinstance(arg1, ast.Name):
                    epoch_count = self.literal_vars.get(arg1.id)
            if epoch_count is not None:
                self.literal_vars['num_epochs'] = epoch_count
                self.lines.append(f"Variable: num_epochs = {epoch_count} (from training loop)")
        # Continue generic traversal
        self.generic_visit(node)