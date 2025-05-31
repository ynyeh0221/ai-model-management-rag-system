import ast
from typing import List, Set, Dict, Any, TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    from src.core.content_analyzer.model_script.ast_summary_generator import ASTSummaryGenerator


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

        # Counters and context tracking
        self.call_counter: Dict[str, int] = {}
        self.current_function: Optional[str] = None
        self.function_references: Dict[str, Set[str]] = {}

    # ---------------------- Visitor Methods ----------------------
    def visit_Import(self, node: ast.Import):
        names = [alias.name for alias in node.names]
        self.lines.append(f"Import: {', '.join(names)}")

    def visit_ImportFrom(self, node: ast.ImportFrom):
        module = node.module or ""
        names = [alias.name for alias in node.names]
        self.lines.append(f"From {module} import {', '.join(names)}")

    def visit_ClassDef(self, node: ast.ClassDef):
        bases = [getattr(b, 'id', getattr(b, 'attr', 'object')) for b in node.bases]
        self.lines.append(f"\nClass: {node.name} (inherits from {', '.join(bases)})")
        doc = ast.get_docstring(node)
        if doc:
            self.lines.append(f"  Docstring: {doc.strip()}")

        # Track hierarchy and detect nn.Module
        self.parent_generator.class_hierarchy[node.name] = bases
        if 'nn.Module' in ' '.join(bases) or 'Module' in ' '.join(bases):
            self.lines.append("  Neural Network Module detected")

        prev_class = self.parent_generator.current_class
        self.parent_generator.current_class = node.name
        self.parent_generator.class_layers.setdefault(node.name, [])

        self.generic_visit(node)
        self.parent_generator.current_class = prev_class

    def visit_FunctionDef(self, node: ast.FunctionDef):
        prev_function = self.current_function
        self.current_function = node.name
        self.function_references.setdefault(node.name, set())

        # 1) Record the signature (params + defaults) and add a line to self.lines
        self._record_signature(node)

        # 2) Detect any default-path arguments and store them
        self._detect_default_paths(node)

        # 3) If this is a `forward` method, attach class info and analyze it
        if node.name == "forward":
            self._handle_forward(node)

        # 4) Append docstring (if present) to self.lines
        self._record_docstring(node)

        # Recurse into child nodes
        self.generic_visit(node)

        # Restore previous function context
        self.current_function = prev_function

    def _record_signature(self, node: ast.FunctionDef) -> None:
        """
        Gather parameter names and default values, store them in
        parent_generator.function_calls, and append a "Function: ..." line.
        """
        # Collect all parameter names (positional-only and normal args)
        params = [arg.arg for arg in node.args.args] if node.args.args else []
        defaults: dict[str, object] = {}

        # If there are default values, map them to the correct param name
        if node.args.defaults:
            offset = len(params) - len(node.args.defaults)
            for i, default_node in enumerate(node.args.defaults):
                param_name = params[offset + i]
                defaults[param_name] = self.parent_generator._eval_constant(default_node)

        # Save into parent_generator
        self.parent_generator.function_calls[node.name] = {
            "params": params,
            "defaults": defaults,
        }

        # Add a line like: "\nFunction: foo(arg1, arg2, ...)"
        self.lines.append(f"\nFunction: {node.name}({', '.join(params)})")

    def _detect_default_paths(self, node: ast.FunctionDef) -> None:
        """
        Look at any default-valued arguments on this function. If the
        parameter name suggests it's a path (e.g. 'save_dir', 'output_dir',
        etc.), evaluate the constant and store its folder via
        parent_generator._determine_folder(...).
        """
        # If there are no defaults, nothing to do
        if not node.args.defaults:
            return

        # Reconstruct a param list again (could cache, but kept simple here)
        params = [arg.arg for arg in node.args.args]

        # Only consider the last len(defaults) parameters
        num_defaults = len(node.args.defaults)
        for name, default_node in zip(params[-num_defaults:], node.args.defaults):
            lower_name = name.lower()
            # Check if the param name contains any of these path-like keywords
            if any(key in lower_name for key in ("save_dir", "results_dir", "save_path", "output_dir")):
                val = self.parent_generator._eval_constant(default_node)
                if isinstance(val, str):
                    folder = self.parent_generator._determine_folder(val, self.parent_generator.base_dir)
                    self.default_paths.add(folder)

    def _handle_forward(self, node: ast.FunctionDef) -> None:
        """
        If this function is named 'forward', attach the current class (if any)
        into parent_generator.function_calls[node.name]['class'], then call
        _analyze_forward_method, and finally record that this class was used.
        """
        cls = self.parent_generator.current_class
        if cls:
            # Ensure that function_calls[forward]['class'] = current_class
            self.parent_generator.function_calls[node.name].setdefault("class", cls)

        # Perform whatever additional analysis is needed for the forward method
        self._analyze_forward_method(node)

        # Mark that this class was used in a forward invocation
        if cls:
            self.parent_generator.used_components.add(cls)

    def _record_docstring(self, node: ast.FunctionDef) -> None:
        """
        If there’s a docstring, append it (indented) to self.lines.
        """
        doc = ast.get_docstring(node)
        if doc:
            self.lines.append(f"  Docstring: {doc.strip()}")

    def visit_Call(self, node: ast.Call):
        # 1) Track calls to known functions
        func_name = self._get_callable_name(node.func)
        if func_name and func_name in self.parent_generator.function_calls:
            self._process_function_call(node, func_name)

        # 2) Track component instantiation
        cls_name = self._get_callable_name(node.func)
        if cls_name and cls_name in self.parent_generator.class_layers:
            self._process_component_instantiation(cls_name)

        # 3) Continue recursion
        self.generic_visit(node)

    def _get_callable_name(self, func_node: ast.expr) -> Union[str, None]:
        """
        Given node.func, return the function/attribute name if it's a simple Name
        or Attribute, else return None.
        """
        if isinstance(func_node, ast.Name):
            return func_node.id
        if isinstance(func_node, ast.Attribute):
            return func_node.attr
        return None

    def _process_function_call(self, node: ast.Call, func_name: str) -> None:
        """
        When calling a function we already know about (in function_calls):
          1) Increment and record the call counter → call_id
          2) Build a mapping from parameter names to literal values (args_map)
             and track which params were provided
          3) Compute any omitted default parameters
          4) Store a new entry in parent_generator.function_calls for this specific call
          5) Update the 'called_by' set on the original function info
        """
        # 1) Bump the counter and build a unique call_id
        count = self.call_counter.setdefault(func_name, 0) + 1
        self.call_counter[func_name] = count
        call_id = f"{func_name}_call_{count}"

        info = self.parent_generator.function_calls[func_name]

        # 2) Gather literal arguments from positional and keyword args
        args_map, provided = self._gather_call_arguments(node, info)

        # 3) Compute which defaulted params were omitted
        omitted = self._compute_omitted_params(info, provided)

        # 4) Store this particular call’s details under its call_id
        self.parent_generator.function_calls[call_id] = {
            "function": func_name,
            "args": args_map,
            "omitted_args": omitted,
        }

        # 5) Record that func_name was called by current_function (if set)
        if self.current_function:
            info.setdefault("called_by", set()).add(self.current_function)

    def _gather_call_arguments(
        self, node: ast.Call, info: dict[str, object]
    ) -> tuple[dict[str, object], set[str]]:
        """
        Loop over node.args (positional) and node.keywords to extract literal values.
        Returns (args_map, provided_set), where:
          - args_map maps param_name → literal_value
          - provided_set is the set of param_names we actually got values for
        """
        args_map: dict[str, object] = {}
        provided: set[str] = set()

        # 2a) Positional args
        for i, arg_node in enumerate(node.args):
            if i < len(info["params"]):
                param_name = info["params"][i]
                val = self.parent_generator._eval_constant(arg_node)
                if val is not None:
                    args_map[param_name] = val
                    provided.add(param_name)

        # 2b) Keyword args
        for kw in node.keywords:
            if kw.arg:
                val = self.parent_generator._eval_constant(kw.value)
                if val is not None:
                    args_map[kw.arg] = val
                    provided.add(kw.arg)
                    # Also record in literal_vars for later use
                    self.literal_vars[kw.arg] = val

        return args_map, provided

    def _compute_omitted_params(
        self, info: dict[str, object], provided: set[str]
    ) -> list[str]:
        """
        Given the original function info and which params were provided,
        return a list of any params that have defaults but were NOT provided.
        """
        omitted: list[str] = []
        defaults_dict = info.get("defaults", {})
        for param in info["params"]:
            if param not in provided and param in defaults_dict:
                omitted.append(param)
        return omitted

    def _process_component_instantiation(self, cls_name: str) -> None:
        """
        When we see a call to a known class (in class_layers):
          1) Mark cls_name as used in parent_generator.used_components
          2) If inside a function, add cls_name to function_references[current_function]
          3) If inside a class, also mark that class as used
        """
        self.parent_generator.used_components.add(cls_name)

        if self.current_function:
            self.function_references.setdefault(self.current_function, set()).add(cls_name)
            if self.parent_generator.current_class:
                self.parent_generator.used_components.add(self.parent_generator.current_class)

    def visit_Assign(self, node: ast.Assign):
        # Break out different assignment handlers
        self._handle_dataloader(node)
        self._handle_simple_assignment(node)
        self._handle_layer_assignment(node)
        self._handle_sequential_assignment(node)
        self._handle_top_level_model_instantiation(node)
        self.generic_visit(node)

    def visit_For(self, node: ast.For):
        """
        If this is a `for … in range(...)` loop, try to extract the number of epochs
        from the range arguments and record it as a literal variable called 'num_epochs'.
        """
        if self._is_range_loop(node.iter):
            count = self._extract_range_count(node.iter.args)
            if isinstance(count, int):
                # Record num_epochs and append a descriptive line
                self.literal_vars['num_epochs'] = count
                self.lines.append(f"Variable: num_epochs = {count} (from training loop)")

        # Continue traversing inside the loop body
        self.generic_visit(node)

    def _is_range_loop(self, iter_expr: ast.expr) -> bool:
        """
        Return True if iter_expr is a Call to range(…).
        """
        if not isinstance(iter_expr, ast.Call):
            return False

        func = iter_expr.func
        return (
            isinstance(func, ast.Name)
            and func.id == "range"
        )

    def _extract_range_count(self, args: list[ast.expr]) -> Union[int, str, None]:
        """
        Given the list of arguments to range(), return:
          - The evaluated constant (if literal)
          - Or a variable name (if args[...] is a Name whose value appears in literal_vars)
          - Otherwise None.
        Priority:
          - If len(args) == 1, use args[0]
          - If len(args) >= 2, use args[1]
        """
        if len(args) == 1:
            return self._resolve_arg_value(args[0])
        elif len(args) >= 2:
            return self._resolve_arg_value(args[1])
        return None

    def _resolve_arg_value(self, arg_node: ast.expr) -> Union[int, str, None]:
        """
        1) Try to evaluate arg_node as a constant via parent_generator._eval_constant.
        2) If that returns None and arg_node is a Name, return the name string.
        3) If it’s a name string, look up in literal_vars; otherwise, return the raw constant.
        """
        # 1) Attempt to eval as a literal (e.g. range(10) → 10)
        literal_val = self.parent_generator._eval_constant(arg_node)
        if literal_val is not None:
            return literal_val

        # 2) If it’s a variable name, return that name
        if isinstance(arg_node, ast.Name):
            var_name = arg_node.id
            # 3) If we already have a literal mapping for that var_name, return it
            return self.literal_vars.get(var_name, var_name)

        return None

    # ---------------------- Helper Methods ------------------------
    def _analyze_forward_method(self, node: ast.FunctionDef):
        """
        Traverse the body of a forward(...) method to:
          1. Track connections between consecutive layer outputs (conns).
          2. Collect any layer-type references (refs).
          3. Update parent_generator.function_calls and model_connections accordingly.
        """
        refs: set[str] = set()
        conns: dict[str, str] = {}
        prev_out: Union[str, None] = None

        for stmt in node.body:
            if self._is_self_layer_call(stmt):
                prev_out = self._process_forward_assign(stmt, refs, conns, prev_out)
            elif self._is_return_call(stmt):
                self._process_forward_return(stmt, refs)

        if refs:
            self._update_function_references(node.name, refs)
        if conns:
            self._update_model_connections(conns)

    def _is_self_layer_call(self, stmt: ast.stmt) -> bool:
        """
        Return True if stmt is an Assign whose value is a Call to self.<layer>(...).
        """
        if not (isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call)):
            return False

        call = stmt.value
        func = call.func
        return (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "self"
        )

    def _process_forward_assign(
        self,
        stmt: ast.Assign,
        refs: set[str],
        conns: dict[str, str],
        prev_out: Union[str, None]
    ) -> Union[str, None]:
        """
        Handle a single assignment of the form:
            some_name = self.<layer>(...)
        Steps:
          1) Extract layer name from the call.
          2) If the target is a Name, link prev_out → layer in connections.
          3) Update prev_out to the new output name.
          4) Scan model_layers for that layer, and add any matching layer_types to ref.
        Returns the updated prev_out (or None).
        """
        call = stmt.value
        layer_name = call.func.attr  # because we already checked func is self.<layer>
        new_prev_out: Union[str, None] = prev_out

        # 1) If the assigned target is a simple Name, record its id and connect from prev_out
        target = stmt.targets[0]
        if isinstance(target, ast.Name):
            out_name = target.id
            if prev_out:
                conns[prev_out] = layer_name
            new_prev_out = out_name

        # 2) Scan through all model_layers looking for this layer in the current class
        self._scan_model_layers_for_layer(layer_name, refs)

        return new_prev_out

    def _scan_model_layers_for_layer(self, layer_name: str, refs: set[str]) -> None:
        """
        Look through parent_generator.model_layers. For any entry l where
          l['name'] == layer_name and l['class'] == current_class,
        grab its layer_type (lt). If lt is a key in class_layers, add lt to:
          1) refs
          2) used_components
        """
        current_cls = self.parent_generator.current_class
        for layer_info in self.parent_generator.model_layers:
            if (
                layer_info["name"] == layer_name
                and layer_info["class"] == current_cls
            ):
                lt = layer_info["layer_type"]
                if lt in self.parent_generator.class_layers:
                    refs.add(lt)
                    self.parent_generator.used_components.add(lt)

    def _is_return_call(self, stmt: ast.stmt) -> bool:
        """
        Return True if stmt is a Return whose value is a Call.
        """
        return (
            isinstance(stmt, ast.Return)
            and isinstance(stmt.value, ast.Call)
        )

    def _process_forward_return(self, stmt: ast.Return, refs: set[str]) -> None:
        """
        When seeing `return self.some_layer(...)` (or any return of a Call),
        pass the Call node to _process_call_for_references, accumulating into refs.
        """
        call_node = stmt.value  # type: ignore
        self._process_call_for_references(call_node, refs)

    def _update_function_references(self, func_name: str, refs: set[str]) -> None:
        """
        Ensure parent_generator.function_calls[func_name]['references'] includes all the refs.
        """
        fc = self.parent_generator.function_calls.setdefault(func_name, {})
        fc.setdefault("references", set()).update(refs)

    def _update_model_connections(self, conns: dict[str, str]) -> None:
        """
        Assign conns into parent_generator.model_connections[current_class] if current_class is set.
        """
        cls = self.parent_generator.current_class
        if cls:
            self.parent_generator.model_connections[cls] = conns

    def _process_call_for_references(self, call: ast.Call, refs: Set[str]):
        """
        1) If this call is `self.<layer>(...)`, scan model_layers for matching entries
           and add their layer_type to refs/used_components.
        2) Recursively process any nested Call nodes in args or keywords.
        """
        # Step 1: Handle top-level “self.<layer>” calls
        self._handle_self_layer_call(call, refs)

        # Step 2: Find any nested ast.Call nodes and recurse
        for nested in self._extract_nested_calls(call):
            self._process_call_for_references(nested, refs)

    def _handle_self_layer_call(self, call: ast.Call, refs: Set[str]) -> None:
        """
        If call.func is an Attribute on self (i.e., self.<layer>), check each entry
        in parent_generator.model_layers. For any entry where name==layer and class==current_class,
        if its layer_type is in class_layers, add that layer_type to refs and used_components.
        """
        func = call.func
        if not (isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id == "self"):
            return

        layer_name = func.attr
        self._scan_model_layers_for_layer(layer_name, refs)

    def _scan_model_layers_for_layer(self, layer_name: str, refs: Set[str]) -> None:
        """
        Iterate over parent_generator.model_layers. For each entry l where
            l['name'] == layer_name and l['class'] == current_class,
        if l['layer_type'] is present in class_layers, add it to refs and used_components.
        """
        current_cls = self.parent_generator.current_class
        for layer_info in self.parent_generator.model_layers:
            if (layer_info["name"] == layer_name
                    and layer_info["class"] == current_cls):
                layer_type = layer_info["layer_type"]
                if layer_type in self.parent_generator.class_layers:
                    refs.add(layer_type)
                    self.parent_generator.used_components.add(layer_type)

    def _extract_nested_calls(self, call_node: ast.Call) -> list[ast.Call]:
        """
        Return a list of any ast.Call nodes found in call_node.args or call_node.keywords.
        """
        nested: list[ast.Call] = []
        for arg in call_node.args:
            if isinstance(arg, ast.Call):
                nested.append(arg)
        for kw in call_node.keywords:
            if isinstance(kw.value, ast.Call):
                nested.append(kw.value)
        return nested

    def _handle_dataloader(self, node: ast.Assign):
        """
        If this assignment is `var = DataLoader(...)`, attempt to
        extract the batch_size (either from keywords or positional args).
        If found, store it in literal_vars['batch_size'] and append a line.
        """
        # 1) Must be a Call node
        if not isinstance(node.value, ast.Call):
            return

        # 2) Get the function name; if it isn’t 'DataLoader', bail out
        func_name = self._get_callable_name(node.value.func)
        if func_name != "DataLoader":
            return

        # 3) Extract batch_size (None if neither keyword nor positional provided)
        batch_size = self._extract_dataloader_batch_size(node.value)
        if batch_size is None:
            return

        # 4) Find the target variable name (first simple Name in node.targets)
        var_name = self._extract_assignment_name(node.targets)
        if var_name is None:
            return

        # 5) Record and append the "Variable: batch_size=..." line
        self.literal_vars["batch_size"] = batch_size
        self.lines.append(
            f"Variable: batch_size = {batch_size} (from DataLoader {var_name})"
        )

    def _get_callable_name(self, func_node: ast.expr) -> Union[str, None]:
        """
        Return the simple name of a Call target:
          - If func_node is ast.Name, return its .id
          - If func_node is ast.Attribute, return its .attr
          - Otherwise return None
        """
        if isinstance(func_node, ast.Name):
            return func_node.id
        if isinstance(func_node, ast.Attribute):
            return func_node.attr
        return None

    def _extract_dataloader_batch_size(self, call_node: ast.Call) -> Union[int, None]:
        """
        Look for batch_size in call_node.keywords first.
        If not found and there are ≥2 positional args, evaluate args[1].
        Return the integer value or None.
        """
        # 1) Check keywords
        for kw in call_node.keywords:
            if kw.arg == "batch_size":
                val = self.parent_generator._eval_constant(kw.value)
                if isinstance(val, int):
                    return val

        # 2) If no keyword, check the second positional argument (if it exists)
        if len(call_node.args) >= 2:
            val = self.parent_generator._eval_constant(call_node.args[1])
            if isinstance(val, int):
                return val

        return None

    def _extract_assignment_name(self, targets: list[ast.expr]) -> Union[str, None]:
        """
        Find the first target that is a plain ast.Name,
        and return its .id. Otherwise, return None.
        """
        for tgt in targets:
            if isinstance(tgt, ast.Name):
                return tgt.id
        return None

    def _handle_simple_assignment(self, node: ast.Assign):
        val = self.parent_generator._eval_constant(node.value)
        for t in node.targets:
            if isinstance(t, ast.Name) and val is not None:
                key = t.id.lower()
                if any(k in key for k in ('dir', 'path', 'folder')):
                    self.literal_vars[t.id] = val
                if any(k in key for k in ('batch', 'lr', 'epoch', 'device', 'seed')):
                    self.literal_vars[t.id] = val

    def _handle_layer_assignment(self, node: ast.Assign):
        """
        If `node.value` is a Call (e.g., nn.Conv2d(…)), iterate over targets.
        For any `self.<name> = <Call>`, record the layer in layer_order,
        build its info dict (with layer_type, args, kwargs), append it to
        model_layers, register it under class_layers, and update used_components.
        """
        if not isinstance(node.value, ast.Call):
            return

        call_node = node.value
        for tgt in node.targets:
            self._process_layer_target(tgt, call_node)

    def _process_layer_target(self, tgt: ast.expr, call_node: ast.Call) -> None:
        """
        Handle a single target in the assignment. If it's of the form
        `self.<layer_name>`, perform all the necessary registrations.
        """
        if not (
            isinstance(tgt, ast.Attribute)
            and isinstance(tgt.value, ast.Name)
            and tgt.value.id == "self"
        ):
            return

        layer_name = tgt.attr

        # 1) Ensure layer_name appears in layer_order
        self._add_to_layer_order(layer_name)

        # 2) Determine the layer_type (e.g., 'Conv2d' or fallback to 'Unknown')
        layer_type = self._determine_layer_type(call_node)

        # 3) Gather positional arguments into a Python list
        args_list = self._gather_layer_args(call_node.args)

        # 4) Gather keyword arguments into a dict
        kwargs_dict = self._gather_layer_kwargs(call_node.keywords)

        # 5) Build the info dict and append to model_layers
        info = {
            "name": layer_name,
            "layer_type": layer_type,
            "args": args_list,
            "kwargs": kwargs_dict,
            "class": self.parent_generator.current_class,
        }
        self.parent_generator.model_layers.append(info)

        # 6) Register this layer under the current class (if any)
        self._register_class_layer(layer_name)

        # 7) If the layer_type itself is in class_layers, mark it (and the class) as used
        self._mark_used_components_for_layer(layer_type)

    def _add_to_layer_order(self, layer_name: str) -> None:
        """
        If `layer_name` is not already in parent_generator.layer_order, append it.
        """
        if layer_name not in self.parent_generator.layer_order:
            self.parent_generator.layer_order.append(layer_name)

    def _determine_layer_type(self, call_node: ast.Call) -> str:
        """
        Extract the 'layer_type' from call_node.func. Examples:
          - If func is a Name (e.g., Conv2d), return func.id
          - If func is an Attribute (e.g., nn.Conv2d), return func.attr
          - Otherwise, fall back to unparsed func and split on '.', taking the last segment.
        If any exception occurs, return 'Unknown'.
        """
        try:
            func = call_node.func
            if isinstance(func, ast.Name):
                return func.id
            if isinstance(func, ast.Attribute):
                return func.attr
            # Fallback: unparse and take the last component after a dot
            return ast.unparse(func).split(".")[-1]
        except Exception:
            return "Unknown"

    def _gather_layer_args(self, arg_nodes: list[ast.expr]) -> list[object]:
        """
        For each positional argument node in arg_nodes:
          - If it's an ast.Constant, evaluate via parent_generator._eval_constant.
          - Otherwise, if it has an 'id' attribute (e.g., a Name), use that.
          - Else, fall back to ast.unparse(arg).
        Return a list of those values in the same order.
        """
        results: list[object] = []
        for arg in arg_nodes:
            if isinstance(arg, ast.Constant):
                # Literal constant like 64, 0.1, etc.
                results.append(self.parent_generator._eval_constant(arg))
            else:
                # If it's a variable name (ast.Name), use arg.id; otherwise, unparse
                val = getattr(arg, "id", None)
                if val is not None:
                    results.append(val)
                else:
                    results.append(ast.unparse(arg))
        return results

    def _gather_layer_kwargs(self, kw_nodes: list[ast.keyword]) -> dict[str, str]:
        """
        Build a dict mapping each keyword argument name to its source-string value.
        We use ast.unparse on kw.value to preserve whatever expression was written.
        """
        result: dict[str, str] = {}
        for kw in kw_nodes:
            if kw.arg:
                result[kw.arg] = ast.unparse(kw.value)
        return result

    def _register_class_layer(self, layer_name: str) -> None:
        """
        If there's a current class, append layer_name to class_layers[class].
        """
        cls = self.parent_generator.current_class
        if cls:
            self.parent_generator.class_layers.setdefault(cls, []).append(layer_name)

    def _mark_used_components_for_layer(self, layer_type: str) -> None:
        """
        If layer_type is itself a class in class_layers, mark it used. Also mark the
        current class used if present.
        """
        if layer_type in self.parent_generator.class_layers:
            self.parent_generator.used_components.add(layer_type)
            current_cls = self.parent_generator.current_class
            if current_cls:
                self.parent_generator.used_components.add(current_cls)

    def _handle_sequential_assignment(self, node: ast.Assign):
        # Early exit if this isn’t “something = Sequential(...)”
        if not self._is_sequential_call(node.value):
            return

        # Process each target in the assignment
        for tgt in node.targets:
            self._process_sequential_target(tgt, node)

    def _is_sequential_call(self, value_node: ast.expr) -> bool:
        """
        Return True if value_node is a Call to Sequential (either via Name or Attribute).
        """
        if not isinstance(value_node, ast.Call):
            return False

        func = value_node.func
        if isinstance(func, ast.Name) and func.id == "Sequential":
            return True
        if isinstance(func, ast.Attribute) and func.attr == "Sequential":
            return True

        return False

    def _process_sequential_target(self, tgt: ast.expr, node: ast.Assign) -> None:
        """
        If the target is self.<name>, then:
          1) add <name> to layer_order (if not already)
          2) extract the sequence elements from the call’s args
          3) update used_components based on which class_layers appear
          4) record a model_layers info dict
          5) register <name> under the current class in class_layers
        """
        # Check for “self.<name> = Sequential(...)”
        if not (
            isinstance(tgt, ast.Attribute)
            and isinstance(tgt.value, ast.Name)
            and tgt.value.id == "self"
        ):
            return

        name = tgt.attr
        self._add_to_layer_order(name)

        # Extract all element‐strings from the Sequential call
        seq_items = self._gather_sequential_items(node.value.args)

        # Update used_components based on which components appear in seq_items
        self._mark_used_components(seq_items)

        # Build and append the model_layers info dict
        info = {
            "name": name,
            "layer_type": "Sequential",
            "args": seq_items,
            "kwargs": {},
            "class": self.parent_generator.current_class,
        }
        self.parent_generator.model_layers.append(info)

        # Finally, register this layer name under the current class
        self._register_class_layer(name)

    def _add_to_layer_order(self, name: str) -> None:
        """
        If ‘name’ isn’t already in layer_order, append it.
        """
        if name not in self.parent_generator.layer_order:
            self.parent_generator.layer_order.append(name)

    def _gather_sequential_items(self, arg_nodes: list[ast.expr]) -> list[str]:
        """
        For each arg in arg_nodes (which should be lists or tuples),
        run ast.unparse on each element and return a flat list of those strings.
        """
        items: list[str] = []
        for arg in arg_nodes:
            if isinstance(arg, (ast.List, ast.Tuple)):
                for elt in arg.elts:
                    # ast.unparse gives us the source code string for the elt
                    items.append(ast.unparse(elt))
        return items

    def _mark_used_components(self, seq_items: list[str]) -> None:
        """
        If any class name from parent_generator.class_layers appears in an item string,
        add that class to used_components; also add current_class if set.
        """
        current_cls = self.parent_generator.current_class
        for item_str in seq_items:
            for comp in self.parent_generator.class_layers:
                if comp in item_str:
                    self.parent_generator.used_components.add(comp)
                    if current_cls:
                        self.parent_generator.used_components.add(current_cls)

    def _register_class_layer(self, name: str) -> None:
        """
        If we have a current class, ensure class_layers[current_class] includes ‘name.’
        """
        current_cls = self.parent_generator.current_class
        if current_cls:
            self.parent_generator.class_layers.setdefault(current_cls, []).append(name)

    def _handle_top_level_model_instantiation(self, node: ast.Assign):
        """
        If this assignment is to a top-level variable whose name contains
        'optimizer', 'model', or 'train', record it as a Variable line. If the
        right-hand side is a Call to a known class in class_layers, mark it used.
        """
        # 1) Unparse the RHS safely
        expr = self._safe_unparse(node.value)

        # 2) Iterate over each target and process if it matches our naming pattern
        for tgt in node.targets:
            if self._is_top_level_variable(tgt):
                self.lines.append(f"Variable: {tgt.id} = {expr}")

                # 3) If the RHS is a Call, extract its class name and mark usage
                if isinstance(node.value, ast.Call):
                    cls_name = self._get_callable_name(node.value.func)
                    if cls_name and cls_name in self.parent_generator.class_layers:
                        self._mark_top_level_instantiation(cls_name, tgt.id)

    def _safe_unparse(self, value_node: ast.expr) -> str:
        """
        Return ast.unparse(value_node), or '<expr>' if unparsing fails.
        """
        try:
            return ast.unparse(value_node)
        except Exception:
            return "<expr>"

    def _is_top_level_variable(self, tgt: ast.expr) -> bool:
        """
        Return True if tgt is a Name whose identifier contains any of:
        'optimizer', 'model', or 'train' (case-insensitive).
        """
        if not isinstance(tgt, ast.Name):
            return False

        lower_id = tgt.id.lower()
        return any(keyword in lower_id for keyword in ("optimizer", "model", "train"))

    def _get_callable_name(self, func_node: ast.expr) -> Union[str, None]:
        """
        Return the class/function name for a Call node’s func:
          - If func_node is ast.Name, return its .id
          - If func_node is ast.Attribute, return its .attr
          - Otherwise return None
        """
        if isinstance(func_node, ast.Name):
            return func_node.id
        if isinstance(func_node, ast.Attribute):
            return func_node.attr
        return None

    def _mark_top_level_instantiation(self, cls_name: str, var_name: str) -> None:
        """
        Add cls_name to used_components and print a debug message indicating
        we found a top-level instantiation of var_name = cls_name(...).
        """
        self.parent_generator.used_components.add(cls_name)
        print(f"Found top-level model instantiation: {var_name} = {cls_name}")