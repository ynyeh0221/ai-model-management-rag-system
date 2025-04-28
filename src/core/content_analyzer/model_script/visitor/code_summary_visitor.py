import ast
from typing import List, Set, Dict, Any, TYPE_CHECKING

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
        self.current_function: str = None
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

        # Record signature
        params = [arg.arg for arg in node.args.args] if node.args.args else []
        defaults = {}
        if node.args.defaults:
            offset = len(params) - len(node.args.defaults)
            for i, d in enumerate(node.args.defaults):
                defaults[params[offset + i]] = self.parent_generator._eval_constant(d)

        self.parent_generator.function_calls[node.name] = {'params': params, 'defaults': defaults}
        self.lines.append(f"\nFunction: {node.name}({', '.join(params)})")

        # Default path detection
        for name, default_node in zip(params[-len(node.args.defaults or []):], node.args.defaults or []):
            lname = name.lower()
            if any(k in lname for k in ('save_dir', 'results_dir', 'save_path', 'output_dir')):
                val = self.parent_generator._eval_constant(default_node)
                if isinstance(val, str):
                    self.default_paths.add(
                        self.parent_generator._determine_folder(val, self.parent_generator.base_dir)
                    )

        # Forward() analysis
        if node.name == 'forward':
            cls = self.parent_generator.current_class
            if cls:
                self.parent_generator.function_calls[node.name].setdefault('class', cls)
            self._analyze_forward_method(node)
            if cls:
                self.parent_generator.used_components.add(cls)

        doc = ast.get_docstring(node)
        if doc:
            self.lines.append(f"  Docstring: {doc.strip()}")

        self.generic_visit(node)
        self.current_function = prev_function

    def visit_Call(self, node: ast.Call):
        # Track calls to known functions
        func_name = None
        if isinstance(node.func, ast.Name): func_name = node.func.id
        elif isinstance(node.func, ast.Attribute): func_name = node.func.attr

        if func_name and func_name in self.parent_generator.function_calls:
            info = self.parent_generator.function_calls[func_name]
            count = self.call_counter.setdefault(func_name, 0) + 1
            self.call_counter[func_name] = count
            call_id = f"{func_name}_call_{count}"

            args_map, provided = {}, set()
            for i, arg in enumerate(node.args):
                if i < len(info['params']):
                    p = info['params'][i]
                    val = self.parent_generator._eval_constant(arg)
                    if val is not None:
                        args_map[p] = val; provided.add(p)

            for kw in node.keywords:
                if kw.arg:
                    val = self.parent_generator._eval_constant(kw.value)
                    if val is not None:
                        args_map[kw.arg] = val
                        self.literal_vars[kw.arg] = val
                        provided.add(kw.arg)

            omitted = [p for p in info['params'] if p not in provided and p in info.get('defaults', {})]
            self.parent_generator.function_calls[call_id] = {
                'function': func_name,
                'args': args_map,
                'omitted_args': omitted
            }
            if self.current_function:
                info.setdefault('called_by', set()).add(self.current_function)

        # Track component instantiation
        cls_name = None
        if isinstance(node.func, ast.Name): cls_name = node.func.id
        elif isinstance(node.func, ast.Attribute): cls_name = node.func.attr
        if cls_name and cls_name in self.parent_generator.class_layers:
            self.parent_generator.used_components.add(cls_name)
            if self.current_function:
                self.function_references.setdefault(self.current_function, set()).add(cls_name)
                if self.parent_generator.current_class:
                    self.parent_generator.used_components.add(self.parent_generator.current_class)

        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign):
        # Break out different assignment handlers
        self._handle_dataloader(node)
        self._handle_simple_assignment(node)
        self._handle_layer_assignment(node)
        self._handle_sequential_assignment(node)
        self._handle_top_level_model_instantiation(node)
        self.generic_visit(node)

    def visit_For(self, node: ast.For):
        if (isinstance(node.iter, ast.Call) and
            isinstance(node.iter.func, ast.Name) and node.iter.func.id == 'range'):
            args = node.iter.args
            count = None
            if len(args) == 1:
                count = self.parent_generator._eval_constant(args[0]) or \
                        (args[0].id if isinstance(args[0], ast.Name) else None)
                count = self.literal_vars.get(count, count) if isinstance(count, str) else count
            elif len(args) >= 2:
                count = self.parent_generator._eval_constant(args[1]) or \
                        (args[1].id if isinstance(args[1], ast.Name) else None)
                count = self.literal_vars.get(count, count) if isinstance(count, str) else count
            if isinstance(count, int):
                self.literal_vars['num_epochs'] = count
                self.lines.append(f"Variable: num_epochs = {count} (from training loop)")
        self.generic_visit(node)

    # ---------------------- Helper Methods ------------------------
    def _analyze_forward_method(self, node: ast.FunctionDef):
        refs, conns, prev_out = set(), {}, None
        for stmt in node.body:
            if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
                call = stmt.value
                if (isinstance(call.func, ast.Attribute) and
                    isinstance(call.func.value, ast.Name) and call.func.value.id == 'self'):
                    layer = call.func.attr
                    if isinstance(stmt.targets[0], ast.Name):
                        out = stmt.targets[0].id
                        if prev_out: conns[prev_out] = layer
                        prev_out = out
                    for l in self.parent_generator.model_layers:
                        if l['name'] == layer and l['class'] == self.parent_generator.current_class:
                            lt = l['layer_type']
                            if lt in self.parent_generator.class_layers:
                                refs.add(lt)
                                self.parent_generator.used_components.add(lt)
            elif isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Call):
                self._process_call_for_references(stmt.value, refs)
        if refs:
            self.parent_generator.function_calls.setdefault(node.name, {})
            self.parent_generator.function_calls[node.name].setdefault('references', set()).update(refs)
        if conns:
            self.parent_generator.model_connections[self.parent_generator.current_class] = conns

    def _process_call_for_references(self, call: ast.Call, refs: Set[str]):
        if (isinstance(call.func, ast.Attribute) and
            isinstance(call.func.value, ast.Name) and call.func.value.id == 'self'):
            layer = call.func.attr
            for l in self.parent_generator.model_layers:
                if l['name'] == layer and l['class'] == self.parent_generator.current_class:
                    lt = l['layer_type']
                    if lt in self.parent_generator.class_layers:
                        refs.add(lt)
                        self.parent_generator.used_components.add(lt)
        for arg in call.args:
            if isinstance(arg, ast.Call):
                self._process_call_for_references(arg, refs)
        for kw in call.keywords:
            if isinstance(kw.value, ast.Call):
                self._process_call_for_references(kw.value, refs)

    def _handle_dataloader(self, node: ast.Assign):
        if isinstance(node.value, ast.Call):
            func = node.value.func
            name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None
            if name == 'DataLoader':
                bs = None
                for kw in node.value.keywords:
                    if kw.arg == 'batch_size':
                        bs = self.parent_generator._eval_constant(kw.value)
                if bs is None and len(node.value.args) >= 2:
                    bs = self.parent_generator._eval_constant(node.value.args[1])
                if bs is not None:
                    var = next((t.id for t in node.targets if isinstance(t, ast.Name)), None)
                    if var:
                        self.literal_vars['batch_size'] = bs
                        self.lines.append(
                            f"Variable: batch_size = {bs} (from DataLoader {var})"
                        )

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
        if not isinstance(node.value, ast.Call):
            return
        call = node.value
        for tgt in node.targets:
            if isinstance(tgt, ast.Attribute) and isinstance(tgt.value, ast.Name) and tgt.value.id == 'self':
                name = tgt.attr
                if name not in self.parent_generator.layer_order:
                    self.parent_generator.layer_order.append(name)
                try:
                    func = call.func
                    layer_type = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ast.unparse(func).split('.')[-1]
                except Exception:
                    layer_type = 'Unknown'
                args = [self.parent_generator._eval_constant(arg) if isinstance(arg, ast.Constant) else getattr(arg, 'id', ast.unparse(arg)) for arg in call.args]
                kwargs = {kw.arg: ast.unparse(kw.value) for kw in call.keywords}
                info = {'name': name, 'layer_type': layer_type, 'args': args, 'kwargs': kwargs, 'class': self.parent_generator.current_class}
                self.parent_generator.model_layers.append(info)
                cl = self.parent_generator.current_class
                if cl:
                    self.parent_generator.class_layers.setdefault(cl, []).append(name)
                if layer_type in self.parent_generator.class_layers:
                    self.parent_generator.used_components.add(layer_type)
                    if cl:
                        self.parent_generator.used_components.add(cl)

    def _handle_sequential_assignment(self, node: ast.Assign):
        if not (isinstance(node.value, ast.Call) and (
            (isinstance(node.value.func, ast.Name) and node.value.func.id == 'Sequential') or
            (isinstance(node.value.func, ast.Attribute) and node.value.func.attr == 'Sequential')
        )):
            return
        for tgt in node.targets:
            if isinstance(tgt, ast.Attribute) and isinstance(tgt.value, ast.Name) and tgt.value.id == 'self':
                name = tgt.attr
                if name not in self.parent_generator.layer_order:
                    self.parent_generator.layer_order.append(name)
                seq = []
                for arg in node.value.args:
                    if isinstance(arg, (ast.List, ast.Tuple)):
                        for elt in arg.elts:
                            s = ast.unparse(elt)
                            seq.append(s)
                            for comp in self.parent_generator.class_layers:
                                if comp in s:
                                    self.parent_generator.used_components.add(comp)
                                    cls = self.parent_generator.current_class
                                    if cls:
                                        self.parent_generator.used_components.add(cls)
                info = {'name': name, 'layer_type': 'Sequential', 'args': seq, 'kwargs': {}, 'class': self.parent_generator.current_class}
                self.parent_generator.model_layers.append(info)
                cl = self.parent_generator.current_class
                if cl:
                    self.parent_generator.class_layers.setdefault(cl, []).append(name)

    def _handle_top_level_model_instantiation(self, node: ast.Assign):
        try:
            expr = ast.unparse(node.value)
        except Exception:
            expr = '<expr>'
        for tgt in node.targets:
            if isinstance(tgt, ast.Name) and any(k in tgt.id.lower() for k in ('optimizer', 'model', 'train')):
                self.lines.append(f"Variable: {tgt.id} = {expr}")
                if isinstance(node.value, ast.Call):
                    func = node.value.func
                    cls = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None
                    if cls and cls in self.parent_generator.class_layers:
                        self.parent_generator.used_components.add(cls)
                        print(f"Found top-level model instantiation: {tgt.id} = {cls}")
