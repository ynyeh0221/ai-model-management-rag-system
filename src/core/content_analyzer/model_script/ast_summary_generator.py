import ast
import os
from typing import Any, Optional


class ASTSummaryGenerator:
    """
    A class responsible for generating human-readable summaries of Python code structure
    using the Abstract Syntax Tree (AST).

    This class analyzes Python code files to extract and format code structure, including:
    - Import statements
    - Variable assignments
    - Function and class definitions (with docstrings)
    - Path variables and image output directories

    Attributes:
        current_file_path (str): The path of the current file being analyzed
        base_dir (str): The base directory of the current file

    Methods:
        generate_summary(code_str, file_path):
            Parses code using AST and generates a human-readable digest of the code structure.
    """

    def __init__(self):
        self.current_file_path = "<unknown>"
        self.base_dir = ""

    def generate_summary(self, code_str: str, file_path: str = "<unknown>") -> str:
        """
        Parse code using AST and generate a human-readable digest of the code structure.
        - Retains original parsing of imports, classes, functions, variables, and layers.
        - Captures literal config vars (batch_size, lr, epochs, device, seed).
        - Captures path-like vars (dir, path, folder) into literal_vars.
        - Detects savefig/imwrite calls and os.path.join usage.
        - Additionally, post-processes literal_vars to extract image output directories for any save_dir, results_dir, save_path, etc.
          Uses heuristic: if the literal string appears to be a file (has extension), take its dirname; otherwise treat it as the directory itself.
          Filters out any folders that match default parameter values for save_* in function signatures.
          If no image directories are found, outputs "Images path: N/A".
        """
        try:
            tree = ast.parse(code_str, filename=file_path)
        except SyntaxError as e:
            return f"# Failed to parse AST: {e}"

        # Store the file path for use in nested class methods
        self.current_file_path = file_path
        self.base_dir = os.path.dirname(os.path.abspath(file_path))

        lines = []
        literal_vars = {}
        image_folders = set()
        default_paths = set()

        visitor = self._create_code_summary_visitor(lines, literal_vars, image_folders, default_paths)
        visitor.visit(tree)

        # Post-process literal_vars for save/results paths
        for name, val in literal_vars.items():
            low = name.lower()
            if any(k in low for k in ('save_dir', 'results_dir', 'save_path', 'output_dir')):
                image_folders.add(self._determine_folder(val, self.base_dir))

        # filter out default signature paths
        image_folders -= default_paths

        # append captured vars
        for name, val in literal_vars.items():
            if any(k in name.lower() for k in ('dir', 'path', 'folder')):
                lines.append(f"Path variable: {name} = {val}")
            else:
                lines.append(f"Variable: {name} = {val}")

        # append image folders if exists, else append missing
        if image_folders:
            for folder in sorted(image_folders):
                # ensure we report an absolute path
                if not os.path.isabs(folder):
                    abs_folder = os.path.abspath(os.path.join(self.base_dir, folder))
                else:
                    abs_folder = folder
                lines.append(f"Images folder: {abs_folder}")
        else:
            lines.append(f"Images folder: missing")

        return '\n'.join(lines)

    def _eval_constant(self, node: ast.AST) -> Optional[Any]:
        """Resolve simple constants, string concatenation, and basic f-strings."""
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
                elif isinstance(v, ast.Str):
                    parts.append(v.s)
                else:
                    return None
            return "".join(parts)
        return None

    def _determine_folder(self, path_str: str, base_dir: str) -> str:
        """Resolve to an absolute folder: if path_str is relative, join with base_dir; strip file name if ext exists."""
        # Resolve absolute path based on base_dir
        if not os.path.isabs(path_str):
            full_path = os.path.abspath(os.path.join(base_dir, path_str))
        else:
            full_path = os.path.abspath(path_str)
        # If it's a file (has extension), return its directory, else the path itself
        base, ext = os.path.splitext(full_path)
        if ext:
            return os.path.dirname(full_path) or full_path
        return full_path

    def _create_code_summary_visitor(self, lines: list, literal_vars: dict,
                                     image_folders: set, default_paths: set) -> ast.NodeVisitor:
        """Create and return the AST visitor for code summary generation."""
        parent_generator = self  # Reference to the parent ASTSummaryGenerator instance

        class CodeSummaryVisitor(ast.NodeVisitor):
            def visit_Import(self, node):
                names = [alias.name for alias in node.names]
                lines.append(f"Import: {', '.join(names)}")

            def visit_ImportFrom(self, node):
                module = node.module or ""
                names = [alias.name for alias in node.names]
                lines.append(f"From {module} import {', '.join(names)}")

            def visit_ClassDef(self, node):
                bases = [getattr(b, 'id', getattr(b, 'attr', 'object')) for b in node.bases]
                lines.append(f"\nClass: {node.name} (inherits from {', '.join(bases)})")
                doc = ast.get_docstring(node)
                if doc:
                    lines.append(f"  Docstring: {doc.strip()}")
                self.generic_visit(node)

            def visit_FunctionDef(self, node):
                # record default paths for save_* parameters
                params = [arg.arg for arg in node.args.args]
                for name_node, default_node in zip(params[-len(node.args.defaults):], node.args.defaults):
                    lname = name_node.lower()
                    if any(k in lname for k in ('save_dir', 'results_dir', 'save_path', 'output_dir')):
                        val = parent_generator._eval_constant(default_node)
                        if isinstance(val, str):
                            default_paths.add(parent_generator._determine_folder(val, parent_generator.base_dir))
                # existing signature logging
                args = [arg.arg for arg in node.args.args]
                defaults = [ast.unparse(d) if hasattr(ast, 'unparse') else '<default>' for d in node.args.defaults]
                pad = [None] * (len(args) - len(defaults))
                arg_list = [f"{a}={d}" if d else a for a, d in zip(args, pad + defaults)]
                lines.append(f"\nFunction: {node.name}({', '.join(arg_list)})")
                doc = ast.get_docstring(node)
                if doc:
                    lines.append(f"  Docstring: {doc.strip()}")
                self.generic_visit(node)

            def visit_Assign(self, node):
                # capture literal vars
                val = parent_generator._eval_constant(node.value)
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name):
                        name = tgt.id
                        key = name.lower()
                        if val is not None:
                            if any(k in key for k in ('dir', 'path', 'folder')):
                                literal_vars[name] = val
                            if any(k in key for k in ('batch', 'lr', 'epoch', 'device', 'seed')):
                                literal_vars[name] = val
                # detect layer definitions
                if isinstance(node.value, ast.Call):
                    call_name = ast.unparse(node.value.func)
                    if call_name and call_name[0].isupper():
                        args_repr = []
                        for a in node.value.args + node.value.keywords:
                            try:
                                args_repr.append(ast.unparse(a))
                            except:
                                args_repr.append('?')
                        for tgt in node.targets:
                            if isinstance(tgt, ast.Name):
                                lines.append(f"Layer: {tgt.id} = {call_name}({', '.join(args_repr)})")
                # log variables
                expr = ast.unparse(node.value) if hasattr(ast, 'unparse') else '<expr>'
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name) and any(k in tgt.id.lower() for k in
                                                         ('batch', 'lr', 'epoch', 'optimizer', 'device', 'model',
                                                          'train')):
                        lines.append(f"Variable: {tgt.id} = {expr}")
                self.generic_visit(node)

            def visit_Call(self, node):
                name = ast.unparse(node.func)
                if name.endswith('savefig') or name.endswith('imwrite'):
                    if node.args:
                        lit = parent_generator._eval_constant(node.args[0])
                        if isinstance(lit, str):
                            image_folders.add(parent_generator._determine_folder(lit, parent_generator.base_dir))
                    for kw in node.keywords:
                        if kw.arg and 'save_dir' in kw.arg and isinstance(kw.value, ast.Name):
                            var = kw.value.id
                            if var in literal_vars:
                                image_folders.add(
                                    parent_generator._determine_folder(literal_vars[var], parent_generator.base_dir))
                if isinstance(node.func, ast.Attribute) and ast.unparse(node.func).endswith('path.join'):
                    parts = [parent_generator._eval_constant(a) for a in node.args]
                    if all(isinstance(p, str) for p in parts):
                        joined = os.path.join(*parts)
                        image_folders.add(parent_generator._determine_folder(joined, parent_generator.base_dir))
                self.generic_visit(node)

        return CodeSummaryVisitor()
