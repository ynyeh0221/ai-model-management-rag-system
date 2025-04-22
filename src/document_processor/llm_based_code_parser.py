import ast
import datetime
import json
import os
import re
from typing import List, Tuple

from git import Repo
from overrides.signature import is_param_defined_in_sub
from torchgen.packaged.autograd.gen_python_functions import is_py_sparse_function

"""
LLMBasedCodeParser: A class for extracting structured metadata from ML model code using AST and LLMs.

This class analyzes Python code files (e.g., .py or .ipynb) to extract machine learning model metadata,
including framework, architecture, datasets, training configuration, and descriptive information. It combines
static analysis (via Python's AST module) with dynamic summarization powered by a language model (LLM).

The parser generates an AST-based digest of the code (functions, classes, variables, imports), summarizes it 
in natural language, and then generates structured JSON metadata.

Attributes:
    schema_validator (optional): A schema validation object to check the final metadata format.
    llm_interface (optional): An interface that communicates with a language model for code summarization.
    llm_metadata_cache (dict): Stores intermediate results from LLM calls for reuse and traceability.

Methods:
    parse(file_path):
        Parses a file to extract model metadata.

        Args:
            file_path (str): Path to a .py or .ipynb file.

        Returns:
            dict or None: Metadata dictionary (if file is supported), or None.

    parse_file(file_path):
        Reads and analyzes the code file using AST + LLM summarization.
        Returns a complete dictionary of extracted metadata fields.

    generate_ast_summary(code_str, file_path):
        Uses the AST module to extract and format the code structure, including:
            - Import statements
            - Variable assignments
            - Function and class definitions (with docstrings)
        Returns a simplified string representation of the code structure.

    extract_chunk_summary(chunk_text, chunk_offset=0, max_retries=3):
        Feeds an AST-based summary chunk to the LLM for natural language summarization.
        Returns a dictionary with the summary and source preview.

    generate_metadata_from_summary(merged_summary, max_retries=3):
        Uses the LLM to convert a natural-language code summary into structured JSON metadata
        that includes fields like:
            - description
            - framework {name, version}
            - architecture {type}
            - dataset {name}
            - training_config {batch_size, learning_rate, optimizer, epochs, hardware_used}

    remove_import_lines(code):
        Removes all lines that start with 'import' or 'from ... import' from a string.
        Useful for summarization cleanup.

    clean_empty_lines(text):
        Removes empty lines from a string, preserving structure.

    _extract_model_info(file_content, file_path):
        Uses the AST to extract model identifiers from class definitions (e.g., model name and family).

    _get_creation_date(file_path):
        Returns the earliest creation date (from Git history or file system).

    _get_last_modified_date(file_path):
        Returns the last modification date (from Git history or file system).

Example:
    parser = LLMBasedCodeParser(schema_validator=my_validator, llm_interface=my_llm)
    metadata = parser.parse("model.py")
    print(metadata["framework"], metadata["architecture"], metadata["dataset"])
"""
class LLMBasedCodeParser:
    def __init__(self, schema_validator=None, llm_interface=None):
        self.schema_validator = schema_validator
        self.llm_interface = llm_interface
        self.llm_metadata_cache = {}

    def parse(self, file_path):
        ext = os.path.splitext(file_path)[1].lower()
        if ext in ['.py', '.ipynb']:
            return self.parse_file(file_path)
        return None

    def parse_file(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            file_content = f.read()

        self.llm_metadata_cache = self._extract_llm_metadata(file_content, file_path, max_retries=15)

        model_info = {
            "creation_date": self._get_creation_date(file_path),
            "last_modified_date": self._get_last_modified_date(file_path)
        }

        extracted_info = self._extract_model_info(file_content, file_path)
        model_info.update(extracted_info)

        # Safely parse framework
        framework = self.llm_metadata_cache.get("framework", {})
        if isinstance(framework, str):
            model_info["framework"] = {"name": framework, "version": None}
        elif isinstance(framework, dict):
            model_info["framework"] = {
                "name": framework.get("name") if isinstance(framework.get("name"), str) else None,
                "version": framework.get("version") if isinstance(framework.get("version"), str) else None
            }
        else:
            model_info["framework"] = {"name": None, "version": None}

        arch = self.llm_metadata_cache.get("architecture", {})
        if isinstance(arch, str):
            model_info["architecture"] = {"type": arch}
        elif isinstance(arch, dict):
            model_info["architecture"] = {
                "type": arch.get("type") if isinstance(arch.get("type"), str) else None
            }
        else:
            model_info["architecture"] = {"type": None}

        dataset = self.llm_metadata_cache.get("dataset", {})
        if isinstance(dataset, str):
            model_info["dataset"] = {"name": dataset}
        elif isinstance(dataset, dict):
            model_info["dataset"] = {
                "name": dataset.get("name") if isinstance(dataset.get("name"), str) else None
            }
        else:
            model_info["dataset"] = {"name": None}

        images_folder = self.llm_metadata_cache.get("images_folder", {})
        if isinstance(images_folder, str):
            model_info["images_folder"] = {"name": images_folder}
        elif isinstance(images_folder, dict):
            model_info["images_folder"] = {
                "name": images_folder.get("name") if isinstance(images_folder.get("name"), str) else None
            }
        else:
            model_info["images_folder"] = {"name": None}

        tc = self.llm_metadata_cache.get("training_config", {})
        if isinstance(tc, dict):
            model_info["training_config"] = {
                "batch_size": tc.get("batch_size") if isinstance(tc.get("batch_size"), int) else None,
                "learning_rate": tc.get("learning_rate") if isinstance(tc.get("learning_rate"), (int, float)) else None,
                "optimizer": tc.get("optimizer") if isinstance(tc.get("optimizer"), str) else None,
                "epochs": tc.get("epochs") if isinstance(tc.get("epochs"), int) else None,
                "hardware_used": tc.get("hardware_used") if isinstance(tc.get("hardware_used"), str) else None
            }
        else:
            model_info["training_config"] = {
                "batch_size": None,
                "learning_rate": None,
                "optimizer": None,
                "epochs": None,
                "hardware_used": None
            }

        desc = self.llm_metadata_cache.get("chunk_descriptions")
        model_info["chunk_descriptions"] = desc if len(desc) > 0 else []

        model_info["ast_summary"] = self.llm_metadata_cache.get("ast_summary")

        model_info["is_model_script"] = True
        model_info["content"] = file_content

        print(f"updated model_info: {model_info}")
        return model_info

    def generate_ast_summary(self, code_str: str, file_path: str = "<unknown>") -> str:
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

        lines = []
        literal_vars = {}
        image_folders = set()
        default_paths = set()

        def eval_constant(node):
            """Resolve simple constants, string concatenation, and basic f-strings."""
            if isinstance(node, ast.Constant) and isinstance(node.value, (str, int, float, bool)):
                return node.value
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
                left = eval_constant(node.left)
                right = eval_constant(node.right)
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

        def determine_folder(path_str: str) -> str:
            """If value has file extension, return dirname; else return itself."""
            base, ext = os.path.splitext(path_str)
            if ext:
                return os.path.dirname(path_str) or path_str
            return path_str

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
                        val = eval_constant(default_node)
                        if isinstance(val, str):
                            default_paths.add(determine_folder(val))
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
                val = eval_constant(node.value)
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
                        lit = eval_constant(node.args[0])
                        if isinstance(lit, str):
                            image_folders.add(os.path.dirname(lit) or '.')
                    for kw in node.keywords:
                        if kw.arg and 'save_dir' in kw.arg and isinstance(kw.value, ast.Name):
                            var = kw.value.id
                            if var in literal_vars:
                                image_folders.add(determine_folder(literal_vars[var]))
                if isinstance(node.func, ast.Attribute) and ast.unparse(node.func).endswith('path.join'):
                    parts = [eval_constant(a) for a in node.args]
                    if all(isinstance(p, str) for p in parts):
                        joined = os.path.join(*parts)
                        image_folders.add(os.path.dirname(joined) or '.')
                self.generic_visit(node)

        visitor = CodeSummaryVisitor()
        visitor.visit(tree)

        # Post-process literal_vars for save/results paths
        for name, val in literal_vars.items():
            low = name.lower()
            if any(k in low for k in ('save_dir', 'results_dir', 'save_path', 'output_dir')):
                image_folders.add(determine_folder(val))

        # filter out default signature paths
        image_folders -= default_paths

        # append captured vars
        for name, val in literal_vars.items():
            if any(k in name.lower() for k in ('dir', 'path', 'folder')):
                lines.append(f"Path variable: {name} = {val}")
            else:
                lines.append(f"Variable: {name} = {val}")

        # append image folders or N/A
        if image_folders:
            for folder in sorted(image_folders):
                lines.append(f"Images folder: {folder}")
        else:
            lines.append("Images folder: N/A")

        return '\n'.join(lines)

    def _remove_import_lines(self, code: str) -> str:
        filtered_lines = [
            line for line in code.splitlines()
            if not line.strip().lower().startswith("import") and not line.strip().lower().startswith("from")
        ]
        return "\n".join(filtered_lines)

    def _extract_llm_metadata(self, code_str: str, file_path: str = "<unknown>", max_retries: int = 15) -> dict:
        # STEP 1: Generate AST digest/summary
        ast_digest = self.clean_empty_lines(self.generate_ast_summary(code_str=code_str, file_path=file_path))
        print(f"Total AST digest: {ast_digest}")

        # STEP 2: Extract natural-language summary for each digest chunk
        summary = self.extract_natural_language_summary(
            chunk_text=ast_digest,
            chunk_offset=0,
            max_retries=max_retries
        )
        print(f"Natural language summary from AST digest: {summary}")

        # STEP 3: Feed merged summary to LLM for structured metadata generation
        final = self.generate_metadata_from_summary(ast_digest, max_retries=max_retries, file_path=file_path)
        final['chunk_descriptions'] = self.split_summary_into_chunks(summary_text=summary.get("summary", ""), overlap_sentences=0, max_sentences_per_chunk=1)
        print(f"Chunk descriptions count: {len(final['chunk_descriptions'])}")

        # STEP 4: Store AST digest summary
        final['ast_summary'] = ast_digest

        # STEP 5: Remove unneeded fields
        final.pop("_trace", None)

        print(f"Final metadata (from AST summary): {final}")
        return final

    def extract_natural_language_summary(self, chunk_text: str, chunk_offset: int = 0, max_retries: int = 3) -> dict:
        """Extract a natural language summary of metadata from a code chunk instead of structured JSON."""
        if not self.llm_interface:
            return {}

        # Reduce input size of LLM
        if len(chunk_text) >= 2500:
            chunk_text = self.filter_ast_summary_for_metadata(chunk_text)

        system_prompt = (
            "You are a senior ML engineer documenting Python training scripts. "
            "Given an AST summary of a script, **carefully examine every detail** to produce a complete, human-readable English report.\n\n"

            "**CRITICAL REQUIREMENTS**:\n"
            "• Thoroughly analyze the AST summary before writing - do not skip or overlook any nodes or attributes.\n"
            "• Ensure all model components, configurations, and operations in the AST are reflected in your summary.\n"
            "• Pay special attention to hyperparameters, layer dimensions, conditional logic, and data transformations.\n\n"

            "Produce a report with these sections:\n\n"
            "1. Purpose:\n"
            "   • In 1–2 sentences, precisely describe what the script accomplishes, mentioning all major features present in the AST.\n\n"

            "2. Data & Preprocessing:\n"
            "   • **Comprehensively document** all dataset classes, sources, and directories from the AST.\n"
            "   • List **every** transformation and preprocessing step, including normalization, augmentation, or custom functions.\n"
            "   • Detail **all** DataLoader configurations: batch sizes, shuffle settings, train/val/test splits, num_workers, etc.\n\n"

            "3. Model Architecture:\n"
            "   • Describe **every layer** or block found in the AST using fluent prose. For example:\n"
            "     \"The first convolutional layer applies 32 filters of size 3×3 to the 1‑channel 28×28 input, producing 32 feature maps of size 26×26. Next, a ReLU activation introduces non‑linearity, followed by a 2×2 max‑pool that halves the spatial dimensions to 13×13.\"\n"
            "   • For each component, explain:\n"
            "     - What it does and its role in the architecture\n"
            "     - Its exact parameters (e.g., kernel sizes, strides, padding)\n"
            "     - How it transforms the data shape through the network\n"
            "     - Any regularization (dropout, batch norm) or special operations\n"
            "   • Do NOT omit auxiliary modules, custom layers, or architectural details.\n\n"
    
            "4. Training Configuration:\n"
            "   • **Document every training parameter** from the AST:\n"
            "     - Optimizer (type and all hyperparameters like lr, momentum, weight_decay)\n"
            "     - Learning rate schedules (type, step size, gamma, etc.)\n"
            "     - Loss function(s) and any weights/parameters\n"
            "     - Exact number of epochs, early stopping criteria\n"
            "     - Hardware settings (CPU/GPU/MPS), device allocations\n"
            "     - Gradient clipping, mixed precision, or other training modifications\n\n"
    
            "5. Evaluation & Testing:\n"
            "   • Detail **all evaluation procedures** found in the AST:\n"
            "     - Validation frequency and process\n"
            "     - Test protocols and checkpointing strategies\n"
            "     - **Every metric** computed (accuracy, F1, confusion matrix, etc.)\n"
            "     - Any custom evaluation logic or callbacks\n\n"
    
            "6. Visualization & Artifacts:\n"
            "   • **List all visualization components**:\n"
            "     - Plot types (loss curves, confusion matrices, embeddings, etc.)\n"
            "     - Saving directories and file formats\n"
            "     - Logging systems (TensorBoard, Wandb, etc.)\n"
            "     - Model checkpoints and saved artifacts\n\n"
    
            "**WRITING GUIDELINES**:\n"
            "• Use clear headings and bullet lists.\n"
            "• Write in natural, fluent prose - no code snippets or raw AST output.\n"
            "• Be comprehensive - if it's in the AST, it must be in your summary.\n"
            "• Double-check that your summary includes ALL components mentioned in the AST.\n"
            "• Ensure an ML engineer could reproduce the exact workflow from your description.\n\n"
    
            "Remember: Missing important details means failing the task. Be meticulous and thorough."
        )

        for attempt in range(max_retries):
            try:
                user_prompt = f"Here is the AST summary:\n{chunk_text}\n```"
                response = self.llm_interface.generate_structured_response(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=0,
                    max_tokens=4000
                )
                summary = response.get("content", "").strip()

                if summary:
                    return {
                        "summary": summary,
                        "source_offset": chunk_offset,
                        "source_preview": chunk_text[:120]
                    }
                else:
                    print(f"[Retry {attempt + 1}] Empty summary response.")
            except Exception as e:
                print(f"[Retry {attempt + 1}] Chunk summary extraction failed: {e}")

        # After all retries, return a minimal structure
        return {
            "summary": "No relevant metadata found in this code chunk.",
            "source_offset": chunk_offset,
            "source_preview": chunk_text[:120]
        }

    def clean_empty_lines(self, text: str) -> str:
        # Split into lines, strip each line, and remove if it's empty
        lines = text.splitlines()
        cleaned = [line.rstrip() for line in lines if line.strip() != '']

        # Join lines back with a single newline
        return '\n'.join(cleaned)

    def sanitize_json_string(self, json_str: str) -> str:
        # Remove JS-style comments
        json_str = re.sub(r"//.*?$", "", json_str, flags=re.MULTILINE)
        # Remove trailing commas
        json_str = re.sub(r",(\s*[}\]])", r"\1", json_str)
        return json_str

    def filter_ast_summary_for_metadata(self, summary: str) -> str:
        """
        Given the output of generate_ast_summary(), return only the lines that
        contain:
          - Docstrings      (model description)
          - Import / From   (framework & dataset)
          - Class           (architecture type)
          - Layer           (component + dims)
          - Variable        (batch_size, lr, optimizer, epochs, device)
        """
        lines = summary.splitlines()
        filtered: List[str] = []
        if len(summary) <= 9000:
            # what prefixes we always keep
            keep_prefixes = ("Docstring:", "Import:", "From ", "Class:", "Layer:", "Images folder:")
            # which variable names to keep
            var_keys = ("batch", "lr", "epoch", "optimizer", "device")
        else: # For long summary, skip some information to trade off llm's input size
            keep_prefixes = ("Docstring:", "Class:", "Layer:", "Images folder:")
            var_keys = ("batch", "lr", "optimizer", "device")

        for line in lines:
            stripped = line.strip()
            # always keep these kinds of lines
            if any(stripped.startswith(pref) for pref in keep_prefixes):
                filtered.append(stripped)
                continue

            # selectively keep Variable: lines
            if stripped.startswith("Variable:"):
                # extract the var name before the '='
                m = re.match(r"Variable:\s*([^=]+)", stripped)
                if m:
                    varname = m.group(1).strip().lower()
                    if any(key in varname for key in var_keys):
                        filtered.append(stripped)

        return "\n".join(filtered)

    def generate_metadata_from_summary(self, summary: str, max_retries: int = 3, file_path: str = "<unknown>") -> dict:
        """Generate structured JSON metadata from the merged summary."""
        if not self.llm_interface:
            return self._create_default_metadata()

        # Reduce input size of LLM
        print(f"len(summary): {len(summary)}, file_path: {file_path}")
        if len(summary) >= 2500:
            summary = self.filter_ast_summary_for_metadata(summary)
            print(f"len(extracted_summary): {len(summary)}, file_path: {file_path}")

        system_prompt = (
            "You are a metadata extractor for machine learning code. "
            "Based on the following model AST digest summary, create a structured representation of the model metadata.\n\n"
            "The output **must strictly follow this exact JSON structure**:\n"
            "{\n"
            '  \"framework\": { \"name\": \"...\", \"version\": \"...\" },\n'
            '  \"architecture\": { \"type\": \"...\" },\n'
            '  \"dataset\": { \"name\": \"...\" },\n'
            '  \"images_folder\": { \"name\": \"...\" },\n'
            '  \"training_config\": {\n'
            '    \"batch_size\": 32,\n'
            '    \"learning_rate\": 0.001,\n'
            '    \"optimizer\": \"Adam\",\n'
            '    \"epochs\": 10,\n'
            '    \"hardware_used\": \"GPU\"\n'
            '  }\n'
            "}\n\n"
            "Extraction hints:\n"
            "• **framework.name**: look for imports like `import torch` or `import tensorflow`; default to “unknown”.\n"
            "• **framework.version**: look for `torch.__version__` or similar; else “unknown”.\n"
            "• **architecture.type**: identify the high‑level components of the model (e.g. “VAE”, “diffusion model”, “UNet”), listing all major submodels separated by commas; do not list low‑level layers like Conv2d or individual blocks; if none found, use “unknown”.\n"
            "• **dataset.name**: look for dataset identifiers (e.g. “FashionMNIST”); else “unknown”.\n"
            "• **imaged_folder.name**: look for image folders (e.g. “/a/b/c”); else “unknown”.\n"
            "• **batch_size**: look for `batch_size=` in DataLoader; else null.\n"
            "• **learning_rate**: look for `lr=` or “learning rate”; else null.\n"
            "• **optimizer**: look for optimizer names (Adam, SGD); else “unknown”.\n"
            "• **epochs**: look for `epochs =`; else null.\n"
            "• **hardware_used**: look for device settings (`cuda`, `mps`, `cpu`); map to “GPU”, “CPU” or “Both”; else “unknown”.\n\n"
            "🚨 **Output ONLY** the JSON object—no commentary, no markdown."
        )

        for attempt in range(max_retries):
            try:
                user_prompt = f"""
                Here is the model AST digest summary:

                {summary}

                Extract the metadata JSON as specified above.
                """
                response = self.llm_interface.generate_structured_response(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=0,
                    max_tokens=4000
                )
                print(f"metadata generation response (attempt {attempt + 1}): {response}")
                raw = response.get("content", "").strip()
                match = re.search(r"\{.*\}", raw, re.DOTALL)
                if match:
                    json_str = self.sanitize_json_string(match.group())
                    try:
                        parsed = json.loads(json_str)
                        is_parse_success, reason = self._validate_metadata_structure(parsed)
                        if is_parse_success:
                            return parsed
                        else:
                            print(f"[Retry {attempt + 1}] Invalid metadata structure: {reason}, file_path: {file_path}")
                    except json.JSONDecodeError:
                        print(f"[Retry {attempt + 1}] Invalid JSON format, file_path: {file_path}")
                else:
                    print(f"[Retry {attempt + 1}] No valid JSON object found in LLM response, file_path: {file_path}")
            except Exception as e:
                print(f"[Retry {attempt + 1}] Metadata generation failed: {e}, file_path: {file_path}")

        # After all retries, return default metadata
        return self._create_default_metadata()

    def _create_default_metadata(self) -> dict:
        """Create default metadata structure with empty/null values."""
        return {
            "framework": {"name": "unknown", "version": "unknown"},
            "architecture": {"type": "unknown"},
            "dataset": {"name": "unknown"},
            "imaged_folder": {"name": "unknown"},
            "training_config": {
                "batch_size": None,
                "learning_rate": None,
                "optimizer": None,
                "epochs": None,
                "hardware_used": None
            }
        }

    def _validate_metadata_structure(self, metadata: dict) -> Tuple[bool, str]:
        """Validate that the metadata has the required structure."""
        # Check that all required top-level fields exist
        required_fields = ["framework", "architecture", "dataset", "images_folder", "training_config"]
        if not all(field in metadata for field in required_fields):
            return False, "Missing required fields"

        # Check that framework has name and version
        if not isinstance(metadata["framework"], dict) or not all(
                field in metadata["framework"] for field in ["name", "version"]):
            return False, "Invalid framework structure"

        # Check that architecture has type
        if not isinstance(metadata["architecture"], dict) or "type" not in metadata["architecture"]:
            return False, "Invalid architecture structure"

        # Check that dataset has name
        if not isinstance(metadata["dataset"], dict) or "name" not in metadata["dataset"]:
            return False, "Invalid dataset structure"

        if not isinstance(metadata["images_folder"], dict) or "name" not in metadata["images_folder"]:
            return False, "Invalid images folder structure"

        # Check that training_config has all required fields
        training_config_fields = ["batch_size", "learning_rate", "optimizer", "epochs", "hardware_used"]
        if not isinstance(metadata["training_config"], dict) or not all(
                field in metadata["training_config"] for field in training_config_fields):
            return False, "Invalid training configuration structure"

        return True, ""

    def _extract_model_info(self, file_content: str, file_path: str) -> dict:
        try:
            tree = ast.parse(file_content, filename=file_path)
        except SyntaxError as e:
            raise ValueError(f"Syntax error while parsing {file_path}: {e}")

        model_info = {}

        class ModelVisitor(ast.NodeVisitor):
            def __init__(self):
                self.model_name = None

            def visit_ClassDef(self, node):
                for base in node.bases:
                    if hasattr(base, 'id') and base.id in ['Module', 'nn.Module']:
                        self.model_name = node.name
                        break

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

    def split_ast_and_subsplit_chunks(self, file_content: str, file_path: str, chunk_size: int = 500, overlap: int = 100):
        try:
            tree = ast.parse(file_content, filename=file_path)
        except SyntaxError as e:
            raise ValueError(f"Syntax error while parsing {file_path}: {e}")

        lines = file_content.splitlines(keepends=True)
        chunks = []

        for node in sorted(tree.body, key=lambda n: getattr(n, "lineno", 0)):
            start_line = getattr(node, "lineno", None)
            end_line = getattr(node, "end_lineno", None)

            if start_line is None:
                continue

            start_idx = start_line - 1
            end_idx = end_line if end_line else (start_idx + 1)

            block_lines = lines[start_idx:end_idx]
            block_code = "".join(block_lines)
            block_start_char_offset = sum(len(l) for l in lines[:start_idx])

            for j in range(0, len(block_code), chunk_size - overlap):
                chunk_text = block_code[j:j + chunk_size]

                if chunk_text.strip():
                    chunks.append({
                        "text": chunk_text,
                        "offset": block_start_char_offset + j,
                        "type": "code",  # you can expand logic later if needed
                        "source_block": block_code
                    })

        return chunks

    def split_summary_into_chunks(self, summary_text, overlap_sentences=2, max_sentences_per_chunk=10):
        """
        Splits a summary into semantic chunks based on headings (markdown # or underlined headings),
        with an overlap of sentences between chunks. Falls back to fixed-size chunking if no headings detected.

        Parameters:
        - summary_text (str): The full summary text.
        - overlap_sentences (int): Number of sentences to overlap between chunks.
        - max_sentences_per_chunk (int): Maximum sentences per chunk when no headings detected.

        Returns:
        - List[str]: A list of text chunks.
        """
        # Import and download NLTK resources
        import nltk
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab')

        sentences = nltk.sent_tokenize(summary_text)

        chunks = []
        total = len(sentences)
        for start in range(0, total, max_sentences_per_chunk):
            chunk_sentences = sentences[start:start + max_sentences_per_chunk]
            if start > 0:
                overlap = sentences[start - overlap_sentences:start]
                chunk_sentences = overlap + chunk_sentences
            chunks.append(' '.join(chunk_sentences))

        # Filter out chunks with less than or equal to 5 characters
        filtered_chunks = [chunk for chunk in chunks if len(chunk) > 5]

        return filtered_chunks

    def _get_creation_date(self, file_path):
        try:
            repo = Repo(os.path.dirname(file_path), search_parent_directories=True)
            commits = list(repo.iter_commits(paths=file_path, max_count=1, reverse=True))
            if commits:
                return datetime.datetime.fromtimestamp(commits[0].committed_date).isoformat()
        except Exception:
            pass

        try:
            stat = os.stat(file_path)
            return datetime.datetime.fromtimestamp(stat.st_ctime).isoformat()
        except Exception:
            return None

    def _get_last_modified_date(self, file_path):
        try:
            repo = Repo(os.path.dirname(file_path), search_parent_directories=True)
            commit = next(repo.iter_commits(paths=file_path, max_count=1))
            return datetime.datetime.fromtimestamp(commit.committed_date).isoformat()
        except Exception:
            pass

        try:
            stat = os.stat(file_path)
            return datetime.datetime.fromtimestamp(stat.st_mtime).isoformat()
        except Exception:
            return None
