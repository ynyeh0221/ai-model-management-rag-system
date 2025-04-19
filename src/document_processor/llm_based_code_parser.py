import ast
import datetime
import json
import os
import re

from git import Repo

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

        self.llm_metadata_cache = self._extract_llm_metadata(file_content, max_retries=5)

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

        model_info["is_model_script"] = True
        model_info["content"] = file_content

        print(f"updated model_info: {model_info}")
        return model_info

    def generate_ast_summary(self, code_str: str, file_path: str = "<unknown>") -> str:
        """
        Parse code using AST and generate a human-readable digest of the code structure,
        including *all* variable assignments and layer definitions for nn.Module subclasses.
        """
        try:
            tree = ast.parse(code_str, filename=file_path)
        except SyntaxError as e:
            return f"# Failed to parse AST: {e}"

        lines = []

        class CodeSummaryVisitor(ast.NodeVisitor):
            def __init__(self):
                super().__init__()
                self.current_bases = []
                self.in_init = False

            def visit_Import(self, node):
                names = [alias.name for alias in node.names]
                lines.append(f"Import: {', '.join(names)}")

            def visit_ImportFrom(self, node):
                module = node.module or ""
                names = [alias.name for alias in node.names]
                lines.append(f"From {module} import {', '.join(names)}")

            def visit_ClassDef(self, node):
                # record base classes for module detection
                bases = [getattr(b, 'id', getattr(b, 'attr', 'object')) for b in node.bases]
                self.current_bases = bases
                lines.append(f"\nClass: {node.name} (inherits from {', '.join(bases)})")
                doc = ast.get_docstring(node)
                if doc:
                    lines.append(f"  Docstring: {doc.strip()}")
                # visit methods and inner assigns
                for child in node.body:
                    self.visit(child)
                self.current_bases = []

            def visit_FunctionDef(self, node):
                prev_init = self.in_init
                if node.name == "__init__" and any(b.endswith("Module") for b in self.current_bases):
                    self.in_init = True

                args = [arg.arg for arg in node.args.args]
                defaults = [ast.unparse(d) for d in node.args.defaults]
                pad = [None] * (len(args) - len(defaults))
                arg_list = [f"{a}={d}" if d else a for a, d in zip(args, pad + defaults)]
                lines.append(f"\nFunction: {node.name}({', '.join(arg_list)})")
                doc = ast.get_docstring(node)
                if doc:
                    lines.append(f"  Docstring: {doc.strip()}")

                for child in node.body:
                    self.visit(child)
                self.in_init = prev_init

            def visit_Assign(self, node):
                # capture layers inside nn.Module.__init__
                if self.in_init:
                    for target in node.targets:
                        if (isinstance(target, ast.Attribute)
                                and isinstance(target.value, ast.Name)
                                and target.value.id == "self"):
                            name = target.attr
                            try:
                                val = ast.unparse(node.value)
                            except Exception:
                                val = "<complex_expression>"
                            lines.append(f"Layer: self.{name} = {val}")
                            return
                # capture all other top‑level assigns as Variables
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        name = target.id
                        try:
                            val = ast.unparse(node.value)
                        except Exception:
                            val = "<complex_expression>"
                        lines.append(f"Variable: {name} = {val}")

        visitor = CodeSummaryVisitor()
        visitor.visit(tree)
        return "\n".join(lines)

    def _remove_import_lines(self, code: str) -> str:
        filtered_lines = [
            line for line in code.splitlines()
            if not line.strip().lower().startswith("import") and not line.strip().lower().startswith("from")
        ]
        return "\n".join(filtered_lines)

    def _extract_llm_metadata(self, code_str: str, file_path: str = "<unknown>", max_retries: int = 15) -> dict:
        # STEP 1: Generate AST digest/summary
        ast_digest = self.clean_empty_lines(self.generate_ast_summary(code_str, file_path=file_path))
        # print(f"Total AST digest: {ast_digest}")

        # STEP 2: Extract natural-language summary for each digest chunk
        summary = self.extract_chunk_summary(
            chunk_text=ast_digest,
            chunk_offset=0,
            max_retries=max_retries
        )
        print(f"Summary from AST digest: {summary}")

        # STEP 3: Feed merged summary to LLM for structured metadata generation
        summary_and_ast = summary.get("summary", "") + ", " + ast_digest
        final = self.generate_metadata_from_summary(summary_and_ast, max_retries=max_retries)
        final['chunk_descriptions'] = [self.split_summary_into_chunks(summary_text=summary.get("summary", ""), overlap_lines=0, max_lines_per_chunk=50)]
        print(f"Chunk descriptions count: {len(final['chunk_descriptions'])}")
        final.pop("description", None)
        final.pop("_trace", None)

        print(f"Final metadata (from AST summary): {final}")
        return final

    def extract_chunk_summary(self, chunk_text: str, chunk_offset: int = 0, max_retries: int = 3) -> dict:
        """Extract a natural language summary of metadata from a code chunk instead of structured JSON."""
        if not self.llm_interface:
            return {}

        system_prompt = (
            "You are a knowledgeable software‑engineer assistant specialized in reading AST‑based code summaries "
            "and translating them into clear, human‑friendly explanations of what the code does. "
            "You will be given an AST summary of a Python file, listing every import, class, function, "
            "and variable or layer assignment (including `self.xxx = …` in `__init__`). Your task is to:\n"
            "1. Convert each line into a descriptive sentence or paragraph.\n"
            "2. Include every element exactly—do not skip or omit any imports, classes, functions, variables, or layers.\n"
            "3. Group related items in this order: Imports, Variables, Classes, Functions, Model Architecture, Forward Pass.\n"
            "4. Preserve names verbatim but explain their role (e.g. “`optimizer` is set to `Adam(lr=0.001)`, which…”).\n"
            "5. If a docstring appears on a class or function, include it verbatim immediately after its header as:\n"
            "     This class/function does: <docstring>\n"
            "6. If an expression is marked <complex_expression>, say “A complex expression was assigned: <complex_expression>.”\n"
            "7. **Imports:** List each `Import:` or `From … import:` line exactly as it appears, one per bullet.\n"
            "8. **Variables:** Immediately after Imports, list every `Variable:` line as “<name> = <value>” plus a one‑sentence purpose.\n"
            "9. **Model Architecture:** For each `Layer: self.<name> = <Ctor>(…)` in an `nn.Module`:\n"
            "     • **Name & type:** `<name>: <Ctor>(…)`\n"
            "     • **Constructor args unpacked:** e.g. `in_channels=1, out_channels=embed_dim, kernel_size=4, stride=4`\n"
            "     • **Tensor shape transformation:** e.g. `(batch, 1, 28, 28) → (batch, embed_dim, 7, 7)`\n"
            "10. **Forward Pass:** Provide a numbered list of each operation, in the same order as the Layers list, with exact input→output tensor shapes, for example:\n"
            "     1. Embed labels: (batch,) → (batch, embed_dim)\n"
            "     2. Conv reduce + BatchNorm + ReLU: (batch,1,28,28) → (batch,embed_dim,7,7)\n"
            "     3. …\n\n"
            "Return your result as a cohesive, well‑structured report with clear headings and bullet points or numbered steps for maximum readability."
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

    def truncate_string(self, s, threshold):
        return s[:threshold] if len(s) > threshold else s

    def merge_chunk_summaries(self, chunk_summaries: list) -> str:
        """Merge multiple chunk summaries into a single comprehensive summary."""
        if not chunk_summaries:
            return "No metadata information found in the code."

        # Extract all summaries
        summaries = [chunk.get("summary", "") for chunk in chunk_summaries if chunk.get("summary")]

        # Remove duplicates and empty strings
        unique_summaries = []
        for summary in summaries:
            if summary and summary not in unique_summaries and summary != "No relevant metadata found in this code chunk.":
                unique_summaries.append(self.clean_empty_lines(summary))

        if not unique_summaries:
            return "No useful metadata found in the code."

        # Combine the summaries into a single document
        merged_summary = self.truncate_string("Combined Code Analysis Summary:\n" + "\n".join(unique_summaries), 12000) # 4,096 tokens ≈ 16,384 characters
        return merged_summary

    def sanitize_json_string(self, json_str: str) -> str:
        # Remove JS-style comments
        json_str = re.sub(r"//.*?$", "", json_str, flags=re.MULTILINE)
        # Remove trailing commas
        json_str = re.sub(r",(\s*[}\]])", r"\1", json_str)
        return json_str

    def generate_metadata_from_summary(self, merged_summary: str, max_retries: int = 3) -> dict:
        """Generate structured JSON metadata from the merged summary."""
        if not self.llm_interface:
            return self._create_default_metadata()

        system_prompt = (
            "You are a metadata extractor for machine learning code. "
            "Based on the following code analysis summary, create a structured representation of the model metadata.\n\n"
            "The output **must strictly follow this exact JSON structure**:\n"
            "{\n"
            '  "description": "Short summary of what the model does",\n'
            '  "framework": { "name": "...", "version": "..." },\n'
            '  "architecture": { "type": "..." },\n'
            '  "dataset": { "name": "..." },\n'
            '  "training_config": {\n'
            '    "batch_size": 32,\n'
            '    "learning_rate": 0.001,\n'
            '    "optimizer": "Adam",\n'
            '    "epochs": 10,\n'
            '    "hardware_used": "GPU"\n'
            '  }\n'
            "}\n\n"
            "Extraction hints:\n"
            "• **framework.name**: look for imports like `import torch` or `import tensorflow`; default to “unknown”.\n"
            "• **framework.version**: look for `torch.__version__` or similar; else “unknown”.\n"
            "• **architecture.type**: look for class names or keywords (e.g. “Transformer”, “CNN”); else “unknown”.\n"
            "• **dataset.name**: look for dataset identifiers (e.g. “FashionMNIST”); else “unknown”.\n"
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
                Here is the merged code analysis summary:

                {merged_summary}

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
                        if self._validate_metadata_structure(parsed):
                            return parsed
                        else:
                            print(f"[Retry {attempt + 1}] Invalid metadata structure.")
                    except json.JSONDecodeError:
                        print(f"[Retry {attempt + 1}] Invalid JSON format.")
                else:
                    print(f"[Retry {attempt + 1}] No valid JSON object found in LLM response.")
            except Exception as e:
                print(f"[Retry {attempt + 1}] Metadata generation failed: {e}")

        # After all retries, return default metadata
        return self._create_default_metadata()

    def _create_default_metadata(self) -> dict:
        """Create default metadata structure with empty/null values."""
        return {
            "description": "Unknown model purpose",
            "framework": {"name": "unknown", "version": "unknown"},
            "architecture": {"type": "unknown"},
            "dataset": {"name": "unknown"},
            "training_config": {
                "batch_size": None,
                "learning_rate": None,
                "optimizer": None,
                "epochs": None,
                "hardware_used": None
            }
        }

    def _validate_metadata_structure(self, metadata: dict) -> bool:
        """Validate that the metadata has the required structure."""
        # Check that all required top-level fields exist
        required_fields = ["description", "framework", "architecture", "dataset", "training_config"]
        if not all(field in metadata for field in required_fields):
            return False

        # Check that framework has name and version
        if not isinstance(metadata["framework"], dict) or not all(
                field in metadata["framework"] for field in ["name", "version"]):
            return False

        # Check that architecture has type
        if not isinstance(metadata["architecture"], dict) or "type" not in metadata["architecture"]:
            return False

        # Check that dataset has name
        if not isinstance(metadata["dataset"], dict) or "name" not in metadata["dataset"]:
            return False

        # Check that training_config has all required fields
        training_config_fields = ["batch_size", "learning_rate", "optimizer", "epochs", "hardware_used"]
        if not isinstance(metadata["training_config"], dict) or not all(
                field in metadata["training_config"] for field in training_config_fields):
            return False

        return True

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


    def split_summary_into_chunks(self, summary_text, overlap_lines=2, max_lines_per_chunk=50):
        """
        Splits a summary into semantic chunks based on headings (markdown # or underlined headings),
        with an overlap of lines between chunks. Falls back to fixed-size chunking if no headings detected.

        Parameters:
        - summary_text (str): The full summary text.
        - overlap_lines (int): Number of lines to overlap between chunks.
        - max_lines_per_chunk (int): Maximum lines per chunk when no headings detected.

        Returns:
        - List[str]: A list of text chunks.
        """
        lines = summary_text.splitlines(keepends=True)
        heading_indices = set()

        # Detect markdown headings (#, ##, ###, etc.)
        for i, line in enumerate(lines):
            if re.match(r'^\s{0,3}#{1,6}\s+', line):
                heading_indices.add(i)

        # Detect underlined headings (text followed by === or ---)
        for i in range(len(lines) - 1):
            if lines[i].strip() and re.match(r'^[=-]{3,}\s*$', lines[i + 1]):
                heading_indices.add(i)

        # If no headings found, fallback to fixed-size chunking
        if not heading_indices:
            chunks = []
            total = len(lines)
            for start in range(0, total, max_lines_per_chunk):
                chunk = lines[start:start + max_lines_per_chunk]
                if start > 0:
                    overlap = lines[start - overlap_lines:start]
                    chunk = overlap + chunk
                chunks.append(''.join(chunk))
            return chunks

        # Otherwise, split at heading lines
        sorted_idxs = sorted(heading_indices)
        chunks = []
        for idx, start in enumerate(sorted_idxs):
            end = sorted_idxs[idx + 1] if idx + 1 < len(sorted_idxs) else len(lines)
            chunk_lines = lines[start:end]
            if idx > 0:
                prev_end = start
                overlap = lines[prev_end - overlap_lines:prev_end]
                chunk_lines = overlap + chunk_lines
            chunks.append(''.join(chunk_lines))

        return chunks

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
