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
        Parse code using AST and generate a human-readable digest of the code structure.
        Useful for LLM-friendly summarization.
        """
        try:
            tree = ast.parse(code_str, filename=file_path)
        except SyntaxError as e:
            return f"# Failed to parse AST: {e}"

        lines = []

        class CodeSummaryVisitor(ast.NodeVisitor):
            def visit_Import(self, node):
                names = [alias.name for alias in node.names]
                lines.append(f"Import: {', '.join(names)}")

            def visit_ImportFrom(self, node):
                module = node.module or ""
                names = [alias.name for alias in node.names]
                lines.append(f"From {module} import {', '.join(names)}")

            def visit_Assign(self, node):
                targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
                try:
                    value = ast.unparse(node.value)
                except Exception:
                    value = "<complex_expression>"
                for t in targets:
                    if any(k in t.lower() for k in ["batch", "lr", "epoch", "optimizer", "device", "model", "train"]):
                        lines.append(f"Variable: {t} = {value}")

            def visit_ClassDef(self, node):
                bases = [getattr(b, "id", getattr(b, "attr", "object")) for b in node.bases]
                docstring = ast.get_docstring(node)
                lines.append(f"\nClass: {node.name} (inherits from {', '.join(bases)})")
                if docstring:
                    lines.append(f"  Docstring: {docstring.strip()}")

            def visit_FunctionDef(self, node):
                args = [arg.arg for arg in node.args.args]
                defaults = [ast.unparse(d) if hasattr(ast, "unparse") else repr(d) for d in node.args.defaults]
                pad = [None] * (len(args) - len(defaults))
                arg_pairs = [f"{a}={d}" if d else a for a, d in zip(args, pad + defaults)]
                docstring = ast.get_docstring(node)
                lines.append(f"\nFunction: {node.name}({', '.join(arg_pairs)})")
                if docstring:
                    lines.append(f"  Docstring: {docstring.strip()}")

        visitor = CodeSummaryVisitor()
        visitor.visit(tree)

        return "\n".join(lines)

    def remove_import_lines(self, code: str) -> str:
        filtered_lines = [
            line for line in code.splitlines()
            if not line.strip().startswith("import") and not line.strip().startswith("from")
        ]
        return "\n".join(filtered_lines)

    def _extract_llm_metadata(self, code_str: str, file_path: str = "<unknown>", max_retries: int = 15) -> dict:
        # STEP 1: Generate AST digest/summary
        ast_digest = self.clean_empty_lines(self.generate_ast_summary(code_str, file_path=file_path))
        print(f"Total AST digest: {ast_digest}")

        # STEP 3: Extract natural-language summary for each digest chunk
        summary = self.extract_chunk_summary(
            chunk_text=ast_digest,
            chunk_offset=0,
            max_retries=max_retries
        )

        # STEP 4: Merge summaries into a single doc
        print(f"Summary from AST digest: {summary}")

        # STEP 5: Feed merged summary to LLM for structured metadata generation
        summary_and_ast = summary.get("summary", "") + ", " + ast_digest
        final = self.generate_metadata_from_summary(summary_and_ast, max_retries=max_retries)
        final['chunk_descriptions'] = [self.remove_import_lines(summary_and_ast)]
        final.pop("description", None)
        final.pop("_trace", None)

        print(f"Final metadata (from AST summary): {final}")
        return final

    def extract_chunk_summary(self, chunk_text: str, chunk_offset: int = 0, max_retries: int = 3) -> dict:
        """Extract a natural language summary of metadata from a code chunk instead of structured JSON."""
        if not self.llm_interface:
            return {}

        system_prompt = (
            "You are an expert machine learning metadata extractor. "
            "You will be given a structured summary of a Python ML script extracted via AST (abstract syntax tree). "
            "From that, extract metadata explicitly mentioned, including:\n"
            "- Description of what the model does\n"
            "- Framework name (like PyTorch, TensorFlow) and version if available\n"
            "- Neural network architecture (e.g. Transformer, CNN, RNN, etc.)\n"
            "- Dataset name (e.g. FashionMNIST)\n"
            "- Training configuration: batch size, learning rate, optimizer, epochs, hardware used\n\n"
            "Provide a **natural language summary under 300 characters** that captures all extractable info.\n"
            "Use compact, direct language. ONLY include details that appear in the input."
        )

        for attempt in range(max_retries):
            try:
                user_prompt = f"Code chunk:\n```python\n{chunk_text}\n```"
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
            "⚠️ The output **must strictly follow this exact JSON structure**:\n"
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
            "⚠️ ALL fields must be included exactly as shown, even if you have to use placeholder values.\n"
            "🟡 If you cannot confidently extract a field, use \"unknown\", null, or a placeholder value.\n"
            "✅ Do not include any additional fields — **only return the fields shown above**.\n"
            "Respond only with valid JSON. No extra text, comments, or explanation."
        )

        for attempt in range(max_retries):
            try:
                user_prompt = f"Code analysis summary:\n{merged_summary}\n\nBased on this information, create the structured metadata JSON:"
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

    def extract_architecture_metadata(self, chunk_text: str, max_retries: int = 3) -> dict:
        """
        Extract metadata from a code chunk and format it directly in the final structure.
        This eliminates the need to call merge_partial_metadata for short scripts.
        """
        if not self.llm_interface:
            return self._create_default_metadata_structure()

        system_prompt = (
            "You are a code analysis assistant. "
            "Analyze the following Python code and extract metadata. "
            "You may include reasoning in a <thinking>...</thinking> block, but must output a valid JSON object afterward.\n\n"
            "⚠️ The output **must strictly follow this exact JSON structure**:\n"
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
            "⚠️ ALL fields must be included exactly as shown, even if you have to use placeholder values.\n"
            "🟡 If you cannot confidently extract a field, use \"unknown\", null, or a placeholder value.\n"
            "✅ Do not include any additional fields — **only return the fields shown above**.\n"
            "❌ Do not return metadata like 'visualization', 'id', 'tags', or any other fields.\n\n"
            "Respond only with <thinking>...</thinking> followed by the valid JSON. "
            "No markdown, no extra comments, and no content outside the JSON structure."
        )

        required_fields = ["description", "framework", "architecture", "dataset", "training_config"]
        training_config_fields = ["batch_size", "learning_rate", "optimizer", "epochs", "hardware_used"]

        # Create the final structure that will be returned
        result = {
            "description": None,
            "framework": {"name": None, "version": None},
            "architecture": {"type": None},
            "dataset": {"name": None},
            "training_config": {
                "batch_size": None,
                "learning_rate": None,
                "optimizer": None,
                "epochs": None,
                "hardware_used": None
            },
            "_trace": {}  # Keep track of where each piece of metadata came from
        }

        source_preview = chunk_text[:120]

        for attempt in range(max_retries):
            try:
                user_prompt = f"Code chunk:\n```python\n{chunk_text}\n```"
                response = self.llm_interface.generate_structured_response(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=0,
                    max_tokens=4000
                )
                print(f"Metadata extraction attempt {attempt + 1}: {response}")
                raw = response.get("content", "").strip()
                raw = re.sub(r"<thinking>.*?</thinking>", "", raw, flags=re.DOTALL).strip()
                json_candidate = re.search(r"\{.*\}", raw, re.DOTALL)

                if json_candidate:
                    try:
                        parsed = json.loads(json_candidate.group())

                        # Validate that all required fields are present
                        missing_fields = [field for field in required_fields if field not in parsed]
                        missing_tc_fields = []
                        if "training_config" in parsed and isinstance(parsed["training_config"], dict):
                            missing_tc_fields = [field for field in training_config_fields
                                                 if field not in parsed["training_config"]]

                        # Handle variants of "training_config" key
                        if "training_config" not in parsed:
                            # Check for variant keys
                            variant_keys = [key for key in parsed.keys()
                                            if key.lower().replace("_", "") == "trainingconfig"]

                            if variant_keys:
                                # Found a variant key, normalize it
                                variant_key = variant_keys[0]
                                parsed["training_config"] = parsed.pop(variant_key)
                                missing_fields = [f for f in missing_fields if f != "training_config"]

                        # Normalize training_config subfields if needed
                        if "training_config" in parsed and isinstance(parsed["training_config"], dict):
                            tc = parsed["training_config"]

                            # Map variant field names to standard field names
                            field_map = {
                                "batchsize": "batch_size",
                                "batch": "batch_size",
                                "learningrate": "learning_rate",
                                "lr": "learning_rate",
                                "numepochs": "epochs",
                                "epoch": "epochs",
                                "hardware": "hardware_used",
                                "gpu": "hardware_used"
                            }

                            # Process the field mapping
                            for variant, standard in field_map.items():
                                if variant in tc and standard not in tc:
                                    tc[standard] = tc.pop(variant)

                            # Recheck for missing fields after normalization
                            missing_tc_fields = [field for field in training_config_fields
                                                 if field not in tc]

                        # If fields are still missing after normalization, retry
                        if missing_fields or missing_tc_fields:
                            fields_str = ", ".join(missing_fields)
                            tc_fields_str = ", ".join(missing_tc_fields)
                            print(
                                f"[Retry {attempt + 1}] Missing fields: {fields_str}, Missing training_config fields: {tc_fields_str}")

                            # If it's the last attempt, continue with what we have
                            if attempt == max_retries - 1:
                                for field in missing_fields:
                                    if field == "description":
                                        parsed["description"] = "Unknown model purpose"
                                    elif field == "framework":
                                        parsed["framework"] = {"name": "unknown", "version": "unknown"}
                                    elif field == "architecture":
                                        parsed["architecture"] = {"type": "unknown"}
                                    elif field == "dataset":
                                        parsed["dataset"] = {"name": "unknown"}
                                    elif field == "training_config":
                                        parsed["training_config"] = {
                                            "batch_size": None,
                                            "learning_rate": None,
                                            "optimizer": None,
                                            "epochs": None,
                                            "hardware_used": None
                                        }

                                if "training_config" in parsed and isinstance(parsed["training_config"], dict):
                                    for field in missing_tc_fields:
                                        parsed["training_config"][field] = None
                            else:
                                # Not the last attempt, so try again
                                continue

                        # Now populate the result structure with the parsed data
                        # This directly creates the same structure that merge_partial_metadata would produce
                        for key in required_fields:
                            if key in parsed:
                                if isinstance(parsed[key], dict):
                                    for subkey, subval in parsed[key].items():
                                        if subval is not None:
                                            result[key][subkey] = subval
                                            result["_trace"][f"{key}.{subkey}"] = source_preview
                                else:
                                    if parsed[key] is not None:
                                        result[key] = parsed[key]
                                        result["_trace"][key] = source_preview

                        return result
                    except json.JSONDecodeError as e:
                        print(f"[Retry {attempt + 1}] Invalid JSON: {e}")
                else:
                    print(f"[Retry {attempt + 1}] No valid JSON object found in LLM response.")
            except Exception as e:
                print(f"[Retry {attempt + 1}] Metadata extraction failed: {e}")

        # After all retries, return the default structure
        default_result = self._create_default_metadata_structure()
        default_result["_trace"] = {"default": "Failed to extract metadata after all retries"}
        return default_result

    def _create_default_metadata_structure(self) -> dict:
        """Create the default metadata structure with null values."""
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

    def split_by_lines(self, file_content: str, chunk_size_in_lines: int = 500, overlap: int = 100):
        """
        Split code into chunks based on lines rather than AST.
        This avoids the problem of AST creating very small chunks.

        Args:
            file_content: The content of the file to split
            chunk_size_in_lines: Approximate target size of each chunk in lines
            overlap: Number of lines to overlap between chunks
            min_chunk_length: Minimum character length for a chunk to be included

        Returns:
            List of dictionaries containing chunk information
        """
        lines = file_content.splitlines(keepends=True)
        chunks = []

        # Helper function to get character offset from line number
        def get_offset(line_index):
            return sum(len(lines[i]) for i in range(min(line_index, len(lines))))

        # Process chunks of lines
        i = 0
        while i < len(lines):
            # Calculate end of this chunk (with respect to chunk_size)
            end = min(i + chunk_size_in_lines, len(lines))

            # Try to find a better splitting point by looking for empty lines
            # or lines that don't start with whitespace (likely a top-level definition)
            if end < len(lines):
                # Look ahead a bit to find a good split point
                for j in range(end, min(end + 20, len(lines))):
                    if not lines[j].strip() or not lines[j].startswith((' ', '\t')):
                        end = j
                        break

            # Extract the chunk
            chunk_lines = lines[i:end]
            chunk_text = "".join(chunk_lines)
            chunks.append({
                "text": chunk_text,
                "offset": get_offset(i),
                "type": "code",
                "line_range": (i, end)
            })

            # Move to next chunk, accounting for overlap
            i = end - overlap if end < len(lines) else end

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
