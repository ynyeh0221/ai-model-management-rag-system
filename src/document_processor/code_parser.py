import ast
import datetime
import json
import os
import re

from git import Repo


class CodeParser:
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

        desc = self.llm_metadata_cache.get("description")
        model_info["description"] = desc if isinstance(desc, str) else "N/A"

        model_info["is_model_script"] = True
        model_info["content"] = file_content

        print(f"updated model_info: {model_info}")
        return model_info

    def _extract_llm_metadata(self, code_str: str, max_retries: int = 10) -> dict:
        # Define a threshold to decide if chunking is needed
        SHORT_SCRIPT_LINE_LIMIT = 100

        lines = code_str.splitlines()

        if len(lines) <= SHORT_SCRIPT_LINE_LIMIT:
            print("Using full script for LLM metadata extraction.")
            # For short scripts, we can still use the direct JSON approach
            final = self.extract_architecture_metadata(chunk_text=code_str, max_retries=max_retries)
            print(f"Final metadata (full): {final}")
        else:
            print(f"Script is large, using chunked extraction with summaries")
            all_chunks = self.split_by_lines(
                file_content=code_str,
                chunk_size=SHORT_SCRIPT_LINE_LIMIT,
                overlap=0,
            )
            print(f"Chunk counts: {len(all_chunks)}")

            # Extract summaries from each chunk
            chunk_summaries = []
            for chunk in all_chunks:
                chunk_text = chunk['text']
                offset = chunk['offset']
                summary = self.extract_chunk_summary(chunk_text, chunk_offset=offset, max_retries=max_retries)
                if summary:
                    chunk_summaries.append(summary)

            # Merge summaries
            merged_summary = self.merge_chunk_summaries(chunk_summaries)
            print(f"Merged summary: {merged_summary}")

            # Generate final JSON from the merged summary
            final = self.generate_metadata_from_summary(merged_summary, max_retries=max_retries)

            print(f"Final metadata (from summaries): {final}")

        final.pop("_trace", None)
        return final

    def extract_chunk_summary(self, chunk_text: str, chunk_offset: int = 0, max_retries: int = 3) -> dict:
        """Extract a natural language summary of metadata from a code chunk instead of structured JSON."""
        if not self.llm_interface:
            return {}

        system_prompt = (
            "You are a code analysis assistant. "
            "Analyze the following Python code chunk and extract any information relevant to the following metadata categories:\n\n"
            "- Description: What the model or code does\n"
            "- Framework: ML framework (like PyTorch, TensorFlow) and version\n"
            "- Architecture: Neural network architecture type\n"
            "- Dataset: Training data used\n"
            "- Training configuration: batch size, learning rate, optimizer, epochs, hardware used\n\n"
            f"Provide an extremely brief summary in 200 characters or less. Be direct and compact. "
            "Focus only on what's explicitly in the code. "
            "If you can't find certain information, don't mention that category. Only include information that appears in this code chunk."
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
                print(f"chunk summary response (attempt {attempt + 1}): {response}")
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
        print(f"Merged summaries: {merged_summary}")
        return merged_summary

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
                json_candidate = re.search(r"\{.*\}", raw, re.DOTALL)

                if json_candidate:
                    try:
                        parsed = json.loads(json_candidate.group())
                        # Check that the JSON has the required structure
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

    def split_by_lines(self, file_content: str, chunk_size: int = 500, overlap: int = 100):
        """
        Split code into chunks based on lines rather than AST.
        This avoids the problem of AST creating very small chunks.

        Args:
            file_content: The content of the file to split
            chunk_size: Approximate target size of each chunk in lines
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
            end = min(i + chunk_size, len(lines))

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