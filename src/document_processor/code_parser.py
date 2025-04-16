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

        self.llm_metadata_cache = self._extract_llm_metadata(file_content, file_path, max_retries=5)

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

    def _extract_llm_metadata(self, code_str: str, file_path: str, max_retries: int = 5) -> dict:
        # Define a threshold to decide if chunking is needed
        SHORT_SCRIPT_CHAR_LIMIT = 8000  # tune this based on LLM input limits

        if len(code_str) <= SHORT_SCRIPT_CHAR_LIMIT:
            print("Using full script for LLM metadata extraction.")

            partial = self.extract_partial_metadata(chunk_text=code_str, chunk_offset=0, max_retries=max_retries)
            if partial:
                final = self.merge_partial_metadata([partial])
            else:
                final = {}
            print(f"Final metadata (full): {final}")
        else:
            print(f"Script is large, using chunked extraction")
            all_chunks = self.split_ast_and_subsplit_chunks(
                file_content=code_str,
                file_path=file_path,
                chunk_size=8000,
                overlap=0,
                min_chunk_length=200
            )

            partials = []
            for chunk in all_chunks:
                chunk_text = chunk['text']
                offset = chunk['offset']
                partial = self.extract_partial_metadata(chunk_text, chunk_offset=offset, max_retries=max_retries)
                if partial:
                    partials.append(partial)

            merged = self.merge_partial_metadata(partials)
            final = self.run_final_consistency_verifier(merged, max_retries=max_retries)

            print(f"Final metadata (chunk merged): {final}")

        final.pop("_trace", None)
        return final

    def extract_partial_metadata(self, chunk_text: str, chunk_offset: int = 0, max_retries: int = 3) -> dict:
        if not self.llm_interface:
            return {}

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

        for attempt in range(max_retries):
            try:
                user_prompt = f"Code chunk:\n```python\n{chunk_text}\n```"
                response = self.llm_interface.generate_structured_response(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=0,
                    max_tokens=50000
                )
                print(f"partial metadata response (attempt {attempt + 1}): {response}")
                raw = response.get("content", "").strip()
                raw = re.sub(r"<thinking>.*?</thinking>", "", raw, flags=re.DOTALL).strip()
                json_candidate = re.search(r"\{.*\}", raw, re.DOTALL)

                if json_candidate:
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

                        # If it's the last attempt, add default values to missing fields
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

                    # At this point we have a valid structure
                    json_result = {
                        "metadata": parsed,
                        "source_offset": chunk_offset,
                        "source_preview": chunk_text[:120]
                    }
                    print(f"parsed partial metadata response: {json_result}")
                    return json_result
                else:
                    print(f"[Retry {attempt + 1}] No valid JSON object found in LLM response.")
            except Exception as e:
                print(f"[Retry {attempt + 1}] Chunk metadata extraction failed: {e}")

        # After all retries, return a basic structure with default values
        return {
            "metadata": {
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
            },
            "source_offset": chunk_offset,
            "source_preview": chunk_text[:120]
        }

    def merge_partial_metadata(self, partials: list) -> dict:
        allowed_keys = {
            "description", "framework", "architecture", "dataset", "training_config"
        }

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
            "_trace": {}
        }

        print(f"partial_metadatas to be merged: {partials}")
        for p in partials:
            meta = p.get("metadata", {})
            src = p.get("source_preview", "unknown chunk")
            for key, val in meta.items():
                if key not in allowed_keys:
                    continue  # ⛔ Ignore extra keys like "visualization"
                if isinstance(val, dict):
                    for subkey, subval in val.items():
                        if subval and not result[key].get(subkey):
                            result[key][subkey] = subval
                            result["_trace"][f"{key}.{subkey}"] = src
                else:
                    if val and not result.get(key):
                        result[key] = val
                        result["_trace"][key] = src

        print(f"merged result: {result}")
        return result

    def run_final_consistency_verifier(self, merged_metadata: dict, max_retries: int = 3) -> dict:
        if not self.llm_interface:
            return merged_metadata

        # Define required fields and structure
        required_fields = ["description", "framework", "architecture", "dataset", "training_config"]
        training_config_fields = ["batch_size", "learning_rate", "optimizer", "epochs", "hardware_used"]

        system_prompt = (
            "You are a metadata consistency checker.\n"
            "Given a possibly noisy JSON object, return a cleaned version that strictly follows the required structure.\n\n"
            "⚠️ The output **must include all of the following fields**:\n"
            "- description (string)\n"
            "- framework (object with name and version)\n"
            "- architecture (object with type)\n"
            "- dataset (object with name)\n"
            "- training_config (object with batch_size, learning_rate, optimizer, epochs, and hardware_used)\n\n"
            "If any of these fields are missing in the input, add them with placeholder values like 'unknown' or null.\n"
            "Ensure all training_config fields (batch_size, learning_rate, optimizer, epochs, hardware_used) are present.\n"
            "Normalize any variant field names to the standard names listed above.\n"
            "Never add values not present in the input unless resolving a clear contradiction or fixing missing required fields.\n"
        )

        for attempt in range(max_retries):
            try:
                user_prompt = f"Clean this metadata:\n```json\n{json.dumps(merged_metadata, indent=2)}\n```"
                response = self.llm_interface.generate_structured_response(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=0,
                    max_tokens=50000
                )
                print(f"final consistency response: {response}")
                raw = response.get("content", "").strip()
                json_candidate = re.search(r"\{.*\}", raw, re.DOTALL)
                if json_candidate:
                    json_result = json.loads(json_candidate.group())

                    # Validate that all required fields are present
                    missing_fields = [field for field in required_fields if field not in json_result]
                    missing_tc_fields = []
                    if "training_config" in json_result and isinstance(json_result["training_config"], dict):
                        missing_tc_fields = [field for field in training_config_fields
                                             if field not in json_result["training_config"]]

                    # If fields are still missing, add defaults
                    if missing_fields or missing_tc_fields:
                        print(
                            f"[Verifier] Adding missing fields: {missing_fields}, Missing TC fields: {missing_tc_fields}")

                        for field in missing_fields:
                            if field == "description":
                                json_result["description"] = merged_metadata.get(
                                    "description") or "Unknown model purpose"
                            elif field == "framework":
                                json_result["framework"] = merged_metadata.get("framework") or {"name": "unknown",
                                                                                                "version": "unknown"}
                            elif field == "architecture":
                                json_result["architecture"] = merged_metadata.get("architecture") or {"type": "unknown"}
                            elif field == "dataset":
                                json_result["dataset"] = merged_metadata.get("dataset") or {"name": "unknown"}
                            elif field == "training_config":
                                json_result["training_config"] = merged_metadata.get("training_config") or {
                                    "batch_size": None,
                                    "learning_rate": None,
                                    "optimizer": None,
                                    "epochs": None,
                                    "hardware_used": None
                                }

                        if "training_config" in json_result and isinstance(json_result["training_config"], dict):
                            orig_tc = merged_metadata.get("training_config", {})
                            for field in missing_tc_fields:
                                json_result["training_config"][field] = orig_tc.get(field, None)

                    print(f"final consistency result: {json_result}")
                    return json_result
                else:
                    print(f"[Verifier Retry {attempt + 1}] No valid JSON object found in response.")
            except Exception as e:
                print(f"[Verifier Retry {attempt + 1}] Metadata post-clean failed: {e}")

        # If all retries fail, ensure the original metadata has all required fields
        for field in required_fields:
            if field not in merged_metadata:
                if field == "description":
                    merged_metadata["description"] = "Unknown model purpose"
                elif field == "framework":
                    merged_metadata["framework"] = {"name": "unknown", "version": "unknown"}
                elif field == "architecture":
                    merged_metadata["architecture"] = {"type": "unknown"}
                elif field == "dataset":
                    merged_metadata["dataset"] = {"name": "unknown"}
                elif field == "training_config":
                    merged_metadata["training_config"] = {
                        "batch_size": None,
                        "learning_rate": None,
                        "optimizer": None,
                        "epochs": None,
                        "hardware_used": None
                    }

        if "training_config" in merged_metadata and isinstance(merged_metadata["training_config"], dict):
            for field in training_config_fields:
                if field not in merged_metadata["training_config"]:
                    merged_metadata["training_config"][field] = None

        return merged_metadata

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

    def split_ast_and_subsplit_chunks(self, file_content: str, file_path: str, chunk_size: int = 500, overlap: int = 100,
                                      min_chunk_length: int = 0):
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

                # Skip short import-only chunks
                if (len(chunk_text.strip()) < min_chunk_length and
                        (chunk_text.strip().startswith("import") or chunk_text.strip().startswith("from"))):
                    continue

                if chunk_text.strip():
                    chunks.append({
                        "text": chunk_text,
                        "offset": block_start_char_offset + j,
                        "type": "code",  # you can expand logic later if needed
                        "source_block": block_code
                    })

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