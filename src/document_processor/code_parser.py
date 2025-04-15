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
        """Parse a Python file and extract model information."""
        with open(file_path, "r", encoding="utf-8") as f:
            file_content = f.read()

        try:
            tree = ast.parse(file_content, filename=file_path)
        except SyntaxError as e:
            raise ValueError(f"Syntax error while parsing {file_path}: {e}")

        # LLM-based extraction (single call)
        self.llm_metadata_cache = self._extract_llm_metadata(file_content)

        model_info = {
            "creation_date": self._get_creation_date(file_path),
            "last_modified_date": self._get_last_modified_date(file_path)
        }

        # AST-based model metadata
        extracted_info = self._extract_model_info(tree)
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

        # Safely parse architecture
        arch = self.llm_metadata_cache.get("architecture", {})
        if isinstance(arch, str):
            model_info["architecture"] = {"type": arch}
        elif isinstance(arch, dict):
            model_info["architecture"] = {
                "type": arch.get("type") if isinstance(arch.get("type"), str) else None
            }
        else:
            model_info["architecture"] = {"type": None}

        # Safely parse dataset
        dataset = self.llm_metadata_cache.get("dataset", {})
        if isinstance(dataset, str):
            model_info["dataset"] = {"name": dataset}
        elif isinstance(dataset, dict):
            model_info["dataset"] = {
                "name": dataset.get("name") if isinstance(dataset.get("name"), str) else None
            }
        else:
            model_info["dataset"] = {"name": None}

        # Safely parse training_config
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

        # Description
        desc = self.llm_metadata_cache.get("description")
        model_info["description"] = desc if isinstance(desc, str) else "N/A"

        # Remaining fields
        model_info["is_model_script"] = True
        model_info["content"] = file_content

        print(f"updated model_info: {model_info}")
        return model_info

    def _extract_llm_metadata(self, code_str: str, max_retries: int = 3) -> dict:
        """Send code to the LLM and get structured metadata, retrying if response isn't valid JSON."""
        if not self.llm_interface:
            return {}

        system_prompt = (
            "You are a code analysis assistant. "
            "Analyze the following Python code and extract metadata. "
            "Your response must be a **single valid JSON object only** — no explanations, no markdown, no comments, no text before or after.\n\n"
            "The JSON should have the following structure:\n"
            '{\n'
            '  "description": "Short summary of what the model does (max 50 characters)",\n'
            '  "framework": { "name": "e.g. PyTorch", "version": "e.g. 1.13" },\n'
            '  "architecture": { "type": "CNN" },\n'
            '  "dataset": { "name": "MNIST" },\n'
            '  "training_config": {\n'
            '    "batch_size": 32,\n'
            '    "learning_rate": 0.001,\n'
            '    "optimizer": "Adam",\n'
            '    "epochs": 10,\n'
            '    "hardware_used": "GPU"\n'
            '  }\n'
            '}\n\n'
            'Allowed values for "architecture.type": "CNN", "RNN", "Transformer", or "other".\n'
            'If a field is not identifiable, use "unknown", "other", or null.\n'
            "All values must be valid JSON values.\n\n"
            "**Before responding, ensure that the output is valid JSON using a JSON validator. "
            "Double check that there is no extra text, no markdown, and no trailing commas.**"
        )

        user_prompt = f"Analyze the following code:\n```python\n{code_str}\n```"

        for attempt in range(1, max_retries + 1):
            try:
                result = self.llm_interface.generate_structured_response(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=0,
                    max_tokens=31000,
                )

                raw_content = result.get("content", "")
                if not isinstance(raw_content, str):
                    raise ValueError(f"Expected string in result['content'], got {type(raw_content)}")

                raw_content = raw_content.strip()
                print(f"[Attempt {attempt}] Raw content: {raw_content}")

                # First try raw
                try:
                    return json.loads(raw_content)
                except json.JSONDecodeError:
                    pass

                # Try simple cleanup: quote unquoted keys
                try:
                    cleaned_content = re.sub(
                        r'(?<!")(?P<key>[a-zA-Z_][a-zA-Z0-9_]*)(?=\s*:)', r'"\g<key>"', raw_content
                    )
                    parsed = json.loads(cleaned_content)
                    return parsed
                except json.JSONDecodeError:
                    print(f"[Attempt {attempt}] Failed to parse cleaned content.")

            except Exception as e:
                print(f"[Attempt {attempt}] LLM error: {e}")

            print(f"[Attempt {attempt}] Retrying...\n")

        print("[LLMParser] Failed to parse valid JSON after retries.")
        return {}

    def _extract_model_info(self, tree):
        """Extract model name and type from AST."""
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

    def _get_creation_date(self, file_path):
        """Get file creation date using Git history or fallback to filesystem."""
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
        """Get last modified date using Git history or fallback to filesystem."""
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

    def split_ast_and_subsplit_chunks(self, code: str, chunk_size: int = 500, overlap: int = 100):
        """
        Split code by AST blocks and subchunk each block with overlap.
        Returns list of dicts: [{ "text": str, "source_block": str, "offset": int }]
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return [{"text": code, "source_block": code, "offset": 0}]

        lines = code.splitlines(keepends=True)
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
                        "source_block": block_code,
                        "offset": block_start_char_offset + j
                    })

        return chunks
