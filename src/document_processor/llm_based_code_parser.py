import ast
import datetime
import json
import os
import re
from typing import List, Tuple

from git import Repo

from src.document_processor.ast_summary_generator import ASTSummaryGenerator

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
        self.ast_summary_generator = ASTSummaryGenerator()

    def parse(self, file_path):
        ext = os.path.splitext(file_path)[1].lower()
        if ext in ['.py', '.ipynb']:
            return self.parse_file(file_path)
        return None

    def parse_file(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            file_content = f.read()

        self.llm_metadata_cache = self._extract_metadata_by_llm(file_content, file_path, max_retries=15)

        model_info = {
            "creation_date": self._get_creation_date(file_path),
            "last_modified_date": self._get_last_modified_date(file_path)
        }

        extracted_info = self._extract_model_info(file_content, file_path)
        model_info.update(extracted_info)

        # Safely parse framework
        framework = self.llm_metadata_cache.get("framework", {})
        if isinstance(framework, dict):
            model_info["framework"] = {
                "name": framework.get("name") if isinstance(framework.get("name"), str) else None,
                "version": framework.get("version") if isinstance(framework.get("version"), str) else None
            }
        else:
            model_info["framework"] = {"name": None, "version": None}

        arch = self.llm_metadata_cache.get("architecture", {})
        if isinstance(arch, dict):
            model_info["architecture"] = arch
        else:
            model_info["architecture"] = {"type": None, "reason": None}

        dataset = self.llm_metadata_cache.get("dataset", {})
        if isinstance(dataset, dict):
            model_info["dataset"] = {
                "name": dataset.get("name") if isinstance(dataset.get("name"), str) else None
            }
        else:
            model_info["dataset"] = {"name": None}

        images_folder = self.llm_metadata_cache.get("images_folder", {})
        if isinstance(images_folder, dict):
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

    def get_images_folder(self, text):
        # splits on the literal and takes the part before the newline
        try:
            return text.split("Images folder:")[1].split("\n", 1)[0]
        except IndexError:
            return None

    def _extract_metadata_by_llm(self, code_str: str, file_path: str = "<unknown>", max_retries: int = 15) -> dict:
        # STEP 1: Generate AST digest/summary
        ast_digest = self.clean_empty_lines(self.ast_summary_generator.generate_summary(code_str=code_str, file_path=file_path))
        # print(f"Total AST digest: {ast_digest}")

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

        # STEP 5: Add framework
        final['framework'] = {'name': self.parse_framework(ast_digest), 'version': 'missing'}

        # STEP 6: Add images folder. Tried to use LLM for parsing, but finally moved back to manual parse way since this field needs exact value
        final['images_folder'] = {'name': self.get_images_folder(ast_digest)}

        # STEP 7: Remove unneeded fields
        final.pop("_trace", None)

        print(f"Final metadata (from AST summary): {final}")
        return final

    def parse_framework(self, ast_summary: str) -> str:
        """
        Detect the deep-learning framework used, based on imports in the AST summary.

        Args:
            ast_summary: A single string containing lines like "Import: torch" or "From tensorflow import ...".

        Returns:
            The framework name: one of "PyTorch", "TensorFlow", "JAX", or "missing" if none recognized.
        """
        # Normalize to lower-case for matching
        txt = ast_summary.lower()

        # Check in priority order
        if re.search(r"\bimport:\s+torch\b|\bfrom\s+torch\b", txt):
            return "PyTorch"
        if re.search(r"\bimport:\s+tensorflow\b|\bfrom\s+tensorflow\b|\btf\.", txt):
            return "TensorFlow"
        if re.search(r"\bimport:\s+jax\b|\bfrom\s+jax\b", txt):
            return "JAX"
        if re.search(r"\bimport:\s+keras\b|\bfrom\s+keras\b", txt):
            # note: standalone Keras
            return "TensorFlow"  # Keras now lives under TF
        return "missing"

    def extract_natural_language_summary(self, chunk_text: str, chunk_offset: int = 0, max_retries: int = 3) -> dict:
        """Extract a natural language summary of metadata from a code chunk instead of structured JSON."""
        if not self.llm_interface:
            return {}

        # Reduce input size of LLM
        if len(chunk_text) >= 2500:
            chunk_text = self.filter_ast_summary_for_metadata(chunk_text)

        system_prompt = (
            "You are a senior machine-learning architect documenting Python training scripts. "
            "Your audience is **junior ML engineers** who will read your report to understand, reproduce, "
            "and extend the model. Therefore your language must be clear, define all technical terms, "
            "and leave no gaps in explanation.\n\n"
            "Given an AST summary of a script, **carefully examine every detail** to produce a complete, human-readable English report.\n\n"

            "**CRITICAL REQUIREMENTS**:\n"
            "• Thoroughly analyze the AST summary before writing – do not skip or overlook any nodes or attributes.\n"
            "• Ensure all model components, configurations, and operations in the AST are reflected in your summary.\n"
            "• Pay special attention to hyperparameters, layer dimensions, conditional logic, and data transformations.\n"
            "• **Do not invent or hallucinate any information**; only describe what is actually present in the AST summary.\n\n"

            "Produce a report with these sections:\n\n"
            "1. Purpose:\n"
            "   • In 1–2 sentences, precisely describe what the script accomplishes, mentioning all major features present in the AST.\n\n"

            "2. Data & Preprocessing:\n"
            "   • **Comprehensively document** all dataset classes, sources, and directories from the AST.\n"
            "   • List **every** transformation and preprocessing step, including normalization, augmentation, or custom functions.\n"
            "   • Detail **all** DataLoader configurations: batch sizes, shuffle settings, train/val/test splits, num_workers, etc.\n\n"

            "3. Model Architecture:\n"
            "   • Describe **every layer** or block found in the AST using fluent prose. For example:\n"
            "     \"The first convolutional layer applies 32 filters of size 3×3 to the 1-channel 28×28 input, producing 32 feature maps of size 26×26. Next, a ReLU activation introduces non-linearity, followed by a 2×2 max-pool that halves the spatial dimensions to 13×13.\"\n"
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
            "• Write in natural, fluent prose – no code snippets or raw AST output.\n"
            "• Be comprehensive – if it's in the AST, it must be in your summary.\n"
            "• Double-check that your summary includes ALL components mentioned in the AST.\n"
            "• Ensure an ML engineer could reproduce the exact workflow from your description.\n\n"

            "Remember: Missing important details or adding information not present in the AST means failing the task. Be meticulous and thorough."
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
            "You are a senior machine-learning architect. Your job is to extract high-quality, expert-level metadata from ML code. "
            "This JSON will feed into our model registry, auto-generate training dashboards, and ensure full reproducibility of experiments. "
            "Based on the following model AST digest summary, create a structured representation of the model metadata.\n\n"
            "The output **must strictly follow this exact JSON structure**:\n"
            "{\n"
            '  \"architecture\": { \"type\": \"...\", \"reason\": \"...\" },\n'
            '  \"dataset\": { \"name\": \"...\", \"reason\": \"...\" },\n'
            '  \"training_config\": {\n'
            '    \"batch_size\": 32,\n'
            '    \"learning_rate\": 0.001,\n'
            '    \"optimizer\": \"Adam\",\n'
            '    \"epochs\": 10,\n'
            '    \"hardware_used\": \"GPU\"\n'
            '  }\n'
            "}\n\n"
            "Extraction hints:\n"
            "• **architecture**: Do NOT simply copy a layer or class name from the AST. Instead, review *all* detected components (e.g., Conv2d, Linear, MultiHeadAttention, reparameterize, UNetAttentionBlock, etc.) and infer the overarching model paradigm (e.g. “Variational Autoencoder”, “Transformer”, “UNet”, “GAN”). Format as: { \"type\": \"<InferredArchitecture>\", \"reason\": \"<concise justification citing multiple AST cues>\" }\n"
            "• **dataset**: scan the *entire* AST summary for dataset definitions or loader calls (e.g. `datasets.MNIST`, custom Dataset subclasses, DataLoader instantiations, file paths). Extract the dataset identifier and **format this field as a JSON object**: `{ \"name\": \"<DatasetName>\", \"reason\": \"<concise reason citing the exact AST cues>\" }`.\n"
            "• **batch_size**: look for `batch_size=` in DataLoader; else null.\n"
            "• **learning_rate**: look for `lr=` or “learning rate”; else null.\n"
            "• **optimizer**: look for optimizer names (Adam, SGD); else “missing”.\n"
            "• **epochs**: look for `epochs =`; else null.\n"
            "• **hardware_used**: look for device settings (`cuda`, `mps`, `cpu`); map to “GPU”, “CePU” or “Both”; else “missing”.\n\n"
            "• **Do not invent or hallucinate** any values—only extract information actually present in the AST summary; if a field cannot be found, use null or “missing” as specified.\n\n"
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
                    except json.JSONDecodeError as e:
                        print(f"[Retry {attempt + 1}] Invalid JSON format: {e}, file_path: {file_path}")
                else:
                    print(f"[Retry {attempt + 1}] No valid JSON object found in LLM response, file_path: {file_path}")
            except Exception as e:
                print(f"[Retry {attempt + 1}] Metadata generation failed: {e}, file_path: {file_path}")

        # After all retries, return default metadata
        return self._create_default_metadata()

    def _create_default_metadata(self) -> dict:
        """Create default metadata structure with empty/null values."""
        return {
            "architecture": {"type": "missing"},
            "dataset": {"name": "missing"},
            "imaged_folder": {"name": "missing"},
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
        required_fields = ["architecture", "dataset", "training_config"]
        if not all(field in metadata for field in required_fields):
            return False, "Missing required fields"

        # Check that architecture has type
        if not isinstance(metadata["architecture"], dict) or "type" not in metadata["architecture"] or metadata["architecture"].get("type") is None or \
                metadata["architecture"].get("type").lower() == "missing" or metadata["architecture"].get("type").lower() == "unknown" or metadata["architecture"].get("type").lower() == "hybrid" or metadata["architecture"].get("type").lower() == "mixed" or metadata["architecture"].get("type").lower() == "both" or \
                metadata["architecture"].get("type").lower() == "n/a" or "reason" not in metadata["architecture"] or metadata["architecture"].get("reason") is None or \
                metadata["architecture"].get("reason").lower() == "missing" or metadata["architecture"].get("type").lower() == "pytorch" or metadata["architecture"].get("type").lower() == "gpu" or metadata["architecture"].get("type").lower() == "cpu":
            return False, "Invalid architecture structure"

        # Check that dataset has name
        if not isinstance(metadata["dataset"], dict) or "name" not in metadata["dataset"]:
            return False, "Invalid dataset structure"

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
