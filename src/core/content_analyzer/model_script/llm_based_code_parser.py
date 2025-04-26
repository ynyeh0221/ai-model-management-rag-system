import ast
import datetime
import json
import os
import re
from typing import List, Tuple

from git import Repo

from src.core.content_analyzer.model_script.ast_summary_generator import ASTSummaryGenerator

def filter_ast_summary_for_metadata(summary: str) -> str:
    """
    Given a AST digest summary, return only the lines that
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
    else: # For long summary, skip some information to trade off LLM's input size
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

def sanitize_json_string(json_str: str) -> str:
    # Remove JS-style comments
    json_str = re.sub(r"//.*?$", "", json_str, flags=re.MULTILINE)
    # Remove trailing commas
    json_str = re.sub(r",(\s*[}\]])", r"\1", json_str)
    return json_str

def clean_empty_lines(text: str) -> str:
    # Split into lines, strip each line, and remove if it's empty
    lines = text.splitlines()
    cleaned = [line.rstrip() for line in lines if line.strip() != '']

    # Join lines back with a single newline
    return '\n'.join(cleaned)

def parse_framework(ast_summary: str) -> str:
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

def get_images_folder(text):
    # splits on the literal and takes the part before the newline
    try:
        return text.split("Images folder:")[1].split("\n", 1)[0]
    except IndexError:
        return None

def get_last_modified_date(file_path):
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

def get_creation_date(file_path):
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


def split_summary_into_chunks(summary_text, overlap_sentences=2, max_sentences_per_chunk=10):
    """
    Splits a summary into semantic chunks based on headings (markdown # or underlined headings),
    with an overlap of sentences between chunks. Falls back to fixed-size chunking if no headings detected.

    Parameters:
    - summary_text (str): The full summary text.
    - overlap_sentences (int): Number of sentences to overlap between chunks.
    - max_sentences_per_chunk (int): Maximum sentences per chunk when no headings detected.

    Returns:
    - List[Dict]: A list of dictionaries with "description" and "offset" keys.
    """
    # Import and download NLTK resources
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    sentences = nltk.sent_tokenize(summary_text)

    chunks = []
    total = len(sentences)
    current_offset = 0

    for start in range(0, total, max_sentences_per_chunk):
        # Calculate starting position for this chunk
        chunk_start = start
        if start > 0:
            chunk_start = start - overlap_sentences

        # Get sentences for this chunk
        chunk_sentences = sentences[chunk_start:start + max_sentences_per_chunk]
        chunk_text = ' '.join(chunk_sentences)

        # Calculate offset in characters from the original text
        if chunk_start == 0:
            offset = 0
        else:
            # Calculate offset by counting characters up to the start of this chunk
            offset = len(' '.join(sentences[:chunk_start]))
            # Add 1 for the space that would follow the last sentence
            if offset > 0:
                offset += 1

        # Only add chunks that have meaningful content
        if len(chunk_text) > 5:
            chunks.append({
                "description": chunk_text,
                "offset": offset
            })

    return chunks

def split_code_chunks_via_ast(file_content: str, file_path: str, chunk_size: int = 500, overlap: int = 100):
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

def validate_llm_metadata_structure(metadata: dict) -> Tuple[bool, str]:
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

def create_default_llm_metadata() -> dict:
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

class LLMBasedCodeParser:
    """
    LLMBasedCodeParser: An advanced parser for extracting structured metadata from machine learning code files.

    This class combines traditional AST (Abstract Syntax Tree) parsing with large language model (LLM)
    analysis to extract comprehensive metadata from Python ML model files. It identifies key architectural
    patterns, training configurations, dataset information, and other critical metadata that describes
    the implementation and purpose of ML models.

    Key Features:
    -------------
    - Hybrid parsing approach combining static AST analysis with LLM-powered semantic understanding
    - Extraction of architectural patterns beyond simple class/function names
    - Identification of training hyperparameters, optimization settings, and hardware requirements
    - Dataset detection and categorization
    - Comprehensive natural language summarization of code functionality
    - Structured JSON metadata generation for model registries and reproducibility systems

    Processing Pipeline:
    -------------------
    1. AST Digest Generation: Creates a structured digest of code cli_response_utils using Python's AST module
    2. Filtered Metadata Extraction: Identifies key imports, classes, docstrings, and variables
    3. LLM-Based Analysis: Leverages LLM capabilities to interpret code structures semantically
    4. Natural Language Summarization: Generates human-readable descriptions of code functionality
    5. Structured Metadata Extraction: Produces standardized JSON metadata for downstream systems

    Attributes:
    ----------
    schema_validator : object, optional
        Validator for ensuring metadata conforms to expected schema structure.
    llm_interface : object, optional
        Interface to language model API for code summarization and analysis.
    llm_metadata_cache : dict
        Storage for intermediate results from LLM API calls and processing.
    ast_summary_generator : ASTSummaryGenerator
        Component for generating AST-based summaries of code files.

    Methods:
    -------
    parse(file_path):
        Main entry point for parsing a file. Validates file type and delegates to appropriate parser.

    parse_file(file_path):
        Comprehensive parser for supported Python files (.py, .ipynb). Extracts metadata and info.

    extract_metadata_by_llm(code_str, file_path, max_retries=15):
        Orchestrates the multi-step process of extracting metadata using AST and LLM analysis.

    extract_natural_language_summary(chunk_text, chunk_offset=0, max_retries=3):
        Generates a comprehensive natural language description of code functionality.

    generate_metadata_from_ast_summary(summary, max_retries=3, file_path="<unknown>"):
        Produces structured JSON metadata from AST digest summaries using LLM analysis.

    extract_model_info(file_content, file_path):
        Extracts basic model identifiers and version info using AST analysis.

    Examples:
    --------
    >>> parser = LLMBasedCodeParser(schema_validator=validator, llm_interface=llm_api)
    >>> metadata = parser.parse("models/transformer.py")
    >>> print(f"Architecture: {metadata['architecture']['type']}")
    >>> print(f"Dataset: {metadata['dataset']['name']}")
    >>> print(f"Training config: {metadata['training_config']}")

    Notes:
    -----
    - The parser requires valid Python syntax in input files.
    - Performance and accuracy depend on the quality of the LLM interface implementation.
    - For large code files, the parser applies selective filtering to focus on key metadata elements.
    - The parser implements multiple retry mechanisms for API resilience.
    - Git integration provides creation and modification timestamps when available.

    See Also:
    --------
    ASTSummaryGenerator : Component class for generating AST digests from code.
    filter_ast_summary_for_metadata : Function for reducing AST summary size.
    validate_llm_metadata_structure : Function for validating metadata structure.
    """

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

        self.llm_metadata_cache = self.extract_metadata_by_llm(file_content, file_path, max_retries=15)

        model_info = {
            "creation_date": get_creation_date(file_path),
            "last_modified_date": get_last_modified_date(file_path)
        }

        extracted_info = self.extract_model_info(file_content, file_path)
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

    def extract_metadata_by_llm(self, code_str: str, file_path: str = "<unknown>", max_retries: int = 15) -> dict:
        # STEP 1: Generate AST digest/summary
        ast_digest = clean_empty_lines(self.ast_summary_generator.generate_summary(code_str=code_str, file_path=file_path))
        # print(f"Total AST digest: {ast_digest}")

        # STEP 2: Extract natural-language summary for each digest chunk
        summary = self.extract_natural_language_summary(
            chunk_text=ast_digest,
            chunk_offset=0,
            max_retries=max_retries
        )
        print(f"Natural language summary from AST digest: {summary}")

        # STEP 3: Feed merged summary to LLM for structured metadata generation
        final = self.generate_metadata_from_ast_summary(ast_digest, max_retries=max_retries, file_path=file_path)
        final['chunk_descriptions'] = split_summary_into_chunks(summary_text=summary.get("summary", ""), overlap_sentences=0, max_sentences_per_chunk=1)
        print(f"Chunk descriptions count: {len(final['chunk_descriptions'])}")

        # STEP 4: Store AST digest summary
        final['ast_summary'] = ast_digest

        # STEP 5: Add framework
        final['framework'] = {'name': parse_framework(ast_digest), 'version': '2.7'}

        # STEP 6: Add image_processing folder. Tried to use LLM for parsing, but finally moved back to manual parse way since this field needs exact value
        final['images_folder'] = {'name': get_images_folder(ast_digest)}

        # STEP 7: Remove unneeded fields
        final.pop("_trace", None)

        print(f"Final metadata (from AST summary): {final}")
        return final

    def extract_natural_language_summary(self, chunk_text: str, chunk_offset: int = 0, max_retries: int = 3) -> dict:
        """Extract a natural language summary of metadata from a code chunk instead of structured JSON."""
        if not self.llm_interface:
            return {}

        # Reduce input size of LLM
        if len(chunk_text) >= 2500:
            chunk_text = filter_ast_summary_for_metadata(chunk_text)

        system_prompt = (
            "### ML SCRIPT DOCUMENTATION SPECIALIST ROLE AND PURPOSE\n\n"
            "You are a senior machine-learning architect specializing in documenting Python training scripts for ML engineers. Your documentation will help them understand, reproduce, and extend machine learning models with clarity and precision. Your expertise bridges the gap between complex code structures and practical implementation guidance.\n\n"

            "### INPUT FORMAT SPECIFICATION\n\n"
            "The input you will receive is an Abstract Syntax Tree (AST) digest summary of a Python ML training model_script. This AST summary contains structured information about:\n"
            "- Imported libraries and modules\n"
            "- Class and function definitions\n"
            "- Model architecture cli_response_utils and parameters\n"
            "- Data loading and preprocessing configurations\n"
            "- Training loop and optimization settings\n"
            "- Evaluation procedures and metrics\n"
            "- Output and logging mechanisms\n\n"
            "The AST digest represents the model_script's structure and logic without including the full source code. Your task is to transform this technical summary into comprehensive, human-readable English documentation.\n\n"

            "### THINKING PROCESS REQUIREMENTS\n\n"
            "Before drafting your documentation, engage in thorough analytical reasoning that demonstrates:\n"
            "- Systematic analysis of every node and relationship in the provided AST summary\n"
            "- Identification of all model cli_response_utils, hyperparameters, and data transformations\n"
            "- Recognition of architectural patterns and design principles in the implementation\n"
            "- Consideration of how each component contributes to the overall training pipeline\n"
            "- Mapping of code structures to ML concepts that junior engineers need to understand\n"
            "- Verification that no elements from the AST summary are overlooked or misinterpreted\n"
            "- Validation that your explanations are derived solely from evidence in the AST\n\n"

            "### DOCUMENTATION STRUCTURE REQUIREMENTS\n\n"
            "Produce a comprehensive report with these clearly defined sections:\n\n"

            "1. **Purpose and Overview**\n"
            "   - Provide a precise 1-2 sentence description of what the model_script accomplishes\n"
            "   - Mention all major features and functionalities present in the AST\n"
            "   - Frame the model_script's role within typical ML workflows\n\n"

            "2. **Data Pipeline and Preprocessing**\n"
            "   - Document all dataset classes, sources, and directory structures\n"
            "   - Detail every transformation and preprocessing step with parameters\n"
            "   - Explain all DataLoader configurations and their significance:\n"
            "     * Batch sizes and their impact on training\n"
            "     * Shuffle settings and randomization strategies\n"
            "     * Train/validation/test splits with ratios\n"
            "     * Worker configurations and data loading optimizations\n"
            "   - Describe any custom preprocessing functions or augmentation techniques\n"
            "   - Explain shape transformations throughout the preprocessing pipeline\n\n"

            "3. **Model Architecture**\n"
            "   - Use fluid, technical prose to describe each layer or component\n"
            "   - For each architectural element, systematically explain:\n"
            "     * Its functional purpose and role in the network\n"
            "     * Exact parameterization (dimensions, kernel sizes, strides, padding)\n"
            "     * Input and output tensor shapes and transformations\n"
            "     * Activation functions and their effects\n"
            "     * Regularization techniques (dropout rates, batch normalization, etc.)\n"
            "   - Document component connections and information flow through the network\n"
            "   - Explain any custom modules, skip connections, or architectural innovations\n"
            "   - Visualize the architecture through clear textual descriptions of layer sequences\n\n"

            "4. **Training Configuration**\n"
            "   - Document all training hyperparameters with precision:\n"
            "     * Optimizer selection with all parameters (learning rate, momentum, weight decay)\n"
            "     * Learning rate scheduling strategies and parameters\n"
            "     * Loss function formulations and any custom modifications\n"
            "     * Training duration (epochs) and convergence criteria\n"
            "     * Batch sizes and their relationship to model performance\n"
            "     * Hardware utilization settings (device allocation, parallelization)\n"
            "   - Explain gradient handling techniques (clipping, accumulation, etc.)\n"
            "   - Detail any mixed precision or performance optimization strategies\n"
            "   - Document seed settings and reproducibility considerations\n\n"

            "5. **Evaluation and Testing Methodology**\n"
            "   - Outline all evaluation procedures and testing protocols:\n"
            "     * Validation frequency, methodology, and criteria\n"
            "     * Performance metrics calculation and thresholds\n"
            "     * Test data handling and evaluation procedures\n"
            "     * Model selection and checkpoint strategies\n"
            "   - Detail all metrics computed and their significance:\n"
            "     * Classification metrics (accuracy, precision, recall, F1, etc.)\n"
            "     * Regression metrics (MSE, MAE, R², etc.)\n"
            "     * Multi-class analysis tools (confusion matrices, ROC curves)\n"
            "   - Explain any custom evaluation logic or specialized testing\n\n"

            "6. **Visualization and Output Artifacts**\n"
            "   - Document all visualization cli_response_utils and logging mechanisms:\n"
            "     * Training progress visualization (loss curves, metric tracking)\n"
            "     * Model performance visualization (confusion matrices, predictions)\n"
            "     * Feature or embedding visualizations\n"
            "     * Logging frameworks configuration (TensorBoard, W&B, MLflow, etc.)\n"
            "   - Detail all saved artifacts and their purposes:\n"
            "     * Model checkpoints format and saving frequency\n"
            "     * Evaluation results storage and formats\n"
            "     * Output directories and file naming conventions\n\n"

            "7. **Reproduction and Extension Guide**\n"
            "   - Synthesize key information needed for reproduction:\n"
            "     * Critical hyperparameters and their sensitivities\n"
            "     * Hardware requirements and environment considerations\n"
            "     * Potential bottlenecks and optimization opportunities\n"
            "   - Suggest clear extension points for junior engineers\n"
            "   - Highlight areas where the implementation could be improved or modified\n\n"

            "### TECHNICAL COMMUNICATION STANDARDS\n\n"
            "Adhere to these communication principles throughout your documentation:\n\n"

            "**Technical Precision**\n"
            "- Define ALL technical terms upon first use\n"
            "- Use mathematically precise descriptions for operations\n"
            "- Maintain consistent terminology throughout the document\n"
            "- Present numerical values with appropriate units and precision\n"
            "- Preserve exact parameter names and values from the AST\n\n"

            "**Clarity and Accessibility**\n"
            "- Write in natural, fluent technical prose—no code snippets or raw AST output\n"
            "- Use clear section headers, subheaders, and bullet lists for structured information\n"
            "- Build conceptual bridges between technical implementations and ML theory\n"
            "- Balance technical precision with explanatory context for junior engineers\n"
            "- Ensure logical progression of information within and across sections\n\n"

            "**Comprehensiveness and Fidelity**\n"
            "- Document EVERY component present in the AST—nothing should be omitted\n"
            "- Do NOT invent, hallucinate, or assume information not present in the AST\n"
            "- Explicitly acknowledge any ambiguities or incomplete information in the AST\n"
            "- Ensure documentation covers both common and edge-case behaviors\n"
            "- Maintain fidelity to the exact implementation details in the AST\n\n"

            "### QUALITY VERIFICATION CHECKLIST\n\n"
            "Before finalizing your documentation, verify that:\n\n"
            "1. Every node and relationship in the AST is reflected in your report\n"
            "2. All numerical parameters and configurations are accurately transcribed\n"
            "3. The architecture description accounts for all layers and transformations\n"
            "4. Training and evaluation procedures are completely and correctly documented\n"
            "5. No information has been invented or assumed beyond what's in the AST\n"
            "6. A junior ML engineer could reproduce the exact model from your description\n"
            "7. All technical terms are defined and explanations leave no conceptual gaps\n"
            "8. The document flows logically and maintains consistent technical language\n\n"

            "Remember: Accuracy and comprehensiveness are paramount. Your documentation must contain all implementation details from the AST while providing the conceptual clarity junior engineers need to understand the ML system fully."
        )

        for attempt in range(max_retries):
            try:
                user_prompt = f"Here is the AST summary:\n{chunk_text}\n```"
                response = self.llm_interface.generate_structured_response(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=0,
                    max_tokens=4096
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

    def generate_metadata_from_ast_summary(self, summary: str, max_retries: int = 3, file_path: str = "<unknown>") -> dict:
        """Generate structured JSON metadata from AST digest summary."""
        if not self.llm_interface:
            return create_default_llm_metadata()

        # Reduce input size of LLM
        print(f"len(summary): {len(summary)}, file_path: {file_path}")
        if len(summary) >= 2500:
            summary = filter_ast_summary_for_metadata(summary)
            print(f"len(extracted_summary): {len(summary)}, file_path: {file_path}")

        system_prompt = (
            "### ML CODE METADATA EXTRACTOR ROLE AND PURPOSE\n\n"
            "You are a senior machine-learning architect specializing in extracting precise, structured metadata from ML code. "
            "Your expertise allows you to identify key architectural patterns, configuration parameters, and implementation details "
            "from Abstract Syntax Tree (AST) digest summaries. The JSON metadata you produce will feed directly into model registries, "
            "auto-generate training dashboards, and ensure complete reproducibility of ML experiments.\n\n"

            "### INPUT FORMAT SPECIFICATION\n\n"
            "You will receive an AST digest summary of a Python ML training model_script. This structured representation contains "
            "information about imports, classes, functions, variable assignments, and method calls that collectively define "
            "the ML model implementation. Your task is to analyze this digest and extract specific metadata fields.\n\n"

            "### EXTRACTION METHODOLOGY\n\n"
            "Apply these systematic extraction principles:\n"
            "- Perform a comprehensive scan of the entire AST digest before making determinations\n"
            "- Identify component relationships and architectural patterns rather than isolated elements\n"
            "- Trace data flow and parameter usage throughout the model_script\n"
            "- Recognize standard ML implementation patterns and their signatures\n"
            "- Maintain absolute fidelity to information present in the AST—never invent or assume values\n"
            "- Use specified fallback values (null or \"missing\") when information cannot be reliably extracted\n\n"

            "### REQUIRED OUTPUT FORMAT\n\n"
            "Your output MUST strictly conform to this exact JSON structure:\n"
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

            "### FIELD EXTRACTION SPECIFICATIONS\n\n"

            "**architecture**\n"
            "- Look beyond individual layer or class names to identify the overarching architectural paradigm\n"
            "- Analyze component combinations, layer sequences, and functional patterns\n"
            "- Consider distinctive implementation signatures (e.g., encoder-decoder pairs, attention mechanisms, GAN generators/discriminators)\n"
            "- Format as: { \"type\": \"<InferredArchitecture>\", \"reason\": \"<concise justification citing multiple specific AST evidence points>\" }\n"
            "- Examples of valid architecture types: \"Transformer\", \"UNet\", \"ResNet\", \"Variational Autoencoder\", \"GAN\", \"LSTM\", etc.\n"
            "- The reason must cite specific cli_response_utils and patterns from the AST (e.g., \"Contains encoder/decoder modules with reparameterization function, characteristic of VAE architecture\")\n\n"

            "**dataset**\n"
            "- Systematically scan for dataset definitions, imports, and loader instantiations\n"
            "- Look for standard dataset classes (e.g., torchvision.datasets.MNIST, tensorflow.keras.datasets.cifar10)\n"
            "- Identify custom Dataset subclass implementations and their data sources\n"
            "- Check for DataLoader instantiations and their source parameters\n"
            "- Examine file paths, directory patterns, and data loading logic\n"
            "- Format as: { \"name\": \"<DatasetName>\", \"reason\": \"<concise reason citing the exact AST evidence>\" }\n"
            "- If no clear dataset is found, use { \"name\": \"unknown\", \"reason\": \"No dataset references found in AST\" }\n\n"

            "**training_config**\n"
            "Extract each parameter using these specific guidelines:\n\n"

            "1. **batch_size**:\n"
            "   - Primary: Look for `batch_size=` parameter in DataLoader instantiations\n"
            "   - Secondary: Check for variable assignments like `batch_size = 32`\n"
            "   - Tertiary: Examine function parameters that might reference batch size\n"
            "   - Return the integer value if found; else null\n\n"

            "2. **learning_rate**:\n"
            "   - Primary: Look for `lr=` or `learning_rate=` in optimizer instantiations\n"
            "   - Secondary: Check for variable assignments to `lr` or `learning_rate`\n"
            "   - Tertiary: Examine scheduler configurations or training function parameters\n"
            "   - Return the floating-point value if found; else null\n\n"

            "3. **optimizer**:\n"
            "   - Primary: Identify optimizer class instantiations (e.g., `Adam`, `SGD`, `RMSprop`)\n"
            "   - Secondary: Look for imported optimizer classes that are later instantiated\n"
            "   - Return the string name of the optimizer if found; else \"missing\"\n\n"

            "4. **epochs**:\n"
            "   - Primary: Look for variables named `epochs`, `num_epochs`, or similar\n"
            "   - Secondary: Check for loop ranges in training loops\n"
            "   - Tertiary: Examine function parameters related to training duration\n"
            "   - Return the integer value if found; else null\n\n"

            "5. **hardware_used**:\n"
            "   - Primary: Look for device settings via `torch.device`, `tf.device`, etc.\n"
            "   - Secondary: Check for CUDA availability checks or GPU-specific code\n"
            "   - Map findings to \"GPU\", \"CPU\", or \"Both\" based on evidence\n"
            "   - Return the mapped value if found; else \"missing\"\n\n"

            "### DATA FIDELITY REQUIREMENTS\n\n"
            "- **NEVER invent or hallucinate values** not present in the AST digest\n"
            "- Do not make assumptions about default values unless explicitly supported by AST evidence\n"
            "- Use null for missing numerical values and \"missing\" for missing string values as specified\n"
            "- Ensure all extracted values reflect the actual implementation in the AST, not theoretical defaults\n"
            "- When multiple values exist for a field (e.g., multiple batch sizes), extract the one most relevant to training\n\n"

            "### FINAL OUTPUT CONSTRAINTS\n\n"
            "🚨 **Output ONLY the JSON object**—no commentary, no explanations, no markdown formatting.\n"
            "- Ensure the JSON is valid and properly formatted\n"
            "- Include all required fields even if values are null or \"missing\"\n"
            "- Do not add any fields beyond those specified in the structure\n"
            "- Maintain the exact field names and nesting structure as specified\n"
            "- Do not include any text before or after the JSON object\n"
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
                    max_tokens=4096
                )
                print(f"metadata generation response (attempt {attempt + 1}): {response}")
                raw = response.get("content", "").strip()
                match = re.search(r"\{.*\}", raw, re.DOTALL)
                if match:
                    json_str = sanitize_json_string(match.group())
                    try:
                        parsed = json.loads(json_str)
                        is_parse_success, reason = validate_llm_metadata_structure(parsed)
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
        return create_default_llm_metadata()

    def extract_model_info(self, file_content: str, file_path: str) -> dict:
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
