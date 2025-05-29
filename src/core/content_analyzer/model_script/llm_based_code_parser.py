"""
LLM-Based Code Parser - Advanced ML Code Analysis System

This system combines traditional AST parsing with Large Language Model analysis
to extract comprehensive, structured metadata from machine learning code files.

OVERALL SYSTEM ARCHITECTURE:
===========================

    Input: ML Code File (.py/.ipynb)
                    |
                    v
    ┌─────────────────────────────────────────────────────────┐
    │                FILE VALIDATION                          │
    │  • Check file extension (.py, .ipynb)                   │
    │  • Validate file accessibility                          │
    │  • Extract file metadata (creation/modification dates)  │
    └─────────────────────────────────────────────────────────┘
                    |
                    v
    ┌─────────────────────────────────────────────────────────┐
    │              HYBRID ANALYSIS PIPELINE                   │
    │                                                         │
    │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
    │  │ AST PARSING │───>│ LLM ANALYSIS│───>│ STRUCTURED  │  │
    │  │ (Static)    │    │ (Semantic)  │    │ METADATA    │  │
    │  │             │    │             │    │ (JSON)      │  │
    │  └─────────────┘    └─────────────┘    └─────────────┘  │
    └─────────────────────────────────────────────────────────┘
                    |
                    v
    ┌─────────────────────────────────────────────────────────┐
    │                   OUTPUT GENERATION                     │
    │  • Structured JSON metadata                             │
    │  • Natural language summaries                           │
    │  • Architecture diagrams                                │
    │  • Component dependency maps                            │
    └─────────────────────────────────────────────────────────┘

DETAILED PROCESSING PIPELINE:
============================

Step 1: AST Digest Generation
┌──────────────────────────────────────────────────────────────┐
│  Input: Raw Python Code                                      │
│     │                                                        │
│     v                                                        │
│  ┌─────────────────┐                                         │
│  │    ASTSummary   │                                         │
│  │     Generator   │ ──> Generates structured digest:        │
│  │ • Classes       │     • Import statements                 │
│  │ • Functions     │     • Variable assignments              │
│  │ • Variables     │     • Model layers                      │
│  │ • Imports       │     • Component relationships           │
│  │ • Layers        │                                         │
│  └─────────────────┘                                         │
│                                                              │
│  Output: AST Digest Text                                     │
│  "Import: torch                                              │
│   Class: ResNet                                              │
│   Function: forward(x)                                       │
│   Variable: batch_size = 32                                  │
│   Dataset: CIFAR10"                                          │
└──────────────────────────────────────────────────────────────┘

Step 2: Filtering and Preprocessing
┌──────────────────────────────────────────────────────────────┐
│  filter_ast_summary_for_metadata()                           │
│                                                              │
│  Keeps ONLY relevant lines:                                  │
│  ┌─────────────────┐    ┌─────────────────┐                  │
│  │ ALWAYS KEEP:    │    │ VARIABLE FILTER:│                  │
│  │ • Dataset:      │    │ Keep if contain:│                  │
│  │ • Images folder:│    │ • batch         │                  │
│  │ • Model Arch.   │    │ • lr            │                  │
│  └─────────────────┘    │ • epoch         │                  │
│                         │ • optimizer     │                  │
│                         │ • device        │                  │
│                         └─────────────────┘                  │
│                                                              │
│  Result: Compact, focused AST summary                        │
└──────────────────────────────────────────────────────────────┘

Step 3: Dual LLM Analysis Strategy
┌──────────────────────────────────────────────────────────────┐
│                    PARALLEL LLM PROCESSING                   │
│                                                              │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              TRACK A: ARCHITECTURE                      │ │
│  │                                                         │ │
│  │  Input: filter_model_architecture_from_ast_summary()    │ │
│  │  ┌───────────────────────────────────────────────────┐  │ │
│  │  │ Filters for ONLY:                                 │  │ │
│  │  │ • "Model Architecture:" and all subsequent lines  │  │ │
│  │  │ • Component definitions                           │  │ │
│  │  │ • Layer specifications                            │  │ │
│  │  │ • Dependencies                                    │  │ │
│  │  └───────────────────────────────────────────────────┘  │ │
│  │                        │                                │ │
│  │                        v                                │ │
│  │  ┌───────────────────────────────────────────────────┐  │ │
│  │  │              LLM ANALYSIS                         │  │ │
│  │  │ System Prompt: Architecture-focused               │  │ │
│  │  │ Expected Output: {"architecture": {               │  │ │
│  │  │   "type": "ResNet/UNet/GAN/Transformer/etc",      │  │ │
│  │  │   "reason": "explanation of identification"       │  │ │
│  │  │ }}                                                │  │ │
│  │  └───────────────────────────────────────────────────┘  │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │               TRACK B: OTHER METADATA                   │ │
│  │                                                         │ │
│  │  Input: filter_ast_summary_for_metadata                 │ │
│  │  ┌───────────────────────────────────────────────────┐  │ │
│  │  │ Filters for:                                      │  │ │
│  │  │ • Dataset information                             │  │ │
│  │  │ • Training variables (batch_size, lr, etc.)       │  │ │
│  │  │ • Hardware specifications                         │  │ │
│  │  │ • Optimizer settings                              │  │ │
│  │  └───────────────────────────────────────────────────┘  │ │
│  │                        │                                │ │
│  │                        v                                │ │
│  │  ┌───────────────────────────────────────────────────┐  │ │
│  │  │              LLM ANALYSIS                         │  │ │
│  │  │ System Prompt: Training/Dataset-focused           │  │ │
│  │  │ Expected Output: {"dataset": {...},               │  │ │
│  │  │   "training_config": {                            │  │ │
│  │  │     "batch_size": int, "learning_rate": float,    │  │ │
│  │  │     "optimizer": str, "epochs": int,              │  │ │
│  │  │     "hardware_used": str}}                        │  │ │
│  │  └───────────────────────────────────────────────────┘  │ │
│  └─────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘

Step 4: Natural Language Summary Generation
┌──────────────────────────────────────────────────────────────┐
│  extract_natural_language_summary()                          │
│                                                              │
│  Input: Full AST Digest (filtered for metadata)              │
│     │                                                        │
│     v                                                        │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                 LLM INTERFACE                           │ │
│  │  System Prompt: "Generate human-readable summary"       │ │
│  │  Temperature: 0 (deterministic)                         │ │
│  │  Max Tokens: 32000                                      │ │
│  └─────────────────────────────────────────────────────────┘ │
│     │                                                        │
│     v                                                        │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              SEMANTIC CHUNKING                          │ │
│  │  split_summary_into_chunks():                           │ │
│  │  • NLTK sentence tokenization                           │ │
│  │  • TF-IDF vectorization                                 │ │
│  │  • Cosine similarity calculation                        │ │
│  │  • Threshold-based grouping                             │ │
│  │                                                         │ │
│  │  Algorithm:                                             │ │
│  │  1. Split summary into sentences                        │ │
│  │  2. Calculate TF-IDF vectors for each sentence          │ │
│  │  3. Group consecutive sentences if similarity>threshold │ │
│  │  4. Respect max_sentences_per_chunk limit               │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                              │
│  Output: List of semantic chunks with offsets                │
└──────────────────────────────────────────────────────────────┘

RETRY MECHANISM AND ERROR HANDLING:
==================================

Each LLM interaction includes robust error handling:

┌─────────────────────────────────────────────────────────────┐
│                    RETRY LOOP PATTERN                       │
│                                                             │
│  for attempt in range(max_retries):  # Default: 3-15        │
│      try:                                                   │
│          ┌────────────────────────────────────────────┐     │
│          │            LLM API CALL                    │     │
│          │  • Generate structured response            │     │
│          │  • Parse JSON from response                │     │
│          │  • Validate structure against schema       │     │
│          └────────────────────────────────────────────┘     │
│                                │                            │
│                Success? ──Yes──> Return Result              │
│                                │                            │
│                               No                            │
│                                │                            │
│          ┌────────────────────────────────────────────┐     │
│          │              ERROR HANDLING                │     │
│          │  • JSON parsing errors                     │     │
│          │  • Schema validation failures              │     │
│          │  • API communication errors                │     │
│          │  • Log error with attempt number           │     │
│          └────────────────────────────────────────────┘     │
│      except Exception as e:                                 │
│          log(f"[Retry {attempt + 1}] Error: {e}")           │
│                                                             │
│  # After all retries fail:                                  │
│  return create_default_metadata()                           │
└─────────────────────────────────────────────────────────────┘

JSON SANITIZATION PIPELINE:
===========================

sanitize_json_string() applies multiple cleaning steps:

Input: Raw LLM Response
│
v
┌─────────────────────────────────────────────────────────────┐
│  Step 1: Remove Thinking Tags                               │
│  Pattern: <think(ing)?>.*?</think(ing)?>                    │
│  Flags: re.DOTALL (multiline matching)                      │
│                                                             │
│  Before: "<thinking>Let me analyze...</thinking>{"key":...} │
│  After:  {"key":...}                                        │
└─────────────────────────────────────────────────────────────┘
│
v
┌─────────────────────────────────────────────────────────────┐
│  Step 2: Remove JS-Style Comments                           │
│  Pattern: //.*?$                                            │
│  Flags: re.MULTILINE                                        │
│                                                             │
│  Before: {"key": "value", // this is a comment              │
│           "other": "data"}                                  │
│  After:  {"key": "value",                                   │
│           "other": "data"}                                  │
└─────────────────────────────────────────────────────────────┘
│
v
┌─────────────────────────────────────────────────────────────┐
│  Step 3: Remove Trailing Commas                             │
│  Pattern: ,(\s*[}\]])                                       │
│  Replacement: \1                                            │
│                                                             │
│  Before: {"key": "value",}  or  ["item1", "item2",]         │
│  After:  {"key": "value"}   or  ["item1", "item2"]          │
└─────────────────────────────────────────────────────────────┘
│
v
Valid JSON String

FRAMEWORK DETECTION LOGIC:
=========================

parse_framework() uses priority-based regex matching:

Input: AST Summary Text
│
v
┌─────────────────────────────────────────────────────────────┐
│                  FRAMEWORK PATTERNS                         │
│                                                             │
│  Priority 1: PyTorch                                        │
│  ┌────────────────────────────────────────────────────┐     │
│  │ Patterns:                                          │     │
│  │ • "Import: torch"                                  │     │
│  │ • "From torch"                                     │     │
│  │ • Case-insensitive matching                        │     │
│  └────────────────────────────────────────────────────┘     │
│                                                             │
│  Priority 2: TensorFlow                                     │
│  ┌────────────────────────────────────────────────────┐     │
│  │ Patterns:                                          │     │
│  │ • "Import: tensorflow"                             │     │
│  │ • "From tensorflow"                                │     │
│  │ • "tf." (TensorFlow alias usage)                   │     │
│  └────────────────────────────────────────────────────┘     │
│                                                             │
│  Priority 3: JAX                                            │
│  ┌────────────────────────────────────────────────────┐     │
│  │ Patterns:                                          │     │
│  │ • "Import: jax"                                    │     │
│  │ • "From jax"                                       │     │
│  └────────────────────────────────────────────────────┘     │
│                                                             │
│  Priority 4: Keras (mapped to TensorFlow)                   │
│  ┌────────────────────────────────────────────────────┐     │
│  │ Patterns:                                          │     │
│  │ • "Import: keras"                                  │     │
│  │ • "From keras"                                     │     │
│  │ Note: Returns "TensorFlow" (Keras is part of TF)   │     │
│  └────────────────────────────────────────────────────┘     │
│                                                             │
│  Default: "missing" (if no patterns match)                  │
└─────────────────────────────────────────────────────────────┘

SEMANTIC TEXT CHUNKING ALGORITHM:
================================

split_summary_into_chunks() implements intelligent text segmentation:

Input: Natural Language Summary
│
v
┌─────────────────────────────────────────────────────────────┐
│              SENTENCE TOKENIZATION                          │
│                                                             │
│  Uses NLTK Punkt Tokenizer:                                 │
│  • Downloads punkt_tab if not present                       │
│  • Generates sentence boundaries with character offsets     │
│  • Filters out empty sentences                              │
│                                                             │
│  Input:  "This model uses ResNet. It has 50 layers. The     │
│          dataset is CIFAR-10. Training uses Adam optimizer."│
│  Output: [("This model uses ResNet.", (0, 22)),             │
│           ("It has 50 layers.", (23, 41)), ...]             │
└─────────────────────────────────────────────────────────────┘
│
v
┌─────────────────────────────────────────────────────────────┐
│                TF-IDF VECTORIZATION                         │
│                                                             │
│  Sklearn TfidfVectorizer:                                   │
│  • min_df=1 (include all terms)                             │
│  • stop_words='english' (remove common words)               │
│  • Creates vector representation for each sentence          │
│                                                             │
│  Sentence: "This model uses ResNet"                         │
│  Vector:   [0.0, 0.7, 0.0, 0.5, 0.0, 0.9, ...]              │
└─────────────────────────────────────────────────────────────┘
│
v
┌─────────────────────────────────────────────────────────────┐
│              SIMILARITY-BASED GROUPING                      │
│                                                             │
│  Algorithm:                                                 │
│  chunks = []                                                │
│  current_chunk = [sentence_0]                               │
│                                                             │
│  for sentence_i in sentences[1:]:                           │
│      # Calculate similarity to chunk boundaries             │
│      sim_first = cosine_similarity(sentence_i, chunk[0])    │
│      sim_last  = cosine_similarity(sentence_i, chunk[-1])   │
│      max_sim = max(sim_first, sim_last)                     │
│                                                             │
│      if max_sim >= threshold AND len(chunk) < max_size:     │
│          current_chunk.append(sentence_i)                   │
│      else:                                                  │
│          chunks.append(join(current_chunk))                 │
│          current_chunk = [sentence_i]                       │
│                                                             │
│  chunks.append(join(current_chunk))  # Final chunk          │
│                                                             │
│  Parameters:                                                │
│  • similarity_threshold: 0.5 (default)                      │
│  • max_sentences_per_chunk: 10 (default)                    │
└─────────────────────────────────────────────────────────────┘

VALIDATION SCHEMAS:
==================

Two separate validation functions ensure metadata quality:

Architecture Metadata Validation:
┌─────────────────────────────────────────────────────────────┐
│  validate_architecture_llm_metadata_structure()             │
│                                                             │
│  Required Structure:                                        │
│  {                                                          │
│    "architecture": {                                        │
│      "type": <string, not null, not "missing"/"unknown">    │
│      "reason": <string, not null, not empty>                │
│    }                                                        │
│  }                                                          │
│                                                             │
│  Forbidden Values for "type":                               │
│  • "missing", "unknown", "hybrid", "mixed", "both"          │
│  • "n/a", "pytorch", "gpu", "cpu"                           │
│                                                             │
│  Returns: (is_valid: bool, error_message: str)              │
└─────────────────────────────────────────────────────────────┘

Other Metadata Validation:
┌─────────────────────────────────────────────────────────────┐
│  validate_other_llm_metadata_structure()                    │
│                                                             │
│  Required Structure:                                        │
│  {                                                          │
│    "dataset": {"name": <string>},                           │
│    "training_config": {                                     │
│      "batch_size": <int|null>,                              │
│      "learning_rate": <float|null>,                         │
│      "optimizer": <string|null>,                            │
│      "epochs": <int|null>,                                  │
│      "hardware_used": <string|null>                         │
│    }                                                        │
│  }                                                          │
│                                                             │
│  All fields must be present, but values can be null         │
│  Returns: (is_valid: bool, error_message: str)              │
└─────────────────────────────────────────────────────────────┘

GIT INTEGRATION FOR TIMESTAMPS:
==============================

The system attempts to get accurate file timestamps through Git:

get_creation_date() / get_last_modified_date():
┌─────────────────────────────────────────────────────────────┐
│                    TIMESTAMP RESOLUTION                     │
│                                                             │
│  Strategy 1: Git Repository Analysis                        │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ try:                                                    ││
│  │     repo = Repo(dirname, search_parent_directories=True)││
│  │     # For creation: get first commit touching file      ││
│  │     commits = repo.iter_commits(paths=file,reverse=True)││
│  │     # For modification: get latest commit               ││
│  │     commit = next(repo.iter_commits(paths=file, max=1)) ││
│  │     return datetime.fromtimestamp(commit.date).isoformat││
│  │ except Exception: pass                                  ││
│  └─────────────────────────────────────────────────────────┘│
│                              │                              │
│                         Git failed?                         │
│                              │                              │
│                              v                              │
│  Strategy 2: Filesystem Metadata                            │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ try:                                                    ││
│  │     stat = os.stat(file_path)                           ││
│  │     # st_ctime for creation, st_mtime for modification  ││
│  │     return datetime.fromtimestamp(stat.time).isoformat()││
│  │ except Exception: return None                           ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘

FINAL METADATA ASSEMBLY:
=======================

parse_file() orchestrates the complete metadata assembly:

┌─────────────────────────────────────────────────────────────┐
│                    METADATA AGGREGATION                     │
│                                                             │
│  Base Information:                                          │
│  • creation_date (from Git/filesystem)                      │
│  • last_modified_date (from Git/filesystem)                 │
│  • model_id, model_family, version (from AST)               │
│                                                             │
│  LLM-Extracted Metadata:                                    │
│  • framework: {name, version}                               │
│  • architecture: {type, reason}                             │
│  • dataset: {name}                                          │
│  • images_folder: {name}                                    │
│  • training_config: {batch_size, learning_rate, ...}        │
│                                                             │
│  Generated Content:                                         │
│  • chunk_descriptions: [semantic chunks with offsets]       │
│  • ast_summary: (full AST digest)                           │
│  • diagram_path: (generated architecture diagram)           │
│  • content: (original file content)                         │
│                                                             │
│  Flags:                                                     │
│  • is_model_script: true                                    │
└─────────────────────────────────────────────────────────────┘

ERROR HANDLING PHILOSOPHY:
=========================

The system implements "graceful degradation":

┌─────────────────────────────────────────────────────────────┐
│              FAULT TOLERANCE DESIGN                         │
│                                                             │
│  Principle: Always return usable metadata                   │
│                                                             │
│  ┌─────────────────┐    ┌─────────────────┐                 │
│  │ BEST CASE:      │    │ WORST CASE:     │                 │
│  │ • Full LLM      │    │ • Syntax errors │                 │
│  │   analysis      │    │ • LLM failures  │                 │
│  │ • Rich metadata │    │ • API timeouts  │                 │
│  │ • Diagrams      │    │ • Invalid JSON  │                 │
│  └─────────────────┘    └─────────────────┘                 │
│                                │                            │
│                                v                            │
│                    ┌─────────────────┐                      │
│                    │ FALLBACK CHAIN: │                      │
│                    │ • Default values│                      │
│                    │ • Partial data  │                      │
│                    │ • Error logging │                      │
│                    │ • Minimal struct│                      │
│                    └─────────────────┘                      │
│                                                             │
│  Result: System never completely fails, always provides     │
│          some level of usable metadata for downstream       │
│          systems and users.                                 │
└─────────────────────────────────────────────────────────────┘

This architecture ensures robust, production-ready ML code analysis
with comprehensive error handling and multiple fallback strategies.
"""
import ast
import datetime
import json
import os
import re
from pathlib import Path
from typing import List, Tuple

from git import Repo

from src.core.content_analyzer.model_script.ast_summary_generator import ASTSummaryGenerator
from src.core.content_analyzer.utils.model_id_utils import ModelIdUtils
from src.core.prompt_manager.ingestion_path_prompt_manager import IngestionPathPromptManager

def filter_model_architecture_from_ast_summary(summary: str) -> str:
    """
    Given an AST digest summary, return only the lines that
    contain:
      - Import / From   (framework & dataset)
      - Variable        (batch_size, lr, optimizer, epochs, device)
      - Model Architecture and all subsequent lines
    """
    lines = summary.splitlines()
    filtered: List[str] = []
    in_model_architecture = False

    for line in lines:
        stripped = line.strip()

        # Check if we've reached the Model Architecture section
        if stripped.startswith("Model Architecture:"):
            in_model_architecture = True
            filtered.append(stripped)
            continue

        # If we're in the Model Architecture section, keep all lines
        if in_model_architecture:
            filtered.append(stripped)
            continue

    return "\n".join(filtered)

def filter_ast_summary_for_metadata(summary: str, include_model_architecture: bool = True) -> str:
    """
    Given an AST digest summary, return only the lines that
    contain:
      - Import / From   (framework & dataset)
      - Variable        (batch_size, lr, optimizer, epochs, device)
      - Model Architecture and all subsequent lines
    """
    lines = summary.splitlines()
    filtered: List[str] = []
    in_model_architecture = False

    # what prefixes we always keep
    keep_prefixes = ("Dataset:", "Images folder:")
    # which variable names to keep
    var_keys = ("batch", "lr", "epoch", "optimizer", "device")

    for line in lines:
        stripped = line.strip()

        # Check if we've reached the Model Architecture section
        if stripped.startswith("Model Architecture:"):
            in_model_architecture = True
            filtered.append(stripped)
            continue

        # If we're in the Model Architecture section, keep all lines
        if in_model_architecture and include_model_architecture:
            filtered.append(stripped)
            continue

        # For other lines, apply the original filtering logic
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
    # Remove thinking tags and their contents
    json_str = re.sub(r"<think(ing)?>.*?</think(ing)?>", "", json_str, flags=re.DOTALL)
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

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def split_summary_into_chunks(summary_text, similarity_threshold=0.5, max_sentences_per_chunk=10):
    """
    Splits a summary into semantic chunks by combining similar contiguous sentences.
    Uses similarity between each new sentence and the last sentence in the current chunk.

    Returns:
      List[Dict]: each dict has "description" (chunk text) and "offset" (char index).
    """
    if not summary_text or not summary_text.strip():
        return []

    # ensure punkt tokenizer
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')

    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    spans = list(tokenizer.span_tokenize(summary_text))
    sentences = [summary_text[s:e] for s, e in spans]

    # filter blank sentences
    filtered = [(s, span) for s, span in zip(sentences, spans) if s.strip()]
    if not filtered:
        return []
    sentences, spans = zip(*filtered)

    if len(sentences) == 1:
        return [{"description": sentences[0], "offset": spans[0][0]}]

    # TF-IDF
    vectorizer = TfidfVectorizer(min_df=1, stop_words='english')
    tfidf = vectorizer.fit_transform(sentences)

    chunks = []
    current_idxs = [0]

    for i in range(1, len(sentences)):
        # Compare to both first and last sentence in current chunk
        first_idx = current_idxs[0]
        last_idx = current_idxs[-1]
        vec_first = tfidf[first_idx].toarray()
        vec_last  = tfidf[last_idx].toarray()
        vec_i     = tfidf[i].toarray()

        # Determine similarity against both endpoints
        sim_first = cosine_similarity(vec_first, vec_i)[0][0]
        sim_last  = cosine_similarity(vec_last,  vec_i)[0][0]
        sim = max(sim_first, sim_last)

        # Decide whether to extend current chunk or start a new one
        if sim >= similarity_threshold and len(current_idxs) < max_sentences_per_chunk:
            current_idxs.append(i)
        else:
            # flush the current chunk
            start_off = spans[current_idxs[0]][0]
            text = " ".join(sentences[j] for j in current_idxs)
            chunks.append({"description": text, "offset": start_off})
            # begin a new chunk
            current_idxs = [i]

    # flush final
    if current_idxs:
        start_off = spans[current_idxs[0]][0]
        text = " ".join(sentences[j] for j in current_idxs)
        chunks.append({"description": text, "offset": start_off})

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

def validate_architecture_llm_metadata_structure(metadata: dict) -> Tuple[bool, str]:
    """Validate that the metadata has the required structure."""
    # Check that all required top-level fields exist
    required_fields = ["architecture"]
    if not all(field in metadata for field in required_fields):
        return False, "Missing required fields"

    # Check that architecture has type
    if not isinstance(metadata["architecture"], dict) or "type" not in metadata["architecture"] or metadata["architecture"].get("type") is None or \
            metadata["architecture"].get("type").lower() == "missing" or metadata["architecture"].get("type").lower() == "unknown" or metadata["architecture"].get("type").lower() == "hybrid" or metadata["architecture"].get("type").lower() == "mixed" or metadata["architecture"].get("type").lower() == "both" or \
            metadata["architecture"].get("type").lower() == "n/a" or "reason" not in metadata["architecture"] or metadata["architecture"].get("reason") is None or \
            metadata["architecture"].get("reason").lower() == "missing" or metadata["architecture"].get("type").lower() == "pytorch" or metadata["architecture"].get("type").lower() == "gpu" or metadata["architecture"].get("type").lower() == "cpu":
        return False, "Invalid architecture structure"

    return True, ""

def validate_other_llm_metadata_structure(metadata: dict) -> Tuple[bool, str]:
    """Validate that the metadata has the required structure."""
    # Check that all required top-level fields exist
    required_fields = ["dataset", "training_config"]
    if not all(field in metadata for field in required_fields):
        return False, "Missing required fields"

    # Check that dataset has name
    if not isinstance(metadata["dataset"], dict) or "name" not in metadata["dataset"]:
        return False, "Invalid dataset structure"

    # Check that training_config has all required fields
    training_config_fields = ["batch_size", "learning_rate", "optimizer", "epochs", "hardware_used"]
    if not isinstance(metadata["training_config"], dict) or not all(
            field in metadata["training_config"] for field in training_config_fields):
        return False, "Invalid training configuration structure"

    return True, ""

def create_default_architecture_llm_metadata() -> dict:
    """Create default metadata structure with empty/null values."""
    return {
        "architecture": {"type": "missing"},
    }

def create_default_other_llm_metadata() -> dict:
    """Create default metadata structure with empty/null values."""
    return {
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
    1. AST Digest Generation: Creates a structured digest of code components using Python's AST module
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
        Orchestrates the multistep process of extracting metadata using AST and LLM analysis.

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

    def __init__(self, schema_validator=None, llm_interface=None, llm_interface_natural_language_summary=None):
        self.schema_validator = schema_validator
        self.llm_interface = llm_interface
        self.llm_interface_natural_language_summary = llm_interface_natural_language_summary
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

        model_info = {
            "creation_date": get_creation_date(file_path),
            "last_modified_date": get_last_modified_date(file_path)
        }

        extracted_info = self.extract_model_info(file_content, file_path)
        model_info.update(extracted_info)

        self.llm_metadata_cache = self.extract_metadata_by_llm(file_content, file_path, max_retries=15)

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

        model_info["diagram_path"] = self.llm_metadata_cache.get("diagram_path")

        model_info["is_model_script"] = True
        model_info["content"] = file_content

        print(f"updated model_info: {model_info}")
        return model_info

    @staticmethod
    def extract_filename_and_directory(file_path):
        # Get the base filename (without directory)
        filename = os.path.basename(file_path)

        # Get the directory containing the file
        directory = os.path.dirname(file_path)

        # Get just the lowest directory name
        lowest_directory = os.path.basename(directory)

        return lowest_directory + "_" + filename

    def extract_metadata_by_llm(self, code_str: str, file_path: str = "<unknown>", max_retries: int = 15) -> dict:
        # STEP 1: Generate AST digest/summary
        ast_digest = clean_empty_lines(self.ast_summary_generator.generate_summary(code_str=code_str, file_path=file_path))
        print(f"Total AST digest: {ast_digest}")

        # STEP 2: Extract natural-language summary for each digest chunk
        summary = self.extract_natural_language_summary(
            chunk_text=ast_digest,
            chunk_offset=0,
            max_retries=max_retries
        )
        print(f"Natural language summary from AST digest: {summary}")

        # STEP 3: Feed merged summary to LLM for structured metadata generation
        final = {}
        # STEP 3-1: Parse architecture metadata
        architecture_metadata = self.generate_architecture_metadata_from_ast_summary(ast_digest,
                                                                                     max_retries=max_retries,
                                                                                     file_path=file_path)
        print(f"architecture_metadata: {architecture_metadata}")
        final.update(architecture_metadata)
        # STEP 3-2: Parse other metadata
        other_metadata = self.generate_other_metadata_from_ast_summary(ast_digest, max_retries=max_retries,
                                                                       file_path=file_path)
        print(f"other_metadata: {other_metadata}")
        final.update(other_metadata)

        final['chunk_descriptions'] = split_summary_into_chunks(summary_text=summary.get("summary", ""), max_sentences_per_chunk=1)
        print(f"Chunk descriptions count: {len(final['chunk_descriptions'])}")

        # STEP 4: Store AST digest summary
        final['ast_summary'] = ast_digest

        # STEP 5: Add framework
        final['framework'] = {'name': parse_framework(ast_digest), 'version': '2.7'}

        # STEP 6: Add image_processing folder. Tried to use LLM for parsing, but finally moved back to manual parse way since this field needs exact value
        final['images_folder'] = {'name': get_images_folder(ast_digest)}

        # STEP 7: Create component diagram for the AST digest
        model_id = ModelIdUtils.get_model_id(file_path)
        diagram_path = Path(f"./model_diagram/{model_id}.png")
        self.ast_summary_generator.analyze_and_visualize_model(file_path, str(diagram_path))
        final['diagram_path'] = str(diagram_path.resolve())

        # STEP 8: Remove unneeded fields
        final.pop("_trace", None)

        print(f"Final metadata (from AST summary): {final}")
        return final

    def extract_natural_language_summary(self, chunk_text: str, chunk_offset: int = 0, max_retries: int = 3) -> dict:
        """Extract a natural language summary of metadata from a code chunk instead of structured JSON."""
        if not self.llm_interface:
            return {}

        # Reduce input size of LLM
        chunk_text = filter_ast_summary_for_metadata(chunk_text, True)

        system_prompt = IngestionPathPromptManager.get_system_prompt_natural_language_summary_creation()

        for attempt in range(max_retries):
            try:
                user_prompt = f"Here is the AST summary:\n{chunk_text}\n```"
                response = self.llm_interface_natural_language_summary.generate_structured_response(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=0,
                    max_tokens=32000
                )
                summary = response.get("content", "").strip()

                if summary:
                    return {
                        "summary": self.remove_thinking_sections(summary),
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

    @staticmethod
    def remove_thinking_sections(text: str) -> str:
        """
        Remove all <thinking>...</thinking> sections from the input text.

        Args:
            text: The input string containing zero or more <thinking> sections.

        Returns:
            A new string with all <thinking> sections and their content removed.
        """
        import re
        # Regex to match <thinking>...</thinking> including multiline content (non-greedy)
        pattern = re.compile(r'<(thinking|think)>(.*?)</\1>', re.DOTALL)
        # Replace matched sections with an empty string
        return pattern.sub('', text)

    def generate_architecture_metadata_from_ast_summary(self, summary: str, max_retries: int = 3, file_path: str = "<unknown>") -> dict:
        """Generate structured JSON metadata from AST digest summary."""
        if not self.llm_interface_natural_language_summary:
            return create_default_architecture_llm_metadata()

        # Reduce input size of LLM
        print(f"len(summary): {len(summary)}, file_path: {file_path}")
        summary = filter_model_architecture_from_ast_summary(summary)
        print(f"len(extracted_summary): {len(summary)}, file_path: {file_path}")

        system_prompt = IngestionPathPromptManager.get_system_prompt_for_architecture_metadata_from_ast_summary_parsing()

        for attempt in range(max_retries):
            try:
                user_prompt = f"""
                Here is the model AST digest summary:

                {summary}

                Extract the JSON according to this schema:
                {{
                  "architecture": {{
                    "type": "string",
                    "reason": "string"
                  }}
                }}

                Your response should be a valid JSON object with ONLY an "architecture" property containing exactly two fields: "type" and "reason". Both must be strings. The "type" should identify the overall architecture (like "ResNet", "UNet", "GAN", etc.) and the "reason" should explain why this architecture type was identified.
                """
                response = self.llm_interface_natural_language_summary.generate_structured_response(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=0,
                    max_tokens=32000
                )
                print(f"Architecture metadata generation response (attempt {attempt + 1}): {response}")
                raw = response.get("content", "").strip()
                match = re.search(r"\{.*\}", raw, re.DOTALL)
                if match:
                    json_str = sanitize_json_string(match.group())
                    try:
                        parsed = json.loads(json_str)
                        is_parse_success, reason = validate_architecture_llm_metadata_structure(parsed)
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
        return create_default_architecture_llm_metadata()

    def generate_other_metadata_from_ast_summary(self, summary: str, max_retries: int = 3, file_path: str = "<unknown>") -> dict:
        """Generate structured JSON metadata from AST digest summary."""
        if not self.llm_interface:
            return create_default_other_llm_metadata()

        # Reduce input size of LLM
        print(f"len(summary): {len(summary)}, file_path: {file_path}")
        summary = filter_ast_summary_for_metadata(summary, False)
        print(f"len(extracted_summary): {len(summary)}, file_path: {file_path}")

        system_prompt = IngestionPathPromptManager.get_system_prompt_for_other_metadata_from_ast_summary_parsing()

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
                print(f"Other metadata generation response (attempt {attempt + 1}): {response}")
                raw = response.get("content", "").strip()
                match = re.search(r"\{.*\}", raw, re.DOTALL)
                if match:
                    json_str = sanitize_json_string(match.group())
                    try:
                        parsed = json.loads(json_str)
                        is_parse_success, reason = validate_other_llm_metadata_structure(parsed)
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
        return create_default_other_llm_metadata()

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
