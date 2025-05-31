# Model-Insight-RAG: An Agentic Retrieval-Augmented Generation System

## Overview

A comprehensive framework for understanding, indexing, and retrieving machine learning models through advanced code analysis and agentic retrieval-augmented generation.

For ML engineers drowning in disparate model folders, Model-Insight-RAG transforms raw code into structured, searchable knowledge with powerful semantic understanding. Unlike traditional RAG systems, our agentic approach autonomously makes decisions about information gathering, result evaluation, and response generation to deliver comprehensive answers with minimal overhead.

[DEMO link](https://github.com/ynyeh0221/model-insight-rag/blob/main/DEMO.md)

[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=ynyeh0221_model-insight-rag&metric=coverage)](https://sonarcloud.io/summary/new_code?id=ynyeh0221_model-insight-rag)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/ynyeh0221/model-insight-rag)

## Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/model-insight-rag.git
cd model-insight-rag

# Install dependencies
pip install -r requirements.txt

# Start the web interface
streamlit run src/streamlit_main.py

# Start the CLI interface
python src/main.py start-cli
```

## Key Features

<details>
<summary><strong>Agentic Capabilities</strong> (click to expand)</summary>

- **Autonomous Decision-Making**: System independently evaluates result quality and decides when more information is needed
- **Self-Directed Information Gathering**: Adaptively retrieves additional results based on analysis of current information
- **Goal-Oriented Behavior**: Pursues comprehensive answers by continuously evaluating if user queries are fully addressed
- **Multi-Step Planning**: Orchestrates complex workflows from query understanding to response synthesis
- **Tool Utilization**: Strategically employs specialized components as tools to accomplish different aspects of information retrieval and processing
</details>

<details>
<summary><strong>Intelligent Model Processing</strong> (click to expand)</summary>

- **AST-Powered Analysis**: Parses Python code using Abstract Syntax Tree analysis to extract model components, layers, and architecture details
- **LLM-Enhanced Metadata Extraction**: Combines static analysis with LLM capabilities to identify frameworks, architectures, datasets, and training configurations
- **Multi-Collection Vector Database**: Separates model metadata into specialized collections with appropriate embedding spaces
- **Auto-Generated Model Visualizations**: Creates component diagrams showing model architecture and layer connections
</details>

<details>
<summary><strong>Advanced Search Capabilities</strong> (click to expand)</summary>

- **Multi-Stage Query Processing**:
  - Query clarity checking and improvement
  - Comparison query detection and splitting
  - Named entity recognition for technical ML concepts
  - Multi-vector search across specialized collections
  - Contextual reranking of search results
- **Cross-Modal Search**: Find models by text query or image similarity
- **Semantic Understanding**: Understands technical concepts in ML code (e.g., "transformer with self-attention")
- **Adaptive Results Pagination**: The system intelligently determines if more results are needed:
  - Analyzes if current results are sufficient to answer the query
  - LLM determines if additional results would provide better context by detecting the "<need_more_results>" marker
  - Optimizes resource usage by only fetching additional pages when necessary
  - Dynamically adjusts based on similarity distance between records in current results
  - Balances between comprehensive answers and system efficiency with configurable page sizes
</details>

<details>
<summary><strong>Rich User Interfaces</strong> (click to expand)</summary>

- **Interactive CLI**: Command-line interface for model queries and administration
- **Streamlit Web UI**: User-friendly interface for exploring models and visualizing results
- **Automated Notebook Generation**: Convert model code to executable notebooks
</details>

<details>
<summary><strong>Enterprise-Ready Features</strong> (click to expand)</summary>

- **Fine-Grained Access Control**: User and group-level permissions for model access
- **Performance Optimization**: Smart reranking and distance normalization
- **Batch Processing**: Efficiently process entire model repositories
- **Multi-Modal Support**: Handle both model code and generated images
</details>

## Architecture

The system follows a modular design with clear separation of concerns:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  User Interface │     │  Query Engine   │     │ Vector Database │
│  - CLI          │────▶│  - Parser       │────▶│  - ChromaDB     │
│  - Streamlit    │     │  - Dispatcher   │     │  - Collections  │
└─────────────────┘     │  - Rerankers    │     └─────────────────┘
                        └─────────────────┘              │
                                │                        │
                                ▼                        ▼
                        ┌─────────────────┐     ┌─────────────────┐
                        │  Core Logic     │     │ Content Analysis│
                        │  - RAG System   │────▶│  - Code Parser  │
                        │  - Access Ctrl  │     │  - AST Generator│
                        └─────────────────┘     │  - Image Process│
                                                └─────────────────┘
```

## Agentic RAG System Overview

Our agentic RAG approach transcends traditional retrieval-augmented generation by introducing autonomous decision-making throughout the information retrieval and synthesis process:

1. **Query Understanding Agent**: Analyzes queries for clarity, detects comparison requests, and improves ambiguous questions
2. **Information Retrieval Agent**: Dynamically determines what information to retrieve and when more data is needed
3. **Content Evaluation Agent**: Assesses retrieved information for relevance, completeness, and whether it satisfies the query intent
4. **Response Generation Agent**: Synthesizes comprehensive answers with appropriate detail level and supporting information

This agentic architecture enables the system to:
- Adapt to query complexity by retrieving just enough information (not too little, not too much)
- Make autonomous decisions about when to fetch additional results
- Detect and handle comparison queries by splitting into sub-queries and synthesizing results
- Balance between information comprehensiveness and response conciseness

### Data model

The system uses specialized collections for different types of model metadata.
[View full data model documentation →](docs/data_model.md)

| Collection | Purpose |
|------------|---------|
| `model_file` | Basic file attributes |
| `model_frameworks` | Framework details |
| `model_architectures` | Model type and structure |
| `model_datasets` | Dataset references |
| `model_descriptions` | Text summaries |
| `generated_images` | Model outputs |

## Processing pipelines

### Model script processing pipeline

1. **File ingestion**: Load Python files or Jupyter notebooks 
2. **AST analysis**: Extract class hierarchies, layers, dependencies
3. **Code chunking**: Segment files while preserving context
4. **LLM enhancement**: Generate descriptions, classify architectures
5. **Metadata extraction**: Create structured metadata for each collection
6. **Schema validation**: Validate against schema definitions
7. **Vector embedding**: Embed text and code with specialized models
8. **Multi-collection storage**: Store across dedicated collections
9. **Visualization generation**: Generate model architecture diagrams

### Image processing pipeline

1. **Image discovery**: Scan folders for model-generated images
2. **Metadata extraction**: Extract technical image metadata
3. **Content analysis**: Analyze images for semantic information
4. **CLIP embedding**: Generate multimodal embeddings
5. **Schema validation**: Validate against image schema
6. **Vector storage**: Store embeddings and metadata
7. **Association**: Link images to source models

### Agentic query processing pipeline

1. **Query preprocessing**: Assess clarity, improve ambiguous queries
2. **Query parsing**: Classify intent, extract entities and parameters
3. **Search orchestration**: Select collections, perform vector search
4. **Result processing**: Rerank, deduplicate, enrich and order results
5. **Response generation**: Synthesize coherent responses with visuals
6. **Comparison handling**: Process model comparison queries
7. **Adaptive result fetching**: 
   - Initial retrieval with a default page size (typically 3 items)
   - LLM analyzes current results to determine if they sufficiently answer the query
   - If more context needed (indicated by "<need_more_results>" marker), system autonomously fetches the next page
   - Process continues until either sufficient information is gathered or max pages reached
   - Optimizes between comprehensive answers and system efficiency

## Usage

### Command line interface

```bash
# Start the CLI
python src/main.py start-cli

# Process a single script
python src/main.py process-single-script path/to/model.py

# Process a directory of scripts
python src/main.py process-scripts path/to/models/

# Process a single image
python src/main.py process-single-image path/to/image.py

# Process a directory of images
python src/main.py process-images path/to/images/
```

### Web interface

[Streamlit UI Demo](https://github.com/ynyeh0221/model-insight-rag/blob/main/demo/screenshot/Model%20Insight%20RAG%20System%20Demo.pdf)

```bash
# Start the web interface
streamlit run src/streamlit_main.py
```

## Technical requirements

- **Python**: 3.9, 3.10, 3.11 (tested)
- **Core dependencies**:
  - Vector database: `chromadb`
  - Embeddings: `sentence-transformers`, `open-clip`
  - LLM integration: `langchain`
  - UI: `streamlit`
  - Code analysis: Standard library `ast`
  - Visualization: `graphviz`, `matplotlib`

## Project structure

```
src/ (~5K LoC)
├── api/                        # API layer
├── cli/                        # Command-line interface
├── streamlit/                  # Streamlit web UI 
├── core/                       # Core system logic
│   ├── content_analyzer/       # Code and image analysis
│   ├── query_engine/           # Query processing pipeline
│   ├── vector_db/              # Database management
│   └── rag_system.py           # Main orchestration
├── main.py                     # CLI entry point
└── streamlit_main.py           # Web UI entry point
```

## Contributing

We welcome contributions! See our [contributing guide](docs/CONTRIBUTING.md) for details on how to get involved.

## License

[MIT License](https://github.com/ynyeh0221/model-insight-rag/blob/main/LICENSE)
