# Model Insight RAG System

A comprehensive framework for understanding, indexing, and retrieving machine learning models through advanced code analysis and retrieval-augmented generation.

[DEMO](https://github.com/ynyeh0221/model-insight-rag/blob/main/DEMO.md)

## Overview

The Model Insight RAG System is a specialized retrieval-augmented generation platform designed to help ML engineers, researchers, and teams manage complex model repositories. It combines deep code understanding, multi-modal search, and LLM-powered analysis to transform raw model code into structured, searchable knowledge.

## Key Features

### Intelligent Model Processing

- **AST-Powered Analysis**: Parses Python code using Abstract Syntax Tree analysis to extract model components, layers, and architecture details
- **LLM-Enhanced Metadata Extraction**: Combines static analysis with LLM capabilities to identify frameworks, architectures, datasets, and training configurations
- **Multi-Collection Vector Database**: Separates model metadata into specialized collections with appropriate embedding spaces
- **Auto-Generated Model Visualizations**: Creates component diagrams showing model architecture and layer connections

### Advanced Search Capabilities

- **Multi-Stage Query Processing**:
  - Query clarity checking and improvement
  - Comparison query detection and splitting
  - Named entity recognition for technical ML concepts
  - Multi-vector search across specialized collections
  - Contextual reranking of search results
- **Cross-Modal Search**: Find models by text query or image similarity
- **Semantic Understanding**: Understands technical concepts in ML code (e.g., "transformer with self-attention")

### Rich User Interfaces

- **Interactive CLI**: Command-line interface for model queries and administration
- **Streamlit Web UI**: User-friendly interface for exploring models and visualizing results
- **Automated Notebook Generation**: Convert model code to executable notebooks

### Enterprise-Ready Features

- **Fine-Grained Access Control**: User and group-level permissions for model access
- **Performance Optimization**: Smart reranking and distance normalization
- **Batch Processing**: Efficiently process entire model repositories
- **Multi-Modal Support**: Handle both model code and generated images

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

### Data Model

Each model is decomposed into a rich set of metadata stored in dedicated collections per the schema definitions:

| Collection | Purpose | Metadata Structure |
|------------|---------|-------------------|
| `model_file` | File-level attributes | `model_id`, `file`: {`size_bytes`, `creation_date`, `last_modified_date`, `file_extension`, `absolute_path`} |
| `model_date` | Timestamp information | `model_id`, `created_at`, `created_month`, `created_year`, `last_modified_month`, `last_modified_year` |
| `model_git` | Version control info | `model_id`, `git`: {`creation_date`, `last_modified_date`, `commit_count`} |
| `model_frameworks` | Framework details | `model_id`, `framework`: {`name`, `version`} |
| `model_datasets` | Dataset references | `model_id`, `dataset`: {`name`} |
| `model_training_configs` | Training parameters | `model_id`, `training_config`: {`batch_size`, `learning_rate`, `optimizer`, `epochs`, `hardware_used`} |
| `model_architectures` | Model type and structure | `model_id`, `architecture`: {`type`, `reason`} |
| `model_descriptions` | Text summaries | `model_id`, `description`, `total_chunks`, `offset` |
| `model_ast_summaries` | AST analysis results | `model_id`, `ast_summary`: {`ast_summary`} |
| `model_scripts_chunks` | Code segments | `model_id`, `chunk_id`, `total_chunks`, `metadata_doc_id`, `offset`, `type` |
| `model_images_folder` | Image storage location | `model_id`, `images_folder`: {`name`} |
| `model_diagram_path` | Diagram storage location | `model_id`, `diagram_path`: {`name`} |
| `generated_images` | Model outputs | `model_id`, `epoch`, `description`, `image_content`: {`subject_type`, `style`, `tags`, etc.}, `image_path`, `thumbnail_path`, `format`, `mode`, `size`, `exif`, `dates` |

All collections include `access_control` attributes for permission management.

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/model-insight-rag.git
cd model-insight-rag

# Install dependencies
pip install -r requirements.txt
```

### Usage

#### Web Interface

```bash
streamlit run src/streamlit_main.py
```

#### Command Line Interface

```bash
python src/main.py start-cli
```

#### Process Model Scripts

```bash
# Process a single script
python src/main.py process-single-script path/to/model.py

# Process a directory of scripts
python src/main.py process-scripts path/to/models/
```

#### Process Generated Images

```bash
# Process a single image
python src/main.py process-single-image path/to/image.png

# Process a directory of images
python src/main.py process-images path/to/image/directory/
```

## Deep Dive: Query Processing

When a user submits a query, the system performs:

1. **Query Clarification**: Checks if the query is clear and suggests improvements if needed
2. **Intent Classification**: Determines if this is a retrieval, comparison, or notebook generation query
3. **Named Entity Recognition**: Extracts ML-specific entities (architectures, datasets, parameters)
4. **Multi-Collection Search**: Searches across relevant vector spaces based on query intent
5. **Result Reranking**: Applies contextual reranking to improve relevance
6. **Response Generation**: Formats results based on query type and available information

### Example Query Processing

For a query like "Compare CNN models trained on CIFAR-10 vs. MNIST":

1. System detects a comparison query and splits into sub-queries:
   - "Find CNN models trained on CIFAR-10"
   - "Find CNN models trained on MNIST"
2. Each sub-query retrieves relevant models
3. System reranks results for each collection
4. LLM synthesizes comparison highlighting key differences in architecture, performance, etc.

## Technical Requirements

- **Python**: 3.9+
- **Core Dependencies**:
  - Vector Database: `chromadb`
  - Embeddings: `sentence-transformers`, `open-clip`
  - LLM Integration: `langchain`
  - UI: `streamlit`
  - Code Analysis: Standard library `ast`
  - Visualization: `graphviz`, `matplotlib`

## Advanced Configuration

The system can be customized through configuration files:

- **Vector Collections**: Configure embedding models and schemas
- **Access Control**: Set up user groups and permissions
- **LLM Models**: Configure which models to use for different analysis tasks
- **Rerankers**: Select and tune reranking strategies

## Project Structure

```
src/
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

## License

[MIT License](https://github.com/ynyeh0221/model-insight-rag/blob/main/LICENSE)
