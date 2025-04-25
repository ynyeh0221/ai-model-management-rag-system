## Model Insight RAG System

### Overview

This system enables structured understanding, indexing, and retrieval of machine learning model code and metadata. It combines static analysis, LLM-based summarization, vector embeddings, and Google Colab automation to support advanced retrieval-augmented workflows.

[DEMO](https://github.com/ynyeh0221/ai-model-management-rag-system/blob/main/DEMO.md)

### Features

- Script processing and code chunking via AST
- LLM-based metadata extraction and summarization
- Schema validation and multi-collection vector indexing
- Support for querying across different embedding spaces (multi-vector support)
- Remote notebook generation and execution via Google Colab API
- Modular access control and metadata schemas

---

### Core Components

#### ScriptProcessorRunner

Processes entire directories of model-related files. For each file:

- Verifies file type and structure
- Parses code using `LLMBasedCodeParser`
- Extracts both metadata and code chunks
- Validates documents using schema definitions
- Embeds and stores both code and metadata in separate Chroma collections

#### LLMBasedCodeParser

Parses `.py` or `.ipynb` files by:

1. Extracting code structure using Python's AST (classes, functions, variables)
2. Summarizing with an LLM (e.g., GPT) to generate structured metadata including:
   - Description of model purpose
   - Framework (name and version)
   - Architecture type
   - Dataset used
   - Training configuration (e.g., batch size, learning rate)

All metadata is normalized and can be extended or validated against versioned schemas.

#### Image Embedding & Similarity Search

The system supports multimodal retrieval by embedding and indexing model-generated or related images. This enables advanced search workflows such as:

- **Text-to-Image Search**: Find model outputs or visuals that best match a natural language query.
- **Image-to-Image Search**: Discover visually similar outputs using image embeddings.

##### Embedding Pipeline

The `ImageEmbedder` module utilizes [OpenCLIP](https://github.com/mlfoundations/open_clip) to compute embeddings for images found in configured directories or generated during model execution. These embeddings are stored in the `model_images_folder` Chroma collection.

- Extracts `.png`, `.jpg`, `.jpeg` files
- Generates embeddings using CLIP’s vision encoder
- Associates metadata like `source_model`, `filepath`, `generation_params`

---

### Data Model

Each piece of metadata is stored as a document in a dedicated vector collection:

| Collection               | Purpose                            | Example Metadata                       |
|--------------------------|------------------------------------|----------------------------------------|
| `model_file`             | File-level attributes              | Path, size, creation/modification dates|
| `model_date`             | Timestamp-based indexing           | Month/year created, last modified      |
| `model_frameworks`       | Framework metadata                 | `{"name": "PyTorch", "version": "2.0"}`|
| `model_architectures`    | Model type and structure           | `{"type": "Transformer"}`              |
| `model_datasets`         | Dataset references                 | `{"name": "CIFAR-10"}`                 |
| `model_training_configs` | Training configuration             | Optimizer, epochs, batch size, etc.    |
| `model_descriptions`     | LLM-generated text summaries       | Concise explanation of each code chunk |
| `model_ast_summaries`    | Model AST digest summaries         | Concise explanation of code            |
| `model_images_folder`    | Folders to store generated images  | Folder in /a/b/c format                |
| `model_scripts_chunks`   | Code snippets and AST sections     | Actual Python source segments          |

---

### Multi-Vector Support

Each Chroma collection supports its own embedding model and schema. This allows different types of information to be indexed, retrieved, and filtered independently.

- `model_descriptions` is optimized for natural language search
- `model_frameworks` supports faceted filtering on version and name
- `model_scripts_chunks` enables deep semantic search over model code
- Queries can be executed across any vector space with optional structured filters

Example:

```python
results = await chroma_manager.search(
    query="transformer trained on CIFAR-10",
    collection_name="model_descriptions",
    where={"framework.name": {"$eq": "PyTorch"}}
)
```

---

### ColabAPIClient

Handles notebook lifecycle management with Google APIs.

- Authenticates with service account or OAuth2
- Creates and uploads Jupyter notebooks from extracted code
- Launches remote execution with:
  - Custom parameters
  - GPU/TPU support
  - Execution monitoring
  - Result downloading and inspection

Example:

```python
nb = codegen.generate_notebook_from_chunks(chunks)
client = ColabAPIClient()
file_id = client.create_notebook(nb, "example_model.ipynb")
execution_id = client.execute_notebook(file_id, accelerator_type="GPU")
status = client.wait_for_execution(execution_id)
```

---

### Access Control

Document-level access control is enforced at the time of ingestion:

- Documents include access metadata derived from `access_control.get_document_permissions`
- These access rights are respected during query filtering and sharing
- Supports user-level, group-level, or public visibility

---

### Response Generation Workflow

When a user submits a query through the CLI interface (`UIRunner`), the system follows a structured multi-stage process to generate a final response:

#### 1. **Query Parsing**
- The raw query is processed by a `query_parser` component.
- Extracts the user's intent (e.g., retrieval, comparison, summarization) and filters (e.g., timeframe, framework).
  
#### 2. **Search Dispatching**
- The parsed query is sent to a `search_dispatcher`, which dispatches it to one or more vector collections:
  - `model_descriptions` for natural language summaries
  - `model_frameworks`, `model_architectures`, `model_training_configs` for structured filtering
- The dispatcher supports hybrid vector+metadata queries and returns relevant matches.

#### 3. **Reranking**
- Raw search results are reranked using a `reranker` (e.g., BGE-Reranker or CrossEncoder).
- Scores are adjusted based on textual similarity and contextual relevance to the query.

#### 4. **Template Selection**
- A `template_manager` selects the appropriate response template based on query type:
  - Retrieval: e.g., “Which models use Vision Transformers on CIFAR-10?”
  - Aggregation: e.g., “What frameworks are most common for diffusion models?”
  - Comparison: e.g., “Compare ModelA vs ModelB”
  - General query fallback if no intent is confidently extracted

#### 5. **Prompt Construction**
- The selected template is rendered using:
  - The original query
  - Top-N reranked search results
  - Parsed metadata (frameworks, datasets, configs, etc.)

#### 6. **LLM Response Generation**
- The `llm_interface` (e.g., OpenAI or other LLM provider) generates a final answer using the templated prompt.
- Responses can include:
  - Summary tables
  - Textual comparisons
  - Recommendations based on metadata

#### 7. **Fallback Strategy**
- If template rendering fails, a backup prompt is constructed using raw results + query.
- A fallback LLM call is made to ensure robustness.

---

### Technology Stack

| Category          | Tools / Libraries |
|-------------------|------------------|
| Language          | Python 3.9+, JavaScript (React) |
| Backend           | FastAPI / Flask |
| Frontend          | React, Tailwind CSS, Zustand |
| Vector Search     | ChromaDB, FAISS, SQLite |
| Embeddings        | SentenceTransformers, OpenCLIP |
| LLM Integration   | LangChain, LLaMA, Deepseek-llm |
| Image Processing  | Pillow, OpenCLIP |
| Notebooks         | nbformat, Papermill, Google Colab API |
| Visualization     | Matplotlib, Plotly |
| Monitoring        | Prometheus, Grafana, OpenTelemetry |
| Logging           | Elasticsearch, Kibana |
| Access Control    | JWT, Role-Based Access Control (RBAC) |

---

### Example

```bash
> query transformer with CIFAR-10
```

The system:

- Parses it as an information retrieval query
- Searches in `model_descriptions` and filters by dataset
- Reranks results based on relevance to “transformer”
- Selects the `information_retrieval` template
- Builds a prompt like:

```txt
Query: transformer with CIFAR-10

Top Results:
1. Model: ViT_CIFAR10
   - Framework: PyTorch
   - Dataset: CIFAR-10
   - Arch: Vision Transformer
   - Training: Adam, batch=32, lr=3e-4
2. Model: TransformerBaseline
   - Framework: TensorFlow
   - Dataset: CIFAR-10
   - Arch: Transformer

Please summarize key differences and trends across these models.
```

LLM returns:

> "Two transformer-based models were found using CIFAR-10. The first uses PyTorch and ViT architecture with Adam optimizer and a learning rate of 3e-4. The second uses a TensorFlow baseline. Most models prefer PyTorch for transformer training on this dataset."

---

### Installation

```bash
pip install -r requirements.txt
```

Requirements include:

- `chromadb`, `sentence-transformers`, `nbformat`
- `openai`, `gitpython`, `google-auth`, `google-api-python-client`
- `torch`, `Pillow`, `jsonschema`, etc.

---

### Usage

Run batch model script processing:

```bash
python run_model_processor.py --directory ./models
```

This will:

- Traverse all supported model scripts
- Parse and chunk using AST
- Extract structured metadata via LLM
- Store into multiple vector collections in Chroma
- Generate Colab notebooks if configured

---

### Folder Structure

```bash
.
├── colab/                     # Google Colab integration
├── codegen/                   # Notebook and code generation
├── processing/                # Script parsing and metadata extraction
├── vector_db/                 # Chroma and embedding management
├── schemas/                   # Metadata schema definitions
├── models/                    # Example model scripts
├── run_model_processor.py     # Entry point script
└── README.md
```

---

### License

MIT License

---
