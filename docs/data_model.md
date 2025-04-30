# Model Insight RAG - Data Model Documentation

This document provides a detailed overview of the data model used in the Model Insight RAG system. The system uses a multi-collection approach to store different aspects of machine learning models, enabling specialized vector embeddings and targeted search.

## Schema Registry

All collections adhere to schemas defined in the registry `MySchemaRegistry` (version 1.0.0).

## Core Collection Structure

Each collection follows a consistent base structure:

```json
{
  "id": "unique_identifier",
  "content": "text_for_embedding",
  "metadata": {
    "model_id": "related_model_identifier",
    "access_control": { /* access control properties */ }
    // Collection-specific properties
  }
}
```

## Collection Details

### 1. Model File Information (`model_file`)

Stores basic file metadata for model scripts.

**Key fields:**
- `model_id`: Unique identifier for the model
- `file`:
  - `size_bytes`: File size in bytes
  - `creation_date`: File creation timestamp
  - `last_modified_date`: Last modification timestamp
  - `file_extension`: File type (e.g., `.py`, `.ipynb`)
  - `absolute_path`: Full path to the file

**Usage:** Enables file-based filtering and retrieving source locations.

### 2. Model Date Information (`model_date`)

Tracks temporal information about models.

**Key fields:**
- `created_at`: Full creation timestamp
- `created_month`, `created_year`: Extracted date components
- `last_modified_month`, `last_modified_year`: Modification date components

**Usage:** Supports time-based queries (e.g., "models created in 2024").

### 3. Git Repository Information (`model_git`)

Stores version control metadata.

**Key fields:**
- `git`:
  - `creation_date`: Initial commit date
  - `last_modified_date`: Latest commit date
  - `commit_count`: Number of commits

**Usage:** Provides development history context and activity metrics.

### 4. Framework Information (`model_frameworks`)

Identifies machine learning frameworks used.

**Key fields:**
- `framework`:
  - `name`: Framework name (e.g., "PyTorch", "TensorFlow")
  - `version`: Framework version

**Usage:** Enables framework-specific searches (e.g., "PyTorch models using version 2.0+").

### 5. Dataset Information (`model_datasets`)

Links models to their training datasets.

**Key fields:**
- `dataset`:
  - `name`: Dataset name (e.g., "MNIST", "CIFAR-10")

**Usage:** Facilitates dataset-specific queries (e.g., "models trained on ImageNet").

### 6. Training Configuration (`model_training_configs`)

Captures training hyperparameters.

**Key fields:**
- `training_config`:
  - `batch_size`: Training batch size
  - `learning_rate`: Learning rate value
  - `optimizer`: Optimizer algorithm (e.g., "Adam", "SGD")
  - `epochs`: Number of training epochs
  - `hardware_used`: Training hardware information

**Usage:** Enables hyperparameter-specific queries and comparisons.

### 7. Architecture Information (`model_architectures`)

Stores model architecture classifications.

**Key fields:**
- `architecture`:
  - `type`: Architecture type (e.g., "CNN", "Transformer")
  - `reason`: Explanation of the classification

**Usage:** Supports architecture-based searches and comparisons.

### 8. Model Descriptions (`model_descriptions`)

Contains natural language descriptions of models.

**Key fields:**
- `description`: Text description of the model
- `total_chunks`: Number of chunks this description is split into
- `offset`: Position within the chunked sequence

**Usage:** Enables natural language searches about model functionality.

### 9. AST Summaries (`model_ast_summaries`)

Stores Abstract Syntax Tree analysis results.

**Key fields:**
- `ast_summary`:
  - `ast_summary`: Structural analysis of the model code

**Usage:** Provides code structure insights for technical searches.

### 10. Code Chunks (`model_chunk`)

Contains segmented code from model scripts.

**Key fields:**
- `chunk_id`: Chunk identifier
- `total_chunks`: Total number of chunks for this model
- `metadata_doc_id`: Reference to parent document
- `offset`: Position within the chunked sequence
- `type`: Content type identifier

**Usage:** Supports code-specific searches with context preservation.

### 11. Images Folder Information (`model_images_folder`)

Tracks folders containing model-generated images.

**Key fields:**
- `images_folder`:
  - `name`: Folder path name

**Usage:** Links models to their image outputs for browsing.

### 12. Diagram Path Information (`model_diagram_path`)

Stores locations of model architecture diagrams.

**Key fields:**
- `diagram_path`:
  - `name`: Path to the diagram file

**Usage:** Enables visualization of model architectures.

### 13. Generated Images (`generated_images`)

Stores metadata about model-generated images with rich content attributes.

**Key fields:**
- `epoch`: Training epoch when image was generated
- `description`: Text description of the image
- `image_content`:
  - `subject_type`: Primary subject category
  - `subject_details`: Additional subject information
  - `scene_type`: Scene context classification
  - `style`: Visual style descriptor
  - `tags`: Content tags
  - `colors`: Dominant colors
  - `objects`: Identified objects
- `image_path`: Path to full-size image
- `thumbnail_path`: Path to thumbnail version
- `format`: Image format (e.g., "PNG", "JPEG")
- `mode`: Color mode (e.g., "RGB", "RGBA")
- `size`: Image dimensions [width, height]
- `exif`: EXIF metadata
- `dates`: Temporal information

**Usage:** Enables sophisticated image search and filtering based on content, style, and technical attributes.

## Relationships Between Collections

All collections are linked by the `model_id` field, which serves as a foreign key reference to create a cohesive view of each model:

```
model_id
   |
   ├── model_file
   ├── model_date
   ├── model_git
   ├── model_frameworks
   ├── model_datasets
   ├── model_training_configs
   ├── model_architectures
   ├── model_descriptions
   ├── model_ast_summaries
   ├── model_chunk
   ├── model_images_folder
   ├── model_diagram_path
   └── generated_images
```

## Vector Embedding Strategy

Each collection uses specialized embedding models appropriate to its content type:

- **Code collections** (`model_chunk`, `model_ast_summaries`): Code-specific embedding models
- **Description collections** (`model_descriptions`): Text embedding models
- **Technical collections** (`model_frameworks`, etc.): Domain-specific embeddings
- **Image collections** (`generated_images`): CLIP-based multi-modal embeddings

This specialized approach enables more accurate similarity search within each domain while maintaining cross-collection relationships through the `model_id` field.

## Example Query Patterns

### Architecture-Specific Search

```
"Find transformer models with self-attention mechanisms"
```

Primary collections: `model_architectures`, `model_ast_summaries`, `model_chunk`

### Dataset Comparison

```
"Compare CNN models trained on CIFAR-10 vs. MNIST"
```

Primary collections: `model_datasets`, `model_architectures`, `model_training_configs`

### Code Pattern Search

```
"Find models implementing gradient clipping"
```

Primary collections: `model_chunk`, `model_ast_summaries`

### Visual Output Search

```
"Show me image generation models that produce photorealistic portraits"
```

Primary collections: `generated_images`, `model_architectures`

## Access Control

All collections include an `access_control` object that enables fine-grained permission management at the model level. This allows organizations to implement:

- User-specific access restrictions
- Team-based sharing
- Role-based permissions
- Visibility controls (public/private/internal)
