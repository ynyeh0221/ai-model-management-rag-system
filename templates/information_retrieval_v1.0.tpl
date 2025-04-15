You are a helpful AI assistant generating a response about AI models.

User Query: `{{ query }}`

I found exactly {{ results|length }} models created in April. Here is the information for each model, based strictly on the provided metadata:

{% for model in results %}
MODEL #{{ loop.index }}:
- Model ID: {{ model.model_id }}
- File Size: {{ model.metadata.file.size_bytes if model.metadata.file and model.metadata.file.size_bytes else "Unknown" }}
- Absolute Path: {{ model.metadata.file.absolute_path if model.metadata.file and model.metadata.file.absolute_path else "Unknown" }}
- Description: {{ model.metadata.description if model.metadata.description and model.metadata.description != "N/A" else "Unknown" }}
- Framework:
  {% set fw = model.metadata.framework %}
  {{ fw.name if fw and fw.name else "Unknown" }}{{ " " + fw.version if fw and fw.version else "" }}
- Architecture: {{ model.metadata.architecture.type if model.metadata.architecture and model.metadata.architecture.type else "Unknown" }}
- Dataset: {{ model.metadata.dataset.name if model.metadata.dataset and model.metadata.dataset.name else "Unknown" }}
- Training Configuration:
  {% set tc = model.metadata.training_config %}
  - Batch Size: {{ tc.batch_size if tc and tc.batch_size is not none else "Unknown" }}
  - Learning Rate: {{ tc.learning_rate if tc and tc.learning_rate is not none else "Unknown" }}
  - Optimizer: {{ tc.optimizer if tc and tc.optimizer else "Unknown" }}
  - Epochs: {{ tc.epochs if tc and tc.epochs is not none else "Unknown" }}
  - Hardware Used: {{ tc.hardware_used if tc and tc.hardware_used else "Unknown" }}
{% endfor %}

---

## CRITICAL INSTRUCTIONS

You must strictly follow the four steps below. Do not invent or assume any data not explicitly available in the model metadata.

---

### Step 1: # My Thinking Process

Explain your *method* — how you are interpreting and parsing the fields **only from what is provided**:
- Clarify that you are only using values from the `results` list, and not inventing model IDs or attributes.
- Identify which fields are frequently missing or incomplete (e.g., framework/version, dataset).
- Avoid assumptions — if something is not present in a field, treat it as `"Unknown"`.

**Do not list or summarize model entries in this step. This is your setup and validation.**

---

### Step 2: List the Top 10 Most Relevant Models

From the list above, identify **up to 20 models** most relevant to this query:

> `{{ query }}`

**Relevance Rules (in priority order):**
1. Models explicitly mentioning a relevant keyword (e.g., “CIFAR” in query → Dataset or Description match “CIFAR”).
2. Prefer models with non-null:
   - `description`
   - `dataset.name`
   - `framework.name`
   - `architecture.type`
   - any `training_config` fields
3. If still equal, prefer models with newer `created_at` date.

For each model, show in bullet format:
- Model ID
- File Size
- Absolute Path
- Description
- Framework
- Architecture
- Dataset
- Training Configuration

**You must use only metadata fields. Never fabricate values.**

---

### Step 3: Summarize Remaining Models (Optional)

You may optionally group remaining models by issues (e.g., “6 models had missing dataset and architecture info”).

Do not repeat full metadata for these models.

---

### Step 4: Final Analysis

Based on the listed models, summarize any observed **data trends**:
- Frequent frameworks, architectures, or datasets
- Quality and completeness of metadata (e.g., "many models are missing training configs")
- Any strong patterns of interest (e.g., "models using CIFAR-10 are mostly PyTorch-based")

**Stick to facts. Do not speculate or generalize beyond what the metadata shows.**

---

## EXAMPLE (Do not output this again; for learning only)

User Query: `please find models using STL`

MODEL #1:
- Model ID: SampleModel1
- File Size: 40120
- Absolute Path: /path/to/sample1.py
- Description: CNN model for classifying STL-10 images
- Framework: PyTorch 2.0
- Architecture: CNN
- Dataset: STL-10
- Training Configuration:
  - Batch Size: 128
  - Learning Rate: 0.001
  - Optimizer: Adam
  - Epochs: 25
  - Hardware Used: GPU

MODEL #2:
...

---

### Step 1: # My Thinking Process

I only used the dataset and description fields to identify references to "STL". I did not use the model ID or filename, even if they contained the word “STL”. I checked which fields were missing or marked as "Unknown", and I did not fabricate any values. Most models lacked dataset metadata, but a few had “STL-10” clearly listed.

---

### Step 2: Top 3 Relevant Models

Here are the most relevant models for the query:

- **Model ID**: SampleModel1
  - File Size: 40120
  - Absolute Path: /path/to/sample1.py
  - Description: CNN model for classifying STL-10 images
  - Framework: PyTorch 2.0
  - Architecture: CNN
  - Dataset: STL-10
  - Training Configuration:
    - Batch Size: 128
    - Learning Rate: 0.001
    - Optimizer: Adam
    - Epochs: 25
    - Hardware Used: GPU

(… more ranked models here …)

---

### Step 3: Remaining Models

12 models had no reference to STL and were missing dataset and architecture fields.

---

### Step 4: Final Analysis

Only 3 models clearly referenced the STL dataset. Most metadata was incomplete. PyTorch was the most commonly listed framework. No models used TensorFlow or JAX. Only two had complete training configuration.

