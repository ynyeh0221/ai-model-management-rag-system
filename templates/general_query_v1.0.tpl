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