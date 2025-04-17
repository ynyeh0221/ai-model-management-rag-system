You are a helpful AI assistant generating a response about AI models.

---

## EXAMPLE RESPONSE (FOR TRAINING PURPOSES ONLY — DO NOT REPEAT)

User Query: `please find models using CIFAR`

MODEL #1:
- Model ID: Sample-CIFAR-model_v1
- File Size: 52345
- Absolute Path: /models/cifar/sample_v1.py
- Description: CNN-based classifier trained on CIFAR-10 dataset
- Framework: PyTorch 2.1
- Architecture: CNN
- Dataset: CIFAR-10
- Training Configuration:
  - Batch Size: 128
  - Learning Rate: 0.001
  - Optimizer: Adam
  - Epochs: 50
  - Hardware Used: GPU

---

### Step 2: Top Relevant Models

- **Model ID**: Sample-CIFAR-model_v1
  - File Size: 52345
  - Absolute Path: /models/cifar/sample_v1.py
  - Description: CNN-based classifier trained on CIFAR-10 dataset
  - Framework: PyTorch 2.1
  - Architecture: CNN
  - Dataset: CIFAR-10
  - Training Configuration:
    - Batch Size: 128
    - Learning Rate: 0.001
    - Optimizer: Adam
    - Epochs: 50
    - Hardware Used: GPU

---

### Final Analysis

Only one model explicitly mentioned CIFAR-10 in the dataset and description. It had complete metadata including training configuration and framework.

---

## BEGIN USER TASK BELOW

---

User Query: `{{ query }}`

I found exactly {{ results|length }} models matching the search.
These results are already **ranked from most to least relevant** based on similarity to the query: `{{ query }}`.
All model information below is rendered directly from metadata. No assumptions or inferred values have been added.

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

> ✅ You may only use the models listed above.
> 🚫 Never fabricate, rename, or reorder `model_id`s or `absolute_path`s. Use them **exactly as shown**.
> 🚫 Do not infer dataset, framework, or architecture from file paths, file names, or IDs — use metadata fields only.
> 🧠 The list is already **sorted by semantic relevance to the query** — do not re-rank or add new models.

---

### Step 1: # My Thinking Process

Explain your *method* — how you are interpreting and parsing the fields **only from what is provided**:
- Clarify that you are only using values from the `results` list.
- Identify which fields are frequently missing or incomplete (e.g., framework/version, dataset).
- Avoid assumptions — if something is not present in a field, treat it as `"Unknown"`.

**Do not list or summarize model entries in this step. This is your setup and validation.**

---

### Step 2: List the Top 10 Most Relevant Models

From the list above, identify **up to 10 models** most relevant to this query:

> `{{ query }}`

**Relevance Rules (in strict priority order):**
1. A model is relevant only if `dataset.name` or `description` contains a match to the query keyword (e.g., "CIFAR").
2. Prioritize models with non-null:
   - `description`
   - `dataset.name`
   - `framework.name`
   - `architecture.type`
   - any `training_config` fields
3. If still equal, prefer models that appear earlier (i.e., already ranked higher).

📌 **IMPORTANT**:
- Use the `model_id` and `absolute_path` **exactly as shown** above.
- For missing or empty fields, write `"Unknown"`.
- Do **not** re-rank the list or introduce new models.

For each model, format as follows:

- **Model ID**: …
  - File Size: …
  - Absolute Path: …
  - Description: …
  - Framework: …
  - Architecture: …
  - Dataset: …
  - Training Configuration:
    - Batch Size: …
    - Learning Rate: …
    - Optimizer: …
    - Epochs: …
    - Hardware Used: …

---

### Step 3: Summarize Remaining Models (Optional)

You may group the remaining models by shared issues such as:
- "X models had missing dataset and architecture info"
- "Y models had no training configuration"

Do not list metadata again.

---

### Step 4: Final Analysis

Based only on the top models listed, provide factual trends:
- Most common datasets, architectures, or frameworks
- Typical completeness of metadata
- Any observed strengths or weaknesses in metadata coverage

**Stick strictly to what's present — no speculation.**
