You are a helpful AI assistant generating a response about AI models.

User Query: `{{ query }}`

I found exactly {{ results|length }} models matching the search.
These results are already ranked from most to least relevant based on similarity to the query.

{% if intent %}Query Intent: {{ intent }}{% endif %}
{% if timeframe %}Timeframe: {{ timeframe }}{% endif %}

---

### Top-Ranked Model Summaries

{% for model in results %}
MODEL #{{ loop.index }}:
- Model ID: {{ model.model_id }}
- File Size: {{ model.metadata.file.size_bytes if model.metadata.file and model.metadata.file.size_bytes else "Unknown" }}
- Created On: {{ model.metadata.file.creation_date if model.metadata.file and model.metadata.file.creation_date else "Unknown" }}
- Last Modified: {{ model.metadata.file.last_modified_date if model.metadata.file and model.metadata.file.last_modified_date else "Unknown" }}
- Framework: {{ model.metadata.framework.name if model.metadata.framework else "Unknown" }}{{ " " + model.metadata.framework.version if model.metadata.framework and model.metadata.framework.version else "" }}
- Architecture: {{ model.metadata.architecture.type if model.metadata.architecture and model.metadata.architecture.type else "Unknown" }}
- Dataset: {{ model.metadata.dataset.name if model.metadata.dataset and model.metadata.dataset.name else "Unknown" }}

{% if model.metadata.training_config %}
- Training Configuration:
  - Batch Size: {{ model.metadata.training_config.batch_size if model.metadata.training_config.batch_size is not none else "Unknown" }}
  - Learning Rate: {{ model.metadata.training_config.learning_rate if model.metadata.training_config.learning_rate is not none else "Unknown" }}
  - Optimizer: {{ model.metadata.training_config.optimizer if model.metadata.training_config.optimizer else "Unknown" }}
  - Epochs: {{ model.metadata.training_config.epochs if model.metadata.training_config.epochs is not none else "Unknown" }}
  - Hardware Used: {{ model.metadata.training_config.hardware_used if model.metadata.training_config.hardware_used else "Unknown" }}
{% endif %}

- Description: {{ model.metadata.description if model.metadata.description and model.metadata.description != "N/A" else "Unknown" }}
  {% if model.metadata.description and query.lower() in model.metadata.description.lower() %}
  🔍 This description contains terms relevant to your query.
  {% endif %}
{% endfor %}

---

### Answer to the User’s Question

{% set answered = false %}
{% for model in results %}
  {% if model.metadata.description and query.lower() in model.metadata.description.lower() %}
    {% set answered = true %}
    Based on **Model #{{ loop.index }} ({{ model.model_id }})**, here’s what we can infer:
    "{{ model.metadata.description }}"
    → This content directly addresses or contains information relevant to your query.
  {% endif %}
{% endfor %}

{% if not answered %}
No explicit match was found in the descriptions or structured metadata for your query: `{{ query }}`.
Try rephrasing the question or checking individual model summaries above for more details.
{% endif %}
