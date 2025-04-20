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
- Framework: {{ model.metadata.framework.name if model.metadata.framework else "Unknown" }}{% if model.metadata.framework and model.metadata.framework.version %} {{ model.metadata.framework.version }}{% endif %}
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

### Detailed Model Comparison

Below you'll find a side‑by‑side comparison. Focus on:
{% if parsed_query.parameters and parsed_query.parameters.comparison_points %}
  {% set pts = parsed_query.parameters.comparison_points %}
{% else %}
  {% set pts = ['Architecture', 'Performance metrics', 'Training methodology', 'Use cases and capabilities'] %}
{% endif %}

Models:
{% for model in results %}
- {{ model.model_id }}
{% endfor %}

---

{% for point in pts %}
#### {{ point }}

{% for model in results %}
- **{{ model.model_id }}**:
  {% if point == 'Architecture' %}
    {{ model.metadata.architecture.type if model.metadata.architecture and model.metadata.architecture.type else "Unknown" }}
  {% elif point == 'Performance metrics' %}
    Accuracy: {{ model.metadata.performance.accuracy if model.metadata.performance and model.metadata.performance.accuracy is not none else "Unknown" }},
    Loss: {{ model.metadata.performance.loss if model.metadata.performance and model.metadata.performance.loss is not none else "Unknown" }}
  {% elif point == 'Training methodology' %}
    Batch Size: {{ model.metadata.training_config.batch_size if model.metadata.training_config and model.metadata.training_config.batch_size is not none else "Unknown" }},
    Learning Rate: {{ model.metadata.training_config.learning_rate if model.metadata.training_config and model.metadata.training_config.learning_rate is not none else "Unknown" }}
  {% elif point == 'Use cases and capabilities' %}
    {{ model.metadata.description if model.metadata.description else "Unknown" }}
  {% else %}
    N/A
  {% endif %}
{% endfor %}

{% endfor %}

---

### Summary

Conclude with a high‑level summary of the trade‑offs between these models and which is most appropriate for different scenarios based on the comparison above.
