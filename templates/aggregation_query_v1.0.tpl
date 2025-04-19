You are a helpful AI assistant generating a response about AI models.

User Query: `{{ query }}`

I found exactly {{ results|length }} models matching the search.
These results are already ranked from most to least relevant based on similarity to the query.

{% if intent %}Query Intent: {{ intent }}{% endif %}
{% if timeframe %}Timeframe: {{ timeframe }}{% endif %}

---

### Aggregated Insights Across All Matching Models

{% set ns = namespace(datasets=[], architectures=[], frameworks=[], optimizers=[]) %}

{% for model in results %}
  {% if model.metadata.dataset and model.metadata.dataset.name %}
    {% set _ = ns.datasets.append(model.metadata.dataset.name) %}
  {% endif %}
  {% if model.metadata.architecture and model.metadata.architecture.type %}
    {% set _ = ns.architectures.append(model.metadata.architecture.type) %}
  {% endif %}
  {% if model.metadata.framework and model.metadata.framework.name %}
    {% set _ = ns.frameworks.append(model.metadata.framework.name ~ " " ~ (model.metadata.framework.version or "")) %}
  {% endif %}
  {% if model.metadata.training_config and model.metadata.training_config.optimizer %}
    {% set _ = ns.optimizers.append(model.metadata.training_config.optimizer) %}
  {% endif %}
{% endfor %}

**All Found Datasets:**
{% if ns.datasets %}
- {{ ns.datasets | join(", ") }}
{% else %}
- No dataset information available.
{% endif %}

**All Found Architectures:**
{% if ns.architectures %}
- {{ ns.architectures | join(", ") }}
{% else %}
- No architecture info found.
{% endif %}

**All Found Frameworks:**
{% if ns.frameworks %}
- {{ ns.frameworks | join(", ") }}
{% else %}
- No framework info found.
{% endif %}

**All Found Optimizers:**
{% if ns.optimizers %}
- {{ ns.optimizers | join(", ") }}
{% else %}
- No optimizer info found.
{% endif %}

---

### Extracted Details from Top-Ranked Models

{% for model in results %}
MODEL #{{ loop.index }}:
- Model ID: {{ model.model_id }}
- File Size: {{ model.metadata.file.size_bytes if model.metadata.file and model.metadata.file.size_bytes else "Unknown" }}
- File Creation Date: {{ model.metadata.file.creation_date if model.metadata.file and model.metadata.file.creation_date else "Unknown" }}
- File Last Modified Date: {{ model.metadata.file.last_modified_date if model.metadata.file and model.metadata.file.last_modified_date else "Unknown" }}
- Framework: {{ model.metadata.framework.name if model.metadata.framework and model.metadata.framework.name else "Unknown" }}{{ " " + model.metadata.framework.version if model.metadata.framework and model.metadata.framework.version else "" }}
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

Based on the query `{{ query }}`, here’s what we found from analyzing all {{ results|length }} matching models:

{% if "dataset" in query.lower() and ns.datasets %}
- **Datasets Found:** {{ ns.datasets | join(", ") }}
{% endif %}

{% if "architecture" in query.lower() and ns.architectures %}
- **Architectures Found:** {{ ns.architectures | join(", ") }}
{% endif %}

{% if "framework" in query.lower() and ns.frameworks %}
- **Frameworks Found:** {{ ns.frameworks | join(", ") }}
{% endif %}

{% if "optimizer" in query.lower() and ns.optimizers %}
- **Optimizers Found:** {{ ns.optimizers | join(", ") }}
{% endif %}

{% if not (
  "dataset" in query.lower() or
  "architecture" in query.lower() or
  "framework" in query.lower() or
  "optimizer" in query.lower()
) %}
- No specific pattern match found based on your query. Try asking about "common dataset", "most used architecture", etc., or inspect the individual model summaries above.
{% endif %}

---
