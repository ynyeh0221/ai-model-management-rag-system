You are a helpful AI assistant generating a response about AI models.

User Query: `{{ query }}`

I found exactly {{ results|length }} models matching the search.
These results are already ranked from most to least relevant based on similarity to the query.

{% if intent %}Query Intent: {{ intent }}{% endif %}
{% if timeframe %}Timeframe: {{ timeframe }}{% endif %}

{# Display top models from ranked results #}
Top ranked models:
{% for model in results %}
MODEL #{{ loop.index }}:
- Model ID: {{ model.model_id }}
- File Size: {{ model.metadata.file.size_bytes if model.metadata.file and model.metadata.file.size_bytes else "Unknown" }}
- File Creation Date: {{ model.metadata.file.creation_date if model.metadata.file and model.metadata.file.creation_date else "Unknown" }}
- File Last Modified Date: {{ model.metadata.file.last_modified_date if model.metadata.file and model.metadata.file.last_modified_date else "Unknown" }}
- Description: {{ model.metadata.description if model.metadata.description and model.metadata.description != "N/A" else "Unknown" }}
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
{% endfor %}

Please analyze these models using the following two-step approach:

### Step 1: Your Thinking Process
<thinking>
- Explain your thought process for analyzing these models
- Which query terms match which models and why?
- Identify which fields are useful and which are missing
- Consider how the models relate to the query `{{ query }}`
{% if intent %}- How well do the results satisfy the {{ intent }} intent?{% endif %}
</thinking>

### Step 2: Final Analysis
Based only on the models you've analyzed, provide factual insights:
- Most common datasets, architectures, or frameworks
- Typical completeness of metadata
- Any observed strengths or weaknesses
- How well the results match the query intent
- Suggested follow-up queries that might yield better results

Stick strictly to what's present in the data — no speculation.