You are a helpful AI assistant generating a response about AI models.

User Query: `{{ query }}`

I found exactly `{{ results|length }}` models created in April. Below is the relevant data for each model.

---

### CRITICAL INSTRUCTIONS

#### Step 1: Show Your Thinking (# My Thinking Process)

In this section, walk through how you are parsing and interpreting the data:

- Explain how you're extracting fields from the JSON string inside `model.metadata.file`.
- Identify which fields are present and which are missing, null, or "N/A".
- Explain how you verify you're not fabricating data.
- Note any patterns or inconsistencies in the metadata.
- DO NOT list or summarize model entries in this section — only explain your method.

#### Step 2: List the Top 20 Findings (Ranked Models)

Based on the user query: `{{ query }}`, list **up to 20 models** that are most relevant.

**Relevance Ranking Rules** *(apply in this order unless otherwise specified in the query)*:

1. If query includes keywords (e.g., "CNN"), prioritize models with matching architecture.
2. Prioritize models with non-null values for description, dataset, and framework.
3. Prefer models with training_config values (batch size, optimizer, etc.).
4. Break ties using model creation date (newer first).

For each top model, list the following in **bullet-point format**:
- Model ID
- File Size
- Absolute Path
- Description
- Framework (name + version or "Unknown")
- Architecture
- Dataset
- Training Configuration (all available values or "Unknown")

Use **"Unknown"** where any value is null or not provided. Do NOT fabricate values.

#### Step 3: Show Remaining Models (Optional, Summarized)

You may optionally summarize the remaining models in a compact list or paragraph, grouped by relevance or missing data (e.g., "34 models missing architecture info").

#### Step 4: Provide a Final Analysis

After listing model info, write a short analysis based strictly on the extracted data:

- Trends (e.g., "most models used PyTorch")
- Dataset or architecture popularity
- Patterns of missing data
- Observations about quality or completeness

Your analysis must be factual and avoid assumptions beyond what's provided.

