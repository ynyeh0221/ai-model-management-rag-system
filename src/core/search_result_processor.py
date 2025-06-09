class SearchResultProcessor:
    """Helper class to process and format search results."""

    def process_search_results(self, search_results, reranker, parsed_query, query_text, rerank_threshold=0.1):
        """Process and optionally rerank search results."""
        if not isinstance(search_results, dict) or 'items' not in search_results:
            return []

        items_to_rerank = search_results['items']

        if reranker and items_to_rerank:
            for item in items_to_rerank:
                item['content'] = (
                    "Model description: " + item.get('merged_description', '') + "\n" +
                    "architecture is: " + item.get('metadata', {}).get('architecture', '') + "\n" +
                    "dataset is: " + item.get('metadata', {}).get('dataset', {})
                )
            all_ranked = reranker.rerank(
                query=parsed_query.get("processed_query", query_text),
                results=items_to_rerank,
                top_k=len(items_to_rerank),
                threshold=rerank_threshold
            )
            return self.remove_field(all_ranked, "content")

        return items_to_rerank

    @staticmethod
    def remove_field(dict_list, field_to_remove):
        """Remove a field from all dictionaries in a list."""
        if not dict_list:
            return []
        return [{k: v for k, v in item.items() if k != field_to_remove} for item in dict_list]

    @staticmethod
    def build_results_text(reranked_results, has_next_page):
        """Build structured text of search results."""
        lines = []
        for idx, item in enumerate(reranked_results, start=1):
            model = item if isinstance(item, dict) else {"model_id": str(item), "metadata": {}}
            md = model.get("metadata") or {}
            md = SearchResultProcessor.parse_metadata_json(md)
            description = SearchResultProcessor.extract_description(model, md)
            block_lines = SearchResultProcessor.format_single_model(
                idx=idx,
                model=model,
                md=md,
                description=description,
                has_next_page=has_next_page
            )
            lines.extend(block_lines)
        return "\n".join(lines)

    @staticmethod
    def parse_metadata_json(md):
        """Safely parse JSON-encoded metadata fields."""
        import json
        for key in ["file", "framework", "architecture", "dataset", "training_config"]:
            val = md.get(key)
            if isinstance(val, str):
                try:
                    md[key] = json.loads(val)
                except json.JSONDecodeError:
                    md[key] = {}
            else:
                md[key] = val or {}
        return md

    @staticmethod
    def extract_description(model, md):
        """Return a single description value with fallbacks."""
        desc = model.get("merged_description") or model.get("description")
        if desc:
            return desc
        return md.get("description") or md.get("merged_description") or "N/A"

    @staticmethod
    def format_training_config(training):
        """Format training configuration details."""
        lines = []
        if training:
            lines.append("- Training Configuration:")
            for field in ["batch_size", "learning_rate", "optimizer", "epochs", "hardware_used"]:
                pretty = field.replace("_", " ").title()
                lines.append(f"  - {pretty}: {training.get(field, 'missing')}")
        return lines

    @staticmethod
    def format_single_model(idx, model, md, description, has_next_page):
        """Build output lines for a single model block."""
        lines = []
        model_id = model.get("model_id") or model.get("id") or "missing"
        rerank_score = model.get("rerank_score", "N/A")
        file_md = md.get("file", {})
        fw = md.get("framework", {})
        arch = md.get("architecture", {})
        ds = md.get("dataset", {})
        training = md.get("training_config", {})

        lines.append(f"Model #{idx}:")
        lines.append(f"- Model ID: {model_id}")
        lines.append(f"- Rerank Score: {rerank_score}")
        lines.append(f"- File Size: {file_md.get('size_bytes', 'missing')}")
        lines.append(f"- Created On: {file_md.get('creation_date', 'missing')}")
        lines.append(f"- Last Modified: {file_md.get('last_modified_date', 'missing')}")
        lines.append(f"- Framework: {fw.get('name', 'missing')} {fw.get('version', '')}".rstrip())
        lines.append(f"- Architecture: {arch.get('type', 'missing')}")
        lines.append(arch.get("reason", "missing"))
        lines.append(f"- Dataset: {ds.get('name', 'missing')}")
        lines.extend(SearchResultProcessor.format_training_config(training))
        lines.append(f"- Description: {description}")
        lines.append(f"Has More Models: {has_next_page}")
        lines.append("")
        return lines
