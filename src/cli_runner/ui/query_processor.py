import json

from cli_runner.ui.image_display_manager import ImageDisplayManager
from cli_runner.ui.llm_response_processor import LLMResponseProcessor
from cli_runner.ui.model_display_manager import ModelDisplayManager
from response_generator.llm_interface import LLMInterface


def remove_thinking_sections(text: str) -> str:
    """
    Remove all <thinking>...</thinking> sections from the input text.

    Args:
        text: The input string containing zero or more <thinking> sections.

    Returns:
        A new string with all <thinking> sections and their content removed.
    """
    import re
    # Regex to match <thinking>...</thinking> including multiline content (non-greedy)
    pattern = re.compile(r'<thinking>.*?</thinking>', re.DOTALL)
    # Replace matched sections with an empty string
    return pattern.sub('', text)


class QueryProcessor:
    """Handles query processing and results."""

    def __init__(self, components, user_id):
        self.components = components
        self.user_id = user_id
        self.model_display = ModelDisplayManager()
        self.image_display = ImageDisplayManager()
        self.llm_processor = LLMResponseProcessor()

    async def process_query(self, query_text):
        """
        Process and execute a query.

        Args:
            query_text (str): The query text to process.
        """
        # Extract required components
        query_parser = self.components["query_engine"]["query_parser"]
        search_dispatcher = self.components["query_engine"]["search_dispatcher"]
        query_analytics = self.components["query_engine"]["query_analytics"]
        reranker = self.components["query_engine"]["reranker"]
        llm_interface = self.components["response_generator"]["llm_interface"]

        # Parse the query
        parsed_query = query_parser.parse_query(query_text)
        print(f"Parsed query: {parsed_query}")

        # Log the query for analytics
        query_analytics.log_query(self.user_id, query_text, parsed_query)

        print("Searching...")

        # Special handling for image searches if needed
        if parsed_query["intent"] == "image_search":
            print("Note: Image uploads for image-to-image similarity search are not supported in CLI mode.")

        # Dispatch the query
        search_results = await search_dispatcher.dispatch(
            query=parsed_query.get("processed_query", query_text),
            intent=parsed_query["intent"],
            parameters=parsed_query["parameters"],
            user_id=self.user_id
        )

        # Different handling based on query type
        if parsed_query["intent"] == "image_search":
            self.image_display.display_image_search_results(search_results)
        else:
            # Process and rerank search results
            reranked_results = self._process_search_results(
                search_results, reranker, parsed_query, query_text, max_to_return=3
            )

            # Remove content field since it's not needed for display
            reranked_results = self._remove_field(reranked_results, "content")

            # Generate response using appropriate template
            self._generate_query_response(query_text, reranked_results, llm_interface)

    def _process_search_results(self, search_results, reranker, parsed_query, query_text,
                                max_to_return=10, rerank_threshold=0.108):
        """
        Process and rerank search results.

        Args:
            search_results (dict): The raw search results from the dispatcher.
            reranker (object): Reranker component for ordering results by relevance.
            parsed_query (dict): The parsed query information.
            query_text (str): The original query text.
            max_to_return (int, optional): Maximum number of results to return.
            rerank_threshold (float, optional): Similarity threshold for reranking.

        Returns:
            list: Reranked search results.
        """
        if not isinstance(search_results, dict) or 'items' not in search_results:
            return []

        # Extract the items from the search results
        items_to_rerank = search_results['items']

        # Loop through each item and add the content field
        for item in items_to_rerank:
            item['content'] = ("Model description: " + item.get('merged_description', '') +
                               ", created year: " + item.get('created_year', '') +
                               ", created month: " + item.get('created_month', '') +
                               ", last modified year: " + item.get('last_modified_year', '') +
                               ", last modified month: " + item.get('last_modified_month', ''))

        if reranker and items_to_rerank:
            print(f"Sending {len(items_to_rerank)} items to reranker")
            return reranker.rerank(
                query=parsed_query.get("processed_query", query_text),
                results=items_to_rerank,
                top_k=max_to_return,
                threshold=rerank_threshold
            )
        else:
            return items_to_rerank

    def _remove_field(self, dict_list, field_to_remove):
        """Remove a field from all dictionaries in a list."""
        return [{k: v for k, v in item.items() if k != field_to_remove} for item in dict_list]

    def _generate_query_response(self, query_text, reranked_results, llm_interface: LLMInterface):
        """
        Generate a response by using an LLM to construct an answer based on
        the user's query and the search results.

        Args:
            query_text (str): The original query text.
            reranked_results (list): The reranked search results.
            llm_interface (object): Component for generating LLM responses.
        """
        # 1. Display reranked results for user visibility
        print("\nRetrieving and displaying reranked search results:\n")
        self.model_display.display_reranked_results_pretty(reranked_results)

        # 2. Build structured text of search results
        results_text = self._build_results_text(reranked_results)

        # 3. Get meta-prompt using the prompt builder
        constructed_prompt = self._build_meta_prompt(query_text, reranked_results, llm_interface)

        # 4. Output the complete prompt stub
        print("\n--- Constructed Prompt for Answer LLM ---")
        print(constructed_prompt)

        system_prompt =  constructed_prompt + f"You are provided user query and its searched results with {len(reranked_results)} records.\nIMPORTANT: Before doing anything else, wrap every step of your internal reasoning in `<thinking></thinking>` tags.\n"

        # 5. Create user prompt
        user_prompt = f"\nUser query: {query_text}\n"
        user_prompt += f"Search results:\n{results_text}"

        # 6. Query the answer LLM with the constructed prompt
        final_response = llm_interface.generate_structured_response(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.5,
            max_tokens=4000
        )
        print("\nLLM Final Response:")
        self.llm_processor.print_llm_content(final_response)

    def _build_results_text(self, reranked_results):
        """Build structured text of search results."""
        results_text = ""
        for idx, model in enumerate(reranked_results, 1):
            # Ensure model dict and safely parse nested metadata
            if not isinstance(model, dict):
                model = {"model_id": str(model), "metadata": {}}

            # Extract description
            description = model.get('merged_description') or model.get('description')
            md = model.get('metadata') or {}
            if not description:
                description = md.get('description') or md.get('merged_description') or 'N/A'

            # Safely parse nested JSON fields in metadata
            for key in ["file", "framework", "architecture", "dataset", "training_config"]:
                val = md.get(key)
                if isinstance(val, str):
                    try:
                        md[key] = json.loads(val)
                    except json.JSONDecodeError:
                        md[key] = {}

            # Extract fields with fallbacks
            model_id = model.get('model_id') or model.get('id') or 'Unknown'
            file_md = md.get('file', {})
            fw = md.get('framework', {})
            arch = md.get('architecture', {})
            ds = md.get('dataset', {})
            training = md.get('training_config', {})

            # Compose block
            results_text += f"Model #{idx}:\n"
            results_text += f"- Model ID: {model_id}\n"
            results_text += f"- File Size: {file_md.get('size_bytes', 'Unknown')}\n"
            results_text += f"- Created On: {file_md.get('creation_date', 'Unknown')}\n"
            results_text += f"- Last Modified: {file_md.get('last_modified_date', 'Unknown')}\n"
            results_text += f"- Framework: {fw.get('name', 'Unknown')} {fw.get('version', '')}\n"
            results_text += f"- Architecture: {arch.get('type', 'Unknown')}\n"
            results_text += f"- Dataset: {ds.get('name', 'Unknown')}\n"
            if training:
                results_text += "- Training Configuration:\n"
                for field in ['batch_size', 'learning_rate', 'optimizer', 'epochs', 'hardware_used']:
                    results_text += f"  - {field.replace('_', ' ').title()}: {training.get(field, 'Unknown')}\n"
            results_text += f"- Description: {description}\n\n"

        return results_text

    def _build_meta_prompt(self, query_text, reranked_results, llm_interface: LLMInterface):
        """Build the meta-prompt for the LLM using the prompt builder."""
        result_count = len(reranked_results)
        result_schema = "Model ID, File Size, Created On, Last Modified, Framework, Architecture, Dataset, Training Configuration, Description"

        # Construct the prompt builder logic
        system_prompt = (
            "You are a senior machine-learning architect. "
            "Your task is to craft a concise meta-prompt to generate clear, comprehensive reports for ML engineers to learn and understand the system. "
            "Define all technical terms and leave no gaps in explanation. "
            "  * Enclose your internal reasoning in `<thinking></thinking>` tags before constructing the meta-prompt."
            "Now, construct a single, high-level meta-prompt that will:\n"
            "  1. Restate the user's original request so the LLM knows what to address.\n"
            "  2. Instruct the LLM to use **only** the provided `search results` and actual runtime data—no hallucinations or invented details.\n"
            "  3. Instruct the LLM that it should explain how the searched results can answer user query, derived solely from the provided data—do not fabricate any information.\n"
            f"  4. Inform the LLM that there are {result_count} searched results available, sorted by similarity in decreasing order.\n"
            "  5. Provide this background: the LLM assumes the role of a senior machine-learning architect writing for ML engineers who will read the report to understand, reproduce, and extend the model; language must be clear, define all technical terms, and leave no gaps.\n"
            "  6. Require the LLM to generate a meaningful, substantive response rather than merely restating the question or remaining silent; if no relevant data is found in the provided results, it must explicitly state 'No data found.'\n"
            "The meta-prompt should not mention any other LLM identifiers; it should be concise and focused on the user's intent. Do **not** answer the query yourself or include any data; return only the meta-prompt text.\n\n"
            "EXAMPLE:\n"
            "  User query: \"Describe the model with ID XYZ.\"\n"
            "  Result schema: { 'model_id': 'string', 'framework': 'string', 'created_date': 'string' }\n"
            "  => Meta-prompt:\n"
            "     <thinking>\n"
            "     I need to restate the user's request, constrain to fields, ensure no hallucinations, note result count, define terms, require substantive response, and instruct on reasoning tags.\n"
            "     </thinking>\n"
            "     \"You are a senior machine-learning architect writing for ML engineers. Please describe the model with ID XYZ using only the provided fields `model_id`, `framework`, and `created_date`, defining all technical terms and leaving no gaps. Please analysis or insights derived solely from those fields. There are 3 results available. Provide a meaningful, substantive response, and if no relevant data is found, explicitly state 'No data found.'\"\n\n"
        )

        prompt_builder = (
            f"User query: {query_text}\n"
            f"Result schema: {result_schema}\n"
        )

        builder_response = llm_interface.generate_structured_response(
            system_prompt=system_prompt,
            user_prompt=prompt_builder,
            temperature=0.5,
            max_tokens=4000
        )

        # Safely extract constructed prompt text
        if isinstance(builder_response, dict):
            constructed_prompt = (
                    builder_response.get('content')
                    or builder_response.get('text')
                    or (builder_response.get('message') or {}).get('content')
                    or str(builder_response))
        else:
            constructed_prompt = str(builder_response)

        print(f"Constructed prompt with thinking: {constructed_prompt}")

        constructed_prompt = remove_thinking_sections(constructed_prompt.strip()).strip()

        if constructed_prompt[0] == '"' and constructed_prompt[-1] == '"':
            constructed_prompt = constructed_prompt[1:-1]

        return constructed_prompt