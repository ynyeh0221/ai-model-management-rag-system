import asyncio
import json
import logging
import os
from datetime import datetime

import nbformat
from nbformat.v4 import new_notebook, new_code_cell
from prettytable import PrettyTable


class UIRunner:
    """
    A command-line interface runner for the AI Model Management RAG system.

    This class handles user interactions through a command-line interface,
    providing functionalities to search, list, compare models, and generate
    notebooks for model analysis.
    """

    def __init__(self):
        self.components = None
        self.user_id = "anonymous"

    def start_ui(self, components, host="localhost", port=8000):
        """Start a command-line interface for the RAG system.

        Args:
            components: Dictionary containing initialized system components
            host: Not used in CLI mode, kept for API compatibility
            port: Not used in CLI mode, kept for API compatibility
        """
        self.components = components
        print(f"Starting command-line interface (Web UI will be developed later)...")

        # Display welcome message
        self._show_welcome_message()

        # Simple user authentication
        self.user_id = input("Enter your user ID (default: anonymous): ") or "anonymous"

        # Main command loop
        self._run_command_loop()

    def _show_welcome_message(self):
        """Display welcome message and system information."""
        print("\nAI Model Management RAG System - Command Line Interface")
        print("=" * 50)
        print("Type 'help' to see available commands or 'exit' to quit.")
        print("=" * 50)

    def _run_command_loop(self):
        """
        Main command processing loop.

        Continuously reads user input, processes commands, and handles exceptions
        until the user exits the application.
        """
        while True:
            try:
                print("\n")
                cmd = input("> ").strip()

                if cmd.lower() == "exit":
                    print("Exiting. Goodbye!")
                    break
                elif cmd.lower() == "help":
                    self._handle_help_command()
                elif cmd.lower() == "list-models":
                    self._handle_list_models_command()
                elif cmd.lower() == "list-images":
                    self._handle_list_images_command()
                elif cmd.lower().startswith("query"):
                    self._handle_query_command(cmd)
                elif cmd.lower().startswith("generate-notebook"):
                    self._handle_generate_notebook_command(cmd)
                else:
                    print(f"Unknown command: {cmd}")
                    print("Type 'help' to see available commands.")

            except KeyboardInterrupt:
                print("\nOperation cancelled.")
            except Exception as e:
                print(f"Error: {str(e)}")

        print("UI session ended.")

    def _handle_help_command(self):
        """
        Display available commands and their descriptions.

        Prints a list of all available commands with brief descriptions
        of their functionality.
        """
        commands = {
            "query": "Search for model scripts or images",
            "list-models": "List available models",
            "list-images": "List available images",
            "generate-notebook": "Generate a Colab notebook for a model",
            "help": "Show available commands",
            "exit": "Exit the program"
        }

        print("\nAvailable commands:")
        for cmd_name, cmd_desc in commands.items():
            print(f"  {cmd_name:<15} - {cmd_desc}")

    def _handle_list_models_command(self):
        """
        List all models accessible to the current user.

        Retrieves and displays models that the current user has permission
        to access in a formatted table.
        """
        access_control = self.components["vector_db_manager"]["access_control"]

        # Get models the user has access to
        available_models = access_control.get_accessible_models(self.user_id)
        print("\nAvailable models:")
        self._display_models_pretty(available_models)

    def _handle_query_command(self, cmd):
        """
        Process and execute a query command.

        Parses the query, dispatches it to the search system, reranks results,
        and generates a response using an LLM.

        Args:
            cmd (str): The command string, which may include the query text.
        """
        # Extract required components
        query_parser = self.components["query_engine"]["query_parser"]
        search_dispatcher = self.components["query_engine"]["search_dispatcher"]
        query_analytics = self.components["query_engine"]["query_analytics"]
        reranker = self.components["query_engine"]["reranker"]
        llm_interface = self.components["response_generator"]["llm_interface"]

        # Get query text
        if cmd.lower() == "query":
            query_text = input("Enter your query: ")
        else:
            query_text = cmd[6:].strip()

        # Parse the query
        parsed_query = query_parser.parse_query(query_text)
        print(f"Parsed query: {parsed_query}")

        # Log the query for analytics
        query_analytics.log_query(self.user_id, query_text, parsed_query)

        print("Searching...")

        # Special handling for image searches if needed
        if parsed_query["intent"] == "image_search":
            # For image uploads (not implemented in CLI mode)
            # In a real UI, this would handle file uploads for image-to-image search
            print("Note: Image uploads for image-to-image similarity search are not supported in CLI mode.")

            # If UI mode had an image upload, you would add:
            # parsed_query["parameters"]["query_image"] = uploaded_image_data

        # Dispatch the query
        search_results = asyncio.run(search_dispatcher.dispatch(
            query=parsed_query.get("processed_query", query_text),
            intent=parsed_query["intent"],
            parameters=parsed_query["parameters"],
            user_id=self.user_id
        ))

        # Different handling based on query type
        if parsed_query["intent"] == "image_search":
            self._display_image_search_results(search_results)
        else:
            # Process and rerank search results
            # Use max_to_return = 3 as a trade off between details in search result and size of LLM input
            reranked_results = self._process_search_results(search_results, reranker, parsed_query, query_text,
                                                            max_to_return=3)

            def remove_field(dict_list, field_to_remove):
                return [{k: v for k, v in item.items() if k != field_to_remove} for item in dict_list]

            reranked_results = remove_field(reranked_results, "content")

            # Generate response using appropriate template
            self._generate_query_response(query_text, reranked_results, llm_interface)

    def _process_search_results(self, search_results, reranker, parsed_query, query_text, max_to_return=10,
                                rerank_threshold=0.05):
        """
        Process and rerank search results.

        Prepares search results for reranking and applies reranking if available.

        Args:
            search_results (dict): The raw search results from the dispatcher.
            reranker (object): Reranker component for ordering results by relevance.
            parsed_query (dict): The parsed query information.
            query_text (str): The original query text.
            max_to_return (int, optional): Maximum number of results to return. Defaults to 10.
            rerank_threshold (float, optional): Similarity threshold for reranking. Defaults to 0.3.

        Returns:
            list: Reranked search results.
        """

        if not isinstance(search_results, dict) or 'items' not in search_results:
            return []

        # Extract the items from the search results
        items_to_rerank = search_results['items']
        # Loop through each item and add the content field
        for item in items_to_rerank:
            item['content'] = item.get('model_id', '') + ", " + item.get('merged_description', '')

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

    def _display_reranked_results_pretty(self, reranked_results):
        """
        Display reranked search results in a nicely formatted table with detailed model metadata.
        Optimized for 15-inch laptop screens.

        Args:
            reranked_results (list): List of reranked search result dictionaries to display.
        """
        import json
        from prettytable import PrettyTable, ALL

        table = PrettyTable()
        table.field_names = ["Rank", "Model ID", "Similarity Score", "Similarity Distance", "Size", "Created", "Modified",
                             "Path", "Description", "Framework", "Arch", "Dataset",
                             "Batch", "LR", "Optimizer", "Epochs", "HW"]

        # Align columns
        table.align = "l"  # Default left alignment for all
        table.align["Rank"] = "c"  # center
        table.align["Similarity Score"] = "r"  # right
        table.align["Similarity Distance"] = "r"  # right
        table.align["Size"] = "r"  # right align file size

        # Set max width for columns - optimized for 15-inch laptop
        max_width = {
            "Rank": 4,
            "Model ID": 15,
            "Similarity Score": 6,
            "Similarity Distance": 6,
            "Size": 7,
            "Created": 10,
            "Modified": 10,
            "Path": 15,
            "Description": 20,
            "Framework": 8,
            "Arch": 10,
            "Dataset": 10,
            "Batch": 5,
            "LR": 4,
            "Optimizer": 7,
            "Epochs": 5,
            "HW": 6
        }

        for column in max_width:
            table.max_width[column] = max_width[column]

        # Add horizontal lines and reduce padding for better display
        table.hrules = ALL
        table.padding_width = 1

        # Helper function to parse JSON string fields
        def parse_nested_json(metadata, fields):
            parsed_metadata = metadata.copy()
            for field in fields:
                raw_value = parsed_metadata.get(field)
                if isinstance(raw_value, str):
                    try:
                        parsed = json.loads(raw_value)
                        parsed_metadata[field] = parsed if isinstance(parsed, dict) else {}
                    except json.JSONDecodeError:
                        parsed_metadata[field] = {}
            return parsed_metadata

        for i, result in enumerate(reranked_results):
            rank = i + 1

            # Get basic fields
            model_id = result.get('model_id', result.get('id', f'Item {rank}'))

            # Get score from various possible fields
            score = result.get('score', result.get('similarity',
                                                   result.get('rank_score', result.get('rerank_score', 'N/A'))))

            # Format score to 3 decimal places if it's a number
            if isinstance(score, (int, float)):
                score = f"{score:.3f}"

            # Get distance
            distance = result.get('distance', 'N/A')

            # Format distance to 3 decimal places if it's a number
            if isinstance(distance, (int, float)):
                distance = f"{distance:.3f}"

            # Get raw metadata
            metadata = result.get('metadata', {}) if isinstance(result.get('metadata'), dict) else {}

            # Parse JSON string fields in metadata
            parsed_metadata = parse_nested_json(
                metadata,
                ['file', 'framework', 'architecture', 'dataset', 'training_config', 'git']
            )

            # Extract description
            description = result.get('merged_description', 'Unknown')
            if description == "N/A":
                description = "Unknown"

            # Extract file metadata
            file_metadata = parsed_metadata.get('file', {})
            size_bytes = file_metadata.get('size_bytes', 'Unknown')

            # Convert file size to compact format
            if isinstance(size_bytes, (int, float)):
                size_mb = size_bytes / 1048576  # 1024 * 1024
                if size_mb >= 1:
                    file_size = f"{size_mb:.1f}MB"
                else:
                    # For small files, show in KB
                    size_kb = size_bytes / 1024
                    if size_kb >= 1:
                        file_size = f"{size_kb:.1f}KB"
                    else:
                        # For very small files, show in bytes
                        file_size = f"{size_bytes}B"
            else:
                file_size = size_bytes  # Keep as "Unknown" or whatever non-numeric value

            # Truncate dates to save space
            creation_date = file_metadata.get('creation_date', 'Unknown')
            if isinstance(creation_date, str) and len(creation_date) > 10:
                creation_date = creation_date[:10]  # Just YYYY-MM-DD

            last_modified = file_metadata.get('last_modified_date', 'Unknown')
            if isinstance(last_modified, str) and len(last_modified) > 10:
                last_modified = last_modified[:10]  # Just YYYY-MM-DD

            # Extract absolute path
            absolute_path = file_metadata.get('absolute_path', 'Unknown')

            # Extract framework
            framework_metadata = parsed_metadata.get('framework', {})
            framework_name = framework_metadata.get('name', 'Unknown')
            framework_version = framework_metadata.get('version', '')
            framework = framework_name
            if framework_version and framework_version.lower() not in ['unknown', 'unspecified']:
                # Just add major version number
                if '.' in framework_version:
                    framework_version = framework_version.split('.')[0]
                framework += f" {framework_version}"

            # Extract architecture and dataset
            architecture_metadata = parsed_metadata.get('architecture', {})
            architecture = architecture_metadata.get('type', 'Unknown')

            dataset_metadata = parsed_metadata.get('dataset', {})
            dataset = dataset_metadata.get('name', 'Unknown')

            # Extract training config with compact formatting
            training_config = parsed_metadata.get('training_config', {})
            batch_size = training_config.get('batch_size', 'N/A')

            learning_rate = training_config.get('learning_rate', 'N/A')
            # Format learning rate in scientific notation if it's a small float
            if isinstance(learning_rate, float) and learning_rate < 0.01:
                learning_rate = f"{learning_rate:.0e}"

            optimizer = training_config.get('optimizer', 'N/A')
            epochs = training_config.get('epochs', 'N/A')
            hardware = training_config.get('hardware_used', 'N/A')

            # Add row to table
            table.add_row([
                rank, model_id, score, distance, file_size, creation_date, last_modified,
                absolute_path, description, framework, architecture, dataset,
                batch_size, learning_rate, optimizer, epochs, hardware
            ])

        print(table)

    def _generate_query_response(self, query_text, reranked_results, llm_interface):
        """
        Generate a response prompt by using an LLM to construct an answer prompt based on
        the user's query and the search results, then querying another LLM.
        """
        # 1. Display reranked results for user visibility
        print("\nRetrieving and displaying reranked search results:\n")
        self._display_reranked_results_pretty(reranked_results)

        # 2. Build structured text of search results (to be attached to the constructed prompt)
        results_text = ""
        for idx, model in enumerate(reranked_results, 1):
            # Ensure model dict and safely parse nested metadata
            if not isinstance(model, dict):
                model = {"model_id": str(model), "metadata": {}}

            # Extract description (prefer top-level merged_description, then metadata fields)
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
            file_md = md.get('file') or {}
            fw = md.get('framework') or {}
            arch = md.get('architecture') or {}
            ds = md.get('dataset') or {}
            training = md.get('training_config') or {}

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

        result_schema = "Model ID, File Size, Created On, Last Modified, Framework, Architecture, Dataset, Training Configuration, Description"

        # 3. Construct the prompt builder logic
        prompt_builder = (
            "You are a prompt engineer. Think from the user’s perspective and craft a single, high‑level meta‑prompt for a second LLM (LLM2) that will:\n"
            "  1. Restate the user’s original request so LLM2 knows what to address.\n"
            "  2. Instruct LLM2 to use **only** the provided `search results` and actual runtime data—no hallucinations or invented details.\n"
            "  3. When token limits permit, include your own analysis or insights derived solely from the provided data—do not fabricate any information.\n"
            "The meta‑prompt should be concise and focused on the user’s intent. Do **not** answer the query yourself or include any real data; return only the meta‑prompt text.\n\n"
            "EXAMPLE:\n"
            "  User query: \"Describe the model with ID XYZ.\"\n"
            "  Result schema: { 'model_id': 'string', 'framework': 'string', 'created_date': 'string' }\n"
            "  => LLM2 meta‑prompt:\n"
            "     \"Describe the model with ID XYZ using only the provided fields `model_id`, `framework`, and `created_date`. "
            "Do not add any details beyond those fields. When token limits permit, include your analysis or insights based solely on that data.\"\n\n"
            f"User query: {query_text}\n"
            f"Result schema: {result_schema}\n"
        )

        builder_response = llm_interface.generate_response(
            prompt=prompt_builder,
            temperature=0.5,
            max_tokens=2000
        )

        # 4. Safely extract constructed prompt text
        if isinstance(builder_response, dict):
            constructed_prompt = (
                    builder_response.get('content')
                    or builder_response.get('text')
                    or (builder_response.get('message') or {}).get('content')
                    or str(builder_response)
            )
        else:
            constructed_prompt = str(builder_response)
        constructed_prompt = constructed_prompt.strip()

        # 5. Output the complete prompt stub
        print("\n--- Constructed Prompt for Answer LLM ---")
        print(constructed_prompt)

        # 6. Append user query and search results
        constructed_prompt += f"\nUser query: {query_text}\n"
        constructed_prompt += f"Search results:\n{results_text}"

        # 7. Query the answer LLM with the constructed prompt
        final_response = llm_interface.generate_response(
            prompt=constructed_prompt,
            temperature=0.5,
            max_tokens=4000
        )
        print("\nLLM Final Response:")
        self._print_llm_content(final_response)

    def _prepare_template_context(self, query_text, results, parsed_query):
        """
        Prepare the context for the template with proper model information.

        Processes and cleans model metadata for use in template rendering.

        Args:
            query_text (str): The original query text.
            results (list): The search results to include in the context.
            parsed_query (dict): The parsed query information.

        Returns:
            dict: Context dictionary for template rendering.
        """

        def parse_nested_json(metadata, fields):
            for field in fields:
                raw_value = metadata.get(field)

                if isinstance(raw_value, str):
                    try:
                        parsed = json.loads(raw_value)
                        metadata[field] = parsed if isinstance(parsed, dict) else {}
                    except json.JSONDecodeError:
                        metadata[field] = {}
            return metadata

        def normalize_values(metadata):
            for key, value in metadata.items():
                if isinstance(value, str) and value.strip().lower() in ["n/a", "null", "none"]:
                    metadata[key] = None
            return metadata

        def preprocess_model(model):
            metadata = model.get("metadata", {})

            # Parse specific fields
            metadata = parse_nested_json(metadata,
                                         ["architecture", "dataset", "framework", "training_config",
                                          "file", "git"])

            # Normalize values like "N/A"
            metadata = normalize_values(metadata)
            model["metadata"] = metadata
            return model

        # Apply preprocessing to each result
        cleaned_results = [preprocess_model(model) for model in results]

        # Build context
        context = {
            "intent": parsed_query.get("intent", "unknown"),
            "query": query_text,
            "results": cleaned_results,
            "parsed_query": parsed_query,
            "timeframe": parsed_query.get("parameters", {}).get("filters", {}).get("created_month", None),
        }

        return context

    def _handle_fallback_response(self, query_text, reranked_results, llm_interface):
        """
        Handle fallback when template rendering fails.

        Creates a simple prompt and attempts to generate a response directly.

        Args:
            query_text (str): The original query text.
            reranked_results (list): The reranked search results.
            llm_interface (object): Component for generating LLM responses.
        """
        print("Falling back to direct LLM call...")

        # Create a simple prompt without using the template system
        fallback_prompt = f"Query: {query_text}\n\nResults:\n"
        for i, r in enumerate(reranked_results):
            fallback_prompt += f"{i + 1}. {r['id']}: {r['content']}\n"
        fallback_prompt += "\nPlease provide a comprehensive response to the query based on these results."

        try:
            # Try again with the fallback prompt
            fallback_response = llm_interface.generate_response(
                prompt=fallback_prompt,
                temperature=0.5,
                max_tokens=4000
            )
            print("\nFallback Response:")
            print(fallback_response)
        except Exception as e2:
            print(f"Fallback also failed: {str(e2)}")
            print("Could not generate LLM response. Check your connection and service status.")

    def _print_llm_content(self, response):
        """
        Extract and print content from an LLM response in various formats.

        Handles different response formats (dict, string, list) and attempts
        to extract and print the most relevant content.

        Args:
            response: The response from the LLM, which could be a string, dict, or list.
        """
        try:
            # Handle dict case
            if isinstance(response, dict):
                if "content" in response:
                    print(response["content"])
                elif "text" in response:
                    print(response["text"])
                elif "response" in response:
                    print(response["response"])
                elif "answer" in response:
                    print(response["answer"])
                elif "message" in response:
                    # Check for OpenAI format
                    if "content" in response["message"]:
                        print(response["message"]["content"])
                    else:
                        print(response["message"])
                else:
                    print("No recognizable content field found in the dictionary response")
                    print(f"Available fields: {list(response.keys())}")

                    # Try to print the full dictionary if it's not too large
                    if len(str(response)) < 1000:
                        print("Response content:")
                        print(response)

            # Handle str case
            elif isinstance(response, str):
                # Try to parse as JSON first
                try:
                    parsed = json.loads(response)

                    if isinstance(parsed, dict):
                        # Recursively call with the parsed dict
                        self._print_llm_content(parsed)
                    else:
                        # Just print the string as is
                        print(response)
                except json.JSONDecodeError:
                    # Not JSON, print as is
                    print(response)

            # Handle list case
            elif isinstance(response, list):
                if response and len(response) > 0:
                    # Try the first element
                    first_elem = response[0]
                    if isinstance(first_elem, dict):
                        # Recursively call with the first dict
                        self._print_llm_content(first_elem)
                    else:
                        print(first_elem)
                else:
                    print("Empty list response")

            else:
                print(f"Unsupported response type: {type(response)}")
                # Try to print it anyway
                print(str(response))

        except Exception as e:
            print(f"Error extracting content: {e}")
            # Try to print the raw response
            try:
                print("Raw response:")
                print(str(response)[:500])  # Print first 500 chars at most
            except:
                print("Could not print raw response")

    def _handle_generate_notebook_command(self, cmd):
        """
        Handle the generate-notebook command.

        Parses command arguments, extracts model ID and output path,
        and generates a notebook for the specified model.

        Args:
            cmd (str): The command string, which may include model ID and options.
        """
        notebook_type = "evaluation"

        if cmd.lower() == "generate-notebook":
            model_id = input("Enter model ID: ")
            output_path = input(
                f"Enter output path [default: ./notebooks/{model_id}.ipynb]: ") or f"./notebooks/{model_id}.ipynb"
        else:
            parts = cmd.split(maxsplit=3)
            model_id = parts[1] if len(parts) > 1 else input("Enter model ID: ")

            if len(parts) > 2:
                if parts[2].startswith("--type=") or parts[2].startswith("-t="):
                    notebook_type = parts[2].split("=", 1)[1]
                else:
                    notebook_type = parts[2]

            if len(parts) > 3:
                if parts[3].startswith("--output=") or parts[3].startswith("-o="):
                    output_path = parts[3].split("=", 1)[1]
                else:
                    output_path = parts[3]
            else:
                output_path = f"./notebooks/{model_id}_{notebook_type}.ipynb"

        result = self.generate_notebook(self.components, model_id, output_path)

        if result:
            print(f"Notebook generated successfully: {result}")
        else:
            print("Failed to generate notebook")

    def _display_models_pretty(self, available_models):
        """
        Display models in a nicely formatted table.

        Creates and prints a table showing model information with proper
        column alignment and truncation for long values.

        Args:
            available_models (list): List of model dictionaries to display.
        """
        table = PrettyTable()
        table.field_names = ["Model ID", "Created", "Last Modified", "Absolute Path"]

        # Align all columns to the left
        table.align["Model ID"] = "l"
        table.align["Created"] = "l"
        table.align["Last Modified"] = "l"
        table.align["Absolute Path"] = "l"

        # Sort models by creation date in descending order
        sorted_models = sorted(
            available_models,
            key=lambda m: datetime.fromisoformat(m['creation_date']),
            reverse=True
        )

        for model in sorted_models:
            model_id = model['model_id']
            created = datetime.fromisoformat(model['creation_date']).strftime("%Y-%m-%dT%H:%M")
            modified = datetime.fromisoformat(model['last_modified_date']).strftime("%Y-%m-%dT%H:%M")
            absolute_path = model['absolute_path']

            # Truncate if needed
            if len(model_id) > 50:
                model_id = model_id[:47] + "..."
            if len(absolute_path) > 100:
                absolute_path = absolute_path[:97] + "..."

            table.add_row([model_id, created, modified, absolute_path])

        print(table)

    def _display_image_search_results(self, search_results):
        """
        Display image search results in a table format with ASCII thumbnails.

        Args:
            search_results (dict): Results from the image search.
        """
        if not search_results.get('success', False):
            print(f"Image search failed: {search_results.get('error', 'Unknown error')}")
            return

        items = search_results.get('items', [])
        if not items:
            print("No images found matching your search criteria.")
            return

        print(f"\nFound {len(items)} images:")

        # Try to import necessary libraries for image processing
        try:
            from PIL import Image
            import numpy as np
            has_pil = True
        except ImportError:
            print("PIL/Pillow not installed. Thumbnails will not be displayed.")
            has_pil = False

        # Define ASCII characters for grayscale representation (from dark to light)
        ascii_chars = '@%#*+=-:. '

        # Function to convert image to ASCII art
        def image_to_ascii(image_path, width=30, height=15):
            try:
                # Open the image and resize it
                img = Image.open(image_path)
                img = img.resize((width, height))
                img = img.convert('L')  # Convert to grayscale

                # Convert pixels to ASCII characters
                pixels = np.array(img)
                ascii_img = []
                for row in pixels:
                    ascii_row = ''
                    for pixel in row:
                        # Map pixel value to ASCII character
                        index = int(pixel * (len(ascii_chars) - 1) / 255)
                        ascii_row += ascii_chars[index]
                    ascii_img.append(ascii_row)

                return ascii_img
            except Exception as e:
                return [f"Error loading image: {e}"]

        # Custom table class to include ASCII thumbnails
        class ThumbnailTable:
            def __init__(self):
                self.rows = []
                self.headers = ["#", "ID", "Source Model", "Prompt", "Thumbnail", "Image Path"]

            def add_row(self, row_data, ascii_img=None):
                if ascii_img:
                    row_data_with_thumbnail = row_data.copy()
                    row_data_with_thumbnail.insert(-1, ascii_img)  # Insert thumbnail before image path
                    self.rows.append(row_data_with_thumbnail)
                else:
                    row_data.insert(-1, ["Thumbnail not available"])  # Insert placeholder
                    self.rows.append(row_data)

            def __str__(self):
                # Calculate column widths for text columns
                col_widths = [max([len(str(row[i])) for row in self.rows] + [len(self.headers[i])])
                              for i in range(len(self.headers)) if i != 4]  # Skip thumbnail column

                # Create header line
                header = " | ".join(
                    f"{self.headers[i]:<{width}}" if i < 4 else
                    (f"{self.headers[i]}" if i == 4 else f"{self.headers[i]:<{col_widths[i - 1]}}")
                    for i, width in enumerate(col_widths + [0])
                )
                separator = "-" * len(header)

                # Build the table string
                result = [header, separator]

                for row in self.rows:
                    # Format each non-thumbnail column
                    formatted_row = [
                        f"{row[0]:>{col_widths[0]}}" if i == 0 else  # Right-align first column (#)
                        f"{str(row[i]):<{col_widths[i]}}" if i < 4 else  # Left-align other text columns
                        f"{str(row[i]):<{col_widths[i - 1]}}" if i == 5 else  # Skip thumbnail index adjustment
                        ""  # Placeholder for thumbnail
                        for i in range(len(row)) if i != 4  # Skip thumbnail in this formatting
                    ]

                    # Add the row header
                    result.append(" | ".join(formatted_row[:4]) + " |")

                    # Add thumbnail lines
                    for line in row[4]:  # Thumbnail is at index 4
                        result.append(f"{' ' * (len(' | '.join(formatted_row[:4])) + 2)}| {line}")

                    # Add the file path
                    result.append(f"{' ' * (len(' | '.join(formatted_row[:4])) + 2)}| {formatted_row[4]}")

                    # Add separator
                    result.append(separator)

                return "\n".join(result)

        # Create and populate the table
        table = ThumbnailTable()

        for i, item in enumerate(items, 1):
            metadata = item.get('metadata', {})

            # Get values with fallbacks
            image_id = item.get('id', 'Unknown')
            model_id = metadata.get('source_model_id', metadata.get('model_id', 'Unknown'))
            prompt = metadata.get('prompt', 'No prompt')
            image_path = item.get('image_path', metadata.get('image_path', 'Not available'))

            # Get thumbnail path if available
            thumbnail_path = item.get('thumbnail_path', metadata.get('thumbnail_path', image_path))

            # Truncate long values
            if len(prompt) > 40:
                prompt = prompt[:37] + "..."
            if len(image_path) > 40:
                image_path = image_path[:37] + "..."

            # Generate ASCII thumbnail if PIL is available
            if has_pil and thumbnail_path and thumbnail_path != 'Not available':
                import os
                if os.path.exists(thumbnail_path):
                    ascii_img = image_to_ascii(thumbnail_path)
                else:
                    ascii_img = ["Image file not found"]
            else:
                ascii_img = ["Thumbnail not available"]

            table.add_row([i, image_id, model_id, prompt, image_path], ascii_img)

        # Print the table
        print(table)

        # Print performance metrics if available
        if 'performance' in search_results:
            perf = search_results['performance']
            print("\nPerformance:")
            for metric, value in perf.items():
                if isinstance(value, (int, float)):
                    print(f"  {metric}: {value:.2f} ms")

    def generate_notebook(self, components, model_id, output_path):
        """
        Generate a Colab notebook for model analysis using full script reconstruction.

        Retrieves model metadata and code chunks, reconstructs the full script,
        creates a notebook, and saves it to the specified path.

        Args:
            components (dict): Dictionary containing initialized system components.
            model_id (str): ID of the model to generate a notebook for.
            output_path (str): Path where the generated notebook will be saved.

        Returns:
            str or None: Path to the generated notebook if successful, None otherwise.
        """
        print(f"Generating notebook for model {model_id}...")

        # Extract components
        code_generator = components["colab_generator"]["code_generator"]
        reproducibility_manager = components["colab_generator"]["reproducibility_manager"]
        chroma_manager = components["vector_db_manager"]["chroma_manager"]

        # Set up logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger("notebook_generator")

        # First retrieve metadata for the model
        logger.info(f"Retrieving metadata for model {model_id}")
        metadata_results = asyncio.run(chroma_manager.get(
            collection_name="model_scripts_metadata",
            where={"model_id": {"$eq": model_id}},
            include=["metadatas"],
            limit=1
        ))

        if not metadata_results or "results" not in metadata_results or not metadata_results["results"]:
            logger.error(f"No metadata found for model {model_id}")
            return None

        # Now retrieve all code chunks for the given model_id
        logger.info(f"Retrieving all code chunks for model {model_id}")
        results = asyncio.run(chroma_manager.get(
            collection_name="model_scripts_chunks",
            where={"model_id": {"$eq": model_id}},
            include=["documents", "metadatas"],
            limit=200  # increase if needed
        ))

        if not results or "results" not in results or not results["results"]:
            logger.error(f"No code chunks found for model {model_id}")
            return None

        # Sort chunks by chunk_id
        chunks = sorted(
            results["results"],
            key=lambda x: x["metadata"].get("chunk_id", 0)
        )

        # Prepare structured chunks
        chunk_contents = self._prepare_chunk_contents(chunks)
        logger.info(f"Found {len(chunk_contents)} chunks. Reconstructing full script...")

        # Reconstruct full code
        full_script = code_generator.generate_full_script(chunk_contents, overlap=100, use_offset=True)

        # Create notebook with reconstructed code
        notebook = new_notebook(cells=[
            new_code_cell(full_script)
        ])

        # Add reproducibility metadata
        notebook = reproducibility_manager.add_reproducibility_info(notebook, model_id)

        # Save notebook
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            nbformat.write(notebook, f)

        logger.info(f"Notebook saved to {output_path}")
        return output_path

    def _prepare_chunk_contents(self, chunks):
        """
        Convert raw chunks to structured format for script reconstruction.

        Transforms document chunks from various formats into a consistent
        structure with text content and offset information.

        Args:
            chunks (list): List of raw document chunks.

        Returns:
            list: List of structured chunk dictionaries with text and offset.
        """
        chunk_contents = []
        for doc in chunks:
            content = doc.get("document", "")
            metadata = doc.get("metadata", {})

            # If document is a string, wrap it into a structured format
            if isinstance(content, str):
                chunk_contents.append({
                    "text": content,
                    "offset": metadata.get("offset", 0)  # default to 0 if offset not present
                })
            elif isinstance(content, dict):
                chunk_contents.append({
                    "text": content.get("content", ""),  # fallback if structure exists
                    "offset": metadata.get("offset", 0)
                })
            else:
                # Just in case it's malformed
                chunk_contents.append({
                    "text": str(content),
                    "offset": metadata.get("offset", 0)
                })

        return chunk_contents

    def _handle_list_images_command(self):
        """
        List all images accessible to the current user.

        Retrieves and displays images that the current user has permission
        to access in a formatted table with thumbnails.
        """
        try:
            access_control = self.components["vector_db_manager"]["access_control"]

            # Get images the user has access to
            # Modified to handle the case where get_accessible_images might fail
            try:
                available_images = access_control.get_accessible_images(self.user_id)
            except AttributeError:
                # Fallback: Get all images if access control is not properly implemented
                print("Warning: Access control not fully implemented. Showing all available images.")
                chroma_manager = self.components["vector_db_manager"]["chroma_manager"]
                # Use a safer method to get images
                available_images = asyncio.run(self._get_all_images(chroma_manager))

            print("\nAvailable images:")
            self._display_images_with_thumbnails(available_images, is_search_result=False)
        except Exception as e:
            print(f"Error listing images: {str(e)}")
            print("Please ensure the vector database and access control are properly configured.")

    async def _get_all_images(self, chroma_manager):
        """
        Fallback method to get all images when access control fails.

        Args:
            chroma_manager: The Chroma database manager

        Returns:
            List of image dictionaries
        """
        try:
            # Query for all images in the generated_images collection
            results = await chroma_manager.get(
                collection_name="generated_images",
                include=["metadatas"],
                limit=100  # Set a reasonable limit
            )

            if results and "results" in results:
                images = []
                for item in results["results"]:
                    metadata = item.get("metadata", {})
                    images.append({
                        "id": item.get("id", "Unknown"),
                        "prompt": metadata.get("prompt", "No prompt"),
                        "image_path": metadata.get("image_path", "Not available"),
                        "thumbnail_path": metadata.get("thumbnail_path", metadata.get("image_path", "Not available")),
                        "metadata": metadata
                    })
                return images
            return []
        except Exception as e:
            print(f"Error retrieving images: {str(e)}")
            return []

    def _display_images_with_thumbnails(self, images, is_search_result=False):
        """
        Display images in a formatted table with ASCII thumbnails.
        Works for both image lists and search results.

        Args:
            images (list): List of image dictionaries to display.
            is_search_result (bool): Whether the images are from search results.
        """
        if not images:
            print("  No images available")
            return

        # Try to import necessary libraries for image processing
        try:
            from PIL import Image
            import numpy as np
            has_pil = True
        except ImportError:
            print("PIL/Pillow not installed. Thumbnails will not be displayed.")
            has_pil = False

        # Define ASCII characters for grayscale representation (from dark to light)
        ascii_chars = '@%#*+=-:. '

        # Function to convert image to ASCII art
        def image_to_ascii(image_path, width=30, height=15):
            try:
                # Check if path exists
                import os
                if not os.path.exists(image_path):
                    return ["Image file not found"]

                # Open the image and resize it
                img = Image.open(image_path)
                img = img.resize((width, height))
                img = img.convert('L')  # Convert to grayscale

                # Convert pixels to ASCII characters
                pixels = np.array(img)
                ascii_img = []
                for row in pixels:
                    ascii_row = ''
                    for pixel in row:
                        # Map pixel value to ASCII character
                        index = int(pixel * (len(ascii_chars) - 1) / 255)
                        ascii_row += ascii_chars[index]
                    ascii_img.append(ascii_row)

                return ascii_img
            except Exception as e:
                return [f"Error loading image: {str(e)}"]

        # Custom table class to include ASCII thumbnails
        class ThumbnailTable:
            def __init__(self, is_search_result):
                self.rows = []
                # Different headers based on the context
                if is_search_result:
                    self.headers = ["#", "ID", "Source Model", "Prompt", "Thumbnail", "Image Path"]
                else:
                    self.headers = ["Image ID", "Prompt", "Thumbnail", "File Path"]

            def add_row(self, row_data, ascii_img=None):
                if ascii_img:
                    row_data_with_thumbnail = row_data.copy()
                    # Insert thumbnail before the last column (file path)
                    row_data_with_thumbnail.insert(len(row_data_with_thumbnail) - 1, ascii_img)
                    self.rows.append(row_data_with_thumbnail)
                else:
                    # Insert placeholder before the last column
                    row_data.insert(len(row_data) - 1, ["Thumbnail not available"])
                    self.rows.append(row_data)

            def __str__(self):
                if not self.rows:
                    return "No rows to display"

                # Calculate column widths for text columns (excluding thumbnail)
                thumbnail_idx = self.headers.index("Thumbnail")
                col_widths = [
                    max([len(str(row[i])) for row in self.rows] + [len(self.headers[i])])
                    for i in range(len(self.headers)) if i != thumbnail_idx
                ]

                # Create header line
                header_parts = []
                col_idx = 0
                for i, header in enumerate(self.headers):
                    if i != thumbnail_idx:
                        width = col_widths[col_idx]
                        # Right align first column if it's a number (#)
                        if i == 0 and header == "#":
                            header_parts.append(f"{header:>{width}}")
                        else:
                            header_parts.append(f"{header:<{width}}")
                        col_idx += 1
                    else:
                        header_parts.append(header)

                header = " | ".join(header_parts)
                separator = "-" * len(header)

                # Build the table string
                result = [header, separator]

                for row in self.rows:
                    # Format each non-thumbnail column
                    formatted_row = []
                    col_idx = 0
                    for i in range(len(row)):
                        if i != thumbnail_idx:
                            width = col_widths[col_idx]
                            # Right align first column if it's a number (#)
                            if i == 0 and self.headers[0] == "#":
                                formatted_row.append(f"{row[i]:>{width}}")
                            else:
                                formatted_row.append(f"{str(row[i]):<{width}}")
                            col_idx += 1

                    # Calculate where to split the row (before thumbnail)
                    first_part = " | ".join(formatted_row[:thumbnail_idx])
                    second_part = " | ".join(formatted_row[thumbnail_idx:])

                    # Add the row header
                    result.append(f"{first_part} |")

                    # Add thumbnail lines
                    for line in row[thumbnail_idx]:
                        result.append(f"{' ' * (len(first_part) + 2)}| {line}")

                    # Add the file path
                    result.append(f"{' ' * (len(first_part) + 2)}| {second_part}")

                    # Add separator
                    result.append(separator)

                return "\n".join(result)

        # Create and populate the table
        table = ThumbnailTable(is_search_result)

        # Ensure images is always treated as a list
        if isinstance(images, dict) and 'items' in images:
            image_list = images.get('items', [])
            performance_data = images.get('performance', {})
        else:
            image_list = images if isinstance(images, list) else []
            performance_data = None

        for i, image in enumerate(image_list, 1):
            if is_search_result:
                # Handle search result format
                metadata = image.get('metadata', {})
                image_id = image.get('id', 'Unknown')
                model_id = metadata.get('source_model_id', metadata.get('model_id', 'Unknown'))
                prompt = metadata.get('prompt', 'No prompt')
                image_path = image.get('image_path', metadata.get('image_path', 'Not available'))
                thumbnail_path = image.get('thumbnail_path', metadata.get('thumbnail_path', image_path))
                row_data = [i, image_id, model_id, prompt, image_path]
            else:
                # Handle list format
                image_id = image.get('id', 'Unknown')
                prompt = image.get('prompt', 'No prompt')
                image_path = image.get('filepath', image.get('image_path', 'No path'))
                thumbnail_path = image.get('thumbnail_path', image_path)
                row_data = [image_id, prompt, image_path]

            # Truncate long values
            for j in range(len(row_data)):
                if isinstance(row_data[j], str) and len(row_data[j]) > 40:
                    row_data[j] = row_data[j][:37] + "..."

            # Generate ASCII thumbnail if PIL is available
            if has_pil and thumbnail_path and thumbnail_path not in ['No path', 'Not available']:
                import os
                if os.path.exists(thumbnail_path):
                    ascii_img = image_to_ascii(thumbnail_path)
                else:
                    ascii_img = ["Image file not found"]
            else:
                ascii_img = ["Thumbnail not available"]

            table.add_row(row_data, ascii_img)

        # Print the table
        print(table)

        # Print performance metrics if available
        if performance_data:
            print("\nPerformance:")
            for metric, value in performance_data.items():
                if isinstance(value, (int, float)):
                    print(f"  {metric}: {value:.2f} ms")

    def _display_image_search_results(self, search_results):
        """
        Display image search results in a table format with ASCII thumbnails.

        Args:
            search_results (dict): Results from the image search.
        """
        if not search_results.get('success', False):
            print(f"Image search failed: {search_results.get('error', 'Unknown error')}")
            return

        items = search_results.get('items', [])
        if not items:
            print("No images found matching your search criteria.")
            return

        print(f"\nFound {len(items)} images:")
        self._display_images_with_thumbnails(items, is_search_result=True)