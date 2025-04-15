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
                elif cmd.lower().startswith("compare-models"):
                    self._handle_compare_models_command(cmd)
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
            "compare-models": "Compare two or more models",
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

    def _handle_list_images_command(self):
        """
        List all images accessible to the current user.

        Retrieves and displays images that the current user has permission
        to access in a formatted table.
        """
        access_control = self.components["vector_db_manager"]["access_control"]

        # Get images the user has access to
        available_images = access_control.get_accessible_images(self.user_id)

        print("\nAvailable images:")
        if not available_images:
            print("  No images available")
        else:
            self._display_images_pretty(available_images)

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
        template_manager = self.components["response_generator"]["template_manager"]

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

        # Dispatch the query
        search_results = asyncio.run(search_dispatcher.dispatch(
            query=parsed_query.get("processed_query", query_text),
            intent=parsed_query["intent"],
            parameters=parsed_query["parameters"],
            user_id=self.user_id
        ))

        # Process and rerank search results
        reranked_results = self._process_search_results(search_results, reranker, parsed_query, query_text)
        print(f"Reranked results: {reranked_results}")

        # Generate response using appropriate template
        self._generate_query_response(query_text, reranked_results, parsed_query, template_manager, llm_interface)

    def _process_search_results(self, search_results, reranker, parsed_query, query_text, max_to_return=10,
                                rerank_threshold=0.3):
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
        print(f"Search results: {search_results}")

        if not isinstance(search_results, dict) or 'items' not in search_results:
            return []

        # Extract the items from the search results
        items_to_rerank = search_results['items']

        # Add a content field if it doesn't exist (reranker might require this)
        for item in items_to_rerank:
            if 'content' not in item:
                # Use some meaningful field as content, like model_id or metadata description
                item['content'] = item.get('model_id', '') + ': ' + item.get('metadata', {}).get(
                    'description', 'No description')

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

    def _generate_query_response(self, query_text, reranked_results, parsed_query, template_manager, llm_interface):
        """
        Generate a response to the query using the appropriate template.

        Selects a template based on query type, renders it with context,
        and generates a response using an LLM.

        Args:
            query_text (str): The original query text.
            reranked_results (list): The reranked search results.
            parsed_query (dict): The parsed query information.
            template_manager (object): Component for rendering templates.
            llm_interface (object): Component for generating LLM responses.
        """
        # Select template based on query type
        if parsed_query["type"] == "comparison":
            template_id = "model_comparison"
        elif parsed_query["type"] == "retrieval":
            template_id = "information_retrieval"
        else:
            template_id = "general_query"

        print("Generating response...")

        # Prepare context for template
        context = self._prepare_template_context(query_text, reranked_results, parsed_query)
        print(f"context: {context}")

        try:
            # Try to render the template with the context
            rendered_prompt = template_manager.render_template(template_id, context)
            print(f"prompt: {rendered_prompt}")

            # Generate response using LLM interface
            llm_response = llm_interface.generate_response(
                prompt=rendered_prompt,
                temperature=0.5,
                max_tokens=31000
            )

            print("Printing LLM Response...")
            self._print_llm_content(llm_response)

        except Exception as e:
            print(f"Error generating response: {str(e)}")
            self._handle_fallback_response(query_text, reranked_results, llm_interface)

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
                max_tokens=31000
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
                    import json
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

    def _handle_compare_models_command(self, cmd):
        """
        Handle the compare-models command.

        Parses model IDs to compare and displays a placeholder message.

        Args:
            cmd (str): The command string, which may include model IDs.
        """
        if cmd.lower() == "compare-models":
            models_to_compare = input("Enter model IDs to compare (comma separated): ").split(",")
        else:
            models_to_compare = cmd[14:].strip().split(",")

        models_to_compare = [model.strip() for model in models_to_compare]

        if len(models_to_compare) < 2:
            print("Please specify at least two models to compare.")
            return

        print(f"Comparing models: {', '.join(models_to_compare)}...")
        print("Model comparison functionality will be implemented in a future update.")

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

    def _display_images_pretty(self, available_images):
        """
        Display images in a nicely formatted table.

        Creates and prints a table showing image information with proper
        column alignment and truncation for long values.

        Args:
            available_images (list): List of image dictionaries to display.
        """
        table = PrettyTable()
        table.field_names = ["Image ID", "Prompt", "File Path"]

        # Align all columns to the left
        table.align["Image ID"] = "l"
        table.align["Prompt"] = "l"
        table.align["File Path"] = "l"

        for image in available_images:
            image_id = image['id']
            prompt = image.get('prompt', 'No prompt')
            filepath = image.get('filepath', 'No path')

            # Optional: truncate long values
            if len(prompt) > 40:
                prompt = prompt[:37] + "..."
            if len(filepath) > 50:
                filepath = filepath[:47] + "..."

            table.add_row([image_id, prompt, filepath])

        print(table)

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