"""
CLI Interface Module - Encapsulating the Command Line Interface

This module provides a command line interface that uses the RAGSystem core module
to process user queries and commands. It refactors the original UIRunner class
into an interface that uses RAGSystem.
"""
import asyncio
import logging
from typing import Dict, Any

from src.core.rag_system import RAGSystem


class CLIInterface:
    """
    CLI Interface Class - Encapsulating command line interaction

    This class provides a command line interface that uses the RAGSystem core
    to process user queries and commands. It implements the following features:
    - Displaying welcome messages
    - Handling user authentication
    - Running command loops
    - Formatting and displaying results
    """

    def __init__(self):
        """Initialize CLI interface"""
        self.rag_system = RAGSystem()
        self.user_id = "anonymous"
        self.components = None

        # Display managers and processors, will be initialized in start_cli
        self.model_display_manager = None
        self.image_display_manager = None
        self.notebook_generator = None
        self.llm_response_processor = None

        # CommandHandler will be initialized after UI cli_response_utils
        self.command_handler = None

    def start_cli(self, components: Dict[str, Any], host: str = "localhost", port: int = 8000):
        """
        Start the command line interface

        Args:
            components: Dictionary containing initialized system cli_response_utils
            host: Hostname, kept for API compatibility, not used in CLI mode
            port: Port number, kept for API compatibility, not used in CLI mode
        """
        print(f"Starting command line interface (Web UI will be developed later)...")

        # Store cli_response_utils reference
        self.components = components

        # Display welcome message
        self._show_welcome_message()

        # Simple user authentication
        self.user_id = input("Enter your user ID (default: anonymous): ") or "anonymous"

        # Initialize RAG system
        success = self.rag_system.initialize(components, self.user_id)
        if not success:
            print("System initialization failed, cannot continue.")
            return

        # Register callbacks
        self._register_callbacks()

        # Initialize UI cli_response_utils
        self._initialize_components()

        # Initialize command handler after UI cli_response_utils
        self.command_handler = CommandHandler(self)

        # Run main command loop
        self._run_command_loop()

    def _initialize_components(self):
        """Initialize component classes needed by the UI"""
        # Import necessary cli_response_utils
        from cli.cli_response_utils.model_display_manager import ModelDisplayManager
        from cli.cli_response_utils.image_display_manager import ImageDisplayManager
        from core.notebook_generator import NotebookGenerator
        from cli.cli_response_utils.llm_response_processor import LLMResponseProcessor

        # Create display managers
        self.model_display_manager = ModelDisplayManager()
        self.image_display_manager = ImageDisplayManager()

        # Create generators and processors
        self.notebook_generator = NotebookGenerator()
        self.llm_response_processor = LLMResponseProcessor()

    def _register_callbacks(self):
        """Register RAG system callbacks"""
        self.rag_system.register_callback("on_log", self._handle_log)
        self.rag_system.register_callback("on_result", self._handle_result)
        self.rag_system.register_callback("on_error", self._handle_error)
        self.rag_system.register_callback("on_status", self._handle_status)

    def _show_welcome_message(self):
        """Display welcome message and system information"""
        print("\nAI Model Management RAG System - Command Line Interface")
        print("=" * 50)
        print("Type 'help' to see available commands or 'exit' to quit.")
        print("=" * 50)

    def _run_command_loop(self):
        """
        Main command processing loop

        Continuously reads user input, processes commands, and handles exceptions
        until the user exits the application.
        """
        while True:
            try:
                print("\n")
                cmd = input("> ").strip()

                # Delegate command handling to CommandHandler instead of RAG system
                continue_processing = self.command_handler.handle_command(cmd)
                if not continue_processing:
                    break

            except KeyboardInterrupt:
                print("\nOperation cancelled.")
            except Exception as e:
                print(f"Error: {str(e)}")
                logging.error(f"Exception in command loop: {str(e)}", exc_info=True)

        print("UI session ended.")

    # Callback handling methods

    def _handle_log(self, log_data):
        """Handle log callback"""
        # Can format log output as needed
        level = log_data.get("level", "info")
        message = log_data.get("message", "")

        # Can use different colors or formats based on log level
        if level == "error":
            print(f"ERROR: {message}")
        elif level == "warning":
            print(f"WARNING: {message}")
        elif level == "debug":
            # Might want to skip debug logs in non-debug mode
            if logging.getLogger().getEffectiveLevel() <= logging.DEBUG:
                print(f"DEBUG: {message}")
        else:
            # Non-critical logs can optionally be skipped
            pass

    def _handle_result(self, result):
        """Handle result callback"""
        if result.get("type") == "text_search":

            # Display full response
            print("\nLLM Final Response:")
            self.llm_response_processor.print_llm_content(result.get("final_response", ""))

            # Display text search results
            print("\nRetrieving and displaying reranked search results:\n")
            self.model_display_manager.display_reranked_results_pretty(result.get("search_results", []))

        elif result.get("type") == "image_search":
            # Display image search results
            print("\nImage search results:")
            self.image_display_manager.display_image_search_results(result.get("results", []))

        elif result.get("type") == "command":
            # Display command results
            if isinstance(result.get("result"), dict):
                # Format dictionary results
                if "available_commands" in result["result"]:
                    print("\nAvailable commands:")
                    for cmd in result["result"]["available_commands"]:
                        print(f"  {cmd}")
                else:
                    # Generic dictionary formatting
                    for key, value in result["result"].items():
                        print(f"{key}: {value}")
            else:
                # Simple string result
                print(result.get("result", ""))

        elif result.get("type") == "needs_clarification":
            # Handle query that needs clarification
            query = result.get("query", "")
            clarity_result = result.get("clarity_result", {})

            print("\nYour query could be clearer:")
            print(f"Reason: {clarity_result.get('reason', 'No reason provided')}")

            improved_query = clarity_result.get('improved_query', query)
            suggestions = clarity_result.get('suggestions', [])

            print("\nSuggestions:")
            print(f"0. Improved query: {improved_query}")

            for i, suggestion in enumerate(suggestions, 1):
                print(f"{i}. {suggestion}")

            print(f"{len(suggestions) + 1}. Use original query: {query}")
            print(f"{len(suggestions) + 2}. Enter a new query")

            while True:
                try:
                    choice = input("Select an option (number): ")
                    choice_num = int(choice)

                    if choice_num == 0:
                        new_query = improved_query
                        break
                    elif 1 <= choice_num <= len(suggestions):
                        new_query = suggestions[choice_num - 1]
                        break
                    elif choice_num == len(suggestions) + 1:
                        new_query = query
                        break
                    elif choice_num == len(suggestions) + 2:
                        new_query = input("Enter your new query: ")
                        break
                    else:
                        print("Invalid selection. Please try again.")
                except ValueError:
                    print("Please enter a number.")

            # Process the new query
            asyncio.run(self.rag_system.process_query(new_query))

    def _handle_error(self, error):
        """Handle error callback"""
        print(f"\nERROR: {str(error)}")

    def _handle_status(self, status):
        """Handle status callback"""
        # Status updates can be displayed as needed
        status_messages = {
            "ready": "System ready",
            "processing": "Processing query...",
            "searching": "Searching...",
            "processing_results": "Processing results...",
            "generating_response": "Generating response...",
            "completed": "Operation completed",
            "error": "Error occurred"
        }

        message = status_messages.get(status, status)
        print(f"Status: {message}")

class CommandHandler:
    """Handles command processing and delegation to appropriate cli_response_utils."""

    def __init__(self, cli_interface):
        """
        Initialize CommandHandler with a reference to the CLIInterface.

        Args:
            cli_interface: Instance of CLIInterface that this handler will work with
        """
        self.cli_interface = cli_interface
        self.rag_system = cli_interface.rag_system
        self.components = self.rag_system.components
        self.user_id = cli_interface.user_id

        # These will be accessed from the CLIInterface
        self.model_display = cli_interface.model_display_manager
        self.image_display = cli_interface.image_display_manager
        self.notebook_generator = cli_interface.notebook_generator

    def handle_command(self, cmd):
        """
        Process a user command and delegate to the appropriate handler.

        Args:
            cmd (str): The command string to process.

        Returns:
            bool: True if processing should continue, False if the user wants to exit.
        """
        cmd = cmd.strip()

        # Basic command handling - will be processed by RAGSystem
        if cmd.lower() == "exit":
            print("Exiting. Goodbye!")
            return False
        elif cmd.lower() == "help":
            self._handle_help_command()
        elif cmd.lower() == "list-models":
            self._handle_list_models_command()
        elif cmd.lower() == "list-image_processing":
            self._handle_list_images_command()
        elif cmd.lower().startswith("query"):
            self._handle_query_command(cmd)
        elif cmd.lower().startswith("generate-notebook"):
            self._handle_generate_notebook_command(cmd)
        else:
            # For unknown commands, let the RAGSystem try to process it
            result = self.rag_system.execute_command(cmd)
            if result.get("type") == "error":
                print(f"Unknown command: {cmd}")
                print("Type 'help' to see available commands.")

        return True

    def _handle_help_command(self):
        """Display available commands and their descriptions."""
        commands = {
            "query": "Search for model scripts or image_processing",
            "list-models": "List available models",
            "list-image_processing": "List available image_processing",
            "generate-notebook": "Generate a Colab notebook for a model",
            "help": "Show available commands",
            "exit": "Exit the program"
        }

        print("\nAvailable commands:")
        for cmd_name, cmd_desc in commands.items():
            print(f"  {cmd_name:<15} - {cmd_desc}")

    def _handle_list_models_command(self):
        """List all models accessible to the current user."""
        access_control = self.components["vector_db"]["access_control"]

        try:
            # Get models the user has access to
            available_models = access_control.get_accessible_models(self.user_id)
            print("\nAvailable models:")
            self.model_display.display_models_pretty(available_models)
        except Exception as e:
            print(f"Error listing models: {str(e)}")

    def _handle_list_images_command(self):
        """List all image_processing accessible to the current user."""
        try:
            access_control = self.components["vector_db"]["access_control"]

            # Get image_processing the user has access to
            try:
                available_images = access_control.get_accessible_images(self.user_id)
            except AttributeError:
                # Fallback: Get all image_processing if access control is not properly implemented
                print("Warning: Access control not fully implemented. Showing all available image_processing.")
                chroma_manager = self.components["vector_db"]["chroma_manager"]
                # Use a safer method to get image_processing
                available_images = asyncio.run(self._get_all_images(chroma_manager))

            print("\nAvailable image_processing:")
            self.image_display.display_images_with_thumbnails(available_images, is_search_result=False)
        except Exception as e:
            print(f"Error listing image_processing: {str(e)}")
            print("Please ensure the vector database and access control are properly configured.")

    async def _get_all_images(self, chroma_manager):
        """
        Fallback method to get all image_processing when access control fails.

        Args:
            chroma_manager: The Chroma database manager

        Returns:
            List of image dictionaries
        """
        try:
            # Query for all image_processing in the generated_images collection
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
                        "thumbnail_path": metadata.get("thumbnail_path",
                                                       metadata.get("image_path", "Not available")),
                        "metadata": metadata
                    })
                return images
            return []
        except Exception as e:
            print(f"Error retrieving image_processing: {str(e)}")
            return []

    def _handle_query_command(self, cmd):
        """Process and execute a query command."""
        # Get query text
        if cmd.lower() == "query":
            query_text = input("Enter your query: ")
        else:
            query_text = cmd[6:].strip()

        # Process the query using RAGSystem
        asyncio.run(self.rag_system.process_query(query_text))

    def _handle_generate_notebook_command(self, cmd):
        """Handle the generate-notebook command."""
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

        result = self.notebook_generator.generate_notebook(self.components, model_id, output_path)

        if result:
            print(f"Notebook generated successfully: {result}")
        else:
            print("Failed to generate notebook")