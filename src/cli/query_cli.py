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
from src.core.base_command_handler import BaseCommandHandler


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

        # Display managers and processors will be initialized in start_cli
        self.model_display_manager = None
        self.image_display_manager = None
        self.notebook_generator = None
        self.llm_response_processor = None

        # CommandHandler will be initialized after UI components
        self.command_handler = None

    def start_cli(self, components: Dict[str, Any], host: str = "localhost", port: int = 8000):
        """
        Start the command line interface

        Args:
            components: Dictionary containing initialized system components
            host: Hostname, kept for API compatibility, not used in CLI mode
            port: Port number, kept for API compatibility, not used in CLI mode
        """
        print("Starting command line interface (Web UI will be developed later)...")
        print(f"Host: {host}. Port: {port}")

        # Store components reference
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

        # Initialize UI components
        self._initialize_components()

        # Initialize command handler after UI components
        self.command_handler = CommandHandler(self)

        # Run main command loop
        self._run_command_loop()

    def _initialize_components(self):
        """Initialize component classes needed by the UI"""
        # Import necessary components
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
        # Can a format log output as needed?
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
        """Route the result to the appropriate handler based on its type."""
        result_type = result.get("type")

        if result_type == "text_search":
            self._handle_text_search(result)

        elif result_type == "image_search":
            self._handle_image_search(result)

        elif result_type == "command":
            self._handle_command(result)

        elif result_type == "needs_clarification":
            self._handle_needs_clarification(result)

        else:
            # You could optionally log or print unknown‐type results here
            print(f"Unknown result type: {result_type}")

    def _handle_text_search(self, result):
        """Display LLM response and any reranked text‐search results."""
        # Display full LLM response
        print("\nLLM Final Response:")
        full_resp = result.get("final_response", "")
        self.llm_response_processor.print_llm_content(full_resp)

        # Display reranked search results (if any)
        print("\nRetrieving and displaying reranked search results:\n")
        reranked = result.get("search_results", [])
        self.model_display_manager.display_reranked_results_pretty(reranked)

    def _handle_image_search(self, result):
        """Display image‐search results."""
        print("\nImage search results:")
        images = result.get("results", [])
        self.image_display_manager.display_image_search_results(images)

    def _handle_command(self, result):
        """Format and display a 'command'‐type result."""
        cmd_res = result.get("result")

        if isinstance(cmd_res, dict):
            # If there is an "available_commands" key, list them
            if "available_commands" in cmd_res:
                print("\nAvailable commands:")
                for cmd in cmd_res["available_commands"]:
                    print(f"  {cmd}")
            else:
                # Otherwise, print each key/value pair in the dictionary
                for key, value in cmd_res.items():
                    print(f"{key}: {value}")
        else:
            # If it's not a dict, treat it as a simple string
            print(cmd_res or "")

    def _handle_needs_clarification(self, result):
        """Ask the user to pick (or type) a clarified query, then re‐run it."""
        query = result.get("query", "")
        clarity_result = result.get("clarity_result", {})

        # 1) Print out why clarification is needed
        self._print_clarification_prompt(query, clarity_result)

        # 2) Loop until the user picks a valid option (0 = improved, 1.N = suggestions,
        #    N+1 = original, N+2 = custom)
        new_query = self._resolve_clarification_choice(query, clarity_result)

        # 3) Re‐run the RAG system on whatever new_query was chosen
        asyncio.run(self.rag_system.process_query(new_query))

    def _print_clarification_prompt(self, original_query, clarity_result):
        """Print the “needs clarification” text and suggested queries."""
        print("\nYour query could be clearer:")
        reason = clarity_result.get("reason", "No reason provided")
        print(f"Reason: {reason}")

        improved_query = clarity_result.get("improved_query", original_query)
        suggestions = clarity_result.get("suggestions", [])

        print("\nSuggestions:")
        print(f"0. Improved query: {improved_query}")

        for i, suggestion in enumerate(suggestions, 1):
            print(f"{i}. {suggestion}")

        print(f"{len(suggestions) + 1}. Use original query: {original_query}")
        print(f"{len(suggestions) + 2}. Enter a new query")

    def _resolve_clarification_choice(self, original_query, clarity_result):
        """
        Prompt the user to pick one of the suggested queries (or enter a new one).
        Returns the chosen query string.
        """
        improved_query = clarity_result.get("improved_query", original_query)
        suggestions = clarity_result.get("suggestions", [])

        max_option = len(suggestions) + 2

        while True:
            choice = input("Select an option (number): ")

            try:
                choice_num = int(choice)
            except ValueError:
                print("Please enter a number.")
                continue

            # 0 => improved_query
            if choice_num == 0:
                return improved_query

            # 1..len(suggestions) => pick from suggestions
            if 1 <= choice_num <= len(suggestions):
                return suggestions[choice_num - 1]

            # len(suggestions)+1 => original query
            if choice_num == len(suggestions) + 1:
                return original_query

            # len(suggestions)+2 => custom (prompt the user)
            if choice_num == max_option:
                return input("Enter your new query: ")

            print("Invalid selection. Please try again.")

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

class CommandHandler(BaseCommandHandler):
    """Handles command processing and delegation to appropriate components."""

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

        super().__init__(self.rag_system, self.components, self.user_id)

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
        try:
            available_models = self.list_models()
            print("\nAvailable models:")
            self.model_display.display_models_pretty(available_models)
        except Exception as e:
            print(f"Error listing models: {str(e)}")

    def _handle_list_images_command(self):
        """List all image_processing accessible to the current user."""
        try:
            available_images = self.list_images()
            print("\nAvailable image_processing:")
            self.image_display.display_images_with_thumbnails(available_images, is_search_result=False)
        except Exception as e:
            print(f"Error listing image_processing: {str(e)}")
            print("Please ensure the vector database and access control are properly configured.")


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
        """
        Handle the “generate-notebook” command. Delegates argument parsing
        to helper methods, then calls the notebook generator and prints the result.
        """
        # 1) If the user just typed “generate-notebook” (no flags/arguments),
        #    prompt them for model_id and output_path.
        if cmd.strip().lower() == "generate-notebook":
            model_id, _, output_path = self._prompt_notebook_defaults()
        else:
            # 2) Otherwise, parse out model_id, --type (or -t), and --output (or -o)
            model_id, _, output_path = self._extract_notebook_args(cmd)

        # 3) Invoke the generator
        result = self.generate_notebook(model_id, output_path)

        # 4) Print success / failure
        if result:
            print(f"Notebook generated successfully: {result}")
        else:
            print("Failed to generate notebook")

    def _prompt_notebook_defaults(self):
        """
        Prompt the user for model_id and output_path when they only typed
        “generate-notebook” with no additional arguments.
        Returns a tuple: (model_id, notebook_type, output_path).
        """
        # Default notebook_type for the bare command is “evaluation”
        notebook_type = "evaluation"

        model_id = input("Enter model ID: ")
        default_path = f"./notebooks/{model_id}.ipynb"
        output_path = input(f"Enter output path [default: {default_path}]: ") or default_path

        return model_id, notebook_type, output_path

    def _extract_notebook_args(self, cmd):
        """
        Parse a command string of the form:
            generate-notebook <model_id> [--type=<type> | -t=<type>] [--output=<path> | -o=<path>]
        or positional arguments in place of flags.
        Returns (model_id, notebook_type, output_path).
        """
        parts = cmd.split(maxsplit=3)

        # 1) Model ID is either the second token, or prompt if missing
        if len(parts) > 1:
            model_id = parts[1]
        else:
            model_id = input("Enter model ID: ")

        # 2) Default type is “evaluation”
        notebook_type = "evaluation"
        if len(parts) > 2:
            raw_type = parts[2]
            if raw_type.startswith("--type=") or raw_type.startswith("-t="):
                notebook_type = raw_type.split("=", 1)[1]
            else:
                # If they passed something like “generate-notebook ID customType …”
                notebook_type = raw_type

        # 3) Default output path (if no --output/-o given)
        default_path = f"./notebooks/{model_id}_{notebook_type}.ipynb"

        if len(parts) > 3:
            raw_output = parts[3]
            if raw_output.startswith("--output=") or raw_output.startswith("-o="):
                output_path = raw_output.split("=", 1)[1]
            else:
                # If they passed something like “generate-notebook ID type /some/path.ipynb”
                output_path = raw_output
        else:
            output_path = default_path

        return model_id, notebook_type, output_path