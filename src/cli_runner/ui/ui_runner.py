import logging

from cli_runner.ui.components.command_hander import CommandHandler
from cli_runner.ui.components.image_display_manager import ImageDisplayManager
from cli_runner.ui.components.llm_response_processor import LLMResponseProcessor
from cli_runner.ui.components.model_display_manager import ModelDisplayManager
from cli_runner.ui.components.notebook_generator import NotebookGenerator
from cli_runner.ui.components.query_processor import QueryProcessor


class UIRunner:
    """
    A command-line interface runner for the AI Model Management RAG system.

    This class is the main orchestrator that initializes all components and
    handles user interactions through a command-line interface. It delegates
    specific functionalities to specialized components.
    """

    def __init__(self):
        self.components = None
        self.user_id = "anonymous"

        # Component instances will be initialized in start_ui
        self.command_handler = None
        self.model_display_manager = None
        self.image_display_manager = None
        self.notebook_generator = None
        self.llm_response_processor = None
        self.query_processor = None

    def start_ui(self, components, host="localhost", port=8000):
        """
        Start a command-line interface for the RAG system.

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

        # Initialize all required components
        self._initialize_components()

        # Main command loop
        self._run_command_loop()

    def _initialize_components(self):
        """Initialize all the component classes needed by the UI."""
        # Create display managers
        self.model_display_manager = ModelDisplayManager()
        self.image_display_manager = ImageDisplayManager()

        # Create generators and processors
        self.notebook_generator = NotebookGenerator()
        self.llm_response_processor = LLMResponseProcessor()

        # Create query processor with necessary dependencies
        self.query_processor = QueryProcessor(
            self.components,
            self.user_id
        )

        # Create command handler with a reference to this runner
        # This allows the command handler to access all other components through the runner
        self.command_handler = CommandHandler(self)

        # Inject component references into the command handler
        self.command_handler.model_display = self.model_display_manager
        self.command_handler.image_display = self.image_display_manager
        self.command_handler.notebook_generator = self.notebook_generator
        self.command_handler.query_processor = self.query_processor

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

                # Delegate command handling to CommandHandler
                continue_processing = self.command_handler.handle_command(cmd)
                if not continue_processing:
                    break

            except KeyboardInterrupt:
                print("\nOperation cancelled.")
            except Exception as e:
                print(f"Error: {str(e)}")
                logging.error(f"Exception in command loop: {str(e)}", exc_info=True)

        print("UI session ended.")