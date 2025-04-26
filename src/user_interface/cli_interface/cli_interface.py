"""
CLI Interface Module - Encapsulating the Command Line Interface

This module provides a command line interface that uses the RAGSystem core module
to process user queries and commands. It refactors the original UIRunner class
into an interface that uses RAGSystem.
"""

import logging
from typing import Dict, Any

from user_interface.cli_interface.command_hander import CommandHandler
from user_interface.core.rag_system import RAGSystem


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
        print(f"Starting command line interface (Web UI will be developed later)...")

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
        from user_interface.cli_interface.components.model_display_manager import ModelDisplayManager
        from user_interface.cli_interface.components.image_display_manager import ImageDisplayManager
        from user_interface.cli_interface.components.notebook_generator import NotebookGenerator
        from user_interface.cli_interface.components.llm_response_processor import LLMResponseProcessor

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