import asyncio


class CommandHandler:
    """Handles command processing and delegation to appropriate components."""

    def __init__(self, ui_runner):
        self.ui_runner = ui_runner
        self.components = ui_runner.components
        self.user_id = ui_runner.user_id

        # These will be set by UIRunner after initialization
        self.model_display = None
        self.image_display = None
        self.notebook_generator = None
        self.query_processor = None

    def handle_command(self, cmd):
        """
        Process a user command and delegate to the appropriate handler.

        Args:
            cmd (str): The command string to process.

        Returns:
            bool: True if processing should continue, False if the user wants to exit.
        """
        cmd = cmd.strip()

        if cmd.lower() == "exit":
            print("Exiting. Goodbye!")
            return False
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

        return True

    def _handle_help_command(self):
        """Display available commands and their descriptions."""
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
        """List all models accessible to the current user."""
        access_control = self.components["vector_db_manager"]["access_control"]

        try:
            # Get models the user has access to
            available_models = access_control.get_accessible_models(self.user_id)
            print("\nAvailable models:")
            self.model_display.display_models_pretty(available_models)
        except Exception as e:
            print(f"Error listing models: {str(e)}")

    def _handle_list_images_command(self):
        """List all images accessible to the current user."""
        try:
            access_control = self.components["vector_db_manager"]["access_control"]

            # Get images the user has access to
            try:
                available_images = access_control.get_accessible_images(self.user_id)
            except AttributeError:
                # Fallback: Get all images if access control is not properly implemented
                print("Warning: Access control not fully implemented. Showing all available images.")
                chroma_manager = self.components["vector_db_manager"]["chroma_manager"]
                # Use a safer method to get images
                available_images = asyncio.run(self._get_all_images(chroma_manager))

            print("\nAvailable images:")
            self.image_display.display_images_with_thumbnails(available_images, is_search_result=False)
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

    def _handle_query_command(self, cmd):
        """Process and execute a query command."""
        # Get query text
        if cmd.lower() == "query":
            query_text = input("Enter your query: ")
        else:
            query_text = cmd[6:].strip()

        # Process the query
        asyncio.run(self.query_processor.process_query(query_text))

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
