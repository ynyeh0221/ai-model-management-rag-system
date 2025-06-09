import asyncio

class BaseCommandHandler:
    """Common command-handling utilities shared by CLI and Streamlit."""

    def __init__(self, rag_system, components, user_id):
        self.rag_system = rag_system
        self.components = components
        self.user_id = user_id
        self.notebook_generator = None

    def list_models(self):
        """Return models accessible to the current user."""
        access_control = self.components["vector_db"]["access_control"]
        return access_control.get_accessible_models(self.user_id)

    def list_images(self):
        """Return images accessible to the current user."""
        access_control = self.components["vector_db"]["access_control"]
        try:
            return access_control.get_accessible_images(self.user_id)
        except AttributeError:
            chroma_manager = self.components["vector_db"]["chroma_manager"]
            return asyncio.run(self._get_all_images(chroma_manager))

    async def _get_all_images(self, chroma_manager):
        """Retrieve all images when access control does not provide them."""
        try:
            results = await chroma_manager.get(
                collection_name="generated_images",
                include=["metadatas"],
                limit=100,
            )
            if results and "results" in results:
                images = []
                for item in results["results"]:
                    metadata = item.get("metadata", {})
                    images.append(
                        {
                            "id": item.get("id", "Unknown"),
                            "prompt": metadata.get("prompt", "No prompt"),
                            "image_path": metadata.get("image_path", "Not available"),
                            "thumbnail_path": metadata.get(
                                "thumbnail_path",
                                metadata.get("image_path", "Not available"),
                            ),
                            "metadata": metadata,
                        }
                    )
                return images
            return []
        except Exception:
            return []

    def generate_notebook(self, model_id, output_path):
        """Generate a notebook using the configured generator."""
        if not self.notebook_generator:
            raise ValueError("Notebook generator not set")
        return self.notebook_generator.generate_notebook(
            self.components, model_id, output_path
        )

