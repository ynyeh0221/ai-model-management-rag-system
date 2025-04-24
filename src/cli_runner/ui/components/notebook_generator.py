import asyncio
import logging
import os

import nbformat
from nbformat.v4 import new_notebook, new_code_cell


class NotebookGenerator:
    """Handles generating notebooks for model analysis."""

    @staticmethod
    def generate_notebook(components, model_id, output_path):
        """
        Generate a Colab notebook for model analysis using full script reconstruction.

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

        try:
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
            chunk_contents = NotebookGenerator._prepare_chunk_contents(chunks)
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

        except Exception as e:
            logger.error(f"Error generating notebook: {str(e)}")
            return None

    @staticmethod
    def _prepare_chunk_contents(chunks):
        """
        Convert raw chunks to structured format for script reconstruction.

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