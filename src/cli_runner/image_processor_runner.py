import asyncio
import concurrent.futures
import glob
import logging
import os
from pathlib import Path


class ImageProcessorRunner:

    def __init__(self):
        pass

    def process_images(self, components, directory_path):
        """Process images in a directory.

        This function walks through the directory to find image files,
        processes them using the image cli_runner component, generates
        embeddings, and stores them in the vector database.

        Args:
            components: Dictionary containing initialized system components
            directory_path: Path to directory containing images
        """
        print(f"Processing images in {directory_path}...")

        # Set up logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger("image_processor")

        # Ensure directory exists
        if not os.path.isdir(directory_path):
            logger.error(f"Directory {directory_path} does not exist")
            return

        # Supported image extensions
        supported_extensions = ['.png', '.jpg', '.jpeg', '.webp']

        # Find all image files
        image_files = []
        for ext in supported_extensions:
            image_files.extend(glob.glob(os.path.join(directory_path, f"**/*{ext}"), recursive=True))

        # Filter out thumbnail images (files that start with "thumb_")
        filtered_image_files = []
        thumbnail_count = 0
        for file_path in image_files:
            file_name = Path(file_path).name
            if file_name.startswith("thumb_"):
                thumbnail_count += 1
                logger.info(f"Skipping thumbnail image: {file_path}")
            else:
                filtered_image_files.append(file_path)

        logger.info(f"Found {len(image_files)} total image files")
        logger.info(f"Skipped {thumbnail_count} thumbnail images")
        logger.info(f"Processing {len(filtered_image_files)} non-thumbnail images")

        # Process files in parallel using a thread pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_file = {executor.submit(self.process_single_image,
                                              components,
                                              file_path): file_path for file_path in filtered_image_files}

            for future in concurrent.futures.as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    if result:
                        document_id, success = result
                        logger.info(
                            f"Processed {file_path} with ID {document_id}: {'Success' if success else 'Failed'}")
                    else:
                        logger.warning(f"Skipped {file_path}: Not a valid image")
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {str(e)}")

        logger.info("Image processing completed")

    def process_single_image(self, components, file_path, logger=None):
        """Process a single image file.

        Args:
            file_path: Path to the image file
            components: Dictionary containing initialized system components
            logger: Logger instance

        Returns:
            Tuple of (document_id, success) if processed, None if skipped
        """
        # Extract required components
        image_processor = components["document_processor"]["image_processor"]
        schema_validator = components["document_processor"]["schema_validator"]
        image_embedder = components["vector_db_manager"]["image_embedder"]
        chroma_manager = components["vector_db_manager"]["chroma_manager"]
        access_control = components["vector_db_manager"]["access_control"]

        # 0. Set up logging and check if image file exists
        if logger is None:
            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger("image_processor")

        if not os.path.isfile(file_path):
            logger.error(f"File {file_path} does not exist")
            return None

        # 1. Process the image and extract metadata
        process_result = image_processor.process_image(file_path)
        if not process_result:
            # Not a valid image, skip it
            return None

        # 2. Generate model_id similar to process_single_script
        file_path_obj = Path(file_path)
        folder_name = file_path_obj.parent.name
        file_stem = file_path_obj.stem
        model_id = f"{folder_name}_{file_stem}"

        # 3. Create document
        document_id = f"generated_image_{model_id}"
        document = {
            "id": document_id,
            "$schema_version": "1.0.0",
            "content": None,  # No text content, just embedding
            "metadata": {
                **process_result["metadata"],
                "model_id": model_id  # Add model_id to metadata
            }
        }

        # 4. Validate against schema
        validation_result = schema_validator.validate(document, "generated_image_schema")
        if not validation_result["valid"]:
            logging.warning(f"Schema validation failed for {file_path}: {validation_result['errors']}")
            return (document_id, False)

        # 5. Generate image embeddings
        # Use global embedding by default, but can use tiled if specified in metadata
        embedding_type = document["metadata"].get("embedding_type", "global")
        if embedding_type == "global":
            embedding = image_embedder.embed_image(process_result["metadata"].get("image_path"))
        else:  # tiled embedding
            embedding = image_embedder.embed_image_tiled(process_result["metadata"].get("image_path"),
                                                         process_result["metadata"].get("tile_config", {}))

        # 6. Apply access control
        access_metadata = access_control.get_document_permissions(document)
        document["metadata"]["access_control"] = access_metadata

        # 7. Store in Chroma
        asyncio.run(chroma_manager.add_document(
            collection_name="generated_images",
            document_id=document_id,
            document=document,
            embed_content=embedding
        ))

        # 8. Create and store thumbnail if it doesn't exist
        thumbnail_path = document["metadata"].get("thumbnail_path")
        # If thumbnail_path is not a valid path type, derive a default thumbnail file path.
        if not isinstance(thumbnail_path, (str, bytes, os.PathLike)):
            thumb_dir = os.path.dirname(file_path)
            thumbnail_path = os.path.join(thumb_dir, f"thumb_{Path(file_path).name}")
            document["metadata"]["thumbnail_path"] = thumbnail_path

        if thumbnail_path and not os.path.exists(thumbnail_path):
            os.makedirs(os.path.dirname(thumbnail_path), exist_ok=True)
            # Pass the loaded image (a PIL Image) and the verified thumbnail_path.
            image_processor.generate_thumbnail(process_result["metadata"].get("image_path"), thumbnail_path)

        logger.info(f"Successfully processed {file_path} with model ID {model_id}")
        return (document_id, True)
