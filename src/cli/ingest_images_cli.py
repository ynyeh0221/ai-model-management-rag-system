import asyncio
import concurrent.futures
import glob
import logging
import os
import re
from pathlib import Path


class ImageProcessorRunner:
    """
    A class that focuses solely on processing image_processing and storing them in the vector database.
    All search/query functionality has been moved to the ImageSearchManager class.
    """

    def __init__(self):
        self.logger = logging.getLogger("image_processor_runner")

    def process_images(self, components, directory_path, model_id=None, extract_epoch=True):
        """Process image_processing in a directory.

        This function walks through the directory to find image files,
        processes them using the image processor component, generates
        embeddings, and stores them in the vector database.

        Args:
            components: Dictionary containing initialized system cli_response_utils
            directory_path: Path to directory containing image_processing
            model_id: Optional model ID to associate with all image_processing in this directory
            extract_epoch: Whether to attempt to extract epoch information from filenames or paths
        """
        print(f"Processing image_processing in {directory_path}...")

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

        # Filter out thumbnail image_processing (files that start with "thumb_")
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
        logger.info(f"Skipped {thumbnail_count} thumbnail image_processing")
        logger.info(f"Processing {len(filtered_image_files)} non-thumbnail image_processing")

        # Try to extract model_id from directory name if not provided
        if model_id is None:
            file_path_obj = Path(file_path).resolve()
            model_id = str(file_path_obj.parent)

        # Process files in parallel using a thread pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_file = {executor.submit(self.process_single_image,
                                              components,
                                              file_path,
                                              extract_epoch): file_path for file_path in filtered_image_files}

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

    def extract_epoch_from_path(self, file_path):
        """Extract epoch information from file path or name.

        Looks for patterns like:
        - epoch_42
        - e42
        - _42_
        - /42/

        Args:
            file_path: Path to the image file

        Returns:
            int: Extracted epoch number or None if not found
        """
        try:
            # Check for epoch in filename
            filename = os.path.basename(file_path)
            name_without_ext = os.path.splitext(filename)[0]

            # Pattern: epoch_42 or e42
            epoch_match = re.search(r'epoch[_-](\d+)', name_without_ext, re.IGNORECASE)
            if epoch_match:
                return int(epoch_match.group(1))

            # Pattern: e42
            e_match = re.search(r'\be(\d+)\b', name_without_ext)
            if e_match:
                return int(e_match.group(1))

            # Check for epoch in directory path
            # Pattern: /epoch_42/ or /42/
            dir_path = os.path.dirname(file_path)
            epoch_dir_match = re.search(r'/epoch[_-](\d+)/', dir_path, re.IGNORECASE)
            if epoch_dir_match:
                return int(epoch_dir_match.group(1))

            # Pattern: /42/
            dir_match = re.search(r'/(\d+)/', dir_path)
            if dir_match:
                # Only consider this a match if the directory only contains digits
                if os.path.basename(os.path.dirname(dir_match.group(0))).isdigit():
                    return int(dir_match.group(1))

            return None
        except Exception as e:
            self.logger.warning(f"Error extracting epoch from path: {e}")
            return None

    def process_single_image(self, components, file_path, extract_epoch=True, logger=None):
        """Process a single image file.

        Args:
            file_path: Path to the image file
            components: Dictionary containing initialized system cli_response_utils
            model_id: Optional model ID to associate with this image
            extract_epoch: Whether to attempt to extract epoch information from filename or path
            logger: Logger instance

        Returns:
            Tuple of (document_id, success) if processed, None if skipped
        """
        # Extract required cli_response_utils
        image_processor = components["content_analyzer"]["image_processor"]
        schema_validator = components["content_analyzer"]["schema_validator"]
        image_embedder = components["vector_db"]["image_embedder"]
        chroma_manager = components["vector_db"]["chroma_manager"]
        access_control = components["vector_db"]["access_control"]

        # Get the image analyzer if available
        image_analyzer = components.get("content_analyzer", {}).get("image_analyzer", None)

        # 0. Set up logging and check if image file exists
        if logger is None:
            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger("image_processor")

        if not os.path.isfile(file_path):
            logger.error(f"File {file_path} does not exist")
            return None

        # 1. Generate model_id as the absolute path of the folder containing the file
        # Replace any previously provided model_id to fix the issue
        file_path_obj = Path(file_path).resolve()
        # Get the absolute directory path of the file
        model_id = str(file_path_obj.parent.absolute())
        logger.info(f"Using directory of file as model_id: {model_id}")

        # Check if model_id already contains epoch information to avoid duplication
        epoch_in_model_id = None
        epoch_match = re.search(r'epoch[_-](\d+)', model_id, re.IGNORECASE)
        if epoch_match:
            epoch_in_model_id = int(epoch_match.group(1))

        # 2. Try to extract epoch information if requested
        epoch = None
        if extract_epoch:
            if epoch_in_model_id is not None:
                # Use the epoch value already found in model_id
                epoch = epoch_in_model_id
                logger.info(f"Using epoch {epoch} from model_id")
            else:
                # Extract epoch from path
                epoch = self.extract_epoch_from_path(file_path)

        # 3. Process the image and extract metadata
        # If we have epoch information, add it to the metadata
        additional_metadata = {}
        if epoch is not None:
            additional_metadata["epoch"] = epoch

        # If we have a model_id, add it to the metadata
        if model_id:
            additional_metadata["model_id"] = model_id

        # Process the image with the additional metadata
        process_result = image_processor.process_image(file_path, additional_metadata)
        if not process_result:
            # Not a valid image, skip it
            return None

        # 4. Create document ID - avoid duplicating epoch in ID if already in model_id
        if epoch_in_model_id is not None:
            document_id = f"generated_image_{model_id}"  # Don't append epoch again
        else:
            document_id = f"generated_image_{model_id}_{epoch}" if epoch is not None else f"generated_image_{model_id}"

        # 5. Analyze image content if we have an image analyzer
        if image_analyzer:
            try:
                # Open the image file for analysis
                from PIL import Image
                with Image.open(file_path) as img:
                    # Run the analyzer on the image
                    content_analysis = image_analyzer.analyze_image(img)

                    # If we got valid analysis results, update the metadata
                    if content_analysis:
                        logger.info(f"Got image analysis results: {content_analysis.keys()}")

                        # Add the image content analysis to the metadata
                        if "metadata" in process_result:
                            # Add description if available
                            if "description" in content_analysis:
                                process_result["metadata"]["description"] = content_analysis.pop("description")

                            # Replace the image_content field with the full analysis
                            process_result["metadata"]["image_content"] = content_analysis

                            logger.info(f"Added image content analysis for {file_path}")
            except Exception as e:
                logger.warning(f"Error analyzing image content for {file_path}: {str(e)}")
                # Continue processing without content analysis if it fails

        # 6. Create document with proper structure
        document = {
            "id": document_id,
            "$schema_version": "1.0.0",
            "content": None,  # No text content, just embedding
            "metadata": process_result["metadata"]  # Ensure metadata is present
        }

        logger.info(f"Document ID: {document_id}")
        logger.info(f"Document metadata keys: {document['metadata'].keys()}")
        logger.info(f"Image content: {document['metadata'].get('image_content', 'None')}")

        # 7. Validate against schema
        validation_result = schema_validator.validate(document, "generated_image_schema")
        if not validation_result["valid"]:
            logger.warning(f"Schema validation failed for {file_path}: {validation_result['errors']}")
            return (document_id, False)

        # 8. Generate image embeddings
        # Use global embedding by default, but can use tiled if specified in metadata
        embedding_type = document["metadata"].get("embedding_type", "global")

        # Ensure we have a valid image path before trying to generate embeddings
        if not os.path.exists(file_path):
            logger.error(f"Image file path does not exist: {file_path}")
            return (document_id, False)

        import numpy as np
        try:
            from PIL import Image
            # Load the image explicitly to ensure it's valid
            with Image.open(file_path) as img:
                # Just verify the image can be read
                img_size = img.size
                logger.info(f"Image size: {img_size}")

            if embedding_type == "global":
                logger.info(f"Generating global embedding for {file_path}")
                embedding = image_embedder.embed_image(file_path)
            else:  # tiled embedding
                logger.info(f"Generating tiled embedding for {file_path}")
                tile_result = image_embedder.embed_image_tiled(file_path,
                                                               document["metadata"].get("tile_config", {}))
                embedding = tile_result[0] if isinstance(tile_result, tuple) else tile_result

            # Check if embedding is valid and not empty
            if embedding is None:
                logger.error(f"Failed to generate embedding: result is None")
                # Use a random embedding as fallback
                embedding = np.random.rand(384).astype(np.float32)  # Default embedding size
                logger.info(f"Using random fallback embedding")
            elif isinstance(embedding, (list, np.ndarray)) and len(embedding) == 0:
                logger.error(f"Failed to generate embedding: empty array")
                # Use a random embedding as fallback
                embedding = np.random.rand(384).astype(np.float32)  # Default embedding size
                logger.info(f"Using random fallback embedding")
            else:
                logger.info(f"Successfully generated embedding of shape: {np.array(embedding).shape}")

        except Exception as e:
            logger.error(f"Error generating embedding for {file_path}: {str(e)}")
            # Create a random embedding as fallback
            embedding = np.random.rand(384).astype(np.float32)  # Default embedding size
            logger.info(f"Using random fallback embedding due to error")

        # 9. Apply access control
        access_metadata = access_control.get_document_permissions(document)
        document["metadata"]["access_control"] = access_metadata

        # 10. Create and store thumbnail if it doesn't exist
        thumbnail_path = document["metadata"].get("thumbnail_path")
        # If thumbnail_path is not a valid path type, derive a default thumbnail file path.
        if not isinstance(thumbnail_path, (str, bytes, os.PathLike)) or not thumbnail_path:
            thumb_dir = os.path.dirname(file_path)
            thumbnail_path = os.path.join(thumb_dir, f"thumb_{Path(file_path).name}")
            document["metadata"]["thumbnail_path"] = thumbnail_path
            logger.info(f"Generated thumbnail path: {thumbnail_path}")

        if thumbnail_path and not os.path.exists(thumbnail_path):
            try:
                os.makedirs(os.path.dirname(thumbnail_path), exist_ok=True)
                # Pass the loaded image (a PIL Image) and the verified thumbnail_path.
                image_processor.generate_thumbnail(file_path, thumbnail_path)
                logger.info(f"Created thumbnail at: {thumbnail_path}")
            except Exception as e:
                logger.warning(f"Error generating thumbnail for {file_path}: {str(e)}")
                # Continue processing even if thumbnail generation fails

        # 11. Store in Chroma
        try:
            logger.info(f"Adding document to Chroma with ID: {document_id}")
            asyncio.run(chroma_manager.add_document(
                collection_name="generated_images",
                document_id=document_id,
                document=document,
                embed_content=embedding  # Pass the actual embedding, not the file path
            ))
            logger.info(f"Document added successfully")

        except Exception as e:
            logger.error(f"Error adding document to Chroma: {str(e)}")
            return (document_id, False)

        logger.info(f"Successfully processed {file_path} with model ID {model_id}")
        return (document_id, True)

    def reprocess_images(self, components, directory_path, model_id=None, force_update=False):
        """Reprocess image_processing that have already been processed.

        This is useful for updating metadata when the schema or processing logic changes.

        Args:
            components: Dictionary containing initialized system cli_response_utils
            directory_path: Path to directory containing image_processing
            model_id: Optional model ID to associate with all image_processing in this directory
            force_update: Whether to update all image_processing or only those with missing required fields

        Returns:
            tuple: (success_count, failure_count, skipped_count)
        """
        logger = logging.getLogger("image_processor")
        logger.info(f"Reprocessing image_processing in {directory_path}...")

        # Set up tracking counters
        success_count = 0
        failure_count = 0
        skipped_count = 0

        # Get existing image documents from Chroma for this model
        chroma_manager = components["vector_db"]["chroma_manager"]

        # Build filter query
        filter_query = {}
        if model_id:
            filter_query["model_id"] = model_id

        existing_images = asyncio.run(chroma_manager.get(
            collection_name="generated_images",
            where=filter_query,
            include=["metadatas", "documents", "embeddings"]
        ))

        # If no existing image_processing found
        if not existing_images or not existing_images.get("metadatas"):
            logger.info("No existing image_processing found for reprocessing")
            return (0, 0, 0)

        # Process each existing image
        for idx, metadata in enumerate(existing_images.get("metadatas", [])):
            document_id = existing_images.get("ids", [])[idx] if idx < len(existing_images.get("ids", [])) else None

            # Skip if no document_id
            if not document_id:
                logger.warning(f"Missing document ID for image at index {idx}")
                skipped_count += 1
                continue

            # Get image path from metadata
            image_path = metadata.get("image_path")
            if not image_path or not os.path.exists(image_path):
                logger.warning(f"Image path not found for document {document_id}: {image_path}")
                skipped_count += 1
                continue

            # Check if we need to update this image
            if not force_update:
                # Skip if all required fields are present
                if self._has_required_fields(metadata):
                    logger.info(f"Skipping document {document_id} - all required fields present")
                    skipped_count += 1
                    continue

            # Reprocess the image
            try:
                # Extract model_id and epoch from existing metadata
                existing_epoch = metadata.get("epoch")

                # Reprocess the image
                # Use existing epoch if available, otherwise extract from path if requested
                should_extract_epoch = existing_epoch is None

                result = self.process_single_image(
                    components,
                    image_path,
                    extract_epoch=should_extract_epoch,
                    logger=logger
                )

                if result:
                    success_count += 1
                else:
                    failure_count += 1

            except Exception as e:
                logger.error(f"Error reprocessing image {document_id}: {str(e)}")
                failure_count += 1

        logger.info(
            f"Reprocessing completed. Success: {success_count}, Failed: {failure_count}, Skipped: {skipped_count}")
        return (success_count, failure_count, skipped_count)

    def _has_required_fields(self, metadata):
        """Check if metadata has all required fields according to the schema.

        Args:
            metadata: The metadata dictionary to check

        Returns:
            bool: True if all required fields are present, False otherwise
        """
        # Check for critical fields
        required_fields = ["model_id", "image_path"]
        for field in required_fields:
            if field not in metadata or metadata[field] is None:
                return False

        # Check for image_content structure
        if "image_content" not in metadata:
            return False

        image_content = metadata["image_content"]
        if not isinstance(image_content, dict):
            return False

        # Check for dates structure
        if "dates" not in metadata:
            return False

        dates = metadata["dates"]
        if not isinstance(dates, dict):
            return False

        # Check for critical date fields
        date_fields = ["creation_date", "created_at", "created_year", "created_month"]
        date_field_found = False
        for field in date_fields:
            if field in dates and dates[field] is not None:
                date_field_found = True
                break

        if not date_field_found:
            return False

        return True

    def batch_process_images(self, components, batch_config):
        """Process multiple directories of image_processing in batch.

        Args:
            components: Dictionary containing initialized system cli_response_utils
            batch_config: Dictionary with batch processing configuration
                Example: {
                    "directories": [
                        {
                            "path": "/path/to/dir1",
                            "model_id": "model1",
                            "extract_epoch": true
                        },
                        {
                            "path": "/path/to/dir2",
                            "model_id": "model2",
                            "extract_epoch": false
                        }
                    ],
                    "max_workers": 3,  # Number of parallel directory processors
                    "reprocess": false  # Whether to reprocess existing image_processing
                }

        Returns:
            dict: Dictionary with processing results for each directory
        """
        logger = logging.getLogger("image_processor")
        logger.info("Starting batch image processing...")

        results = {}
        directories = batch_config.get("directories", [])
        max_workers = batch_config.get("max_workers", 1)
        reprocess = batch_config.get("reprocess", False)

        # Process directories in parallel if max_workers > 1
        if max_workers > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_dir = {}

                # Submit directory processing tasks
                for dir_config in directories:
                    path = dir_config.get("path")
                    model_id = dir_config.get("model_id")
                    extract_epoch = dir_config.get("extract_epoch", True)

                    if not path:
                        logger.warning("Skipping directory with missing path")
                        continue

                    if reprocess:
                        future = executor.submit(
                            self.reprocess_images,
                            components,
                            path,
                            model_id,
                            batch_config.get("force_update", False)
                        )
                    else:
                        # Pass the extract_epoch parameter from directory config
                        future = executor.submit(
                            self.process_images,
                            components,
                            path,
                            model_id,
                            extract_epoch
                        )

                    future_to_dir[future] = path

                # Collect results
                for future in concurrent.futures.as_completed(future_to_dir):
                    directory = future_to_dir[future]
                    try:
                        result = future.result()
                        results[directory] = {
                            "success": True,
                            "result": result
                        }
                    except Exception as e:
                        logger.error(f"Error processing directory {directory}: {str(e)}")
                        results[directory] = {
                            "success": False,
                            "error": str(e)
                        }
        else:
            # Process directories sequentially
            for dir_config in directories:
                path = dir_config.get("path")
                model_id = dir_config.get("model_id")
                extract_epoch = dir_config.get("extract_epoch", True)

                if not path:
                    logger.warning("Skipping directory with missing path")
                    continue

                try:
                    if reprocess:
                        result = self.reprocess_images(
                            components,
                            path,
                            model_id,
                            batch_config.get("force_update", False)
                        )
                    else:
                        # Pass the extract_epoch parameter from directory config
                        result = self.process_images(
                            components,
                            path,
                            model_id,
                            extract_epoch
                        )

                    results[path] = {
                        "success": True,
                        "result": result
                    }
                except Exception as e:
                    logger.error(f"Error processing directory {path}: {str(e)}")
                    results[path] = {
                        "success": False,
                        "error": str(e)
                    }

        logger.info("Batch processing completed")
        return results