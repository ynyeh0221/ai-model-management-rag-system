import glob
import asyncio
import logging
import concurrent.futures
import os
from pathlib import Path
from datetime import datetime


class ScriptProcessorRunner:

    def __init__(self):
        pass

    def _clean_iso_timestamp(self, ts: str) -> str:
        """Remove microseconds from ISO format like 2025-03-02T10:53:24.620782 -> 2025-03-02T10:53:24"""
        try:
            dt = datetime.fromisoformat(ts)
            return dt.replace(microsecond=0).isoformat()
        except Exception:
            return ts  # fallback to original if parsing fails

    def _format_natural_date(self, iso_date: str):
        """Format ISO date to natural month.

        Args:
            iso_date: ISO formatted date string

        Returns:
            Natural month representation
        """
        try:
            dt = datetime.fromisoformat(iso_date)
            return dt.strftime("%B")  # e.g. "April 2025"
        except Exception:
            return "Unknown"

    def process_model_scripts(self, components, directory_path):
        """Process model scripts in a directory."""
        print(f"Processing model scripts in {directory_path}...")

        # Set up logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger("model_script_processor")

        # Ensure directory exists
        if not os.path.isdir(directory_path):
            logger.error(f"Directory {directory_path} does not exist")
            return

        # Supported file extensions for model scripts
        supported_extensions = ['.py', '.ipynb', '.json', '.yaml', '.yml']

        # Find all model script files recursively
        script_files = []
        for ext in supported_extensions:
            script_files.extend(glob.glob(os.path.join(directory_path, f"**/*{ext}"), recursive=True))

        # Exclude files from the virtual environment folder, e.g., any path containing '/~/myenv/'
        script_files = [f for f in script_files if '/~/myenv/' not in f]

        logger.info(f"Found {len(script_files)} potential model script files after filtering")

        total_files = len(script_files)
        processed_count = 0

        # Process files in parallel using a thread pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_file = {executor.submit(self.process_single_script, components, file_path, logger): file_path
                              for file_path in script_files}

            for future in concurrent.futures.as_completed(future_to_file):
                file_path = future_to_file[future]
                processed_count += 1
                try:
                    result = future.result()
                    if result:
                        document_id, success = result
                        logger.info(
                            f"Processed {file_path} with ID {document_id}: {'Success' if success else 'Failed'}")
                    else:
                        logger.warning(f"Skipped {file_path}: Not a model script")
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {str(e)}")

                progress = (processed_count / total_files) * 100
                logger.info(f"[Progress] {processed_count}/{total_files} files processed ({progress:.1f}%)")

        logger.info("Model script processing completed")

    def process_single_script(self, components, file_path, logger=None):
        """Process a single model script file.

        Args:
            components: Dictionary containing initialized system components
            file_path: Path to the model script file
            logger: Logger instance

        Returns:
            Tuple of (document_id, success) if processed, None if skipped
        """
        # Extract required components
        code_parser = components["document_processor"]["code_parser"]
        metadata_extractor = components["document_processor"]["metadata_extractor"]
        schema_validator = components["document_processor"]["schema_validator"]
        text_embedder = components["vector_db_manager"]["text_embedder"]
        chroma_manager = components["vector_db_manager"]["chroma_manager"]
        access_control = components["vector_db_manager"]["access_control"]

        # Set up logging
        if logger is None:
            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger("model_script_processor")

        # Check if file exists
        if not os.path.isfile(file_path):
            logger.error(f"File {file_path} does not exist")
            return None

        # Parse the code to determine if it's a model script
        parse_result = code_parser.parse(file_path)
        if not parse_result or not parse_result.get("is_model_script", False):
            # Not a model script, skip it
            return None

        # Split into chunks for processing
        chunks = code_parser.split_ast_and_subsplit_chunks(
            file_content=parse_result["content"],
            file_path=file_path,
            chunk_size=5000,
            overlap=1000
        )

        # Extract metadata and prepare metadata document
        model_id, metadata_document = self._extract_and_prepare_metadata(
            metadata_extractor,
            parse_result,
            file_path,
            chunks
        )

        # Validate and store metadata document
        metadata_stored = self._validate_and_store_metadata(
            schema_validator,
            metadata_document,
            text_embedder,
            chroma_manager,
            access_control
        )

        if not metadata_stored:
            logger.warning(f"Failed to store metadata for {file_path}")
            return None, False

        # Process and store code chunks
        chunk_documents = self._process_and_store_chunks(
            chunks,
            model_id,
            metadata_document["id"],
            schema_validator,
            text_embedder,
            chroma_manager
        )

        if chunk_documents:
            logger.info(f"Successfully processed {file_path} with model ID {metadata_document['id']}")
            return metadata_document["id"], True
        else:
            logger.warning(f"Skipped {file_path}: Failed to process model script")
            return None, False

    def _extract_and_prepare_metadata(self, metadata_extractor, parse_result, file_path, chunks):
        """Extract metadata and prepare metadata document.

        Args:
            metadata_extractor: The metadata extractor component
            parse_result: Result from code parsing
            file_path: Path to the file
            chunks: Code chunks

        Returns:
            Tuple of (model_id, metadata_document)
        """
        # Extract metadata
        metadata = metadata_extractor.extract_metadata(file_path)

        # Prepare model ID
        file_path_obj = Path(file_path)
        folder_name = file_path_obj.parent.name
        file_stem = file_path_obj.stem
        model_id = f"{folder_name}_{file_stem}"

        # Format dates
        creation_date_raw = self._clean_iso_timestamp(metadata.get("file", {}).get("creation_date", "N/A"))
        last_modified_raw = self._clean_iso_timestamp(metadata.get("file", {}).get("last_modified_date", "N/A"))

        creation_natural_month = self._format_natural_date(creation_date_raw)
        last_modified_natural_month = self._format_natural_date(last_modified_raw)

        # Extract additional metadata from LLM parse_result
        llm_fields = self._extract_llm_fields(parse_result)

        # Prepare metadata document
        metadata_document = {
            "id": f"model_metadata_{model_id}",
            "$schema_version": "1.0.0",
            "content": f"Model: {model_id}",  # Simple summary for embedding
            "metadata": {
                **metadata,
                "model_id": model_id,
                "description": llm_fields["description"],
                "framework": llm_fields["framework"],
                "architecture": llm_fields["architecture"],
                "dataset": llm_fields["dataset"],
                "training_config": llm_fields["training_config"],
                "created_at": creation_date_raw,
                "created_month": creation_natural_month,
                "created_year": creation_date_raw[:4],
                "last_modified_month": last_modified_natural_month,
                "last_modified_year": last_modified_raw[:4],
                "total_chunks": len(chunks)
            }
        }

        return model_id, metadata_document

    def _extract_llm_fields(self, parse_result):
        """Extract and format LLM parsed metadata fields from parse result.

        Args:
            parse_result: Result from code parsing

        Returns:
            Dictionary of formatted LLM parsed metadata fields
        """
        llm_fields = {
            "description": parse_result.get("description", "No description"),
            "framework": parse_result.get("framework", {}),
            "architecture": parse_result.get("architecture", {}),
            "dataset": parse_result.get("dataset", {}),
            "training_config": parse_result.get("training_config", {})
        }

        # Ensure all fields are the correct type
        if isinstance(llm_fields["framework"], str):
            llm_fields["framework"] = {"name": llm_fields["framework"], "version": "unknown"}
        if isinstance(llm_fields["architecture"], str):
            llm_fields["architecture"] = {"type": llm_fields["architecture"]}
        if isinstance(llm_fields["dataset"], str):
            llm_fields["dataset"] = {"name": llm_fields["dataset"]}
        if not isinstance(llm_fields["training_config"], dict):
            llm_fields["training_config"] = {}

        return llm_fields

    def _validate_and_store_metadata(self, schema_validator, metadata_document, text_embedder, chroma_manager,
                                     access_control):
        """Validate and store metadata document.

        Args:
            schema_validator: The schema validator component
            metadata_document: The metadata document
            text_embedder: The text embedder component
            chroma_manager: The chroma manager component
            access_control: The access control component

        Returns:
            True if successful, False otherwise
        """
        # Validate using the metadata schema
        validation_result = schema_validator.validate(metadata_document, "model_metadata_schema")
        if not validation_result["valid"]:
            logging.warning(f"Schema validation failed for metadata document: {validation_result['errors']}")
            return False

        # Add access control metadata
        access_metadata = access_control.get_document_permissions(metadata_document)
        metadata_document["metadata"]["access_control"] = access_metadata

        # Create metadata embedding content
        metadata_content = self._create_metadata_content(metadata_document)
        metadata_embedding = text_embedder.embed_mixed_content(metadata_content)

        # Store metadata document
        asyncio.run(chroma_manager.add_document(
            collection_name="model_scripts_metadata",
            document_id=metadata_document["id"],
            document=metadata_document,
            embed_content=metadata_embedding
        ))

        return True

    def _create_metadata_content(self, metadata_document):
        """Create metadata content for embedding.

        Args:
            metadata_document: The metadata document

        Returns:
            Dictionary with title and description for embedding
        """
        metadata = metadata_document["metadata"]

        return {
            "title": metadata["model_id"],
            "description": f"""
                Model created in {metadata["created_month"]}.
                Created in month: {metadata["created_month"]}.
                Created in year: {metadata["created_year"]}.
                Created on {metadata["created_at"]}.
                Last modified in {metadata["last_modified_month"]}.
                Last modified in year: {metadata["last_modified_year"]}.
                Last modified on {metadata["created_at"]}.
                Size: {metadata.get("file", {}).get('size_bytes', 'N/A')} bytes.

                Description: {metadata["description"]}.
                Framework: {metadata["framework"]}.
                Architecture: {metadata["architecture"]}.
                Dataset: {metadata["dataset"]}.
                Training config: {metadata["training_config"]}.
            """
        }

    def _process_and_store_chunks(self, chunks, model_id, metadata_doc_id, schema_validator, text_embedder,
                                  chroma_manager):
        """Process and store code chunks.

        Args:
            chunks: Code chunks
            model_id: Model ID
            metadata_doc_id: Metadata document ID
            schema_validator: The schema validator component
            text_embedder: The text embedder component
            chroma_manager: The chroma manager component

        Returns:
            List of processed chunk documents
        """
        chunk_documents = []

        for i, chunk_obj in enumerate(chunks):
            chunk_text = chunk_obj.get("text", "")
            if not isinstance(chunk_text, str):
                logging.warning(f"Invalid chunk text type for chunk {i}. Skipping.")
                continue

            chunk_document = {
                "id": f"model_chunk_{model_id}_{i}",
                "$schema_version": "1.0.0",
                "content": chunk_text,
                "metadata": {
                    "model_id": model_id,
                    "chunk_id": i,
                    "total_chunks": len(chunks),
                    "metadata_doc_id": metadata_doc_id,
                    "offset": chunk_obj.get("offset", 0),
                    "type": chunk_obj.get("type", "code"),
                }
            }

            # Validate using the chunk schema
            validation_result = schema_validator.validate(chunk_document, "model_chunk_schema")
            if not validation_result["valid"]:
                logging.warning(
                    f"Schema validation failed for chunk schema, chunk {i}: {validation_result['errors']}")
                continue

            # Create chunk embedding
            chunk_embedding = text_embedder.embed_text(chunk_text)

            # Store chunk document
            asyncio.run(chroma_manager.add_document(
                collection_name="model_scripts_chunks",
                document_id=chunk_document["id"],
                document=chunk_document,
                embed_content=chunk_embedding
            ))

            chunk_documents.append(chunk_document)

        return chunk_documents
