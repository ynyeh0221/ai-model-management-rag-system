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
            future_to_file = {executor.submit(self.process_single_script, file_path, components): file_path
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


    def process_single_script(self, file_path, components):
        """Process a single model script file.

        Args:
            file_path: Path to the model script file
            components: Dictionary containing initialized system components

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

        # 1. Parse the code to determine if it's a model script and extract relevant parts
        parse_result = code_parser.parse(file_path)
        if not parse_result or not parse_result.get("is_model_script", False):
            # Not a model script, skip it
            return None

        # 2. Extract metadata
        metadata = metadata_extractor.extract_metadata(file_path)

        # 3. Split into chunks for processing
        chunks = code_parser.split_ast_and_subsplit_chunks(file_content=parse_result["content"], file_path=file_path, chunk_size=5000, overlap=1000)

        file_path_obj = Path(file_path)
        folder_name = file_path_obj.parent.name
        file_stem = file_path_obj.stem
        model_id = f"{folder_name}_{file_stem}"

        # Create metadata document first
        creation_date_raw = self._clean_iso_timestamp(metadata.get("file", {}).get("creation_date", "N/A"))
        last_modified_raw = self._clean_iso_timestamp(metadata.get("file", {}).get("last_modified_date", "N/A"))

        def format_natural_date(iso_date: str):
            try:
                dt = datetime.fromisoformat(iso_date)
                return dt.strftime("%B")  # e.g. "April 2025"
            except Exception:
                return "Unknown"

        creation_natural_month = format_natural_date(creation_date_raw)
        last_modified_natural_month = format_natural_date(last_modified_raw)

        # Extract additional metadata from LLM parse_result
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

        # Validate using the metadata schema
        validation_result = schema_validator.validate(metadata_document, "model_metadata_schema")
        if not validation_result["valid"]:
            logging.warning(f"Schema validation failed for metadata document of {file_path}: {validation_result['errors']}")
            return None

        # Add access control metadata
        access_metadata = access_control.get_document_permissions(metadata_document)
        metadata_document["metadata"]["access_control"] = access_metadata

        # Create metadata embedding content
        metadata_content = {
            "title": model_id,
            "description": f"""
                Model created in {creation_natural_month}.
                Created in month: {creation_natural_month}.
                Created in year: {creation_date_raw[:4]}.
                Created on {creation_date_raw}.
                Last modified in {last_modified_natural_month}.
                Last modified in year: {last_modified_raw[:4]}.
                Last modified on {last_modified_raw}.
                Size: {metadata.get("file", {}).get('size_bytes', 'N/A')} bytes.
    
                Description: {llm_fields["description"]}.
                Framework: {llm_fields["framework"]}.
                Architecture: {llm_fields["architecture"]}.
                Dataset: {llm_fields["dataset"]}.
                Training config: {llm_fields["training_config"]}.
            """
        }

        metadata_embedding = text_embedder.embed_mixed_content(metadata_content)

        # Store metadata document
        asyncio.run(chroma_manager.add_document(
            collection_name="model_scripts_metadata",
            document_id=metadata_document["id"],
            document=metadata_document,
            embed_content=metadata_embedding
        ))

        # Process and store code chunks
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
                    "metadata_doc_id": metadata_document["id"],
                    "offset": chunk_obj.get("offset", 0),
                    "type": chunk_obj.get("type", "code"),
                }
            }

            # Validate using the chunk schema
            validation_result = schema_validator.validate(chunk_document, "model_chunk_schema")
            if not validation_result["valid"]:
                logging.warning(
                    f"Schema validation failed for chunk schema of {file_path}, chunk {i}: {validation_result['errors']}")
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

        return (metadata_document["id"], True) if chunk_documents else (None, False)