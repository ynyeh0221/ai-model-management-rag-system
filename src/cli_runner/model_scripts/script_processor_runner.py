import glob
import asyncio
import logging
import concurrent.futures
import os
from pathlib import Path
from datetime import datetime

from document_processor.llm_based_code_parser import split_code_chunks_via_ast


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
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future_to_file = {executor.submit(self.process_single_script, components, file_path, logger): file_path
                              for file_path in script_files}

            for future in concurrent.futures.as_completed(future_to_file):
                file_path = future_to_file[future]
                processed_count += 1
                try:
                    result = future.result()
                    if result:
                        model_id, success = result
                        logger.info(
                            f"Processed {file_path} with ID {model_id}: {'Success' if success else 'Failed'}")
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
            Tuple of (model_id, success) if processed, None if skipped
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
        chunks = split_code_chunks_via_ast(
            file_content=parse_result["content"],
            file_path=file_path,
            chunk_size=20000,
            overlap=1000
        )

        # Extract metadata and prepare metadata documents for different tables
        model_id, metadata_documents = self._extract_and_prepare_metadata(
            metadata_extractor,
            parse_result,
            file_path
        )

        # Validate and store metadata documents in different tables
        metadata_stored = self._validate_and_store_metadata_documents(
            schema_validator,
            metadata_documents,
            text_embedder,
            chroma_manager,
            access_control
        )

        if not metadata_stored:
            logger.warning(f"Failed to store metadata for {file_path}")
            return None, False

        chunk_documents = self._process_and_store_chunks(
            chunks,
            model_id,
            metadata_documents["model_date"]["id"],
            schema_validator,
            text_embedder,
            chroma_manager
        )

        if chunk_documents:
            logger.info(f"Successfully processed {file_path} with model ID {model_id}")
            return model_id, True
        else:
            logger.warning(f"Skipped {file_path}: Failed to process model script")
            return None, False

    def _extract_and_prepare_metadata(self, metadata_extractor, parse_result, file_path):
        """Extract metadata and prepare metadata documents for different tables.

        Args:
            metadata_extractor: The metadata extractor component
            parse_result: Result from code parsing
            file_path: Path to the file

        Returns:
            Tuple of (model_id, metadata_documents_dict)
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

        # Extract other fields that might be in the metadata
        offset = metadata.get("offset", -999)

        # Get chunk descriptions from parse_result
        chunk_descriptions = parse_result.get("chunk_descriptions", [])

        # Create individual model_descriptions documents for each chunk
        model_descriptions_documents = {}
        for i, description in enumerate(chunk_descriptions):
            doc_id = f"model_descriptions_{model_id}_chunk_{i}"
            model_descriptions_documents[doc_id] = {
                "id": doc_id,
                "$schema_version": "1.0.0",
                "content": f"Description for chunk {i} of model {model_id}",
                "metadata": {
                    "model_id": model_id,
                    "chunk_id": i,
                    "total_chunks": len(chunk_descriptions),
                    "offset": offset,
                    "description": description
                }
            }

        # Prepare metadata documents for different tables
        metadata_documents = {
            # 1. Model file information
            "model_file": {
                "id": f"model_file_{model_id}",
                "$schema_version": "1.0.0",
                "content": f"Model file: {model_id}",
                "metadata": {
                    "model_id": model_id,
                    "file": metadata.get("file", {})
                }
            },

            # 2. Model date information
            "model_date": {
                "id": f"model_date_{model_id}",
                "$schema_version": "1.0.0",
                "content": f"Model date: {model_id}",
                "metadata": {
                    "model_id": model_id,
                    "created_at": creation_date_raw,
                    "created_month": creation_natural_month,
                    "created_year": creation_date_raw[:4] if creation_date_raw != "N/A" else "N/A",
                    "last_modified_month": last_modified_natural_month,
                    "last_modified_year": last_modified_raw[:4] if last_modified_raw != "N/A" else "N/A"
                }
            },

            # 3. Model git information
            "model_git": {
                "id": f"model_git_{model_id}",
                "$schema_version": "1.0.0",
                "content": f"Model git: {model_id}",
                "metadata": {
                    "model_id": model_id,
                    "git": metadata.get("git", {})
                }
            },

            # 4. Model frameworks information
            "model_frameworks": {
                "id": f"model_frameworks_{model_id}",
                "$schema_version": "1.0.0",
                "content": f"Model frameworks: {model_id}",
                "metadata": {
                    "model_id": model_id,
                    "framework": llm_fields["framework"]
                }
            },

            # 5. Model datasets information
            "model_datasets": {
                "id": f"model_datasets_{model_id}",
                "$schema_version": "1.0.0",
                "content": f"Model datasets: {model_id}",
                "metadata": {
                    "model_id": model_id,
                    "dataset": llm_fields["dataset"]
                }
            },

            # 6. Model training configs information
            "model_training_configs": {
                "id": f"model_training_configs_{model_id}",
                "$schema_version": "1.0.0",
                "content": f"Model training configs: {model_id}",
                "metadata": {
                    "model_id": model_id,
                    "training_config": llm_fields["training_config"]
                }
            },

            # 7. Model architectures information
            "model_architectures": {
                "id": f"model_architectures_{model_id}",
                "$schema_version": "1.0.0",
                "content": f"Model architectures: {model_id}",
                "metadata": {
                    "model_id": model_id,
                    "architecture": llm_fields["architecture"]
                }
            },

            # 8. Model descriptions information
            "model_descriptions": [],  # List of chunk descriptions

            # 9. Model AST summaries information
            "model_ast_summaries": {
                "id": f"model_ast_summaries_{model_id}",
                "$schema_version": "1.0.0",
                "content": f"Model AST summaries: {model_id}",
                "metadata": {
                    "model_id": model_id,
                    "ast_summary": llm_fields["ast_summary"]
                }
            },

            # 10. Model images path information
            "model_images_folder": {
                "id": f"model_images_folder_{model_id}",
                "$schema_version": "1.0.0",
                "content": f"Model images folder: {model_id}",
                "metadata": {
                    "model_id": model_id,
                    "images_folder": llm_fields["images_folder"]
                }
            },
        }

        # Add all model_descriptions documents
        for i, description in enumerate(chunk_descriptions):
            doc_id = f"model_descriptions_{model_id}_chunk_{i}"
            metadata_documents["model_descriptions"].append({
                "id": doc_id,
                "$schema_version": "1.0.0",
                "content": f"Description for chunk {i} of model {model_id}",
                "metadata": {
                    "model_id": model_id,
                    "chunk_id": i,
                    "total_chunks": len(chunk_descriptions),
                    "offset": offset,
                    "description": description or "No description available"
                }
            })

        return model_id, metadata_documents

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
            "training_config": parse_result.get("training_config", {}),
            "ast_summary": parse_result.get("ast_summary", {}),
            "images_folder": parse_result.get("images_folder", {})
        }

        # Ensure all fields are the correct type
        if isinstance(llm_fields["framework"], str):
            llm_fields["framework"] = {"name": llm_fields["framework"], "version": "unknown"}
        if not isinstance(llm_fields["architecture"], dict):
            llm_fields["architecture"] = {}
        if isinstance(llm_fields["dataset"], str):
            llm_fields["dataset"] = {"name": llm_fields["dataset"]}
        if not isinstance(llm_fields["training_config"], dict):
            llm_fields["training_config"] = {}
        if isinstance(llm_fields["ast_summary"], str):
            llm_fields["ast_summary"] = {"ast_summary": llm_fields["ast_summary"]}
        if isinstance(llm_fields["images_folder"], str):
            llm_fields["images_folder"] = {"name": llm_fields["images_folder"]}

        return llm_fields

    def _validate_and_store_metadata_documents(self, schema_validator, metadata_documents, text_embedder,
                                               chroma_manager, access_control):
        """Validate and store metadata documents in different tables."""
        critical_failure = False
        critical_collections = ["model_file", "model_architectures", "model_frameworks", "model_descriptions", "model_ast_summaries", "model_images_folder"]

        for key, value in metadata_documents.items():
            documents = value if isinstance(value, list) else [value]

            for document in documents:
                doc_id = document.get("id", "")

                collection_name = key
                schema_id = f"{key}_schema"

                if not collection_name or not schema_id:
                    logging.warning(
                        f"Could not determine collection name or schema ID for document {doc_id}. Skipping.")
                    continue

                try:
                    validation_result = schema_validator.validate(document, schema_id)
                    if not validation_result["valid"]:
                        logging.warning(
                            f"Schema validation failed for {schema_id} document: {validation_result['errors']}")
                        if collection_name in critical_collections:
                            critical_failure = True
                        continue
                except ValueError as e:
                    logging.error(f"Schema validation error for {doc_id}: {str(e)}")
                    if collection_name in critical_collections:
                        critical_failure = True
                    continue

                access_metadata = access_control.get_document_permissions(document)
                document["metadata"]["access_control"] = access_metadata

                try:
                    embedding_content = self._create_metadata_content_for_type(collection_name, document)
                    document_embedding = text_embedder.embed_mixed_content(embedding_content)
                except Exception as e:
                    logging.error(f"Error creating embedding for document {doc_id}: {str(e)}")
                    if collection_name in critical_collections:
                        critical_failure = True
                    continue

                try:
                    asyncio.run(chroma_manager.add_document(
                        collection_name=collection_name,
                        document_id=document["id"],
                        document=document,
                        embed_content=document_embedding
                    ))
                except Exception as e:
                    logging.error(f"Error storing document {doc_id} in collection {collection_name}: {str(e)}")
                    if collection_name in critical_collections:
                        critical_failure = True

        return not critical_failure

    def _create_metadata_content_for_type(self, doc_type, document):
        """Create metadata content for embedding based on document type.

        Args:
            doc_type: Type of the document (e.g., 'model_file', 'model_date')
            document: The metadata document

        Returns:
            Dictionary with title and description for embedding
        """
        metadata = document["metadata"]
        model_id = metadata["model_id"]

        # Create type-specific embedding content
        if doc_type == "model_file":
            file_info = metadata.get("file", {})
            return {
                "title": f"File information for {model_id}",
                "description": f"""
                    Model file size: {file_info.get('size_bytes', 'N/A')} bytes.
                    File extension: {file_info.get('file_extension', 'N/A')}.
                    Path: {file_info.get('absolute_path', 'N/A')}.
                    Creation date: {file_info.get('creation_date', 'N/A')}.
                    Last modified date: {file_info.get('last_modified_date', 'N/A')}.
                """
            }

        elif doc_type == "model_date":
            return {
                "title": f"Date information for {model_id}",
                "description": f"""
                    Model created in {metadata.get('created_month', 'N/A')}.
                    Created in year: {metadata.get('created_year', 'N/A')}.
                    Created on {metadata.get('created_at', 'N/A')}.
                    Last modified in {metadata.get('last_modified_month', 'N/A')}.
                    Last modified in year: {metadata.get('last_modified_year', 'N/A')}.
                """
            }

        elif doc_type == "model_git":
            git_info = metadata.get("git", {})
            return {
                "title": f"Git information for {model_id}",
                "description": f"""
                    Git creation date: {git_info.get('creation_date', 'N/A')}.
                    Git last modified date: {git_info.get('last_modified_date', 'N/A')}.
                    Commit count: {git_info.get('commit_count', 'N/A')}.
                """
            }

        elif doc_type == "model_frameworks":
            framework = metadata.get("framework", {})
            return {
                "title": f"Framework information for {model_id}",
                "description": f"""
                    Framework name: {framework.get('name', 'N/A')}.
                    Framework version: {framework.get('version', 'N/A')}.
                """
            }

        elif doc_type == "model_datasets":
            dataset = metadata.get("dataset", {})
            return {
                "title": f"Dataset information for {model_id}",
                "description": f"""
                    Dataset name: {dataset.get('name', 'N/A')}.
                """
            }

        elif doc_type == "model_training_configs":
            training_config = metadata.get("training_config", {})
            return {
                "title": f"Training configuration for {model_id}",
                "description": f"""
                    Batch size: {training_config.get('batch_size', 'N/A')}.
                    Optimizer: {training_config.get('optimizer', 'N/A')}.
                    Epochs: {training_config.get('epochs', 'N/A')}.
                    Learning rate: {training_config.get('learning_rate', 'N/A')}.
                    Hardware used: {training_config.get('hardware_used', 'N/A')}.
                    Associated configs: {metadata.get('associated_configs', '[]')}.
                """
            }

        elif doc_type == "model_architectures":
            architecture = metadata.get("architecture", {})
            return {
                "title": f"Architecture information for {model_id}",
                "description": f"""
                    Architecture type: {architecture.get('type', 'N/A')}.
                    Architecture reason: {architecture.get('reason', 'N/A')}.
                """
            }

        elif doc_type == "model_descriptions":
            # For individual chunk descriptions
            chunk_id = metadata.get("chunk_id", "N/A")
            description = metadata.get("description", "No description available")

            return {
                "title": f"Description for chunk {chunk_id} of model {model_id}",
                "description": f"""
                    Model: {model_id}
                    Chunk: {chunk_id} of {metadata.get('total_chunks', 0)}
                    Description: {description}
                    Offset: {metadata.get('offset', -999)}.
                """
            }

        elif doc_type == "model_ast_summary":
            ast_summary = metadata.get("ast_summary", "No ast_summary available")

            return {
                "title": f"AST summary information for {model_id}",
                "description": f"""
                                AST summary type: {ast_summary.get('type', 'N/A')}.
                            """
            }

        elif doc_type == "model_images_folder":
            images_folder = metadata.get("images_folder", {})
            return {
                "title": f"Images path information for {model_id}",
                "description": f"""
                    Images path: {images_folder.get('path', 'N/A')}.
                """
            }

        # Default case
        return {
            "title": model_id,
            "description": f"Metadata for {model_id} of type {doc_type}"
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