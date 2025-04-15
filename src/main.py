# main.py
import argparse
import asyncio
import concurrent.futures
import glob
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import nbformat
from nbformat.v4 import new_notebook, new_code_cell
from prettytable import PrettyTable

from src.colab_generator.code_generator import CodeGenerator
from src.colab_generator.colab_api_client import ColabAPIClient
from src.colab_generator.reproducibility_manager import ReproducibilityManager
from src.colab_generator.resource_quota_manager import ResourceQuotaManager
from src.document_processor.code_parser import CodeParser
from src.document_processor.image_processor import ImageProcessor
from src.document_processor.metadata_extractor import MetadataExtractor
from src.document_processor.schema_validator import SchemaValidator
from src.query_engine.query_analytics import QueryAnalytics
from src.query_engine.query_parser import QueryParser
from src.query_engine.result_reranker import CrossEncoderReranker
from src.query_engine.search_dispatcher import SearchDispatcher
from src.response_generator.llm_interface import LLMInterface
from src.response_generator.prompt_visualizer import PromptVisualizer
from src.response_generator.template_manager import TemplateManager
from src.vector_db_manager.access_control import AccessControlManager
from src.vector_db_manager.chroma_manager import ChromaManager
from src.vector_db_manager.image_embedder import ImageEmbedder
from src.vector_db_manager.text_embedder import TextEmbedder


def initialize_components(config_path="./config"):
    """Initialize all components of the RAG system."""

    llm_interface = LLMInterface(model_name="deepseek-r1:7b", timeout=20000)

    # Initialize document processor components
    schema_validator = SchemaValidator(os.path.join(config_path, "schema_registry.json"))
    code_parser = CodeParser(schema_validator=schema_validator, llm_interface=llm_interface)
    metadata_extractor = MetadataExtractor()
    image_processor = ImageProcessor(schema_validator)
    
    # Initialize vector database components
    text_embedder = TextEmbedder()
    image_embedder = ImageEmbedder()
    chroma_manager = ChromaManager(text_embedder, image_embedder, "./chroma_db")
    access_control = AccessControlManager(chroma_manager)
    
    # Initialize query engine components
    query_parser = QueryParser(llm_model_name="deepseek-r1:7b")
    search_dispatcher = SearchDispatcher(chroma_manager, text_embedder, image_embedder)
    query_analytics = QueryAnalytics()
    result_reranker = CrossEncoderReranker(device="mps")

    # Initialize response generator components
    template_manager = TemplateManager("./templates")
    prompt_visualizer = PromptVisualizer(template_manager)
    
    # Initialize Colab notebook generator components
    code_generator = CodeGenerator()
    colab_api_client = ColabAPIClient()
    reproducibility_manager = ReproducibilityManager()
    resource_quota_manager = ResourceQuotaManager()

    return {
        "document_processor": {
            "schema_validator": schema_validator,
            "code_parser": code_parser,
            "metadata_extractor": metadata_extractor,
            "image_processor": image_processor
        },
        "vector_db_manager": {
            "text_embedder": text_embedder,
            "image_embedder": image_embedder,
            "chroma_manager": chroma_manager,
            "access_control": access_control
        },
        "query_engine": {
            "query_parser": query_parser,
            "search_dispatcher": search_dispatcher,
            "query_analytics": query_analytics,
            "reranker": result_reranker
        },
        "response_generator": {
            "llm_interface": llm_interface,
            "template_manager": template_manager,
            "prompt_visualizer": prompt_visualizer
        },
        "colab_generator": {
            "code_generator": code_generator,
            "colab_api_client": colab_api_client,
            "reproducibility_manager": reproducibility_manager,
            "resource_quota_manager": resource_quota_manager
        }
    }

def clean_iso_timestamp(ts: str) -> str:
    """Remove microseconds from ISO format like 2025-03-02T10:53:24.620782 -> 2025-03-02T10:53:24"""
    try:
        dt = datetime.fromisoformat(ts)
        return dt.replace(microsecond=0).isoformat()
    except Exception:
        return ts  # fallback to original if parsing fails

def process_model_scripts(components, directory_path):
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

    # Process files in parallel using a thread pool
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_file = {executor.submit(process_single_script, file_path, components): file_path
                          for file_path in script_files}

        for future in concurrent.futures.as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                result = future.result()
                if result:
                    document_id, success = result
                    logger.info(f"Processed {file_path} with ID {document_id}: {'Success' if success else 'Failed'}")
                else:
                    logger.warning(f"Skipped {file_path}: Not a model script")
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")

    logger.info("Model script processing completed")


def process_single_script(file_path, components):
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
    chunks = code_parser.split_ast_and_subsplit_chunks(parse_result["content"], chunk_size=500, overlap=100)

    file_path_obj = Path(file_path)
    folder_name = file_path_obj.parent.name
    file_stem = file_path_obj.stem
    model_id = f"{folder_name}_{file_stem}"

    # Create metadata document first
    creation_date_raw = clean_iso_timestamp(metadata.get("file", {}).get("creation_date", "N/A"))
    last_modified_raw = clean_iso_timestamp(metadata.get("file", {}).get("last_modified_date", "N/A"))

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

    # Create metadata embedding

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
        if isinstance(chunk_obj, dict):
            chunk_text = chunk_obj.get("text", "")
            chunk_metadata = {k: v for k, v in chunk_obj.items() if k != "text"}
        else:
            chunk_text = chunk_obj
            chunk_metadata = {}

        chunk_document = {
            "id": f"model_chunk_{model_id}_{i}",
            "$schema_version": "1.0.0",
            "content": chunk_text,
            "metadata": {
                **chunk_metadata,  # <- store offset, type, etc.
                "model_id": model_id,
                "chunk_id": i,
                "total_chunks": len(chunks),
                "metadata_doc_id": metadata_document["id"]  # Reference to metadata document
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

def process_images(components, directory_path):
    """Process images in a directory.
    
    This function walks through the directory to find image files,
    processes them using the image processor component, generates
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
    
    logger.info(f"Found {len(image_files)} image files")
    
    # Process files in parallel using a thread pool
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_file = {executor.submit(process_single_image, 
                                          file_path, 
                                          components): file_path for file_path in image_files}
        
        for future in concurrent.futures.as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                result = future.result()
                if result:
                    document_id, success = result
                    logger.info(f"Processed {file_path} with ID {document_id}: {'Success' if success else 'Failed'}")
                else:
                    logger.warning(f"Skipped {file_path}: Not a valid image")
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
    
    logger.info("Image processing completed")


def process_single_image(file_path, components):
    """Process a single image file.

    Args:
        file_path: Path to the image file
        components: Dictionary containing initialized system components

    Returns:
        Tuple of (document_id, success) if processed, None if skipped
    """
    # Extract required components
    image_processor = components["document_processor"]["image_processor"]
    schema_validator = components["document_processor"]["schema_validator"]
    image_embedder = components["vector_db_manager"]["image_embedder"]
    chroma_manager = components["vector_db_manager"]["chroma_manager"]
    access_control = components["vector_db_manager"]["access_control"]

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

    return (document_id, True)

def display_models_pretty(available_models):
    table = PrettyTable()
    table.field_names = ["Model ID", "Created", "Last Modified", "Absolute Path"]

    # Align all columns to the left
    table.align["Model ID"] = "l"
    table.align["Created"] = "l"
    table.align["Last Modified"] = "l"
    table.align["Absolute Path"] = "l"

    # Sort models by creation date in descending order
    sorted_models = sorted(
        available_models,
        key=lambda m: datetime.fromisoformat(m['creation_date']),
        reverse=True
    )

    for model in sorted_models:
        model_id = model['model_id']
        created = datetime.fromisoformat(model['creation_date']).strftime("%Y-%m-%dT%H:%M")
        modified = datetime.fromisoformat(model['last_modified_date']).strftime("%Y-%m-%dT%H:%M")
        absolute_path = model['absolute_path']

        # Truncate if needed
        if len(model_id) > 50:
            model_id = model_id[:47] + "..."
        if len(absolute_path) > 100:
            absolute_path = absolute_path[:97] + "..."

        table.add_row([model_id, created, modified, absolute_path])

    print(table)

def start_ui(components, host="localhost", port=8000):
    """Start a command-line interface for the RAG system.
    
    This function provides a simple command-line interface to interact with
    the RAG system components. The web UI will be developed later.
    
    Args:
        components: Dictionary containing initialized system components
        host: Not used in CLI mode, kept for API compatibility
        port: Not used in CLI mode, kept for API compatibility
    """
    print(f"Starting command-line interface (Web UI will be developed later)...")
    
    # Extract required components
    query_parser = components["query_engine"]["query_parser"]
    search_dispatcher = components["query_engine"]["search_dispatcher"]
    query_analytics = components["query_engine"]["query_analytics"]
    reranker = components["query_engine"]["reranker"]
    llm_interface = components["response_generator"]["llm_interface"]
    template_manager = components["response_generator"]["template_manager"]
    access_control = components["vector_db_manager"]["access_control"]
    
    # Define available commands
    commands = {
        "query": "Search for model scripts or images",
        "list-models": "List available models",
        "list-images": "List available images",
        "compare-models": "Compare two or more models",
        "generate-notebook": "Generate a Colab notebook for a model",
        "help": "Show available commands",
        "exit": "Exit the program"
    }
    
    # Display welcome message
    print("\nAI Model Management RAG System - Command Line Interface")
    print("=" * 50)
    print("Type 'help' to see available commands or 'exit' to quit.")
    print("=" * 50)
    
    # Simple user authentication
    user_id = input("Enter your user ID (default: anonymous): ") or "anonymous"
    
    # Command loop
    while True:
        try:
            print("\n")
            cmd = input("> ").strip()
            
            if cmd.lower() == "exit":
                print("Exiting. Goodbye!")
                break
                
            elif cmd.lower() == "help":
                print("\nAvailable commands:")
                for cmd_name, cmd_desc in commands.items():
                    print(f"  {cmd_name:<15} - {cmd_desc}")

            elif cmd.lower() == "list-models":
                # Get models the user has access to
                available_models = access_control.get_accessible_models(user_id)
                print("\nAvailable models:")
                display_models_pretty(available_models)

            elif cmd.lower() == "list-images":
                # Get images the user has access to
                available_images = access_control.get_accessible_images(user_id)

                print("\nAvailable images:")
                if not available_images:
                    print("  No images available")
                else:
                    table = PrettyTable()
                    table.field_names = ["Image ID", "Prompt", "File Path"]
                    # Align all columns to the left
                    table.align["Model ID"] = "l"
                    table.align["Prompt"] = "l"
                    table.align["File Path"] = "l"

                    for image in available_images:
                        image_id = image['id']
                        prompt = image.get('prompt', 'No prompt')
                        filepath = image.get('filepath', 'No path')

                        # Optional: truncate long values
                        if len(prompt) > 40:
                            prompt = prompt[:37] + "..."
                        if len(filepath) > 50:
                            filepath = filepath[:47] + "..."
                        table.add_row([image_id, prompt, filepath])
                    print(table)

            elif cmd.lower().startswith("query"):
                if cmd.lower() == "query":
                    query_text = input("Enter your query: ")
                else:
                    query_text = cmd[6:].strip()

                # Parse the query
                parsed_query = query_parser.parse_query(query_text)
                print(f"Parsed query: {parsed_query}")

                # Log the query for analytics
                query_analytics.log_query(user_id, query_text, parsed_query)

                # Check access permissions
                # if not access_control.verify_user_permissions(user_id, parsed_query):
                #    print("Access denied. You don't have permission to access this information.")
                #    continue

                print("Searching...")

                # Dispatch the query
                search_results = asyncio.run(search_dispatcher.dispatch(
                    query=parsed_query["processed_query"] if "processed_query" in parsed_query else query_text,
                    intent=parsed_query["intent"],
                    parameters=parsed_query["parameters"],
                    user_id=user_id
                ))

                # Get reranking parameters from config
                max_to_return = 10  # Default max to return
                rerank_threshold = 0.3  # Default threshold

                print(f"Search results: {search_results}")
                # Rank the results and filter out low-similarities
                if isinstance(search_results, dict) and 'items' in search_results:
                    # Extract the items from the search results
                    items_to_rerank = search_results['items']

                    # Add a content field if it doesn't exist (reranker might require this)
                    for item in items_to_rerank:
                        if 'content' not in item:
                            # Use some meaningful field as content, like model_id or metadata description
                            item['content'] = item.get('model_id', '') + ': ' + item.get('metadata', {}).get(
                                'description', 'No description')

                    if reranker and items_to_rerank:
                        print(f"Sending {len(items_to_rerank)} items to reranker")
                        reranked_results = reranker.rerank(
                            query=parsed_query.get("processed_query", query_text),
                            results=items_to_rerank,
                            top_k=max_to_return,
                            threshold=rerank_threshold
                        )
                    else:
                        reranked_results = items_to_rerank
                else:
                    reranked_results = []
                print(f"Reranked results: {reranked_results}")

                # Select template based on query type
                if parsed_query["type"] == "comparison":
                    template_id = "model_comparison"
                elif parsed_query["type"] == "retrieval":
                    template_id = "information_retrieval"
                else:
                    template_id = "general_query"

                print("Generating response...")

                def prepare_template_context(query_text, results, parsed_query):
                    """
                    Prepare the context for the template with proper model information.

                    Args:
                        query_text: The original query text
                        results: List of simplified search results
                        parsed_query: The parsed query information

                    Returns:
                        Dictionary containing the template context
                    """
                    def parse_nested_json(metadata, fields):
                        for field in fields:
                            raw_value = metadata.get(field)

                            if isinstance(raw_value, str):
                                try:
                                    parsed = json.loads(raw_value)
                                    metadata[field] = parsed if isinstance(parsed, dict) else {}
                                except json.JSONDecodeError:
                                    metadata[field] = {}
                        return metadata

                    def normalize_values(metadata):
                        for key, value in metadata.items():
                            if isinstance(value, str) and value.strip().lower() in ["n/a", "null", "none"]:
                                metadata[key] = None
                        return metadata

                    def preprocess_model(model):
                        metadata = model.get("metadata", {})

                        # Parse specific fields
                        metadata = parse_nested_json(metadata,
                                                     ["architecture", "dataset", "framework", "training_config", "file",
                                                      "git"])

                        # Normalize values like "N/A"
                        metadata = normalize_values(metadata)
                        model["metadata"] = metadata
                        return model

                    # Apply preprocessing to each result
                    cleaned_results = [preprocess_model(model) for model in results]

                    # Build context
                    context = {
                        "intent": parsed_query.get("intent", "unknown"),
                        "query": query_text,
                        "results": cleaned_results,
                        "parsed_query": parsed_query,
                        "timeframe": parsed_query.get("parameters", {}).get("filters", {}).get("created_month", None),
                    }

                    return context

                # Usage
                context = prepare_template_context(query_text, reranked_results, parsed_query)
                print(f"context: {context}")

                try:
                    # First, try to render the template with the context
                    rendered_prompt = template_manager.render_template(template_id, context)
                    print(f"prompt: {rendered_prompt}")
                    def print_llm_content(response):
                        """
                        Extract and print content from an LLM response in various formats.

                        Args:
                            response: The response from the LLM, which could be a string, dict, or list
                        """

                        try:
                            # Handle dict case
                            if isinstance(response, dict):
                                if "content" in response:
                                    print(response["content"])
                                elif "text" in response:
                                    print(response["text"])
                                elif "response" in response:
                                    print(response["response"])
                                elif "answer" in response:
                                    print(response["answer"])
                                elif "message" in response:
                                    # Check for OpenAI format
                                    if "content" in response["message"]:
                                        print(response["message"]["content"])
                                    else:
                                        print(response["message"])

                                else:
                                    print("No recognizable content field found in the dictionary response")
                                    print(f"Available fields: {list(response.keys())}")

                                    # Try to print the full dictionary if it's not too large
                                    if len(str(response)) < 1000:
                                        print("Response content:")
                                        print(response)

                            # Handle str case
                            elif isinstance(response, str):
                                # Try to parse as JSON first
                                try:
                                    import json
                                    parsed = json.loads(response)

                                    if isinstance(parsed, dict):
                                        # Recursively call with the parsed dict
                                        print_llm_content(parsed)
                                    else:
                                        # Just print the string as is
                                        print(response)
                                except json.JSONDecodeError:
                                    # Not JSON, print as is
                                    print(response)

                            # Handle list case
                            elif isinstance(response, list):
                                if response and len(response) > 0:
                                    # Try the first element
                                    first_elem = response[0]
                                    if isinstance(first_elem, dict):
                                        # Recursively call with the first dict
                                        print_llm_content(first_elem)
                                    else:
                                        print(first_elem)
                                else:
                                    print("Empty list response")

                            else:
                                print(f"Unsupported response type: {type(response)}")
                                # Try to print it anyway
                                print(str(response))

                        except Exception as e:
                            print(f"Error extracting content: {e}")
                            # Try to print the raw response
                            try:
                                print("Raw response:")
                                print(str(response)[:500])  # Print first 500 chars at most
                            except:
                                print("Could not print raw response")

                    # Usage
                    llm_response = llm_interface.generate_response(
                        prompt=rendered_prompt,
                        temperature=0.5,
                        max_tokens=31000
                    )

                    print("Printing LLM Response...")
                    print_llm_content(llm_response)

                except Exception as e:
                    print(f"Error generating response: {str(e)}")
                    print("Falling back to direct LLM call...")

                    # Fallback: Create a simple prompt without using the template system
                    fallback_prompt = f"Query: {query_text}\n\nResults:\n"
                    for i, r in enumerate(reranked_results):
                        fallback_prompt += f"{i + 1}. {r['id']}: {r['content']}\n"
                    fallback_prompt += "\nPlease provide a comprehensive response to the query based on these results."

                    try:
                        # Try again with the fallback prompt
                        fallback_response = llm_interface.generate_response(
                            prompt=fallback_prompt,
                            temperature=0.5,
                            max_tokens=31000
                        )
                        print("\nFallback Response:")
                        print(fallback_response)

                    except Exception as e2:
                        print(f"Fallback also failed: {str(e2)}")
                        print("Could not generate LLM response. Check your connection and service status.")

            elif cmd.lower().startswith("compare-models"):
                if cmd.lower() == "compare-models":
                    models_to_compare = input("Enter model IDs to compare (comma separated): ").split(",")
                else:
                    models_to_compare = cmd[14:].strip().split(",")
                
                models_to_compare = [model.strip() for model in models_to_compare]
                
                if len(models_to_compare) < 2:
                    print("Please specify at least two models to compare.")
                    continue
                
                # Check access permissions for all models
                all_accessible = True
                # for model_id in models_to_compare:
                    # if not access_control.verify_model_access(user_id, model_id):
                        # print(f"Access denied for model: {model_id}")
                        # all_accessible = False
                        # break
                
                if not all_accessible:
                    continue
                
                print(f"Comparing models: {', '.join(models_to_compare)}...")
                
                # Here you would implement the model comparison logic
                # For now, just a placeholder
                print("Model comparison functionality will be implemented in a future update.")

            elif cmd.lower().startswith("generate-notebook"):
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
                    else:
                        notebook_type = "evaluation"

                    if len(parts) > 3:
                        if parts[3].startswith("--output=") or parts[3].startswith("-o="):
                            output_path = parts[3].split("=", 1)[1]
                        else:
                            output_path = parts[3]
                    else:
                        output_path = f"./notebooks/{model_id}_{notebook_type}.ipynb"

                # Check access permissions
                # if not access_control.verify_model_access(user_id, model_id):
                #    print(f"Access denied for model: {model_id}")
                #    continue

                result = generate_notebook(components, model_id, output_path)

                if result:
                    print(f"Notebook generated successfully: {result}")
                else:
                    print("Failed to generate notebook")
                
            else:
                print(f"Unknown command: {cmd}")
                print("Type 'help' to see available commands.")
                
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
        except Exception as e:
            print(f"Error: {str(e)}")
    
    print("UI session ended.")


def generate_notebook(components, model_id, output_path):
    """Generate a Colab notebook for model analysis using full script reconstruction."""
    print(f"Generating notebook for model {model_id}...")

    # Extract components
    code_generator = components["colab_generator"]["code_generator"]
    reproducibility_manager = components["colab_generator"]["reproducibility_manager"]
    chroma_manager = components["vector_db_manager"]["chroma_manager"]

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("notebook_generator")

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

    model_metadata = metadata_results["results"][0].get("metadata", {})

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

    # Reconstruct full code
    full_script = code_generator.generate_full_script(chunk_contents, overlap=100, use_offset=True)

    logger.info(f"Found {len(chunk_contents)} chunks. Reconstructing full script...")

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

def main():
    parser = argparse.ArgumentParser(description="AI Model Management RAG System")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Process model scripts command
    process_scripts_parser = subparsers.add_parser("process-scripts", help="Process model scripts")
    process_scripts_parser.add_argument("directory", help="Directory containing model scripts")

    # Process images command
    process_images_parser = subparsers.add_parser("process-images", help="Process images")
    process_images_parser.add_argument("directory", help="Directory containing images")

    # Start UI command
    ui_parser = subparsers.add_parser("start-ui", help="Start the user interface")
    ui_parser.add_argument("--host", default="localhost", help="Host to bind the UI to")
    ui_parser.add_argument("--port", type=int, default=8000, help="Port to bind the UI to")

    args = parser.parse_args()

    # Initialize components
    components = initialize_components()

    # Execute command
    if args.command == "process-scripts":
        process_model_scripts(components, args.directory)
    elif args.command == "process-images":
        process_images(components, args.directory)
    elif args.command == "start-ui":
        start_ui(components, args.host, args.port)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
