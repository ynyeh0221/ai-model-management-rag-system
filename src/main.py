# main.py
import asyncio
import argparse
import os
import glob
import logging
import concurrent.futures
from pathlib import Path

from src.document_processor.code_parser import CodeParser
from src.document_processor.metadata_extractor import MetadataExtractor
from src.document_processor.image_processor import ImageProcessor
from src.document_processor.schema_validator import SchemaValidator

from src.vector_db_manager.text_embedder import TextEmbedder
from src.vector_db_manager.image_embedder import ImageEmbedder
from src.vector_db_manager.chroma_manager import ChromaManager
from src.vector_db_manager.access_control import AccessControlManager

from src.query_engine.query_parser import QueryParser
from src.query_engine.search_dispatcher import SearchDispatcher
from src.query_engine.result_ranker import ResultRanker
from src.query_engine.query_analytics import QueryAnalytics

from src.response_generator.llm_interface import LLMInterface
from src.response_generator.template_manager import TemplateManager
from src.response_generator.response_formatter import ResponseFormatter
from src.response_generator.prompt_visualizer import PromptVisualizer

# from src.colab_generator.template_engine import NotebookTemplateEngine
# from src.colab_generator.code_generator import CodeGenerator
# from src.colab_generator.colab_api_client import ColabAPIClient
# from src.colab_generator.reproducibility_manager import ReproducibilityManager
# from src.colab_generator.resource_quota_manager import ResourceQuotaManager

def initialize_components(config_path="./config"):
    """Initialize all components of the RAG system."""
    # Initialize document processor components
    schema_validator = SchemaValidator(os.path.join(config_path, "schema_registry"))
    code_parser = CodeParser(schema_validator)
    metadata_extractor = MetadataExtractor()
    image_processor = ImageProcessor(schema_validator)
    
    # Initialize vector database components
    text_embedder = TextEmbedder()
    image_embedder = ImageEmbedder()
    chroma_manager = ChromaManager("./chroma_db")
    access_control = AccessControlManager()
    
    # Initialize query engine components
    query_parser = QueryParser()
    search_dispatcher = SearchDispatcher(chroma_manager, text_embedder, image_embedder)
    result_ranker = ResultRanker()
    query_analytics = QueryAnalytics()
    
    # Initialize response generator components
    llm_interface = LLMInterface()
    template_manager = TemplateManager("./templates")
    response_formatter = ResponseFormatter(template_manager)
    prompt_visualizer = PromptVisualizer(template_manager)
    
    # Initialize Colab notebook generator components
    # notebook_template_engine = NotebookTemplateEngine("./notebook_templates")
    # code_generator = CodeGenerator()
    # colab_api_client = ColabAPIClient()
    # reproducibility_manager = ReproducibilityManager()
    # resource_quota_manager = ResourceQuotaManager()
    
    return {
        "document_processor": {
            "schema_validator": schema_validator,
            "code_parser": code_parser,
            "metadata_extractor": metadata_extractor,
            # "image_processor": image_processor
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
            "result_ranker": result_ranker,
            "query_analytics": query_analytics
        },
        "response_generator": {
            "llm_interface": llm_interface,
            "template_manager": template_manager,
            "response_formatter": response_formatter,
            "prompt_visualizer": prompt_visualizer
        }
        # "colab_generator": {
        #     "notebook_template_engine": notebook_template_engine,
        #     "code_generator": code_generator,
        #     "colab_api_client": colab_api_client,
        #     "reproducibility_manager": reproducibility_manager,
        #     "resource_quota_manager": resource_quota_manager
        # }
    }

def process_model_scripts(components, directory_path):
    """Process model scripts in a directory.
    
    This function walks through the directory to find model script files,
    processes them using the document processor components, generates
    embeddings, and stores them in the vector database.
    
    Args:
        components: Dictionary containing initialized system components
        directory_path: Path to directory containing model scripts
    """
    print(f"Processing model scripts in {directory_path}...")
    
    # Extract required components
    code_parser = components["document_processor"]["code_parser"]
    metadata_extractor = components["document_processor"]["metadata_extractor"]
    schema_validator = components["document_processor"]["schema_validator"]
    text_embedder = components["vector_db_manager"]["text_embedder"]
    chroma_manager = components["vector_db_manager"]["chroma_manager"]
    access_control = components["vector_db_manager"]["access_control"]
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("model_script_processor")
    
    # Ensure directory exists
    if not os.path.isdir(directory_path):
        logger.error(f"Directory {directory_path} does not exist")
        return
    
    # Supported file extensions for model scripts
    supported_extensions = ['.py', '.ipynb', '.json', '.yaml', '.yml']
    
    # Find all model script files
    script_files = []
    for ext in supported_extensions:
        script_files.extend(glob.glob(os.path.join(directory_path, f"**/*{ext}"), recursive=True))
    
    logger.info(f"Found {len(script_files)} potential model script files")
    
    # Process files in parallel using a thread pool
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_file = {executor.submit(process_single_script, 
                                          file_path, 
                                          components): file_path for file_path in script_files}
        
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
    metadata = metadata_extractor.extract_metadata(file_path, parse_result)
    
    # 3. Split into chunks for processing
    chunks = code_parser.split_into_chunks(parse_result["content"], chunk_size=1000, overlap=200)
    
    documents = []
    for i, chunk in enumerate(chunks):
        # Create document for each chunk
        document = {
            "id": f"model_script_{Path(file_path).stem}_{i}",
            "$schema_version": "1.0.0",
            "content": chunk,
            "metadata": {
                **metadata,
                "chunk_id": i,
                "total_chunks": len(chunks)
            }
        }
        
        # 4. Validate against schema
        validation_result = schema_validator.validate(document, "model_script_schema")
        if not validation_result["valid"]:
            logging.warning(f"Schema validation failed for {file_path}, chunk {i}: {validation_result['errors']}")
            continue
        
        # 5. Generate embeddings
        embedding = text_embedder.embed_text(chunk)
        
        # 6. Apply access control
        access_metadata = access_control.get_document_permissions(document)
        document["metadata"]["access_control"] = access_metadata
        
        # 7. Store in Chroma
        chroma_manager.add_document(
            collection="model_scripts",
            document_id=document["id"],
            document=document,
            embedding=embedding
        )
        
        documents.append(document)
    
    # Return the first document ID and success status
    return (documents[0]["id"], True) if documents else (None, False)

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
    
    # Extract required components
    image_processor = components["document_processor"]["image_processor"]
    schema_validator = components["document_processor"]["schema_validator"]
    image_embedder = components["vector_db_manager"]["image_embedder"]
    chroma_manager = components["vector_db_manager"]["chroma_manager"]
    access_control = components["vector_db_manager"]["access_control"]
    
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
    process_result = image_processor.process(file_path)
    if not process_result:
        # Not a valid image, skip it
        return None
    
    # 2. Create document
    document_id = f"generated_image_{Path(file_path).stem}"
    document = {
        "id": document_id,
        "$schema_version": "1.0.0",
        "content": None,  # No text content, just embedding
        "metadata": process_result["metadata"]
    }
    
    # 3. Validate against schema
    validation_result = schema_validator.validate(document, "generated_image_schema")
    if not validation_result["valid"]:
        logging.warning(f"Schema validation failed for {file_path}: {validation_result['errors']}")
        return (document_id, False)
    
    # 4. Generate image embeddings
    # Use global embedding by default, but can use tiled if specified in metadata
    embedding_type = document["metadata"].get("embedding_type", "global")
    if embedding_type == "global":
        embedding = image_embedder.embed_image(process_result["image"])
    else:  # tiled embedding
        embedding = image_embedder.embed_image_tiled(process_result["image"], 
                                                   process_result["metadata"].get("tile_config", {}))
    
    # 5. Apply access control
    access_metadata = access_control.get_document_permissions(document)
    document["metadata"]["access_control"] = access_metadata
    
    # 6. Store in Chroma
    chroma_manager.add_document(
        collection="generated_images",
        document_id=document_id,
        document=document,
        embedding=embedding
    )
    
    # 7. Create and store thumbnail if it doesn't exist
    thumbnail_path = document["metadata"].get("thumbnail_path")
    if thumbnail_path and not os.path.exists(thumbnail_path):
        thumb_dir = os.path.dirname(thumbnail_path)
        os.makedirs(thumb_dir, exist_ok=True)
        image_processor.create_thumbnail(process_result["image"], thumbnail_path)
    
    return (document_id, True)

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
    result_ranker = components["query_engine"]["result_ranker"]
    query_analytics = components["query_engine"]["query_analytics"]
    llm_interface = components["response_generator"]["llm_interface"]
    template_manager = components["response_generator"]["template_manager"]
    response_formatter = components["response_generator"]["response_formatter"]
    access_control = components["vector_db_manager"]["access_control"]
    
    # Define available commands
    commands = {
        "query": "Search for model scripts or images",
        "list-models": "List available models",
        "list-images": "List available images",
        "compare-models": "Compare two or more models",
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
                if not available_models:
                    print("  No models available")
                else:
                    for model in available_models:
                        print(f"  {model['model_id']} - {model.get('description', 'No description')}")
                        
            elif cmd.lower() == "list-images":
                # Get images the user has access to
                available_images = access_control.get_accessible_images(user_id)
                print("\nAvailable images:")
                if not available_images:
                    print("  No images available")
                else:
                    for image in available_images:
                        print(f"  {image['id']} - {image.get('prompt', 'No prompt')} - {image.get('filepath', 'No path')}")
                        
            elif cmd.lower().startswith("query"):
                if cmd.lower() == "query":
                    query_text = input("Enter your query: ")
                else:
                    query_text = cmd[6:].strip()
                    
                # Parse the query
                parsed_query = query_parser.parse_query(query_text)
                
                # Log the query for analytics
                query_analytics.log_query(user_id, query_text, parsed_query)
                
                # Check access permissions
                # if not access_control.verify_user_permissions(user_id, parsed_query):
                #    print("Access denied. You don't have permission to access this information.")
                #    continue
                
                print("Searching...")
                # Dispatch the query
                # Dispatch the query
                search_results = asyncio.run(search_dispatcher.dispatch(
                    query=parsed_query["processed_query"] if "processed_query" in parsed_query else query_text,
                    intent=parsed_query["intent"],
                    parameters=parsed_query["parameters"],
                    user_id=user_id
                ))
                
                # Rank the results
                if isinstance(search_results, dict) and 'items' in search_results:
                    ranked_results = result_ranker.rank_results(search_results['items'])
                else:
                    ranked_results = []
                
                # Generate response using LLM
                if parsed_query["type"] == "comparison":
                    template = template_manager.get_template("model_comparison")
                elif parsed_query["type"] == "retrieval":
                    template = template_manager.get_template("information_retrieval")
                else:
                    template = template_manager.get_template("general_query")
                    
                print("Generating response...")
                llm_response = llm_interface.generate_response(template, {
                    "query": query_text,
                    "results": ranked_results,
                    "parsed_query": parsed_query
                })
                
                # Format the response
                formatted_response = response_formatter.format_response([llm_response], parsed_query, parsed_query["type"])
                
                # Print the response
                print("\nResponse:")
                print(formatted_response)
                
                # Show top results
                print("\nTop Results:")
                for i, result in enumerate(ranked_results[:5]):
                    print(f"  {i+1}. {result.get('id', 'Unknown')} - {result.get('score', 0):.2f}")
                    
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
                for model_id in models_to_compare:
                    if not access_control.verify_model_access(user_id, model_id):
                        print(f"Access denied for model: {model_id}")
                        all_accessible = False
                        break
                
                if not all_accessible:
                    continue
                
                print(f"Comparing models: {', '.join(models_to_compare)}...")
                
                # Here you would implement the model comparison logic
                # For now, just a placeholder
                print("Model comparison functionality will be implemented in a future update.")
                
            else:
                print(f"Unknown command: {cmd}")
                print("Type 'help' to see available commands.")
                
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
        except Exception as e:
            print(f"Error: {str(e)}")
    
    print("UI session ended.")

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
