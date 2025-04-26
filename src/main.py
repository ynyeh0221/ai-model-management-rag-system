import argparse
import os

from src.cli.ingest_images import ImageProcessorRunner
from src.cli.ingest_model_scripts import ScriptProcessorRunner
from src.cli.query_cli import CLIInterface
from src.core.colab_generator.code_generator import CodeGenerator
from src.core.colab_generator.colab_api_client import ColabAPIClient
from src.core.colab_generator.reproducibility_manager import ReproducibilityManager
from src.core.colab_generator.resource_quota_manager import ResourceQuotaManager
from src.core.content_analyzer.image.image_processor import ImageProcessor
from src.core.content_analyzer.metadata.metadata_extractor import MetadataExtractor
from src.core.content_analyzer.metadata.schema_validator import SchemaValidator
from src.core.content_analyzer.model_script.llm_based_code_parser import LLMBasedCodeParser
from src.core.query_engine.llm_interface import LLMInterface
from src.core.query_engine.query_analytics import QueryAnalytics
from src.core.query_engine.query_parser import QueryParser
from src.core.query_engine.result_reranker import CrossEncoderReranker
from src.core.query_engine.search_dispatcher import SearchDispatcher
from src.core.vector_db.access_control import AccessControlManager
from src.core.vector_db.chroma_manager import ChromaManager
from src.core.vector_db.image_embedder import ImageEmbedder
from src.core.vector_db.text_embedder import TextEmbedder


def initialize_components(config_path="./config", llm_model_name: str = "llm"):
    """Initialize all cli_response_utils of the RAG system."""

    llm_interface = LLMInterface(model_name=llm_model_name, timeout=60000)

    # Initialize document cli_runner cli_response_utils
    schema_validator = SchemaValidator(os.path.join(config_path, "schema_registry.json"))
    code_parser = LLMBasedCodeParser(schema_validator=schema_validator, llm_interface=llm_interface)
    metadata_extractor = MetadataExtractor()
    image_processor = ImageProcessor(schema_validator)
    
    # Initialize vector database cli_response_utils
    text_embedder = TextEmbedder(device="mps")
    image_embedder = ImageEmbedder()
    chroma_manager = ChromaManager(text_embedder, image_embedder, "./chroma_db")
    access_control = AccessControlManager(chroma_manager)
    
    # Initialize query engine cli_response_utils
    query_parser = QueryParser()
    search_dispatcher = SearchDispatcher(chroma_manager, text_embedder, image_embedder)
    query_analytics = QueryAnalytics()
    result_reranker = CrossEncoderReranker(device="mps")
    
    # Initialize Colab notebook generator cli_response_utils
    code_generator = CodeGenerator()
    colab_api_client = ColabAPIClient()
    reproducibility_manager = ReproducibilityManager()
    resource_quota_manager = ResourceQuotaManager()

    return {
        "content_analyzer": {
            "schema_validator": schema_validator,
            "code_parser": code_parser,
            "metadata_extractor": metadata_extractor,
            "image_processor": image_processor
        },
        "vector_db": {
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
        },
        "colab_generator": {
            "code_generator": code_generator,
            "colab_api_client": colab_api_client,
            "reproducibility_manager": reproducibility_manager,
            "resource_quota_manager": resource_quota_manager
        }
    }

def main():
    script_processor_runner = ScriptProcessorRunner()
    image_processor_runner = ImageProcessorRunner()
    cli_interface = CLIInterface()

    parser = argparse.ArgumentParser(description="AI Model Management RAG System")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Process model scripts command
    process_scripts_parser = subparsers.add_parser("process-scripts", help="Process model scripts")
    process_scripts_parser.add_argument("directory", help="Directory containing model scripts")

    # Process single model model_script command
    process_single_script_parser = subparsers.add_parser("process-single-model_script", help="Process a single model model_script")
    process_single_script_parser.add_argument("file_path", help="Absolute path to target model model_script file")

    # Process image_processing command
    process_images_parser = subparsers.add_parser("process-image_processing", help="Process image_processing")
    process_images_parser.add_argument("directory", help="Directory containing image_processing")

    # Process single image command
    process_single_image_parser = subparsers.add_parser("process-single-image", help="Process a single image")
    process_single_image_parser.add_argument("file_path", help="Absolute path to target image")

    # Start UI command
    ui_parser = subparsers.add_parser("start-user_interface", help="Start the user interface")
    ui_parser.add_argument("--host", default="localhost", help="Host to bind the UI to")
    ui_parser.add_argument("--port", type=int, default=8000, help="Port to bind the UI to")

    args = parser.parse_args()

    # Execute command
    if args.command == "process-scripts":
        # Use deepseek-llm for speed and token length consideration
        components = initialize_components(llm_model_name="deepseek-llm:7b")
        script_processor_runner.process_model_scripts(components, args.directory)
    elif args.command == "process-single-model_script":
        # Use deepseek-llm for speed and token length consideration
        components = initialize_components(llm_model_name="deepseek-llm:7b")
        script_processor_runner.process_single_script(components, args.file_path)
    elif args.command == "process-image_processing":
        # Use deepseek-llm for speed and token length consideration
        components = initialize_components(llm_model_name="deepseek-llm:7b")
        image_processor_runner.process_images(components, args.directory)
    elif args.command == "process-single-image":
        # Use deepseek-llm for speed and token length consideration
        components = initialize_components(llm_model_name="deepseek-llm:7b")
        image_processor_runner.process_single_image(components, args.file_path)
    elif args.command == "start-user_interface":
        # Use deepseek-r1 to better show thinking steps
        components = initialize_components(llm_model_name="deepseek-r1:7b")
        cli_interface.start_cli(components, args.host, args.port)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
