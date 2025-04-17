import argparse
import os

from cli_runner.image_processor_runner import ImageProcessorRunner
from cli_runner.script_processor_runner import ScriptProcessorRunner
from cli_runner.ui_runner import UIRunner
from src.colab_generator.code_generator import CodeGenerator
from src.colab_generator.colab_api_client import ColabAPIClient
from src.colab_generator.reproducibility_manager import ReproducibilityManager
from src.colab_generator.resource_quota_manager import ResourceQuotaManager
from src.document_processor.llm_based_code_parser import LLMBasedCodeParser
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

    llm_interface = LLMInterface(model_name="deepseek-llm:7b", timeout=30000)

    # Initialize document cli_runner components
    schema_validator = SchemaValidator(os.path.join(config_path, "schema_registry.json"))
    code_parser = LLMBasedCodeParser(schema_validator=schema_validator, llm_interface=llm_interface)
    metadata_extractor = MetadataExtractor()
    image_processor = ImageProcessor(schema_validator)
    
    # Initialize vector database components
    text_embedder = TextEmbedder()
    image_embedder = ImageEmbedder()
    chroma_manager = ChromaManager(text_embedder, image_embedder, "./chroma_db")
    access_control = AccessControlManager(chroma_manager)
    
    # Initialize query engine components
    query_parser = QueryParser(llm_model_name="deepseek-llm:7b")
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

def main():
    script_processor_runner = ScriptProcessorRunner()
    image_processor_runner = ImageProcessorRunner()
    ui_runner = UIRunner()

    parser = argparse.ArgumentParser(description="AI Model Management RAG System")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Process model scripts command
    process_scripts_parser = subparsers.add_parser("process-scripts", help="Process model scripts")
    process_scripts_parser.add_argument("directory", help="Directory containing model scripts")

    # Process single model script command
    process_single_script_parser = subparsers.add_parser("process-single-script", help="Process a single model script")
    process_single_script_parser.add_argument("file_path", help="Absolute path to target model script file")

    # Process images command
    process_images_parser = subparsers.add_parser("process-images", help="Process images")
    process_images_parser.add_argument("directory", help="Directory containing images")

    # Process single image command
    process_single_image_parser = subparsers.add_parser("process-single-image", help="Process a single image")
    process_single_image_parser.add_argument("file_path", help="Absolute path to target image")

    # Start UI command
    ui_parser = subparsers.add_parser("start-ui", help="Start the user interface")
    ui_parser.add_argument("--host", default="localhost", help="Host to bind the UI to")
    ui_parser.add_argument("--port", type=int, default=8000, help="Port to bind the UI to")

    args = parser.parse_args()

    # Initialize components
    components = initialize_components()

    # Execute command
    if args.command == "process-scripts":
        script_processor_runner.process_model_scripts(components, args.directory)
    elif args.command == "process-single-script":
        script_processor_runner.process_single_script(components, args.file_path)
    elif args.command == "process-images":
        image_processor_runner.process_images(components, args.directory)
    elif args.command == "process-single-image":
        image_processor_runner.process_single_image(components, args.file_path)
    elif args.command == "start-ui":
        ui_runner.start_ui(components, args.host, args.port)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
