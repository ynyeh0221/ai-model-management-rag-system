import logging
import time
from typing import Dict, Any, Optional, Union

from query_engine.handlers.comparison_handler import ComparisonHandler
from query_engine.handlers.fallback_search_handler import FallbackSearchHandler
from query_engine.handlers.image_search_handler import ImageSearchHandler
from query_engine.handlers.metadata_search_handler import MetadataSearchHandler
from query_engine.handlers.notebook_request_handler import NotebookRequestHandler
from query_engine.handlers.utils.comparison_generator import ComparisonGenerator
from query_engine.handlers.utils.distance_normalizer import DistanceNormalizer
from query_engine.handlers.utils.filter_translator import FilterTranslator
from query_engine.handlers.utils.metadata_table_manager import MetadataTableManager
from query_engine.handlers.utils.model_data_fetcher import ModelDataFetcher
from query_engine.handlers.utils.performance_metrics_calculator import PerformanceMetricsCalculator
from query_engine.query_intent import QueryIntent


class SearchDispatcher:
    """
    Dispatcher that routes queries to appropriate search handlers
    based on the classified intent.
    """

    def __init__(self, chroma_manager, text_embedder, image_embedder,
                 access_control_manager=None, analytics=None, image_search_manager=None):
        """
        Initialize the SearchDispatcher with required dependencies.

        Args:
            chroma_manager: Manager for Chroma vector database interactions
            text_embedder: Component for generating text embeddings
            image_embedder: Component for generating image embeddings
            access_control_manager: Optional manager for access control
            analytics: Optional analytics collector
            image_search_manager: Optional manager for image searches
        """
        self.chroma_manager = chroma_manager
        self.text_embedder = text_embedder
        self.image_embedder = image_embedder
        self.access_control_manager = access_control_manager
        self.analytics = analytics
        self.logger = logging.getLogger(__name__)

        # Initialize utility classes
        self.distance_normalizer = DistanceNormalizer()
        self.filter_translator = FilterTranslator()
        self.model_data_fetcher = ModelDataFetcher(chroma_manager, access_control_manager)
        self.performance_metrics = PerformanceMetricsCalculator(analytics)
        self.comparison_generator = ComparisonGenerator()
        self.metadata_table_manager = MetadataTableManager(chroma_manager, access_control_manager)

        # Initialize handlers
        self.metadata_search_manager = MetadataSearchHandler(self.chroma_manager, self.filter_translator, self.distance_normalizer, self.access_control_manager)
        self.comparison_manager = ComparisonHandler(self.metadata_table_manager, self.metadata_search_manager, self.access_control_manager, self.filter_translator, self.chroma_manager, self.distance_normalizer)
        self.image_search_manager = image_search_manager or ImageSearchHandler(
            chroma_manager=chroma_manager,
            image_embedder=image_embedder,
            access_control_manager=access_control_manager,
            analytics=analytics
        )
        self.notebook_manager = NotebookRequestHandler(
            chroma_manager=chroma_manager,
            model_data_fetcher=self.model_data_fetcher,
            access_control_manager=access_control_manager,
            analytics=analytics
        )
        self.fallback_search_manager = FallbackSearchHandler(self.metadata_search_manager)

        # Define handlers mapping for dispatching
        self.handlers = {
            QueryIntent.RETRIEVAL: self.metadata_search_manager.handle_metadata_search,
            QueryIntent.COMPARISON: self.comparison_manager.handle_comparison,
            QueryIntent.NOTEBOOK: self.notebook_manager.handle_notebook_request,  # Updated line
            QueryIntent.IMAGE_SEARCH: self.image_search_manager.handle_image_search,
            QueryIntent.METADATA: self.metadata_search_manager.handle_metadata_search,
            QueryIntent.UNKNOWN: self.fallback_search_manager.handle_fallback_search
        }

    async def dispatch(self, query: str, intent: Union[str, QueryIntent],
                       parameters: Dict[str, Any], user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Dispatch a query to the appropriate search handler based on intent.

        Args:
            query: The processed query text
            intent: The classified intent (string or enum)
            parameters: Dictionary of extracted parameters
            user_id: Optional user identifier for access control

        Returns:
            Dictionary containing search results and metadata
        """
        start_time = time.time()
        self.logger.info(f"Dispatching query with intent: {intent}")

        # Convert string intent to enum if needed
        if isinstance(intent, str):
            try:
                intent = QueryIntent(intent)
            except ValueError:
                self.logger.warning(f"Unknown intent: {intent}, falling back to RETRIEVAL")
                intent = QueryIntent.RETRIEVAL

        # Add user_id to parameters for access control in handlers
        if user_id:
            parameters['user_id'] = user_id

        # Get the appropriate handler
        handler = self.handlers.get(intent, self.fallback_search_manager.handle_fallback_search)

        try:
            # Call the handler
            results = await handler(query, parameters)

            # Calculate execution time
            execution_time = (time.time() - start_time) * 1000  # Convert to ms

            # Log analytics if available
            if self.analytics:
                self.analytics.log_performance_metrics(
                    query_id=parameters.get('query_id', 'unknown'),
                    total_time_ms=int(execution_time),
                    search_time_ms=int(execution_time)  # More detailed metrics would be set in handlers
                )

            # Add metadata to results
            results['metadata'] = {
                'intent': intent.value if isinstance(intent, QueryIntent) else intent,
                'execution_time_ms': execution_time,
                'result_count': len(results.get('items', [])),
                'parameters': self.performance_metrics.sanitize_parameters(parameters)
            }

            return results

        except Exception as e:
            self.logger.error(f"Error in search dispatch: {e}", exc_info=True)

            # Log failed search if analytics available
            if self.analytics:
                self.analytics.update_query_status(
                    query_id=parameters.get('query_id', 'unknown'),
                    status='failed'
                )

            # Return error information
            return {
                'success': False,
                'error': str(e),
                'metadata': {
                    'intent': intent.value if isinstance(intent, QueryIntent) else intent,
                    'execution_time_ms': (time.time() - start_time) * 1000
                }
            }