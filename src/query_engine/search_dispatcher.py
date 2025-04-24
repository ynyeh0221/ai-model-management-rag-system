import logging
import time
from typing import Dict, Any, Optional, Union

from query_engine.query_intent import QueryIntent


class SearchDispatcher:
    """
    Leaner dispatcher that routes queries to appropriate search handlers based on intent.
    """

    def __init__(self, handlers_factory, access_control_manager=None, analytics=None):
        self.handlers_factory = handlers_factory
        self.access_control_manager = access_control_manager
        self.analytics = analytics
        self.logger = logging.getLogger(__name__)

    async def dispatch(self, query: str, intent: Union[str, QueryIntent],
                       parameters: Dict[str, Any], user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Dispatch a query to the appropriate search handler based on intent.
        """
        start_time = time.time()
        self.logger.info(f"Dispatching query with intent: {intent}")

        # Convert string intent to enum if needed
        if isinstance(intent, str):
            try:
                intent = QueryIntent(intent)
            except ValueError:
                self.logger.warning(f"Unknown intent: {intent}, falling back to RETRIEVAL")
                intent = QueryIntent.UNKNOWN

        # Add user_id to parameters for access control in handlers
        if user_id:
            parameters['user_id'] = user_id

        # Get the appropriate handler
        handler = self.handlers_factory.get_handler(intent)

        try:
            # Call the handler
            results = await handler.handle(query, parameters)

            # Calculate execution time
            execution_time = (time.time() - start_time) * 1000  # Convert to ms

            # Log analytics if available
            if self.analytics:
                self.analytics.log_performance_metrics(
                    query_id=parameters.get('query_id', 'unknown'),
                    total_time_ms=int(execution_time),
                    search_time_ms=int(execution_time)
                )

            # Add metadata to results
            results['metadata'] = {
                'intent': intent.value if isinstance(intent, QueryIntent) else intent,
                'execution_time_ms': execution_time,
                'result_count': len(results.get('items', [])),
                'parameters': self._sanitize_parameters(parameters)
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

    def _sanitize_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize parameters for inclusion in response metadata.
        Remove sensitive or internal fields.
        """
        if not parameters:
            return {}

        # Create a copy to avoid modifying the original
        sanitized = parameters.copy()

        # Remove sensitive fields
        sensitive_fields = ['user_id', 'access_token', 'auth_context', 'raw_query', 'query_id']
        for field in sensitive_fields:
            if field in sanitized:
                del sanitized[field]

        # Remove image data (could be large)
        if 'image_data' in sanitized:
            sanitized['image_data'] = "[binary data removed]"

        return sanitized