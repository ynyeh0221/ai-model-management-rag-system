import time
from typing import Dict, Any


class PerformanceMetricsCalculator:
    """
    Utility class for calculating and logging performance metrics related to search operations.

    This class provides functionality to measure and record various timing metrics
    for search operations, as well as utilities to sanitize parameters for inclusion
    in response metadata.

    Attributes:
        analytics: Optional analytics service for logging performance metrics.
                   If provided, metrics will be logged through this service.

    Examples:
        Basic usage without analytics:
        >>> calculator = PerformanceMetricsCalculator()
        >>> start_time = time.time()
        >>> # ... perform search operations ...
        >>> metrics = calculator.calculate_text_search_performance(
        ...     start_time=start_time,
        ...     metadata_search_time=150.0,
        ...     chunks_search_time=250.0,
        ...     parameters={'query': 'example search'}
        ... )
        >>> print(metrics['total_search_time_ms'])
        400.0

        Usage with analytics service:
        >>> analytics_service = AnalyticsService()
        >>> calculator = PerformanceMetricsCalculator(analytics=analytics_service)
        >>> # ... perform search with query_id ...
        >>> metrics = calculator.calculate_text_search_performance(
        ...     start_time=start_time,
        ...     metadata_search_time=100.0,
        ...     chunks_search_time=200.0,
        ...     parameters={'query': 'example', 'query_id': 'abc123'}
        ... )
        # Performance metrics will be logged to analytics service
    """

    def __init__(self, analytics=None):
        self.analytics = analytics

    def calculate_text_search_performance(
            self, start_time: float, metadata_search_time: float, chunks_search_time: float,
            parameters: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate performance metrics for text search."""
        total_search_time = metadata_search_time + chunks_search_time
        total_time = (time.time() - start_time) * 1000

        # Log performance metrics if analytics available
        if self.analytics and 'query_id' in parameters:
            self.analytics.log_performance_metrics(
                query_id=parameters['query_id'],
                search_time_ms=int(total_search_time),
                total_time_ms=int(total_time)
            )

        return {
            'metadata_search_time_ms': metadata_search_time,
            'chunks_search_time_ms': chunks_search_time,
            'total_search_time_ms': total_search_time,
            'total_time_ms': total_time
        }

    def sanitize_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize parameters for inclusion in response metadata.
        Remove sensitive or internal fields.

        Args:
            parameters: Original parameters dictionary

        Returns:
            Sanitized parameters dictionary
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