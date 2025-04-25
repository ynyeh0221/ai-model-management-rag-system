import time
from typing import Dict, Any


class PerformanceMetricsCalculator:
    """Utility class for calculating and logging performance metrics."""

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