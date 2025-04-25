import logging
from typing import Dict, Any

from src.query_engine.handlers.metadata_search_handler import MetadataSearchHandler


class FallbackSearchHandler:
    def __init__(self, metadata_search_manager: MetadataSearchHandler):
        self.metadata_search_manager = metadata_search_manager
        self.logger = logging.getLogger(__name__)

    async def handle_fallback_search(self, query: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle queries with unknown intent with a best-effort approach.

        Args:
            query: The processed query text
            parameters: Dictionary of extracted parameters

        Returns:
            Dictionary containing search results
        """
        self.logger.warning(f"Using fallback search for query: {query}")

        # Try a combination of text search and metadata search
        try:
            # Run text search
            text_results = await self.handle_text_search(query, parameters)

            # If text search yielded results, return them
            if text_results.get('success', False) and text_results.get('total_found', 0) > 0:
                return text_results

            # If no results from text search, try metadata search
            try:
                metadata_results = await self.metadata_search_manager.handle_metadata_search(query, parameters)

                if metadata_results.get('success', False) and metadata_results.get('total_found', 0) > 0:
                    return metadata_results
            except Exception as e:
                self.logger.error(f"Error in metadata search during fallback: {e}", exc_info=True)

            # If still no results, return empty results with proper structure
            return {
                'success': True,
                'type': 'fallback_search',
                'items': [],
                'total_found': 0,
                'message': "No results found using various search strategies"
            }

        except Exception as e:
            self.logger.error(f"Error in fallback search: {e}", exc_info=True)

            # Return a properly structured error response
            return {
                'success': False,
                'error': "An error occurred during the search",
                'type': 'fallback_search',
                'items': [],
                'total_found': 0
            }