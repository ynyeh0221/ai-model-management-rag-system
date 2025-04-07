# src/query_engine/search_dispatcher.py
class SearchDispatcher:
    def __init__(self, chroma_manager, text_embedder, image_embedder):
        self.chroma_manager = chroma_manager
        self.text_embedder = text_embedder
        self.image_embedder = image_embedder
    
    def dispatch(self, query, intent, parameters):
        """Dispatch a query to the appropriate search handler."""
        pass
    
    def handle_text_search(self, query, parameters):
        """Handle a text search query."""
        pass
    
    def handle_image_search(self, query, parameters):
        """Handle an image search query."""
        pass
    
    def handle_comparison(self, parameters):
        """Handle a comparison query."""
        pass
    
    def handle_notebook_request(self, parameters):
        """Handle a notebook generation request."""
        pass
