# src/response_generator/response_formatter.py
class ResponseFormatter:
    def __init__(self, template_manager):
        self.template_manager = template_manager
    
    def format_response(self, results, query, response_type="text"):
        """Format results into a response."""
        pass
    
    def format_comparison(self, model_a, model_b, comparison_points):
        """Format a comparison of two models."""
        pass
    
    def format_image_gallery(self, images):
        """Format an image gallery."""
        pass
    
    def include_citations(self, response, results):
        """Include citations in a response."""
        pass
