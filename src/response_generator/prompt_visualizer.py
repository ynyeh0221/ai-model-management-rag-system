# src/response_generator/prompt_visualizer.py
class PromptVisualizer:
    def __init__(self, template_manager):
        self.template_manager = template_manager
    
    def render_preview(self, template_id, context, version=None):
        """Render a preview of a prompt template."""
        pass
    
    def generate_diff(self, template_id, version_a, version_b):
        """Generate a diff between two versions of a template."""
        pass
    
    def create_html_preview(self, rendered_prompt):
        """Create an HTML preview of a rendered prompt."""
        pass
