import difflib
import logging
from typing import Dict, Any, Optional, List, Tuple
import html
import json
import markdown
from datetime import datetime

class PromptVisualizer:
    """
    Visualizes prompt templates with previews and diffs.
    
    This class provides functionality to render previews of prompt templates,
    visualize differences between template versions, and create HTML previews
    for the interactive prompt studio.
    """
    
    def __init__(self, template_manager):
        """
        Initialize the PromptVisualizer with a template manager.
        
        Args:
            template_manager: Manager for accessing and manipulating prompt templates
        """
        self.template_manager = template_manager
        self.logger = logging.getLogger(__name__)
        
    def render_preview(self, template_id: str, context: Dict[str, Any], 
                      version: Optional[str] = None) -> Dict[str, Any]:
        """
        Render a preview of a prompt template with the given context.
        
        Args:
            template_id: Identifier of the template to render
            context: Context data to use for template rendering
            version: Optional specific version of the template to render
            
        Returns:
            Dictionary containing the rendered preview and metadata
        """
        self.logger.info(f"Rendering preview for template: {template_id}, version: {version or 'latest'}")
        
        try:
            # Get template from manager
            template = self.template_manager.get_template(template_id, version)
            if not template:
                return {
                    "success": False,
                    "error": f"Template {template_id} not found",
                    "preview": None,
                    "metadata": None
                }
            
            # Get template metadata
            metadata = self.template_manager.get_template_metadata(template_id, version)
            
            # Render the template with provided context
            rendered_content = template.render(**context)
            
            # Create preview response
            preview_data = {
                "success": True,
                "preview": rendered_content,
                "metadata": metadata,
                "rendered_at": datetime.now().isoformat(),
                "context_sample": self._truncate_context(context),
                "html_preview": self.create_html_preview(rendered_content)
            }
            
            return preview_data
            
        except Exception as e:
            self.logger.error(f"Error rendering template preview: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "preview": None,
                "metadata": None
            }
    
    def generate_diff(self, template_id: str, version_a: str, 
                     version_b: str) -> Dict[str, Any]:
        """
        Generate a diff between two versions of a template.
        
        Args:
            template_id: Identifier of the template
            version_a: First version for comparison
            version_b: Second version for comparison
            
        Returns:
            Dictionary containing diff information in various formats
        """
        self.logger.info(f"Generating diff for template {template_id} between versions {version_a} and {version_b}")
        
        try:
            # Get template content for both versions
            template_a_content = self.template_manager.get_template_content(template_id, version_a)
            template_b_content = self.template_manager.get_template_content(template_id, version_b)
            
            if not template_a_content or not template_b_content:
                return {
                    "success": False,
                    "error": "One or both template versions not found",
                    "diff": None
                }
            
            # Get metadata for both versions
            metadata_a = self.template_manager.get_template_metadata(template_id, version_a)
            metadata_b = self.template_manager.get_template_metadata(template_id, version_b)
            
            # Generate text diff
            text_diff = self._generate_text_diff(template_a_content, template_b_content)
            
            # Generate HTML diff
            html_diff = self._generate_html_diff(template_a_content, template_b_content)
            
            # Generate unified diff (git-style)
            unified_diff = self._generate_unified_diff(
                template_a_content, 
                template_b_content,
                f"{template_id} ({version_a})",
                f"{template_id} ({version_b})"
            )
            
            # Create diff response
            diff_data = {
                "success": True,
                "template_id": template_id,
                "version_a": {
                    "id": version_a,
                    "metadata": metadata_a
                },
                "version_b": {
                    "id": version_b,
                    "metadata": metadata_b
                },
                "diff": {
                    "text": text_diff,
                    "html": html_diff,
                    "unified": unified_diff
                },
                "stats": self._calculate_diff_stats(template_a_content, template_b_content),
                "generated_at": datetime.now().isoformat()
            }
            
            return diff_data
            
        except Exception as e:
            self.logger.error(f"Error generating template diff: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "diff": None
            }
    
    def create_html_preview(self, rendered_prompt: str) -> str:
        """
        Create an HTML preview of a rendered prompt.
        
        Args:
            rendered_prompt: The rendered prompt content
            
        Returns:
            HTML formatted preview with syntax highlighting
        """
        # Escape HTML entities to prevent XSS
        escaped_content = html.escape(rendered_prompt)
        
        # Detect if content is JSON and format accordingly
        try:
            json_obj = json.loads(rendered_prompt)
            formatted_json = json.dumps(json_obj, indent=2)
            escaped_content = html.escape(formatted_json)
            html_preview = f"""
            <div class="prompt-preview">
                <div class="prompt-preview-header">
                    <span class="preview-title">JSON Preview</span>
                </div>
                <div class="prompt-preview-content json-content">
                    <pre><code class="language-json">{escaped_content}</code></pre>
                </div>
            </div>
            """
        except (json.JSONDecodeError, TypeError):
            # Check if it looks like markdown
            if '##' in rendered_prompt or '*' in rendered_prompt:
                # Convert markdown to HTML
                md_html = markdown.markdown(rendered_prompt)
                html_preview = f"""
                <div class="prompt-preview">
                    <div class="prompt-preview-header">
                        <span class="preview-title">Markdown Preview</span>
                    </div>
                    <div class="prompt-preview-content markdown-rendered">
                        {md_html}
                    </div>
                    <div class="prompt-preview-source">
                        <details>
                            <summary>View Source</summary>
                            <pre><code class="language-markdown">{escaped_content}</code></pre>
                        </details>
                    </div>
                </div>
                """
            else:
                # Plain text preview
                html_preview = f"""
                <div class="prompt-preview">
                    <div class="prompt-preview-header">
                        <span class="preview-title">Text Preview</span>
                    </div>
                    <div class="prompt-preview-content">
                        <pre>{escaped_content}</pre>
                    </div>
                </div>
                """
        
        return html_preview
    
    def visualize_template_history(self, template_id: str) -> Dict[str, Any]:
        """
        Create a visualization of a template's version history.
        
        Args:
            template_id: Identifier of the template
            
        Returns:
            Dictionary containing template history visualization data
        """
        self.logger.info(f"Visualizing history for template: {template_id}")
        
        try:
            # Get template versions
            versions = self.template_manager.get_template_versions(template_id)
            if not versions:
                return {
                    "success": False,
                    "error": f"No versions found for template {template_id}",
                    "history": None
                }
            
            # Get metadata for each version
            version_data = []
            for version in versions:
                metadata = self.template_manager.get_template_metadata(template_id, version)
                if metadata:
                    version_data.append({
                        "version": version,
                        "metadata": metadata,
                        "created_at": metadata.get("created_at"),
                        "author": metadata.get("author"),
                        "message": metadata.get("message", "")
                    })
            
            # Sort by creation date
            version_data.sort(key=lambda x: x.get("created_at", ""), reverse=True)
            
            # Generate timeline visualization data
            timeline_data = self._generate_timeline_data(version_data)
            
            # Create history response
            history_data = {
                "success": True,
                "template_id": template_id,
                "versions": version_data,
                "timeline": timeline_data,
                "total_versions": len(versions),
                "generated_at": datetime.now().isoformat()
            }
            
            return history_data
            
        except Exception as e:
            self.logger.error(f"Error visualizing template history: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "history": None
            }
    
    def _generate_text_diff(self, text_a: str, text_b: str) -> List[Dict[str, Any]]:
        """Generate a structured text diff between two strings."""
        differ = difflib.SequenceMatcher(None, text_a, text_b)
        diff = []
        
        for tag, i1, i2, j1, j2 in differ.get_opcodes():
            if tag == 'equal':
                diff.append({
                    'type': 'equal',
                    'content': text_a[i1:i2]
                })
            elif tag == 'insert':
                diff.append({
                    'type': 'insert',
                    'content': text_b[j1:j2]
                })
            elif tag == 'delete':
                diff.append({
                    'type': 'delete',
                    'content': text_a[i1:i2]
                })
            elif tag == 'replace':
                diff.append({
                    'type': 'delete',
                    'content': text_a[i1:i2]
                })
                diff.append({
                    'type': 'insert',
                    'content': text_b[j1:j2]
                })
        
        return diff
    
    def _generate_html_diff(self, text_a: str, text_b: str) -> str:
        """Generate an HTML visualization of the diff."""
        diff = self._generate_text_diff(text_a, text_b)
        html_parts = []
        
        for part in diff:
            content = html.escape(part['content'])
            if part['type'] == 'equal':
                html_parts.append(f'<span class="diff-equal">{content}</span>')
            elif part['type'] == 'insert':
                html_parts.append(f'<span class="diff-insert">{content}</span>')
            elif part['type'] == 'delete':
                html_parts.append(f'<span class="diff-delete">{content}</span>')
        
        return ''.join(html_parts)
    
    def _generate_unified_diff(self, text_a: str, text_b: str, 
                              file_a: str, file_b: str) -> str:
        """Generate a unified diff (git-style) between two strings."""
        lines_a = text_a.splitlines(keepends=True)
        lines_b = text_b.splitlines(keepends=True)
        
        unified_diff = difflib.unified_diff(
            lines_a, 
            lines_b, 
            fromfile=file_a, 
            tofile=file_b, 
            lineterm=''
        )
        
        return ''.join(unified_diff)
    
    def _calculate_diff_stats(self, text_a: str, text_b: str) -> Dict[str, Any]:
        """Calculate statistics about the diff."""
        differ = difflib.SequenceMatcher(None, text_a, text_b)
        opcodes = differ.get_opcodes()
        
        stats = {
            "total_chars_a": len(text_a),
            "total_chars_b": len(text_b),
            "total_lines_a": len(text_a.splitlines()),
            "total_lines_b": len(text_b.splitlines()),
            "added_chars": 0,
            "deleted_chars": 0,
            "unchanged_chars": 0,
            "similarity_ratio": differ.ratio()
        }
        
        for tag, i1, i2, j1, j2 in opcodes:
            if tag == 'equal':
                stats["unchanged_chars"] += (i2 - i1)
            elif tag == 'insert':
                stats["added_chars"] += (j2 - j1)
            elif tag == 'delete':
                stats["deleted_chars"] += (i2 - i1)
            elif tag == 'replace':
                stats["deleted_chars"] += (i2 - i1)
                stats["added_chars"] += (j2 - j1)
        
        return stats
    
    def _truncate_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create a truncated version of the context for display."""
        truncated = {}
        for key, value in context.items():
            if isinstance(value, str) and len(value) > 100:
                truncated[key] = value[:100] + '...'
            elif isinstance(value, dict):
                truncated[key] = self._truncate_context(value)
            elif isinstance(value, list) and len(value) > 5:
                truncated[key] = value[:5] + ['...']
            else:
                truncated[key] = value
        return truncated
    
    def _generate_timeline_data(self, version_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate timeline visualization data for template history."""
        timeline = {
            "points": [],
            "connections": []
        }
        
        for i, version in enumerate(version_data):
            # Add timeline point
            point = {
                "id": f"v{i}",
                "version": version["version"],
                "date": version["created_at"],
                "author": version["author"],
                "message": version["message"]
            }
            timeline["points"].append(point)
            
            # Add connection to previous version (if not first)
            if i > 0:
                connection = {
                    "from": f"v{i}",
                    "to": f"v{i-1}",
                    "type": "version"
                }
                timeline["connections"].append(connection)
        
        return timeline
    
    def create_template_report(self, template_id: str) -> Dict[str, Any]:
        """
        Create a comprehensive report about a template, including usage stats.
        
        Args:
            template_id: Identifier of the template
            
        Returns:
            Dictionary containing template report data
        """
        self.logger.info(f"Creating report for template: {template_id}")
        
        try:
            # Get current template
            template = self.template_manager.get_template(template_id)
            if not template:
                return {
                    "success": False,
                    "error": f"Template {template_id} not found",
                    "report": None
                }
            
            # Get template metadata
            metadata = self.template_manager.get_template_metadata(template_id)
            
            # Get usage statistics
            usage_stats = self.template_manager.get_template_usage_stats(template_id)
            
            # Get version history
            history_data = self.visualize_template_history(template_id)
            
            # Get performance metrics if available
            performance_metrics = self.template_manager.get_template_performance_metrics(template_id)
            
            # Create report data
            report_data = {
                "success": True,
                "template_id": template_id,
                "metadata": metadata,
                "content_sample": self.template_manager.get_template_content(template_id)[:200] + '...',
                "usage_stats": usage_stats,
                "history": history_data.get("versions", []),
                "performance_metrics": performance_metrics,
                "generated_at": datetime.now().isoformat()
            }
            
            return report_data
            
        except Exception as e:
            self.logger.error(f"Error creating template report: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "report": None
            }
