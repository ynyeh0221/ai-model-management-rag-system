import json
import markdown
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime
import logging
from jinja2 import Template, Environment, FileSystemLoader

class ResponseFormatter:
    """
    Formats retrieved data into coherent, informative responses.
    
    This class handles different response types including text, comparison reports,
    image galleries, and ensures proper citation of information sources.
    """
    
    def __init__(self, template_manager):
        """
        Initialize the ResponseFormatter with a template manager.
        
        Args:
            template_manager: Manager for prompt templates used in formatting responses
        """
        self.template_manager = template_manager
        self.logger = logging.getLogger(__name__)
        
        # Configure Jinja2 environment for template rendering
        self.env = Environment(
            loader=FileSystemLoader("templates/"),
            autoescape=True
        )

    def _get_default_template(self, response_type: str) -> Template:
        """Get a default template for the given response type."""
        template = self.template_manager.get_template(f"default_{response_type}")

        if not template:
            # Create a very basic default template if no default template is found
            if response_type == "html":
                template_str = '<div><p>{{ results|length }} results found</p>{% for result in results %}<div><h3>{{ result.id }}</h3><p>{{ result.content }}</p></div>{% endfor %}</div>'
            elif response_type == "markdown":
                template_str = '## Results\n\n{{ results|length }} results found\n\n{% for result in results %}### {{ result.id }}\n\n{{ result.content }}\n\n{% endfor %}'
            else:  # text
                template_str = 'Results:\n\n{{ results|length }} results found\n\n{% for result in results %}{{ result.id }}:\n{{ result.content }}\n\n{% endfor %}'

            # Create a template from the string
            template = self.env.from_string(template_str)

        return template

    def format_response(self, results: List[Dict[str, Any]], query: Union[Dict[str, Any], str],
                        response_type: str = "text") -> Dict[str, Any]:
        """
        Format results into a response based on the query type and desired response format.
        """
        # Convert string query to dict if needed.
        if isinstance(query, str):
            query = {"intent": response_type, "type": response_type}
        elif isinstance(query, dict) and "type" not in query and "intent" in query:
            query["type"] = query["intent"]

        self.logger.info(
            f"Formatting response of type: {response_type} for query intent: {query.get('intent', 'general')}")

        # Filter out fallback LLM responses (with id starting "llm_response")
        filtered_results = [r for r in results if
                            not (isinstance(r, dict) and r.get("id", "").startswith("llm_response"))]
        if not filtered_results:
            rendered_content = "No results found."
            return {
                'content': rendered_content,
                'citations': [],
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'result_count': 0
                }
            }
        # Use the filtered results for further processing.
        results = filtered_results

        # Get appropriate template based on query intent and response type.
        # Try specific template combinations first, then fall back to more general ones
        template = None

        # Try intent_response_type combination
        template_key = f"{query.get('intent', 'general')}_{response_type}"
        template = self.template_manager.get_template(template_key)

        # If not found, try just the intent
        if not template:
            template_key = f"{query.get('intent', 'general')}"
            template = self.template_manager.get_template(template_key)

        # If still not found, try information_retrieval for retrieval intents
        if not template and query.get('intent') == 'retrieval':
            template = self.template_manager.get_template("information_retrieval")

        # Finally, fall back to default
        if not template:
            self.logger.warning(
                f"No template found for {query.get('intent', 'general')}_{response_type}, using default")
            template = self._get_default_template(response_type)

        # Process results based on query intent.
        if query.get('intent') == 'model_info':
            return self._format_model_info(results, template, response_type)
        elif query.get('intent') == 'model_comparison':
            model_ids = query.get('parameters', {}).get('model_ids', [])
            comparison_points = query.get('parameters', {}).get('comparison_points', [])
            return self.format_comparison(results, model_ids, comparison_points, template, response_type)
        elif query.get('intent') == 'image_search':
            return self.format_image_gallery(results, template, response_type)
        else:
            # General information response.
            return self._format_general_info(results, template, response_type)

    def _format_general_info(self, results: List[Dict[str, Any]],
                            template: Template, response_type: str) -> Dict[str, Any]:
        """Format general information response."""
        # Extract relevant information from results
        formatted_results = []
        for result in results:
            formatted_result = {
                'id': result.get('id'),
                'content': result.get('content', ''),
                'metadata': {
                    'model_id': result.get('metadata', {}).get('model_id', 'Unknown'),
                    'version': result.get('metadata', {}).get('version', 'Unknown'),
                    'framework': result.get('metadata', {}).get('framework', {}).get('name', 'Unknown'),
                    'description': self._generate_description(result),
                    'source': result.get('metadata', {}).get('filepath', 'Unknown')
                }
            }
            formatted_results.append(formatted_result)
        
        # Render template with results
        context = {
            'results': formatted_results,
            'timestamp': datetime.now().isoformat(),
            'total_results': len(results)
        }
        
        rendered_content = template.render(**context)
        
        # Post-process based on response type
        if response_type == "markdown":
            rendered_content = self._format_markdown(rendered_content)
        elif response_type == "html":
            rendered_content = self._format_html(rendered_content)
        
        # Include citations
        response = {
            'content': rendered_content,
            'citations': self._extract_citations(results),
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'result_count': len(results)
            }
        }
        
        return response
    
    def _format_model_info(self, results: List[Dict[str, Any]], 
                          template: Template, response_type: str) -> Dict[str, Any]:
        """Format model information response."""
        # For model info, we may need to combine information from multiple results
        if not results:
            return {'content': "No model information found.", 'citations': []}
        
        # Group results by model_id
        model_info = {}
        for result in results:
            model_id = result.get('metadata', {}).get('model_id')
            if not model_id:
                continue
                
            if model_id not in model_info:
                model_info[model_id] = {
                    'id': model_id,
                    'version': result.get('metadata', {}).get('version', 'Unknown'),
                    'framework': result.get('metadata', {}).get('framework', {}).get('name', 'Unknown'),
                    'architecture_type': result.get('metadata', {}).get('architecture_type', {}).get('value', 'Unknown'),
                    'dimensions': result.get('metadata', {}).get('model_dimensions', {}),
                    'performance': result.get('metadata', {}).get('performance', {}),
                    'training_config': result.get('metadata', {}).get('training_config', {}),
                    'dataset': result.get('metadata', {}).get('dataset', {}),
                    'code_samples': [],
                    'citations': []
                }
            
            # Add code sample if this result has content
            if result.get('content'):
                model_info[model_id]['code_samples'].append({
                    'section_type': result.get('metadata', {}).get('section_type', 'Unknown'),
                    'content': result.get('content'),
                    'citation_id': result.get('id')
                })
            
            # Add citation for this result
            model_info[model_id]['citations'].append({
                'id': result.get('id'),
                'filepath': result.get('metadata', {}).get('filepath', 'Unknown')
            })
        
        # Render template with model information
        context = {
            'models': list(model_info.values()),
            'timestamp': datetime.now().isoformat()
        }
        
        rendered_content = template.render(**context)
        
        # Post-process based on response type
        if response_type == "markdown":
            rendered_content = self._format_markdown(rendered_content)
        elif response_type == "html":
            rendered_content = self._format_html(rendered_content)
        
        # Include citations
        response = {
            'content': rendered_content,
            'citations': self._extract_citations(results),
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'model_count': len(model_info)
            }
        }
        
        return response
    
    def format_comparison(self, results: List[Dict[str, Any]], model_ids: List[str], 
                         comparison_points: List[str], template: Optional[Template] = None,
                         response_type: str = "markdown") -> Dict[str, Any]:
        """
        Format a comparison of two or more models.
        
        Args:
            results: List of documents containing model information
            model_ids: List of model IDs to compare
            comparison_points: List of specific points to compare
            template: Optional template to use for formatting
            response_type: Response format type
            
        Returns:
            Formatted comparison response
        """
        self.logger.info(f"Formatting comparison for models: {model_ids}")
        
        if not template:
            template = self.template_manager.get_template(f"comparison_{response_type}")
        
        # Group results by model_id
        model_data = {}
        for model_id in model_ids:
            model_data[model_id] = {point: None for point in comparison_points}
            model_data[model_id]['citations'] = []
        
        # Extract relevant comparison data from results
        for result in results:
            model_id = result.get('metadata', {}).get('model_id')
            if model_id not in model_ids:
                continue
            
            # For each comparison point, extract data if available
            for point in comparison_points:
                if point == "architecture":
                    if not model_data[model_id][point]:
                        model_data[model_id][point] = {
                            'type': result.get('metadata', {}).get('architecture_type', {}).get('value'),
                            'dimensions': result.get('metadata', {}).get('model_dimensions', {})
                        }
                elif point == "performance":
                    if not model_data[model_id][point]:
                        model_data[model_id][point] = result.get('metadata', {}).get('performance', {})
                elif point == "training":
                    if not model_data[model_id][point]:
                        model_data[model_id][point] = {
                            'config': result.get('metadata', {}).get('training_config', {}),
                            'dataset': result.get('metadata', {}).get('dataset', {})
                        }
                elif point == "framework":
                    if not model_data[model_id][point]:
                        model_data[model_id][point] = result.get('metadata', {}).get('framework', {})
            
            # Add citation
            model_data[model_id]['citations'].append({
                'id': result.get('id'),
                'filepath': result.get('metadata', {}).get('filepath', 'Unknown')
            })
        
        # Create comparison table data
        comparison_tables = {}
        for point in comparison_points:
            if point == "performance":
                comparison_tables[point] = self._create_performance_comparison_table(model_data, point)
            elif point == "architecture":
                comparison_tables[point] = self._create_architecture_comparison_table(model_data, point)
            elif point == "training":
                comparison_tables[point] = self._create_training_comparison_table(model_data, point)
            else:
                # Generic comparison table
                comparison_tables[point] = self._create_generic_comparison_table(model_data, point)
        
        # Render template with comparison data
        context = {
            'model_ids': model_ids,
            'comparison_points': comparison_points,
            'model_data': model_data,
            'comparison_tables': comparison_tables,
            'timestamp': datetime.now().isoformat()
        }
        
        rendered_content = template.render(**context)
        
        # Post-process based on response type
        if response_type == "markdown":
            rendered_content = self._format_markdown(rendered_content)
        elif response_type == "html":
            rendered_content = self._format_html(rendered_content)
        
        # Collect all citations
        all_citations = []
        for model_id in model_ids:
            all_citations.extend(model_data[model_id]['citations'])
        
        response = {
            'content': rendered_content,
            'citations': all_citations,
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'model_count': len(model_ids),
                'comparison_points': comparison_points
            }
        }
        
        return response
    
    def _create_performance_comparison_table(self, model_data: Dict[str, Any], 
                                            point: str) -> Dict[str, Any]:
        """Create a performance metrics comparison table."""
        # Extract performance metrics for each model
        metrics = set()
        for model_id, data in model_data.items():
            if data[point]:
                metrics.update(data[point].keys())
        
        # Create table data
        table_data = {'headers': ['Metric'] + list(model_data.keys())}
        rows = []
        
        for metric in metrics:
            if metric == 'eval_dataset':  # Skip this as it's metadata, not a metric
                continue
                
            row = {'metric': metric}
            for model_id in model_data.keys():
                if model_data[model_id][point] and metric in model_data[model_id][point]:
                    metric_data = model_data[model_id][point][metric]
                    if isinstance(metric_data, dict) and 'value' in metric_data:
                        row[model_id] = metric_data['value']
                    else:
                        row[model_id] = metric_data
                else:
                    row[model_id] = 'N/A'
            rows.append(row)
        
        table_data['rows'] = rows
        return table_data
    
    def _create_architecture_comparison_table(self, model_data: Dict[str, Any], 
                                             point: str) -> Dict[str, Any]:
        """Create an architecture comparison table."""
        # For architecture, we have two sub-tables: type and dimensions
        
        # Type table
        type_table = {'headers': ['Model', 'Architecture Type']}
        type_rows = []
        
        for model_id, data in model_data.items():
            if data[point] and 'type' in data[point]:
                type_rows.append({'model': model_id, 'type': data[point]['type']})
            else:
                type_rows.append({'model': model_id, 'type': 'N/A'})
        
        # Dimensions table
        dimension_metrics = set()
        for model_id, data in model_data.items():
            if data[point] and 'dimensions' in data[point]:
                dimension_metrics.update(data[point]['dimensions'].keys())
        
        dimensions_table = {'headers': ['Dimension'] + list(model_data.keys())}
        dimension_rows = []
        
        for metric in dimension_metrics:
            row = {'metric': metric}
            for model_id in model_data.keys():
                if (data[point] and 'dimensions' in data[point] and 
                    metric in data[point]['dimensions']):
                    dim_data = data[point]['dimensions'][metric]
                    if isinstance(dim_data, dict) and 'value' in dim_data:
                        row[model_id] = dim_data['value']
                    else:
                        row[model_id] = dim_data
                else:
                    row[model_id] = 'N/A'
            dimension_rows.append(row)
        
        dimensions_table['rows'] = dimension_rows
        
        return {
            'type': {'table': type_table, 'rows': type_rows},
            'dimensions': {'table': dimensions_table, 'rows': dimension_rows}
        }
    
    def _create_training_comparison_table(self, model_data: Dict[str, Any], 
                                         point: str) -> Dict[str, Any]:
        """Create a training configuration comparison table."""
        # For training, we have two sub-tables: config and dataset
        
        # Training config table
        config_metrics = set()
        for model_id, data in model_data.items():
            if data[point] and 'config' in data[point]:
                config_metrics.update(data[point]['config'].keys())
        
        config_table = {'headers': ['Parameter'] + list(model_data.keys())}
        config_rows = []
        
        for metric in config_metrics:
            row = {'metric': metric}
            for model_id in model_data.keys():
                if (data[point] and 'config' in data[point] and 
                    metric in data[point]['config']):
                    config_data = data[point]['config'][metric]
                    if isinstance(config_data, dict) and 'value' in config_data:
                        row[model_id] = config_data['value']
                    else:
                        row[model_id] = config_data
                else:
                    row[model_id] = 'N/A'
            config_rows.append(row)
        
        config_table['rows'] = config_rows
        
        # Dataset table
        dataset_metrics = set()
        for model_id, data in model_data.items():
            if data[point] and 'dataset' in data[point]:
                dataset_metrics.update(data[point]['dataset'].keys())
        
        dataset_table = {'headers': ['Parameter'] + list(model_data.keys())}
        dataset_rows = []
        
        for metric in dataset_metrics:
            row = {'metric': metric}
            for model_id in model_data.keys():
                if (data[point] and 'dataset' in data[point] and 
                    metric in data[point]['dataset']):
                    dataset_data = data[point]['dataset'][metric]
                    if isinstance(dataset_data, dict) and 'value' in dataset_data:
                        row[model_id] = dataset_data['value']
                    else:
                        row[model_id] = dataset_data
                else:
                    row[model_id] = 'N/A'
            dataset_rows.append(row)
        
        dataset_table['rows'] = dataset_rows
        
        return {
            'config': {'table': config_table, 'rows': config_rows},
            'dataset': {'table': dataset_table, 'rows': dataset_rows}
        }
    
    def _create_generic_comparison_table(self, model_data: Dict[str, Any], 
                                        point: str) -> Dict[str, Any]:
        """Create a generic comparison table for other types of data."""
        table_data = {'headers': ['Model', point.capitalize()]}
        rows = []
        
        for model_id, data in model_data.items():
            if data[point]:
                # Convert to string representation if it's a complex object
                if isinstance(data[point], dict):
                    value = json.dumps(data[point], indent=2)
                else:
                    value = str(data[point])
                rows.append({'model': model_id, 'value': value})
            else:
                rows.append({'model': model_id, 'value': 'N/A'})
        
        table_data['rows'] = rows
        return table_data
    
    def format_image_gallery(self, images: List[Dict[str, Any]], 
                            template: Optional[Template] = None,
                            response_type: str = "html") -> Dict[str, Any]:
        """
        Format an image gallery response.
        
        Args:
            images: List of image documents with metadata
            template: Optional template to use for formatting
            response_type: Response format type
            
        Returns:
            Formatted image gallery response
        """
        self.logger.info(f"Formatting image gallery with {len(images)} images")
        
        if not template:
            template = self.template_manager.get_template(f"image_gallery_{response_type}")
        
        # Process image data
        formatted_images = []
        for image in images:
            formatted_image = {
                'id': image.get('id'),
                'thumbnail_path': image.get('metadata', {}).get('thumbnail_path'),
                'image_path': image.get('metadata', {}).get('image_path'),
                'prompt': image.get('metadata', {}).get('prompt', {}).get('value', ''),
                'model_id': image.get('metadata', {}).get('source_model_id', 'Unknown'),
                'resolution': image.get('metadata', {}).get('resolution', {}),
                'style_tags': image.get('metadata', {}).get('style_tags', {}).get('value', []),
                'creation_date': image.get('metadata', {}).get('creation_date'),
                'clip_score': image.get('metadata', {}).get('clip_score', {}).get('value', 'N/A'),
                'generation_params': {
                    'guidance_scale': image.get('metadata', {}).get('guidance_scale', {}).get('value', 'N/A'),
                    'steps': image.get('metadata', {}).get('num_inference_steps', {}).get('value', 'N/A'),
                    'seed': image.get('metadata', {}).get('seed', {}).get('value', 'N/A')
                }
            }
            formatted_images.append(formatted_image)
        
        # Group images by model
        images_by_model = {}
        for image in formatted_images:
            model_id = image['model_id']
            if model_id not in images_by_model:
                images_by_model[model_id] = []
            images_by_model[model_id].append(image)
        
        # Render template with image data
        context = {
            'images': formatted_images,
            'images_by_model': images_by_model,
            'model_list': list(images_by_model.keys()),
            'total_images': len(formatted_images),
            'timestamp': datetime.now().isoformat()
        }
        
        rendered_content = template.render(**context)
        
        # Post-process based on response type
        if response_type == "markdown":
            rendered_content = self._format_markdown(rendered_content)
        elif response_type == "html":
            rendered_content = self._format_html(rendered_content)
        
        # Include citations
        response = {
            'content': rendered_content,
            'citations': self._extract_citations(images),
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'image_count': len(images),
                'model_count': len(images_by_model)
            }
        }
        
        return response
    
    def include_citations(self, response: Dict[str, Any], 
                         results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Include citations in a response.
        
        Args:
            response: Formatted response content
            results: Original retrieved documents
            
        Returns:
            Response with citations added
        """
        if 'citations' not in response:
            response['citations'] = self._extract_citations(results)
        
        # Add citation references to the content
        content = response['content']
        citations = response['citations']
        
        # Add citation markers to content if not already present
        # This would depend on how citations are formatted in templates
        
        return response
    
    def _extract_citations(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract citation information from results."""
        citations = []
        for result in results:
            citation = {
                'id': result.get('id'),
                'type': result.get('metadata', {}).get('type', 'Unknown'),
                'filepath': result.get('metadata', {}).get('filepath', 'Unknown'),
                'model_id': result.get('metadata', {}).get('model_id', 
                                                         result.get('metadata', {}).get('source_model_id')),
                'timestamp': datetime.now().isoformat()
            }
            citations.append(citation)
        return citations
    
    def _generate_description(self, result: Dict[str, Any]) -> str:
        """Generate a description for a result."""
        metadata = result.get('metadata', {})
        if metadata.get('type') == 'model_script':
            return f"Model script for {metadata.get('model_id')} (v{metadata.get('version')})"
        elif metadata.get('type') == 'generated_image':
            return f"Image generated by {metadata.get('source_model_id')} with prompt: {metadata.get('prompt', {}).get('value', 'No prompt')}"
        else:
            return f"Document: {metadata.get('filepath', 'Unknown')}"
    
    def _format_markdown(self, content: str) -> str:
        """Format content as markdown."""
        # No additional processing needed, as the content is already in markdown
        return content
    
    def _format_html(self, content: str) -> str:
        """Format content as HTML."""
        # Convert markdown to HTML if needed
        if not content.strip().startswith('<'):  # Simple check if content is not already HTML
            content = markdown.markdown(content)
        return content
