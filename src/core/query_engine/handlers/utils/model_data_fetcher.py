import logging
from typing import Dict, List, Any, Optional


class ModelDataFetcher:
    """Utility class for fetching model data and metadata."""

    def __init__(self, chroma_manager, access_control_manager=None):
        self.chroma_manager = chroma_manager
        self.access_control_manager = access_control_manager
        self.logger = logging.getLogger(__name__)

    async def fetch_model_metadata(self, model_id: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Fetch metadata for a specific model with access control.

        Args:
            model_id: The model identifier
            user_id: Optional user ID for access control

        Returns:
            Dictionary containing model metadata or None if not found or no access
        """
        # Prepare filters to get model metadata
        filters = {'model_id': {'$eq': model_id}}

        # Apply access control filter if applicable
        if self.access_control_manager and user_id:
            access_filter = self.access_control_manager.create_access_filter(user_id)

            # Combine filters with access control
            filters = {
                "$and": [
                    filters,
                    access_filter
                ]
            }

        # Fetch model metadata from Chroma
        try:
            metadata_results = await self.chroma_manager.get(
                collection_name="model_scripts_metadata",
                where=filters,
                include=["metadatas"]
            )

            # Return the first result's metadata if available
            if metadata_results and metadata_results.get('results'):
                return metadata_results['results'][0].get('metadata', {})

        except Exception as e:
            self.logger.error(f"Error fetching model metadata for {model_id}: {e}")

        return None

    async def fetch_model_data(self, model_id: str, dimensions: List[str], user_id: Optional[str] = None) -> Dict[
        str, Any]:
        """
        Fetch data for a specific model with access control.

        Args:
            model_id: The model identifier
            dimensions: List of data dimensions to fetch
            user_id: Optional user ID for access control

        Returns:
            Dictionary containing model data
        """
        # Prepare filters to get model data
        filters = {'model_id': {'$eq': model_id}}

        # Apply access control filter if applicable
        if self.access_control_manager and user_id:
            access_filter = self.access_control_manager.create_access_filter(user_id)

            # Combine filters with access control
            filters = {
                "$and": [
                    filters,
                    access_filter
                ]
            }

        # Fetch model data from Chroma
        model_data = await self.chroma_manager.get(
            collection_name="model_script_processing",
            where=filters,
            include=["metadata"]
        )

        # Process the results
        if not model_data.get('results'):
            return {'model_id': model_id, 'found': False}

        result = {'model_id': model_id, 'found': True}

        # Extract metadata from the first document (should be the main model document)
        metadata = model_data['results'][0].get('metadata', {})

        # Extract dimensions
        for dimension in dimensions:
            if dimension == 'architecture' and 'architecture_type' in metadata:
                result['architecture'] = {
                    'type': metadata.get('architecture_type', {}).get('value', 'unknown'),
                    'hidden_size': metadata.get('model_dimensions', {}).get('hidden_size', {}).get('value'),
                    'num_layers': metadata.get('model_dimensions', {}).get('num_layers', {}).get('value'),
                    'num_attention_heads': metadata.get('model_dimensions', {}).get('num_attention_heads', {}).get(
                        'value'),
                    'total_parameters': metadata.get('model_dimensions', {}).get('total_parameters', {}).get('value')
                }
            elif dimension == 'performance' and 'performance' in metadata:
                result['performance'] = {
                    'accuracy': metadata.get('performance', {}).get('accuracy', {}).get('value'),
                    'loss': metadata.get('performance', {}).get('loss', {}).get('value'),
                    'perplexity': metadata.get('performance', {}).get('perplexity', {}).get('value'),
                    'eval_dataset': metadata.get('performance', {}).get('eval_dataset', {}).get('value')
                }
            elif dimension == 'training' and 'training_config' in metadata:
                result['training'] = {
                    'batch_size': metadata.get('training_config', {}).get('batch_size', {}).get('value'),
                    'learning_rate': metadata.get('training_config', {}).get('learning_rate', {}).get('value'),
                    'optimizer': metadata.get('training_config', {}).get('optimizer', {}).get('value'),
                    'epochs': metadata.get('training_config', {}).get('epochs', {}).get('value'),
                    'training_time_hours': metadata.get('training_config', {}).get('training_time_hours', {}).get(
                        'value'),
                    'hardware_used': metadata.get('training_config', {}).get('hardware_used', {}).get('value')
                }
            elif dimension == 'dataset' and 'dataset' in metadata:
                result['dataset'] = {
                    'name': metadata.get('dataset', {}).get('name', {}).get('value'),
                    'version': metadata.get('dataset', {}).get('version', {}).get('value'),
                    'num_samples': metadata.get('dataset', {}).get('num_samples', {}).get('value')
                }
            elif dimension == 'framework' and 'framework' in metadata:
                result['framework'] = {
                    'name': metadata.get('framework', {}).get('name'),
                    'version': metadata.get('framework', {}).get('version')
                }

        # Add basic metadata
        result['basic'] = {
            'version': metadata.get('version'),
            'creation_date': metadata.get('creation_date'),
            'last_modified_date': metadata.get('last_modified_date'),
            'predecessor_models': metadata.get('predecessor_models', [])
        }

        return result