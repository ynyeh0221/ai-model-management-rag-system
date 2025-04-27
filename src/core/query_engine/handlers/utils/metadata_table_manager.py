import logging
from typing import Dict


class MetadataTableManager:
    """
    Utility class for handling metadata table operations.

    This class provides functionality for searching and retrieving data from metadata tables
    stored in a Chroma vector database. It supports weighted searches across multiple tables,
    filtering results based on query criteria, and access control mechanisms.

    Attributes:
        chroma_manager: An instance of a Chroma database manager that provides search and retrieval capabilities.
        access_control_manager: Optional manager for handling access control permissions.
        logger: Logger instance for error reporting and debugging.

    Example:
        >>> chroma_mgr = ChromaManager()
        >>> access_ctrl = AccessControlManager()
        >>> metadata_mgr = MetadataTableManager(chroma_mgr, access_ctrl)
        >>> weights = metadata_mgr.get_metadata_table_weights()
        >>> async def search_example():
        ...     results = await metadata_mgr.search_metadata_table("model_descriptions", "transformer", {}, 10, 0.27)
    """

    def __init__(self, chroma_manager, access_control_manager=None):
        self.chroma_manager = chroma_manager
        self.access_control_manager = access_control_manager
        self.logger = logging.getLogger(__name__)

    def get_metadata_table_weights(self) -> Dict[str, float]:
        """Define metadata tables to search with their weights."""
        return {
            "model_descriptions": 0.27,
            "model_architectures": 0.27,
            "model_frameworks": 0.0,
            "model_datasets": 0.26,
            "model_training_configs": 0.20,
            "model_date": 0.0,
            "model_file": 0.0,
            "model_git": 0.0
        }

    async def search_metadata_table(self, table_name, query, filters, limit, weight):
        """Search a specific metadata table and return weighted results."""
        try:
            search_params = {
                'query': query,
                'where': filters,
                'limit': limit,
                'include': ["metadatas", "documents", "distances"]
            }

            results = await self.chroma_manager.search(
                collection_name=table_name,
                **search_params
            )

            # Add the table name and weight to the results
            return {
                'table_name': table_name,
                'weight': weight,
                'results': results.get('results', [])
            }
        except Exception as e:
            self.logger.error(f"Error searching metadata table {table_name}: {e}")
            return {
                'table_name': table_name,
                'weight': weight,
                'results': []
            }

    async def get_metadata_table(self, table_name, filters, limit, weight):
        """Get data from a specific metadata table based on filters."""
        try:
            get_params = {
                'where': filters,
                'limit': limit,
                'include': ["metadatas"]
            }

            results = await self.chroma_manager.get(
                collection_name=table_name,
                **get_params
            )

            # Add the table name and weight to the results
            return {
                'table_name': table_name,
                'weight': weight,
                'results': results.get('results', [])
            }
        except Exception as e:
            self.logger.error(f"Error getting metadata from table {table_name}: {e}")
            return {
                'table_name': table_name,
                'weight': weight,
                'results': []
            }

    async def fetch_model_metadata_by_id(self, table_name, model_id, user_id=None):
        """Fetch metadata for a specific model from a specific table."""
        try:
            # Prepare filters to get model metadata
            filters = {'model_id': {'$eq': model_id}}

            # Apply access control filter if applicable
            if self.access_control_manager and user_id:
                access_filter = self.access_control_manager.create_access_filter(user_id)
                filters = {
                    "$and": [
                        filters,
                        access_filter
                    ]
                }

            # Fetch model metadata from specific table
            result = await self.chroma_manager.get(
                collection_name=table_name,
                where=filters,
                include=["metadatas"]
            )

            # Return the first result if available
            if result and result.get('results'):
                metadata = result['results'][0].get('metadata', {})
                # Create a single top-level field for this table's data to avoid conflicts
                table_key = table_name.replace('model_', '')  # e.g., "model_frameworks" becomes "frameworks"
                return {'metadata': {table_key: metadata}}

        except Exception as e:
            self.logger.error(f"Error fetching model metadata for {model_id} from {table_name}: {e}")

        return None