import unittest
from unittest.mock import MagicMock, AsyncMock

from src.core.query_engine.handlers.utils.metadata_table_manager import MetadataTableManager


class TestMetadataTableManager(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Create mock objects
        self.chroma_manager_mock = AsyncMock()
        self.access_control_manager_mock = MagicMock()

        # Initialize the MetadataTableManager with mock objects
        self.metadata_table_manager = MetadataTableManager(
            chroma_manager=self.chroma_manager_mock,
            access_control_manager=self.access_control_manager_mock
        )

        # Configure the mock logger to avoid actual logging
        self.metadata_table_manager.logger = MagicMock()

    def test_get_metadata_table_weights(self):
        weights = self.metadata_table_manager.get_metadata_table_weights()
        expected_weights = {
            "model_descriptions": 0.27,
            "model_architectures": 0.27,
            "model_frameworks": 0.0,
            "model_datasets": 0.26,
            "model_training_configs": 0.20,
            "model_date": 0.0,
            "model_file": 0.0,
            "model_git": 0.0
        }
        self.assertEqual(weights, expected_weights)

    async def test_search_metadata_table(self):
        # Mock response from chroma_manager.search
        self.chroma_manager_mock.search.return_value = {
            'results': [{'id': '1', 'metadata': {'model_id': 'model1'}, 'document': 'doc1', 'distance': 0.1}]
        }

        # Test parameters
        table_name = 'model_descriptions'
        query = 'test query'
        filters = {'model_id': {'$eq': 'model1'}}
        limit = 10
        weight = 0.27

        # Call the method
        result = await self.metadata_table_manager.search_metadata_table(
            table_name, query, filters, limit, weight
        )

        # Check if chroma_manager.search was called with correct parameters
        self.chroma_manager_mock.search.assert_called_once_with(
            collection_name=table_name,
            query=query,
            where=filters,
            limit=limit,
            include=["metadatas", "documents", "distances"]
        )

        # Check the result
        expected_result = {
            'table_name': table_name,
            'weight': weight,
            'results': [{'id': '1', 'metadata': {'model_id': 'model1'}, 'document': 'doc1', 'distance': 0.1}]
        }
        self.assertEqual(result, expected_result)

    async def test_search_metadata_table_exception(self):
        # Mock chroma_manager.search to raise an exception
        self.chroma_manager_mock.search.side_effect = Exception("Search failed")

        # Test parameters
        table_name = 'model_descriptions'
        query = 'test query'
        filters = {'model_id': {'$eq': 'model1'}}
        limit = 10
        weight = 0.27

        # Call the method
        result = await self.metadata_table_manager.search_metadata_table(
            table_name, query, filters, limit, weight
        )

        # Check if logger.error was called
        self.metadata_table_manager.logger.error.assert_called_once()

        # Check the result (should be empty results with table_name and weight)
        expected_result = {
            'table_name': table_name,
            'weight': weight,
            'results': []
        }
        self.assertEqual(result, expected_result)

    async def test_get_metadata_table(self):
        # Mock response from chroma_manager.get
        self.chroma_manager_mock.get.return_value = {
            'results': [{'id': '1', 'metadata': {'model_id': 'model1'}}]
        }

        # Test parameters
        table_name = 'model_descriptions'
        filters = {'model_id': {'$eq': 'model1'}}
        limit = 10
        weight = 0.27

        # Call the method
        result = await self.metadata_table_manager.get_metadata_table(
            table_name, filters, limit, weight
        )

        # Check if chroma_manager.get was called with correct parameters
        self.chroma_manager_mock.get.assert_called_once_with(
            collection_name=table_name,
            where=filters,
            limit=limit,
            include=["metadatas"]
        )

        # Check the result
        expected_result = {
            'table_name': table_name,
            'weight': weight,
            'results': [{'id': '1', 'metadata': {'model_id': 'model1'}}]
        }
        self.assertEqual(result, expected_result)

    async def test_get_metadata_table_exception(self):
        # Mock chroma_manager.get to raise an exception
        self.chroma_manager_mock.get.side_effect = Exception("Get failed")

        # Test parameters
        table_name = 'model_descriptions'
        filters = {'model_id': {'$eq': 'model1'}}
        limit = 10
        weight = 0.27

        # Call the method
        result = await self.metadata_table_manager.get_metadata_table(
            table_name, filters, limit, weight
        )

        # Check if logger.error was called
        self.metadata_table_manager.logger.error.assert_called_once()

        # Check the result (should be empty results with table_name and weight)
        expected_result = {
            'table_name': table_name,
            'weight': weight,
            'results': []
        }
        self.assertEqual(result, expected_result)

    async def test_fetch_model_metadata_by_id(self):
        # Mock response from chroma_manager.get
        self.chroma_manager_mock.get.return_value = {
            'results': [{'metadata': {'model_id': 'model1', 'name': 'test model'}}]
        }

        # Test parameters
        table_name = 'model_descriptions'
        model_id = 'model1'

        # Call the method
        result = await self.metadata_table_manager.fetch_model_metadata_by_id(
            table_name, model_id
        )

        # Check if chroma_manager.get was called with correct parameters
        self.chroma_manager_mock.get.assert_called_once_with(
            collection_name=table_name,
            where={'model_id': {'$eq': model_id}},
            include=["metadatas"]
        )

        # Check the result
        expected_result = {'metadata': {'descriptions': {'model_id': 'model1', 'name': 'test model'}}}
        self.assertEqual(result, expected_result)

    async def test_fetch_model_metadata_by_id_with_access_control(self):
        # Mock response from chroma_manager.get
        self.chroma_manager_mock.get.return_value = {
            'results': [{'metadata': {'model_id': 'model1', 'name': 'test model'}}]
        }

        # Mock access_control_manager.create_access_filter
        access_filter = {'user_id': {'$eq': 'user1'}}
        self.access_control_manager_mock.create_access_filter.return_value = access_filter

        # Test parameters
        table_name = 'model_descriptions'
        model_id = 'model1'
        user_id = 'user1'

        # Call the method
        result = await self.metadata_table_manager.fetch_model_metadata_by_id(
            table_name, model_id, user_id
        )

        # Check if access_control_manager.create_access_filter was called with correct parameters
        self.access_control_manager_mock.create_access_filter.assert_called_once_with(user_id)

        # Check if chroma_manager.get was called with correct parameters (including access filter)
        self.chroma_manager_mock.get.assert_called_once_with(
            collection_name=table_name,
            where={'$and': [{'model_id': {'$eq': model_id}}, access_filter]},
            include=["metadatas"]
        )

        # Check the result
        expected_result = {'metadata': {'descriptions': {'model_id': 'model1', 'name': 'test model'}}}
        self.assertEqual(result, expected_result)

    async def test_fetch_model_metadata_by_id_exception(self):
        # Mock chroma_manager.get to raise an exception
        self.chroma_manager_mock.get.side_effect = Exception("Get failed")

        # Test parameters
        table_name = 'model_descriptions'
        model_id = 'model1'

        # Call the method
        result = await self.metadata_table_manager.fetch_model_metadata_by_id(
            table_name, model_id
        )

        # Check if logger.error was called
        self.metadata_table_manager.logger.error.assert_called_once()

        # Check the result (should be None)
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()