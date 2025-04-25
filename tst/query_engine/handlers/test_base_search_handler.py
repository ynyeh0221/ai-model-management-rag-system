import unittest
from unittest.mock import AsyncMock, MagicMock

# Import the class we're testing
from src.query_engine.handlers.base_search_handler import BaseSearchHandler
# Import dependencies
from src.query_engine.handlers.utils.distance_normalizer import DistanceNormalizer
from src.query_engine.handlers.utils.filter_translator import FilterTranslator
from src.vector_db_manager.access_control import AccessControlManager
from src.vector_db_manager.chroma_manager import ChromaManager


class TestBaseSearchHandler(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Create mock dependencies
        self.chroma_manager = MagicMock(spec=ChromaManager)
        self.access_control_manager = MagicMock(spec=AccessControlManager)
        self.filter_translator = MagicMock(spec=FilterTranslator)
        self.distance_normalizer = MagicMock(spec=DistanceNormalizer)

        # Make search method an AsyncMock
        self.chroma_manager.search = AsyncMock()

        # Create the instance we're testing
        self.handler = BaseSearchHandler(
            chroma_manager=self.chroma_manager,
            access_control_manager=self.access_control_manager,
            filter_translator=self.filter_translator,
            distance_normalizer=self.distance_normalizer
        )

    async def test_get_collection_distance_stats_for_query(self):
        # Setup mock return values
        self.chroma_manager.search.return_value = {
            'results': [
                {'metadata': {'id': '1'}},
                {'metadata': {'id': '2'}}
            ]
        }
        self.distance_normalizer.extract_search_distance.side_effect = [0.1, 0.2]

        # Call the method
        collections = ["collection1", "collection2"]
        user_id = "user123"
        result = await self.handler._get_collection_distance_stats_for_query(
            query="test query", collections=collections, user_id=user_id
        )

        # Assertions
        self.assertEqual(len(result), 2)
        self.assertIn("collection1", result)
        self.assertIn("collection2", result)
        self.assertEqual(result["collection1"]["min"], 0.1)
        self.assertEqual(result["collection1"]["max"], 0.2)
        self.assertEqual(self.chroma_manager.search.call_count, 2)

        # Verify the access filter was created
        self.access_control_manager.create_access_filter.assert_called_with(user_id)

    def test_extract_text_search_parameters(self):
        # Setup mock return values
        self.access_control_manager.create_access_filter.return_value = {"user_filter": "test"}
        self.filter_translator.translate_to_chroma.return_value = {"translated_filter": "test"}

        # Test with valid parameters
        parameters = {
            "user_id": "user123",
            "limit": 20,
            "filters": {"field": "value"}
        }
        user_id, limit, filters = self.handler._extract_text_search_parameters(parameters)

        # Assertions
        self.assertEqual(user_id, "user123")
        self.assertEqual(limit, 20)
        self.assertEqual(filters, {"translated_filter": "test"})

        # Test with invalid limit
        parameters = {
            "user_id": "user123",
            "limit": -5,
            "filters": {"field": "value"}
        }
        user_id, limit, filters = self.handler._extract_text_search_parameters(parameters)
        self.assertEqual(limit, 10)  # Should default to 10

        # Test with no filters
        parameters = {
            "user_id": "user123",
            "limit": 20
        }
        user_id, limit, filters = self.handler._extract_text_search_parameters(parameters)
        self.assertEqual(filters, {"translated_filter": "test"})

    async def test_search_all_metadata_tables(self):
        # Setup mock return values
        table_weights = {
            "model_descriptions": 0.5,
            "model_params": 0.3,
            "model_tags": 0.2
        }

        # Mock search results for each table
        self.chroma_manager.search.side_effect = [
            # model_descriptions results
            {
                'results': [
                    {'metadata': {'model_id': 'model1', 'description': 'Test description'}},
                    {'metadata': {'model_id': 'model2', 'description': 'Another description'}}
                ]
            },
            # model_params results
            {
                'results': [
                    {'metadata': {'model_id': 'model1', 'param': 'value1'}},
                    {'metadata': {'model_id': 'model3', 'param': 'value2'}}
                ]
            },
            # model_tags results
            {
                'results': [
                    {'metadata': {'model_id': 'model2', 'tag': 'tag1'}},
                    {'metadata': {'model_id': 'model4', 'tag': 'tag2'}}
                ]
            }
        ]

        # Call the method
        user_id = "user123"
        query = "test query"
        chroma_filters = {"field": "value"}
        requested_limit = 10

        result = await self.handler._search_all_metadata_tables(
            query=query,
            chroma_filters=chroma_filters,
            requested_limit=requested_limit,
            table_weights=table_weights,
            user_id=user_id
        )

        # Assertions
        self.assertEqual(len(result), 4)  # 4 unique models
        self.assertIn('model1', result)
        self.assertIn('model2', result)
        self.assertIn('model3', result)
        self.assertIn('model4', result)

        # Check tables listed for model1
        model1_tables = result['model1']['tables']
        self.assertEqual(len(model1_tables), 2)
        self.assertIn('model_descriptions', model1_tables)
        self.assertIn('model_params', model1_tables)

        # Check if search was called correctly
        self.assertEqual(self.chroma_manager.search.call_count, 3)

    async def test_fetch_complete_model_metadata(self):
        # Setup initial results
        all_results = {
            'model1': {
                'model_id': 'model1',
                'tables': ['model_descriptions'],
                'table_initial_distances': {},
                'match_source': 'metadata'
            },
            'model2': {
                'model_id': 'model2',
                'tables': ['model_tags'],
                'table_initial_distances': {},
                'match_source': 'metadata'
            }
        }

        table_weights = {
            'model_descriptions': 0.5,
            'model_params': 0.3,
            'model_tags': 0.2
        }

        # Mock search results
        self.chroma_manager.search.side_effect = [
            # model_params for model1
            {
                'results': [
                    {
                        'metadata': {'model_id': 'model1', 'param': 'value1'},
                        'distance': 0.1
                    }
                ]
            },
            # model_tags for model1
            {
                'results': [
                    {
                        'metadata': {'model_id': 'model1', 'tag': 'tag1'},
                        'distance': 0.2
                    }
                ]
            },
            # model_params for model2
            {
                'results': []
            },
            # model_tags for model2 (already in initial results)
            {
                'results': [
                    {
                        'metadata': {'model_id': 'model2', 'tag': 'tag2'},
                        'distance': 0.3
                    }
                ]
            }
        ]

        self.distance_normalizer.extract_search_distance.side_effect = [0.1, 0.2, 0.3]

        # Call the method
        user_id = "user123"
        query = "test query"

        result = await self.handler._fetch_complete_model_metadata(
            query=query,
            all_results=all_results,
            table_weights=table_weights,
            user_id=user_id
        )

        # Assertions
        self.assertEqual(len(result), 2)

        # Check model1 metadata was updated
        self.assertIn('metadata', result['model1'])
        self.assertEqual(result['model1']['metadata']['param'], 'value1')
        self.assertEqual(result['model1']['metadata']['tag'], 'tag1')

        # Check distances were stored
        self.assertEqual(result['model1']['table_initial_distances']['model_params'], 0.1)
        self.assertEqual(result['model1']['table_initial_distances']['model_tags'], 0.2)

        # Check model2 metadata was updated
        self.assertIn('metadata', result['model2'])
        self.assertEqual(result['model2']['metadata']['tag'], 'tag2')

        # Check tables list was updated
        self.assertIn('model_params', result['model1']['tables'])
        self.assertIn('model_tags', result['model1']['tables'])

    async def test_process_model_descriptions_text_search(self):
        # Setup initial results
        all_results = {
            'model1': {
                'model_id': 'model1',
                'tables': ['model_params'],
                'table_initial_distances': {},
                'match_source': 'metadata',
                'metadata': {}
            }
        }

        # Mock search results for model_descriptions
        self.chroma_manager.search.return_value = {
            'results': [
                {
                    'metadata': {'model_id': 'model1', 'description': 'Chunk 1'},
                    'distance': 0.1
                },
                {
                    'metadata': {'model_id': 'model1', 'description': 'Chunk 2'},
                    'distance': 0.2
                }
            ]
        }

        # Call the method
        query = "test query"

        result = await self.handler._process_model_descriptions_text_search(
            query=query,
            all_results=all_results,
            search_limit=5
        )

        # Assertions
        self.assertEqual(len(result), 1)

        # Check model descriptions were collected
        self.assertEqual(len(result['model1']['chunk_descriptions']), 2)
        self.assertEqual(result['model1']['chunk_descriptions'][0], 'Chunk 1')
        self.assertEqual(result['model1']['chunk_descriptions'][1], 'Chunk 2')

        # Check distances were collected
        self.assertEqual(len(result['model1']['chunk_description_distances']), 2)
        self.assertEqual(result['model1']['chunk_description_distances'][0], 0.1)
        self.assertEqual(result['model1']['chunk_description_distances'][1], 0.2)

        # Check merged description
        self.assertEqual(result['model1']['merged_description'], 'Chunk 1 Chunk 2')

        # Check average distance - use assertAlmostEqual for floating point values
        self.assertAlmostEqual(
            result['model1']['table_initial_distances']['model_descriptions'],
            0.15000000000000002,
            places=15
        )

        # Check metadata was updated
        self.assertEqual(result['model1']['metadata']['description'], 'Chunk 1 Chunk 2')

        # Check tables list was updated
        self.assertIn('model_descriptions', result['model1']['tables'])

    def test_calculate_model_distances(self):
        # Setup test data
        all_results = {
            'model1': {
                'model_id': 'model1',
                'tables': ['model_params', 'model_descriptions'],
                'table_initial_distances': {
                    'model_params': 0.1,
                    'model_descriptions': 0.2
                },
                'match_source': 'metadata'
            },
            'model2': {
                'model_id': 'model2',
                'tables': ['model_params'],
                'table_initial_distances': {
                    'model_params': 0.3
                },
                'match_source': 'metadata+chunks',
                'chunk_initial_distance': 0.4
            },
            'model3': {
                'model_id': 'model3',
                'tables': [],
                'table_initial_distances': {},
                'match_source': 'chunks',
                'chunk_initial_distance': 0.5
            }
        }

        table_weights = {
            'model_params': 0.6,
            'model_descriptions': 0.4
        }

        collection_stats = {
            'model_params': {
                'min': 0.0,
                'max': 1.0,
                'percentile_10': 0.05,
                'percentile_90': 0.8
            },
            'model_descriptions': {
                'min': 0.0,
                'max': 1.0,
                'percentile_10': 0.1,
                'percentile_90': 0.9
            },
            'model_scripts_chunks': {
                'min': 0.0,
                'max': 1.0,
                'percentile_10': 0.1,
                'percentile_90': 0.9
            }
        }

        # Mock normalize_distance to return simple values
        self.distance_normalizer.normalize_distance.side_effect = lambda d, stats: d

        # Call the method
        result = self.handler._calculate_model_distances(
            all_results=all_results,
            table_weights=table_weights,
            collection_stats=collection_stats
        )

        # Assertions
        self.assertEqual(len(result), 3)

        # Check model1 distances
        self.assertEqual(result['model1']['table_normalized_distances']['model_params'], 0.1)
        self.assertEqual(result['model1']['table_normalized_distances']['model_descriptions'], 0.2)
        self.assertEqual(result['model1']['metadata_distance'], 0.1 * 0.6 + 0.2 * 0.4)
        self.assertEqual(result['model1']['distance'], 0.1 * 0.6 + 0.2 * 0.4)

        # Check model2 distances (with chunks)
        self.assertEqual(result['model2']['metadata_distance'], 0.3 * 0.6 + 1.0 * 0.4)  # Missing table gets 1.0
        self.assertEqual(result['model2']['chunk_normalized_distance'], 0.4)

        # Use the expected value from the formula
        expected_value = 0.9 * (0.3 * 0.6 + 1.0 * 0.4) + 0.1 * 0.4
        self.assertAlmostEqual(result['model2']['distance'], expected_value, places=15)

        # Check model3 distances (chunks only)
        self.assertEqual(result['model3']['metadata_distance'], 1.0)  # All missing tables get 1.0
        self.assertEqual(result['model3']['chunk_normalized_distance'], 0.5)
        self.assertEqual(result['model3']['distance'], 0.5)  # Chunks only

    def test_sort_and_limit_search_results(self):
        # Setup test data
        all_results = {
            'model1': {
                'model_id': 'model1',
                'distance': 0.3
            },
            'model2': {
                'model_id': 'model2',
                'distance': 0.1
            },
            'model3': {
                'model_id': 'model3',
                'distance': 0.2
            },
            'model4': {
                'model_id': 'model4',
                'distance': 0.4
            }
        }

        # Call the method
        result = self.handler._sort_and_limit_search_results(
            all_results=all_results,
            requested_limit=2
        )

        # Assertions
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['model_id'], 'model2')  # Lowest distance first
        self.assertEqual(result[1]['model_id'], 'model3')  # Second lowest distance

    def test_prepare_text_search_items(self):
        # Setup test data
        output_list = [
            {
                'model_id': 'model1',
                'metadata': {'name': 'Model 1', 'description': 'Description 1'},
                'match_source': 'metadata',
                'distance': 0.1,
                'merged_description': 'Merged description 1'
            },
            {
                'model_id': 'model2',
                'metadata': {'name': 'Model 2', 'description': 'Description 2'},
                'match_source': 'metadata+chunks',
                'distance': 0.2,
                'merged_description': 'Merged description 2'
            }
        ]

        # Call the method
        result = self.handler._prepare_text_search_items(output_list)

        # Assertions
        self.assertEqual(len(result), 2)

        # Check first item
        self.assertEqual(result[0]['id'], 'model_metadata_model1')
        self.assertEqual(result[0]['model_id'], 'model1')
        self.assertEqual(result[0]['metadata']['name'], 'Model 1')
        self.assertEqual(result[0]['rank'], 1)
        self.assertEqual(result[0]['match_source'], 'metadata')
        self.assertEqual(result[0]['distance'], 0.1)
        self.assertEqual(result[0]['merged_description'], 'Merged description 1')

        # Check second item
        self.assertEqual(result[1]['id'], 'model_metadata_model2')
        self.assertEqual(result[1]['model_id'], 'model2')
        self.assertEqual(result[1]['metadata']['name'], 'Model 2')
        self.assertEqual(result[1]['rank'], 2)
        self.assertEqual(result[1]['match_source'], 'metadata+chunks')
        self.assertEqual(result[1]['distance'], 0.2)
        self.assertEqual(result[1]['merged_description'], 'Merged description 2')


if __name__ == '__main__':
    unittest.main()