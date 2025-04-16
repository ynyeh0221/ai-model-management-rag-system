import asyncio
import unittest
from unittest.mock import MagicMock, patch, AsyncMock

# Import the AccessControlManager class - adjust the import path as needed
from src.vector_db_manager.access_control import AccessControlManager


class TestAccessControlManager(unittest.TestCase):
    """Test suite for the AccessControlManager class."""

    def setUp(self):
        """Set up test fixtures, if any."""
        self.mock_db_client = MagicMock()
        self.mock_db_client.get.return_value = {"id": "doc456", "metadata": {}}

        self.manager = AccessControlManager(self.mock_db_client)

        self.test_user_id = "user123"
        self.test_document_id = "doc456"
        self.test_document = {
            "id": self.test_document_id,
            "metadata": {
                "access_control": {
                    "owner": "owner789",
                    "view_permissions": ["user123", "group1"],
                    "edit_permissions": ["editor1"],
                    "share_permissions": []
                },
                "description": "Test Document"
            }
        }

        self.mock_async_db_client = MagicMock()
        self.mock_async_db_client.get = AsyncMock()
        self.mock_async_db_client.get.return_value = self.test_document
        self.mock_async_db_client.update = AsyncMock()

        self.original_iscoro = asyncio.iscoroutinefunction
        asyncio.iscoroutinefunction = MagicMock(return_value=True)

        self.async_manager = AccessControlManager(self.mock_async_db_client)

    def tearDown(self):
        asyncio.iscoroutinefunction = self.original_iscoro

    def test_check_access(self):
        result = self.manager.check_access(self.test_document, self.test_user_id, "view")
        self.assertTrue(result)

    @patch('asyncio.get_event_loop')
    def test_grant_access(self, mock_get_event_loop):
        asyncio.iscoroutinefunction = MagicMock(return_value=False)
        self.mock_db_client.get.return_value = self.test_document

        self.manager.grant_access(self.test_document_id, "newuser", "view")

        self.mock_db_client.update.assert_called_once()
        call_args = self.mock_db_client.update.call_args[1]
        self.assertEqual(call_args["ids"], [self.test_document_id])
        metadata = call_args["metadatas"][0]
        self.assertIn("newuser", metadata["access_control"]["view_permissions"])

    @patch('asyncio.get_event_loop')
    def test_revoke_access(self, mock_get_event_loop):
        asyncio.iscoroutinefunction = MagicMock(return_value=False)
        self.mock_db_client.get.return_value = self.test_document

        self.manager.revoke_access(self.test_document_id, self.test_user_id, "view")

        self.mock_db_client.update.assert_called_once()
        call_args = self.mock_db_client.update.call_args[1]
        metadata = call_args["metadatas"][0]
        self.assertNotIn(self.test_user_id, metadata["access_control"]["view_permissions"])

    def test_create_access_filter(self):
        self.manager._get_user_groups = MagicMock(return_value=["group1", "group2"])
        filter_dict = self.manager.create_access_filter(self.test_user_id)

        self.assertIn("$or", filter_dict)
        expected_length = 5 + (3 * 2)
        self.assertEqual(len(filter_dict["$or"]), expected_length)

        self.assertIn({"metadata.access_control.owner": self.test_user_id}, filter_dict["$or"])
        self.assertIn({"metadata.access_control.view_permissions": {"$contains": self.test_user_id}}, filter_dict["$or"])
        self.assertIn({"metadata.access_control.view_permissions": {"$contains": "group1"}}, filter_dict["$or"])

    @patch('asyncio.get_event_loop')
    def test_transfer_ownership(self, mock_get_event_loop):
        asyncio.iscoroutinefunction = MagicMock(return_value=False)
        self.mock_db_client.get.return_value = self.test_document

        current_owner = self.test_document["metadata"]["access_control"]["owner"]
        new_owner = "newowner123"

        result = self.manager.transfer_ownership(self.test_document_id, current_owner, new_owner)
        self.assertTrue(result)

        self.mock_db_client.update.assert_called_once()
        call_args = self.mock_db_client.update.call_args[1]
        metadata = call_args["metadatas"][0]
        self.assertEqual(metadata["access_control"]["owner"], new_owner)
        self.assertIn(new_owner, metadata["access_control"]["view_permissions"])
        self.assertIn(new_owner, metadata["access_control"]["edit_permissions"])
        self.assertIn(new_owner, metadata["access_control"]["share_permissions"])

    @patch('asyncio.get_event_loop')
    def test_get_accessible_models_async(self, mock_get_event_loop):
        model_results = {
            'results': [
                {
                    'id': 'model1',
                    'metadata': {
                        'model_id': 'model1',
                        'description': 'Test Model 1',
                        'framework': 'PyTorch',
                        'file': '{"creation_date": "2025-04-01T10:00:00", "last_modified_date": "2025-04-02T10:00:00"}'
                    }
                },
                {
                    'id': 'model2',
                    'metadata': {
                        'model_id': 'model2',
                        'description': 'Test Model 2',
                        'framework': 'TensorFlow',
                        'version': '2.0',
                        'file': '{"creation_date": "2025-03-01T08:00:00", "last_modified_date": "2025-03-03T09:00:00"}'
                    }
                }
            ]
        }

        mock_loop = MagicMock()
        mock_get_event_loop.return_value = mock_loop
        mock_loop.run_until_complete.return_value = model_results

        result = self.async_manager.get_accessible_models(self.test_user_id)

        mock_get_event_loop.assert_called()
        mock_loop.run_until_complete.assert_called()

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['model_id'], 'model1')
        self.assertEqual(result[1]['model_id'], 'model2')

    def test_get_accessible_models_sync(self):
        asyncio.iscoroutinefunction = MagicMock(return_value=False)

        model_results = {
            'results': [
                {
                    'id': 'model1',
                    'metadata': {
                        'model_id': 'model1',
                        'description': 'Test Model 1',
                        'framework': 'PyTorch',
                        'file': '{"creation_date": "2025-04-01T10:00:00", "last_modified_date": "2025-04-02T10:00:00"}'
                    }
                }
            ]
        }
        self.mock_db_client.get.return_value = model_results

        result = self.manager.get_accessible_models(self.test_user_id)

        self.mock_db_client.get.assert_called_with(
            collection_name="model_scripts",
            include=["metadatas"]
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['model_id'], 'model1')

    @patch('asyncio.get_event_loop')
    def test_get_accessible_images_async(self, mock_get_event_loop):
        image_results = {
            'results': [
                {
                    'id': 'image1',
                    'metadata': {
                        'prompt': 'A beautiful landscape',
                        'image_path': '/path/to/image1.jpg',
                        'style_tags': '["landscape", "realistic"]',
                        'clip_score': '0.95'
                    }
                },
                {
                    'id': 'image2',
                    'metadata': {
                        'prompt': 'A portrait of a cat',
                        'image_path': '/path/to/image2.jpg'
                    }
                }
            ]
        }

        mock_loop = MagicMock()
        mock_get_event_loop.return_value = mock_loop
        mock_loop.run_until_complete.return_value = image_results

        result = self.async_manager.get_accessible_images(self.test_user_id)

        mock_get_event_loop.assert_called()
        mock_loop.run_until_complete.assert_called()

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['id'], 'image1')
        self.assertEqual(result[1]['id'], 'image2')

    def test_get_accessible_images_sync(self):
        asyncio.iscoroutinefunction = MagicMock(return_value=False)

        image_results = {
            'results': [
                {
                    'id': 'image1',
                    'metadata': {
                        'prompt': 'A beautiful landscape',
                        'image_path': '/path/to/image1.jpg'
                    }
                }
            ]
        }
        self.mock_db_client.get.return_value = image_results

        result = self.manager.get_accessible_images(self.test_user_id)

        self.mock_db_client.get.assert_called_with(
            collection_name="generated_images",
            include=["metadatas"]
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['id'], 'image1')

    def test_empty_results_handling(self):
        asyncio.iscoroutinefunction = MagicMock(return_value=False)
        self.mock_db_client.get.return_value = {'results': []}

        models = self.manager.get_accessible_models(self.test_user_id)
        self.assertEqual(models, [])

        images = self.manager.get_accessible_images(self.test_user_id)
        self.assertEqual(images, [])

    def test_error_handling(self):
        asyncio.iscoroutinefunction = MagicMock(return_value=False)
        self.mock_db_client.get.side_effect = Exception("Database error")

        models = self.manager.get_accessible_models(self.test_user_id)
        self.assertEqual(models, [])

        images = self.manager.get_accessible_images(self.test_user_id)
        self.assertEqual(images, [])


if __name__ == '__main__':
    unittest.main()
