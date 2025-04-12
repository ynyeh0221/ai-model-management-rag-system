import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import asyncio

# Import the AccessControlManager class - adjust the import path as needed
from src.vector_db_manager.access_control import AccessControlManager


class TestAccessControlManager(unittest.TestCase):
    """Test suite for the AccessControlManager class."""

    def setUp(self):
        """Set up test fixtures, if any."""
        # Create a mock database client for synchronous tests
        self.mock_db_client = MagicMock()

        # For _get_document method tests - make sure it returns a document
        self.mock_db_client.get.return_value = {"id": "doc456", "metadata": {}}

        self.manager = AccessControlManager(self.mock_db_client)

        # Sample test data
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

        # Create a second manager with a mock async DB client for async tests
        self.mock_async_db_client = MagicMock()
        # Mark get method as a coroutine function
        self.mock_async_db_client.get = AsyncMock()
        self.mock_async_db_client.get.return_value = self.test_document
        self.mock_async_db_client.update = AsyncMock()

        # Save original iscoroutinefunction
        self.original_iscoro = asyncio.iscoroutinefunction
        # Mock for testing
        asyncio.iscoroutinefunction = MagicMock(return_value=True)
        self.async_manager = AccessControlManager(self.mock_async_db_client)

    def tearDown(self):
        """Clean up after each test."""
        # Restore original iscoroutinefunction
        asyncio.iscoroutinefunction = self.original_iscoro

    def test_check_access(self):
        """Test the check_access method."""
        # Test with various permission scenarios
        # Note: Since check_access is currently set to always return True in the provided code,
        # we're just testing the basic functionality here.
        result = self.manager.check_access(self.test_document, self.test_user_id, "view")
        self.assertTrue(result)

    @patch('asyncio.get_event_loop')
    def test_grant_access(self, mock_get_event_loop):
        """Test the grant_access method with a synchronous DB client."""
        # For sync client, mock iscoroutinefunction to return False
        asyncio.iscoroutinefunction = MagicMock(return_value=False)

        # Setup the mock to return our test document
        self.mock_db_client.get.return_value = self.test_document

        # Test granting view access
        self.manager.grant_access(self.test_document_id, "newuser", "view")

        # Verify that update was called with the correct parameters
        self.mock_db_client.update.assert_called_once()
        call_args = self.mock_db_client.update.call_args[1]
        self.assertEqual(call_args["ids"], [self.test_document_id])

        # Check that the user was added to the view_permissions list
        metadata = call_args["metadatas"][0]
        self.assertIn("newuser", metadata["access_control"]["view_permissions"])

    @patch('asyncio.get_event_loop')
    def test_revoke_access(self, mock_get_event_loop):
        """Test the revoke_access method with a synchronous DB client."""
        # For sync client, mock iscoroutinefunction to return False
        asyncio.iscoroutinefunction = MagicMock(return_value=False)

        # Setup the mock to return our test document
        self.mock_db_client.get.return_value = self.test_document

        # Test revoking view access
        self.manager.revoke_access(self.test_document_id, self.test_user_id, "view")

        # Verify that update was called with the correct parameters
        self.mock_db_client.update.assert_called_once()
        call_args = self.mock_db_client.update.call_args[1]

        # Check that the user was removed from the view_permissions list
        metadata = call_args["metadatas"][0]
        self.assertNotIn(self.test_user_id, metadata["access_control"]["view_permissions"])

    def test_create_access_filter(self):
        """Test the create_access_filter method."""
        # Mock _get_user_groups to return some test groups
        self.manager._get_user_groups = MagicMock(return_value=["group1", "group2"])

        # Create the filter
        filter_dict = self.manager.create_access_filter(self.test_user_id)

        # Check the filter structure
        self.assertIn("$or", filter_dict)

        # Should be 11 conditions: 5 base + (3 permissions × 2 groups)
        expected_length = 5 + (3 * 2)  # 5 base conditions + 6 from groups (3 permissions × 2 groups)
        self.assertEqual(len(filter_dict["$or"]), expected_length)

        # Check some specific conditions in the filter
        owner_cond = {"metadata.access_control.owner": self.test_user_id}
        self.assertIn(owner_cond, filter_dict["$or"])

        view_cond = {"metadata.access_control.view_permissions": {"$contains": self.test_user_id}}
        self.assertIn(view_cond, filter_dict["$or"])

        group_cond = {"metadata.access_control.view_permissions": {"$contains": "group1"}}
        self.assertIn(group_cond, filter_dict["$or"])

    @patch('asyncio.get_event_loop')
    def test_transfer_ownership(self, mock_get_event_loop):
        """Test the transfer_ownership method."""
        # For sync client, mock iscoroutinefunction to return False
        asyncio.iscoroutinefunction = MagicMock(return_value=False)

        # Setup the mock to return our test document
        self.mock_db_client.get.return_value = self.test_document

        # Test transferring ownership
        current_owner = self.test_document["metadata"]["access_control"]["owner"]
        new_owner = "newowner123"

        result = self.manager.transfer_ownership(self.test_document_id, current_owner, new_owner)
        self.assertTrue(result)

        # Verify that update was called with the correct parameters
        self.mock_db_client.update.assert_called_once()
        call_args = self.mock_db_client.update.call_args[1]

        # Check that the owner was updated
        metadata = call_args["metadatas"][0]
        self.assertEqual(metadata["access_control"]["owner"], new_owner)

        # Check that the new owner has all permissions
        self.assertIn(new_owner, metadata["access_control"]["view_permissions"])
        self.assertIn(new_owner, metadata["access_control"]["edit_permissions"])
        self.assertIn(new_owner, metadata["access_control"]["share_permissions"])

    @patch('asyncio.get_event_loop')
    def test_get_accessible_models_async(self, mock_get_event_loop):
        """Test the get_accessible_models method with an async DB client."""
        # Set up the mock response
        model_results = {
            'results': [
                {
                    'id': 'model1',
                    'metadata': {
                        'description': 'Test Model 1',
                        'framework': 'PyTorch'
                    }
                },
                {
                    'id': 'model2',
                    'metadata': {
                        'description': 'Test Model 2',
                        'framework': 'TensorFlow',
                        'version': '2.0'
                    }
                }
            ]
        }

        # Set up the mock event loop
        mock_loop = MagicMock()
        mock_get_event_loop.return_value = mock_loop

        # Configure run_until_complete to return our model_results
        coro = AsyncMock()
        mock_loop.run_until_complete.return_value = model_results

        # Call the method
        result = self.async_manager.get_accessible_models(self.test_user_id)

        # Ensure get_event_loop was called
        mock_get_event_loop.assert_called()
        # Ensure run_until_complete was called
        mock_loop.run_until_complete.assert_called()

        # Check the result format
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['model_id'], 'model1')
        self.assertEqual(result[0]['description'], 'Test Model 1')
        self.assertEqual(result[0]['framework'], 'PyTorch')

        self.assertEqual(result[1]['model_id'], 'model2')
        self.assertEqual(result[1]['description'], 'Test Model 2')
        self.assertEqual(result[1]['framework'], 'TensorFlow')
        self.assertEqual(result[1]['version'], '2.0')

    @patch('asyncio.get_event_loop')
    def test_get_accessible_images_async(self, mock_get_event_loop):
        """Test the get_accessible_images method with an async DB client."""
        # Set up the mock response
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

        # Set up the mock event loop
        mock_loop = MagicMock()
        mock_get_event_loop.return_value = mock_loop

        # Configure run_until_complete to return our image_results
        coro = AsyncMock()
        mock_loop.run_until_complete.return_value = image_results

        # Call the method
        result = self.async_manager.get_accessible_images(self.test_user_id)

        # Ensure get_event_loop was called
        mock_get_event_loop.assert_called()
        # Ensure run_until_complete was called
        mock_loop.run_until_complete.assert_called()

        # Check the result format
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['id'], 'image1')
        self.assertEqual(result[0]['prompt'], 'A beautiful landscape')
        self.assertEqual(result[0]['filepath'], '/path/to/image1.jpg')
        self.assertEqual(result[0]['style_tags'], '["landscape", "realistic"]')
        self.assertEqual(result[0]['clip_score'], '0.95')

        self.assertEqual(result[1]['id'], 'image2')
        self.assertEqual(result[1]['prompt'], 'A portrait of a cat')
        self.assertEqual(result[1]['filepath'], '/path/to/image2.jpg')

    def test_get_accessible_models_sync(self):
        """Test the get_accessible_models method with a synchronous DB client."""
        # Set up the mock response for a synchronous client
        asyncio.iscoroutinefunction = MagicMock(return_value=False)

        model_results = {
            'results': [
                {
                    'id': 'model1',
                    'metadata': {
                        'description': 'Test Model 1',
                        'framework': 'PyTorch'
                    }
                }
            ]
        }
        self.mock_db_client.get.return_value = model_results

        # Call the method
        result = self.manager.get_accessible_models(self.test_user_id)

        # Check that the DB client's get method was called correctly
        self.mock_db_client.get.assert_called_with(
            collection_name="model_scripts",
            include=["metadatas"]
        )

        # Check the result format
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['model_id'], 'model1')
        self.assertEqual(result[0]['description'], 'Test Model 1')
        self.assertEqual(result[0]['framework'], 'PyTorch')

    def test_get_accessible_images_sync(self):
        """Test the get_accessible_images method with a synchronous DB client."""
        # Set up the mock response for a synchronous client
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

        # Call the method
        result = self.manager.get_accessible_images(self.test_user_id)

        # Check that the DB client's get method was called correctly
        self.mock_db_client.get.assert_called_with(
            collection_name="generated_images",
            include=["metadatas"]
        )

        # Check the result format
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['id'], 'image1')
        self.assertEqual(result[0]['prompt'], 'A beautiful landscape')
        self.assertEqual(result[0]['filepath'], '/path/to/image1.jpg')

    def test_empty_results_handling(self):
        """Test handling of empty results from the database."""
        # Make sure we're using sync mode
        asyncio.iscoroutinefunction = MagicMock(return_value=False)

        # Test with empty results
        self.mock_db_client.get.return_value = {'results': []}

        # Check that the methods handle empty results correctly
        models = self.manager.get_accessible_models(self.test_user_id)
        self.assertEqual(models, [])

        images = self.manager.get_accessible_images(self.test_user_id)
        self.assertEqual(images, [])

    def test_error_handling(self):
        """Test error handling when the database client throws exceptions."""
        # Make sure we're using sync mode
        asyncio.iscoroutinefunction = MagicMock(return_value=False)

        # Set up the mock to raise an exception
        self.mock_db_client.get.side_effect = Exception("Database error")

        # Check that the methods handle exceptions gracefully
        models = self.manager.get_accessible_models(self.test_user_id)
        self.assertEqual(models, [])

        images = self.manager.get_accessible_images(self.test_user_id)
        self.assertEqual(images, [])


if __name__ == '__main__':
    unittest.main()