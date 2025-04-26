import json
import unittest
from unittest.mock import MagicMock, patch

from src.core.vector_db.access_control import AccessControlManager


class TestAccessControlManager(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a mock database client
        self.mock_db_client = MagicMock()

        # Initialize the AccessControlManager with the mock client
        self.access_manager = AccessControlManager(self.mock_db_client)

        # Sample documents for testing
        self.doc_with_public_view = {
            "id": "doc1",
            "metadata": {
                "access_control": json.dumps({"view": ["public"], "edit": []})
            }
        }

        self.doc_with_user_perms = {
            "id": "doc2",
            "metadata": {
                "access_control": json.dumps({"view": ["user1", "user2"], "edit": ["user1"]})
            }
        }

        self.doc_with_group_perms = {
            "id": "doc3",
            "metadata": {
                "access_control": json.dumps({"view": ["group1"], "edit": []})
            }
        }

        self.doc_with_dict_access_control = {
            "id": "doc4",
            "metadata": {
                "access_control": {"view": ["user3"], "edit": ["user3"]}
            }
        }

        self.doc_with_no_access_control = {
            "id": "doc5",
            "metadata": {}
        }

        # Configure mock db_client.get_user to return user groups
        def mock_get_user(user_id):
            user_groups = {
                "user1": {"groups": ["group1", "group2"]},
                "user2": {"groups": ["group3"]},
                "user3": {"groups": []},
            }
            return user_groups.get(user_id, {"groups": []})

        self.mock_db_client.get_user = MagicMock(side_effect=mock_get_user)

        # Configure mock db_client.get to return documents
        def mock_get_document(ids, **kwargs):
            doc_id = ids[0] if ids else None
            docs = {
                "doc1": self.doc_with_public_view,
                "doc2": self.doc_with_user_perms,
                "doc3": self.doc_with_group_perms,
                "doc4": self.doc_with_dict_access_control,
                "doc5": self.doc_with_no_access_control,
            }
            return docs.get(doc_id)

        self.mock_db_client.get = MagicMock(side_effect=mock_get_document)

        # Configure mock db_client.update to return True
        self.mock_db_client.update = MagicMock(return_value=True)

    def tearDown(self):
        """Clean up after each test method."""
        pass

    def test_get_access_control_with_json_string(self):
        """Test _get_access_control with JSON string access_control."""
        access_control = self.access_manager._get_access_control(self.doc_with_public_view)
        self.assertEqual(access_control, {"view": ["public"], "edit": []})

    def test_get_access_control_with_dict(self):
        """Test _get_access_control with dictionary access_control."""
        access_control = self.access_manager._get_access_control(self.doc_with_dict_access_control)
        self.assertEqual(access_control, {"view": ["user3"], "edit": ["user3"]})

    def test_get_access_control_missing(self):
        """Test _get_access_control when access_control is missing."""
        access_control = self.access_manager._get_access_control(self.doc_with_no_access_control)
        self.assertIsNone(access_control)

    def test_get_access_control_invalid_json(self):
        """Test _get_access_control with invalid JSON string."""
        doc_with_invalid_json = {
            "id": "doc_invalid",
            "metadata": {
                "access_control": "{invalid json}"
            }
        }
        access_control = self.access_manager._get_access_control(doc_with_invalid_json)
        self.assertIsNone(access_control)

    def test_check_access_public_view(self):
        """Test check_access with public view permission."""
        # Any user should have view access to a public document
        self.assertTrue(self.access_manager.check_access(
            self.doc_with_public_view, "any_user", "view"))

        # No user should have edit access to this document
        self.assertFalse(self.access_manager.check_access(
            self.doc_with_public_view, "any_user", "edit"))

    def test_check_access_user_specific(self):
        """Test check_access with user-specific permissions."""
        # user1 should have both view and edit access
        self.assertTrue(self.access_manager.check_access(
            self.doc_with_user_perms, "user1", "view"))
        self.assertTrue(self.access_manager.check_access(
            self.doc_with_user_perms, "user1", "edit"))

        # user2 should have only view access
        self.assertTrue(self.access_manager.check_access(
            self.doc_with_user_perms, "user2", "view"))
        self.assertFalse(self.access_manager.check_access(
            self.doc_with_user_perms, "user2", "edit"))

        # user3 should have no access
        self.assertFalse(self.access_manager.check_access(
            self.doc_with_user_perms, "user3", "view"))
        self.assertFalse(self.access_manager.check_access(
            self.doc_with_user_perms, "user3", "edit"))

    def test_check_access_group_permission(self):
        """Test check_access with group permissions."""
        # user1 is in group1, so should have view access
        self.assertTrue(self.access_manager.check_access(
            self.doc_with_group_perms, "user1", "view"))

        # user2 is not in group1, so should not have view access
        self.assertFalse(self.access_manager.check_access(
            self.doc_with_group_perms, "user2", "view"))

    def test_check_access_permission_hierarchy(self):
        """Test check_access respects permission hierarchy."""
        # Create a document where edit permission implies view permission
        doc_with_edit_only = {
            "id": "doc_edit_only",
            "metadata": {
                "access_control": json.dumps({"view": [], "edit": ["user4"]})
            }
        }

        # user4 has edit permission, which should imply view permission
        self.assertTrue(self.access_manager.check_access(
            doc_with_edit_only, "user4", "view"))
        self.assertTrue(self.access_manager.check_access(
            doc_with_edit_only, "user4", "edit"))

    def test_grant_access(self):
        """Test grant_access method."""
        # Configure mock to return a document when _get_document is called
        self.access_manager._get_document = MagicMock(return_value=self.doc_with_public_view.copy())
        self.access_manager._update_document_metadata = MagicMock(return_value=True)

        # Grant edit access to user1
        result = self.access_manager.grant_access("doc1", "user1", "edit")

        # Verify the result and that _update_document_metadata was called
        self.assertTrue(result)
        self.access_manager._update_document_metadata.assert_called_once()

        # Get the updated metadata that was passed to _update_document_metadata
        call_args = self.access_manager._update_document_metadata.call_args[0]
        updated_metadata = call_args[1]

        # Verify the edit permission was added for user1
        updated_access_control = json.loads(updated_metadata["access_control"])
        self.assertIn("user1", updated_access_control["edit"])

    def test_revoke_access(self):
        """Test revoke_access method."""
        # Create a document copy with user1 having edit permission
        doc_with_user1_edit = self.doc_with_user_perms.copy()

        # Configure mock to return the document when _get_document is called
        self.access_manager._get_document = MagicMock(return_value=doc_with_user1_edit)
        self.access_manager._update_document_metadata = MagicMock(return_value=True)

        # Revoke edit access for user1
        result = self.access_manager.revoke_access("doc2", "user1", "edit")

        # Verify the result and that _update_document_metadata was called
        self.assertTrue(result)
        self.access_manager._update_document_metadata.assert_called_once()

        # Get the updated metadata that was passed to _update_document_metadata
        call_args = self.access_manager._update_document_metadata.call_args[0]
        updated_metadata = call_args[1]

        # Verify the edit permission was removed for user1
        updated_access_control = json.loads(updated_metadata["access_control"])
        self.assertNotIn("user1", updated_access_control["edit"])

    def test_set_public_access_enable(self):
        """Test setting public access."""
        doc = self.doc_with_user_perms.copy()

        # Configure mock to return the document when _get_document is called
        self.access_manager._get_document = MagicMock(return_value=doc)
        self.access_manager._update_document_metadata = MagicMock(return_value=True)

        # Set public view access
        result = self.access_manager.set_public_access("doc2", "view", True)

        # Verify the result and that _update_document_metadata was called
        self.assertTrue(result)
        self.access_manager._update_document_metadata.assert_called_once()

        # Get the updated metadata that was passed to _update_document_metadata
        call_args = self.access_manager._update_document_metadata.call_args[0]
        updated_metadata = call_args[1]

        # Verify "public" was added to view permissions
        updated_access_control = json.loads(updated_metadata["access_control"])
        self.assertIn("public", updated_access_control["view"])

    def test_set_public_access_disable(self):
        """Test removing public access."""
        doc = self.doc_with_public_view.copy()

        # Configure mock to return the document when _get_document is called
        self.access_manager._get_document = MagicMock(return_value=doc)
        self.access_manager._update_document_metadata = MagicMock(return_value=True)

        # Remove public view access
        result = self.access_manager.set_public_access("doc1", "view", False)

        # Verify the result and that _update_document_metadata was called
        self.assertTrue(result)
        self.access_manager._update_document_metadata.assert_called_once()

        # Get the updated metadata that was passed to _update_document_metadata
        call_args = self.access_manager._update_document_metadata.call_args[0]
        updated_metadata = call_args[1]

        # Verify "public" was removed from view permissions
        updated_access_control = json.loads(updated_metadata["access_control"])
        self.assertNotIn("public", updated_access_control["view"])

    def test_create_access_filter(self):
        """Test create_access_filter method."""
        # Test for a user with groups
        filter_user1 = self.access_manager.create_access_filter("user1")
        self.assertIn("$or", filter_user1)
        # Should have entries for user1, public, group1, and group2
        self.assertEqual(len(filter_user1["$or"]), 4)

        # Test for a user without groups
        filter_user3 = self.access_manager.create_access_filter("user3")
        self.assertIn("$or", filter_user3)
        # Should have entries for user3 and public only
        self.assertEqual(len(filter_user3["$or"]), 2)

    def test_get_document_permissions(self):
        """Test get_document_permissions method."""
        # Document with access control
        permissions = self.access_manager.get_document_permissions(self.doc_with_user_perms)
        self.assertEqual(permissions, {"view": ["user1", "user2"], "edit": ["user1"]})

        # Document without access control
        permissions = self.access_manager.get_document_permissions(self.doc_with_no_access_control)
        self.assertEqual(permissions, {"view": ["public"], "edit": []})

    @patch('asyncio.iscoroutinefunction')
    def test_async_get_document(self, mock_iscoroutinefunction):
        """Test _get_document with async db client."""
        # Mock asyncio.iscoroutinefunction to return True
        mock_iscoroutinefunction.return_value = True

        # Create a mock async get method result
        async def mock_async_get(ids, **kwargs):
            return {"result": "async_document"}

        # Set up the mock db client with an async get method
        async_db_client = MagicMock()
        async_db_client.get = mock_async_get

        # Create a manager with the async db client
        manager = AccessControlManager(async_db_client)

        # Mock asyncio.get_event_loop and run_until_complete
        mock_loop = MagicMock()
        mock_loop.run_until_complete = MagicMock(return_value={"result": "async_document"})

        with patch('asyncio.get_event_loop', return_value=mock_loop):
            # Call _get_document
            result = manager._get_document("async_doc_id")

            # Verify the result
            self.assertEqual(result, {"result": "async_document"})
            mock_loop.run_until_complete.assert_called_once()

    def test_error_handling_in_get_document(self):
        """Test error handling in _get_document method."""
        # Configure mock to raise an exception
        self.mock_db_client.get = MagicMock(side_effect=Exception("Test error"))

        # Call _get_document
        result = self.access_manager._get_document("error_doc")

        # Verify the result is None when an error occurs
        self.assertIsNone(result)

    def test_no_db_client(self):
        """Test behavior when no db_client is provided."""
        # Create an AccessControlManager with no db_client
        manager = AccessControlManager()

        # Test _get_document
        self.assertIsNone(manager._get_document("doc1"))

        # Test _update_document_metadata
        self.assertFalse(manager._update_document_metadata("doc1", {}))

        # Test _get_user_groups
        self.assertEqual(manager._get_user_groups("user1"), [])

        # Test grant_access raises ValueError
        with self.assertRaises(ValueError):
            manager.grant_access("doc1", "user1")


if __name__ == '__main__':
    unittest.main()
