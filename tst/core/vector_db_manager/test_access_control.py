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
        """Test _get_document with an async db client."""
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

    def test_check_access_with_none_document(self):
        """Test check_access when a document is None."""
        result = self.access_manager.check_access(None, "user1", "view")
        self.assertFalse(result)

    def test_check_access_with_document_no_metadata(self):
        """Test check_access when a document has no metadata."""
        doc = {"id": "doc1"}
        result = self.access_manager.check_access(doc, "user1", "view")
        self.assertFalse(result)

    def test_check_access_with_non_list_permission_list(self):
        """Test check_access when a permission list is not a list/set/tuple."""
        doc = {
            "metadata": {
                "access_control": {"view": "not_a_list", "edit": []}
            }
        }
        result = self.access_manager.check_access(doc, "user1", "view")
        self.assertFalse(result)

    def test_grant_access_document_not_found(self):
        """Test grant_access when a document is not found."""
        self.access_manager._get_document = MagicMock(return_value=None)

        result = self.access_manager.grant_access("nonexistent_doc", "user1", "view")
        self.assertFalse(result)

    def test_grant_access_json_decode_error(self):
        """Test grant_access when JSON decoding fails."""
        doc = {
            "metadata": {
                "access_control": "{invalid json}"
            }
        }
        self.access_manager._get_document = MagicMock(return_value=doc)
        self.access_manager._update_document_metadata = MagicMock(return_value=True)

        result = self.access_manager.grant_access("doc1", "user1", "view")
        self.assertTrue(result)

        # Verify the default access_control was used
        call_args = self.access_manager._update_document_metadata.call_args[0]
        updated_metadata = call_args[1]
        updated_access_control = json.loads(updated_metadata["access_control"])
        self.assertIn("user1", updated_access_control["view"])

    def test_grant_access_user_already_has_permission(self):
        """Test grant_access when a user already has the permission."""
        doc = {
            "metadata": {
                "access_control": json.dumps({"view": ["user1"], "edit": []})
            }
        }
        self.access_manager._get_document = MagicMock(return_value=doc)
        self.access_manager._update_document_metadata = MagicMock(return_value=True)

        result = self.access_manager.grant_access("doc1", "user1", "view")
        self.assertTrue(result)

        # Verify user1 is still in the list (and not duplicated)
        call_args = self.access_manager._update_document_metadata.call_args[0]
        updated_metadata = call_args[1]
        updated_access_control = json.loads(updated_metadata["access_control"])
        self.assertEqual(updated_access_control["view"].count("user1"), 1)

    def test_revoke_access_document_not_found(self):
        """Test revoke_access when a document is not found."""
        self.access_manager._get_document = MagicMock(return_value=None)

        result = self.access_manager.revoke_access("nonexistent_doc", "user1", "view")
        self.assertFalse(result)

    def test_revoke_access_json_decode_error(self):
        """Test revoke_access when JSON decoding fails."""
        doc = {
            "metadata": {
                "access_control": "{invalid json}"
            }
        }
        self.access_manager._get_document = MagicMock(return_value=doc)

        result = self.access_manager.revoke_access("doc1", "user1", "view")
        self.assertFalse(result)

    def test_revoke_access_permission_type_not_exists(self):
        """Test revoke_access when a permission type doesn't exist in access_control."""
        doc = {
            "metadata": {
                "access_control": json.dumps({"view": ["user1"]})  # No "edit" key
            }
        }
        self.access_manager._get_document = MagicMock(return_value=doc)
        self.access_manager._update_document_metadata = MagicMock(return_value=True)

        result = self.access_manager.revoke_access("doc1", "user1", "edit")
        self.assertTrue(result)  # Should still succeed

    def test_revoke_access_user_not_in_permission_list(self):
        """Test revoke_access when a user doesn't have the permission being revoked."""
        doc = {
            "metadata": {
                "access_control": json.dumps({"view": ["user2"], "edit": []})
            }
        }
        self.access_manager._get_document = MagicMock(return_value=doc)
        self.access_manager._update_document_metadata = MagicMock(return_value=True)

        result = self.access_manager.revoke_access("doc1", "user1", "view")
        self.assertTrue(result)  # Should still succeed

    def test_set_public_access_document_not_found(self):
        """Test set_public_access when a document is not found."""
        self.access_manager._get_document = MagicMock(return_value=None)

        result = self.access_manager.set_public_access("nonexistent_doc", "view", True)
        self.assertFalse(result)

    def test_set_public_access_json_decode_error(self):
        """Test set_public_access when JSON decoding fails."""
        doc = {
            "metadata": {
                "access_control": "{invalid json}"
            }
        }
        self.access_manager._get_document = MagicMock(return_value=doc)
        self.access_manager._update_document_metadata = MagicMock(return_value=True)

        result = self.access_manager.set_public_access("doc1", "view", True)
        self.assertTrue(result)

    @patch('asyncio.iscoroutinefunction')
    @patch('asyncio.get_event_loop')
    def test_async_update_document_metadata(self, mock_get_loop, mock_iscoroutinefunction):
        """Test _update_document_metadata with an async db client."""
        mock_iscoroutinefunction.return_value = True
        mock_loop = MagicMock()
        mock_get_loop.return_value = mock_loop

        async def mock_async_update(ids, metadatas):
            return True

        async_db_client = MagicMock()
        async_db_client.update = mock_async_update

        manager = AccessControlManager(async_db_client)

        result = manager._update_document_metadata("doc1", {"test": "metadata"})
        self.assertTrue(result)
        mock_loop.run_until_complete.assert_called_once()

    def test_update_document_metadata_sync_error(self):
        """Test _update_document_metadata with sync db client error."""
        self.mock_db_client.update.side_effect = Exception("Sync error")

        result = self.access_manager._update_document_metadata("doc1", {"test": "metadata"})
        self.assertFalse(result)

    def test_get_user_groups_no_get_user_method(self):
        """Test _get_user_groups when db_client doesn't have get_user method."""
        db_client_no_get_user = MagicMock()
        del db_client_no_get_user.get_user  # Remove the method

        manager = AccessControlManager(db_client_no_get_user)
        result = manager._get_user_groups("user1")
        self.assertEqual(result, [])

    def test_get_user_groups_exception(self):
        """Test _get_user_groups when get_user raises an exception."""
        self.mock_db_client.get_user.side_effect = Exception("Database error")

        result = self.access_manager._get_user_groups("user1")
        self.assertEqual(result, [])

    def test_get_user_groups_no_groups_in_user_record(self):
        """Test _get_user_groups when user record doesn't have groups."""
        # Use a different user not in the setUp mock configuration
        self.mock_db_client.get_user.return_value = {"id": "user_no_groups", "name": "User No Groups"}

        result = self.access_manager._get_user_groups("user_no_groups")
        self.assertEqual(result, [])

    def test_get_accessible_models_no_db_client(self):
        """Test get_accessible_models when no db_client is available."""
        manager = AccessControlManager()
        result = manager.get_accessible_models("user1")
        self.assertEqual(result, [])

    def test_get_accessible_models_empty_results(self):
        """Test get_accessible_models when no models are returned."""
        self.access_manager._fetch_all_model_metadata = MagicMock(return_value={})

        result = self.access_manager.get_accessible_models("user1")
        self.assertEqual(result, [])

    def test_get_accessible_models_no_accessible_ids(self):
        """Test get_accessible_models when no models are accessible."""
        mock_models = {
            "results": [
                {"metadata": {"model_id": "model1", "access_control": json.dumps({"view": ["other_user"], "edit": []})}}
            ]
        }
        self.access_manager._fetch_all_model_metadata = MagicMock(return_value=mock_models)

        result = self.access_manager.get_accessible_models("user1")
        self.assertEqual(result, [])

    def test_get_accessible_models_successful(self):
        """Test get_accessible_models with accessible models."""
        mock_models = {
            "results": [
                {"metadata": {"model_id": "model1", "access_control": json.dumps({"view": ["user1"], "edit": []})}},
                {"metadata": {"model_id": "model2", "access_control": json.dumps({"view": ["public"], "edit": []})}}
            ]
        }
        self.access_manager._fetch_all_model_metadata = MagicMock(return_value=mock_models)
        self.access_manager._consolidate_model_info = MagicMock(
            side_effect=lambda x: {"model_id": x, "info": "consolidated"})

        result = self.access_manager.get_accessible_models("user1")
        self.assertEqual(len(result), 2)

        # Check that both models are present (order may vary due to set operations)
        model_ids = [model["model_id"] for model in result]
        self.assertIn("model1", model_ids)
        self.assertIn("model2", model_ids)

    def test_get_accessible_images_no_db_client(self):
        """Test get_accessible_images when no db_client is available."""
        manager = AccessControlManager()
        result = manager.get_accessible_images("user1")
        self.assertEqual(result, [])

    def test_get_accessible_images_successful(self):
        """Test get_accessible_images with accessible images."""
        mock_images = {
            "results": [
                {
                    "id": "img1",
                    "metadata": {
                        "prompt": "A beautiful sunset",
                        "image_path": "/path/to/img1.jpg",
                        "access_control": json.dumps({"view": ["user1"], "edit": []}),
                        "style_tags": ["landscape", "sunset"],
                        "clip_score": 0.95
                    }
                },
                {
                    "id": "img2",
                    "metadata": {
                        "prompt": "A cute cat",
                        "image_path": "/path/to/img2.jpg",
                        "access_control": json.dumps({"view": ["public"], "edit": []})
                    }
                }
            ]
        }
        self.access_manager._fetch_all_images = MagicMock(return_value=mock_images)

        result = self.access_manager.get_accessible_images("user1")
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["id"], "img1")
        self.assertEqual(result[0]["prompt"], "A beautiful sunset")
        self.assertEqual(result[0]["filepath"], "/path/to/img1.jpg")
        self.assertIn("style_tags", result[0])
        self.assertIn("clip_score", result[0])
        self.assertEqual(result[1]["id"], "img2")
        self.assertNotIn("style_tags", result[1])
        self.assertNotIn("clip_score", result[1])

    def test_fetch_all_model_metadata_exception(self):
        """Test _fetch_all_model_metadata when exception occurs."""
        self.mock_db_client.get.side_effect = Exception("Database error")

        result = self.access_manager._fetch_all_model_metadata()
        self.assertEqual(result, {})

    def test_filter_accessible_model_ids_invalid_input(self):
        """Test _filter_accessible_model_ids with invalid input."""
        result = self.access_manager._filter_accessible_model_ids("invalid", "user1")
        self.assertEqual(result, set())

        result = self.access_manager._filter_accessible_model_ids({"no_results": []}, "user1")
        self.assertEqual(result, set())

    def test_filter_accessible_model_ids_no_model_id(self):
        """Test _filter_accessible_model_ids when metadata has no model_id."""
        mock_models = {
            "results": [
                {"metadata": {"name": "model_without_id"}}
            ]
        }

        result = self.access_manager._filter_accessible_model_ids(mock_models, "user1")
        self.assertEqual(result, set())

    def test_consolidate_model_info_no_metadata_found(self):
        """Test _consolidate_model_info when no metadata is found."""
        self.access_manager._fetch_one_metadata = MagicMock(return_value=None)

        result = self.access_manager._consolidate_model_info("model1")
        expected = {"model_id": "model1"}
        self.assertEqual(result, expected)

    def test_consolidate_model_info_with_structured_info(self):
        """Test _consolidate_model_info with valid structured model info."""

        def mock_fetch_metadata(table, model_id):
            if table == "model_file":
                return {
                    "file": json.dumps({
                        "creation_date": "2023-01-01",
                        "last_modified_date": "2023-01-02",
                        "absolute_path": "/path/to/model"
                    })
                }
            elif table == "model_images_folder":
                return {
                    "images_folder": json.dumps({
                        "name": "model_images"
                    })
                }
            return {"framework": "pytorch", "version": "1.0"}

        self.access_manager._fetch_one_metadata = MagicMock(side_effect=mock_fetch_metadata)

        result = self.access_manager._consolidate_model_info("model1")

        self.assertEqual(result["model_id"], "model1")
        self.assertEqual(result["creation_date"], "2023-01-01")
        self.assertEqual(result["last_modified_date"], "2023-01-02")
        self.assertEqual(result["absolute_path"], "/path/to/model")
        self.assertEqual(result["images_folder"], "model_images")

    def test_fetch_one_metadata_exception(self):
        """Test _fetch_one_metadata when exception occurs."""
        self.mock_db_client.get.side_effect = Exception("Database error")

        result = self.access_manager._fetch_one_metadata("test_table", "model1")
        self.assertIsNone(result)

    def test_fetch_one_metadata_invalid_results(self):
        """Test _fetch_one_metadata with invalid results format."""
        self.mock_db_client.get.return_value = {"results": []}

        result = self.access_manager._fetch_one_metadata("test_table", "model1")
        self.assertIsNone(result)

    def test_build_structured_model_info_missing_data(self):
        """Test _build_structured_model_info with missing required data."""
        consolidated = {"model_id": "model1"}  # Missing file and images_folder

        result = AccessControlManager._build_structured_model_info("model1", consolidated)
        self.assertIsNone(result)

    def test_build_structured_model_info_json_parse_error(self):
        """Test _build_structured_model_info with JSON parse error."""
        consolidated = {
            "model_id": "model1",
            "file": "{invalid json}",
            "images_folder": json.dumps({"name": "test"})
        }

        result = AccessControlManager._build_structured_model_info("model1", consolidated)
        self.assertIsNone(result)

    def test_build_basic_model_info(self):
        """Test _build_basic_model_info."""
        consolidated = {
            "model_id": "model1",
            "framework": "pytorch",
            "version": "1.0",
            "file": "should_be_excluded",
            "description": "should_be_excluded",
            "access_control": "should_be_excluded"
        }

        result = AccessControlManager._build_basic_model_info("model1", consolidated)

        expected = {
            "model_id": "model1",
            "framework": "pytorch",
            "version": "1.0"
        }
        self.assertEqual(result, expected)

    def test_merge_metadata_with_description(self):
        """Test _merge_metadata when meta contains description."""
        consolidated = {"model_id": "model1"}
        meta = {"description": "test desc", "framework": "pytorch", "version": "1.0"}

        AccessControlManager._merge_metadata(consolidated, meta)

        expected = {"model_id": "model1", "framework": "pytorch", "version": "1.0"}
        self.assertEqual(consolidated, expected)

    def test_merge_metadata_without_description(self):
        """Test _merge_metadata when meta doesn't contain description."""
        consolidated = {"model_id": "model1"}
        meta = {"framework": "pytorch", "version": "1.0"}

        AccessControlManager._merge_metadata(consolidated, meta)

        expected = {"model_id": "model1", "framework": "pytorch", "version": "1.0"}
        self.assertEqual(consolidated, expected)

    def test_fetch_all_images_exception(self):
        """Test _fetch_all_images when exception occurs."""
        self.mock_db_client.get.side_effect = Exception("Database error")

        result = self.access_manager._fetch_all_images()
        self.assertEqual(result, {})

    def test_filter_accessible_images_invalid_input(self):
        """Test _filter_accessible_images with invalid input."""
        result = self.access_manager._filter_accessible_images("invalid", "user1")
        self.assertEqual(result, [])

        result = self.access_manager._filter_accessible_images({"no_results": []}, "user1")
        self.assertEqual(result, [])

    def test_filter_accessible_images_missing_metadata_fields(self):
        """Test _filter_accessible_images with missing metadata fields."""
        mock_images = {
            "results": [
                {
                    "id": "img1",
                    "metadata": {
                        "access_control": json.dumps({"view": ["user1"], "edit": []})
                        # Missing prompt and image_path
                    }
                }
            ]
        }

        result = self.access_manager._filter_accessible_images(mock_images, "user1")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["prompt"], "No prompt")
        self.assertEqual(result[0]["filepath"], "No path")

    @patch('asyncio.new_event_loop')
    @patch('asyncio.set_event_loop')
    @patch('asyncio.get_event_loop')
    def test_get_document_runtime_error_new_loop(self, mock_get_loop, mock_set_loop, mock_new_loop):
        """Test _get_document when RuntimeError is raised and a new loop is created."""
        mock_get_loop.side_effect = RuntimeError("No event loop")
        mock_new_loop_instance = MagicMock()
        mock_new_loop.return_value = mock_new_loop_instance
        mock_new_loop_instance.run_until_complete.return_value = {"result": "test"}

        async def mock_async_get(ids, **kwargs):
            return {"result": "test"}

        with patch('asyncio.iscoroutinefunction', return_value=True):
            async_db_client = MagicMock()
            async_db_client.get = mock_async_get

            manager = AccessControlManager(async_db_client)
            result = manager._get_document("test_doc")

            mock_new_loop.assert_called_once()
            mock_set_loop.assert_called_once_with(mock_new_loop_instance)
            self.assertEqual(result, {"result": "test"})

if __name__ == '__main__':
    unittest.main()
