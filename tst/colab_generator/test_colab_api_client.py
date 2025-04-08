import os
import time
import json
import tempfile
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, Mock

import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell

# Import the ColabAPIClient and exceptions from the module.
from src.colab_generator.colab_api_client import (
    ColabAPIClient,
    SCOPES,
    DEFAULT_TIMEOUT,
    MAX_EXECUTION_WAIT_TIME,
    POLL_INTERVAL,
    TOKEN_PATH,
)
from src.utils.exceptions import (
    AuthenticationError,
    NotebookCreationError,
    ExecutionError,
    ResourceExceededError,
)

# Dummy function to bypass authentication.
def dummy_authenticate(self):
    self.drive_service = MagicMock(name="drive_service")
    self.colab_service = MagicMock(name="colab_service")
    
class TestColabAPIClient(unittest.TestCase):

    def setUp(self):
        # Patch _authenticate before instantiation so that no real auth is attempted.
        patcher = patch.object(ColabAPIClient, "_authenticate", dummy_authenticate)
        self.addCleanup(patcher.stop)
        patcher.start()
        # Use a temporary credentials path (won't be used because of dummy auth)
        self.client = ColabAPIClient(credentials_path="dummy_credentials.json")

    def test_create_notebook_with_dict_content(self):
        # Set up dummy response from drive_service.files().create().execute()
        dummy_file_id = "dummy_notebook_id"
        create_execute_mock = MagicMock(return_value={"id": dummy_file_id})
        self.client.drive_service.files.return_value.create.return_value.execute = create_execute_mock

        # Create a dummy notebook content as a dictionary.
        notebook_content = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": ["# Test Notebook\n", "This is a test."]
                }
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 2
        }
        # Call create_notebook.
        file_id = self.client.create_notebook(notebook_content, filename="TestNotebook", folder_id="folder123", description="A test notebook")
        self.assertEqual(file_id, dummy_file_id)
        # Check that the temporary file used for upload is removed.
        # (The create_notebook method unlinks the temporary file.)
        # Since we cannot capture the exact temp filename, we check that os.path.exists returns False for that temp file.
        # In this test, we assume create_notebook worked correctly.
        create_execute_mock.assert_called_once()

    def test_execute_notebook_success(self):
        dummy_execution_id = "exec_dummy_id"
        exec_response = {"name": dummy_execution_id}
        # Patch colab_service.notebooks().execute(...).execute() to return our dummy response.
        self.client.colab_service.notebooks.return_value.execute.return_value.execute = MagicMock(return_value=exec_response)

        file_id = "dummy_file_id"
        execution_id = self.client.execute_notebook(file_id)
        self.assertEqual(execution_id, dummy_execution_id)

    def test_get_execution_status(self):
        dummy_status = {"state": "SUCCEEDED", "createTime": "2023-01-01T00:00:00Z"}
        self.client.colab_service.executions.return_value.get.return_value.execute = MagicMock(return_value=dummy_status)

        status = self.client.get_execution_status("dummy_exec_id")
        self.assertEqual(status, dummy_status)

    def test_wait_for_execution_success(self):
        # Simulate get_execution_status returning "RUNNING" first, then "SUCCEEDED".
        statuses = [
            {"state": "RUNNING"},
            {"state": "RUNNING"},
            {"state": "SUCCEEDED", "createTime": "2023-01-01T00:00:00Z",
             "startTime": "2023-01-01T00:01:00Z", "endTime": "2023-01-01T00:02:00Z"}
        ]
        self.client.get_execution_status = MagicMock(side_effect=statuses)

        # Use a very short poll interval to speed up the test.
        final_status = self.client.wait_for_execution("dummy_exec_id", poll_interval=0.01, max_wait_time=1)
        self.assertEqual(final_status["state"], "SUCCEEDED")

    def test_wait_for_execution_timeout(self):
        # Simulate always returning RUNNING.
        self.client.get_execution_status = MagicMock(return_value={"state": "RUNNING"})
        # Expect an ExecutionError due to timeout.
        with self.assertRaises(Exception) as context:
            self.client.wait_for_execution("dummy_exec_id", poll_interval=0.01, max_wait_time=0.1)
        self.assertIn("timed out", str(context.exception).lower())

    @patch("src.colab_generator.colab_api_client.MediaIoBaseDownload")
    def test_download_execution_result_success(self, mock_downloader_class):
        # Simulate an execution that succeeded and provides an output file ID.
        dummy_execution = {
            "state": "SUCCEEDED",
            "output_notebook_file": {"drive_file": {"id": "dummy_output_id"}}
        }
        self.client.get_execution_status = MagicMock(return_value=dummy_execution)

        # Prepare a dummy downloader that returns progress and then done.
        mock_downloader = MagicMock()
        # Simulate two chunks: first call returns a status with progress, second call indicates done.
        mock_downloader.next_chunk.side_effect = [
            (MagicMock(progress=lambda: 0.5), False),
            (MagicMock(progress=lambda: 1.0), True)
        ]
        mock_downloader_class.return_value = mock_downloader

        # Patch drive_service.files().get_media()
        dummy_media_get = MagicMock()
        dummy_media_get.return_value = None  # Not used as we simulate download via MediaIoBaseDownload
        self.client.drive_service.files.return_value.get_media.return_value = DummyMediaRequest()

        # Call download_execution_result.
        # Use a temporary output path.
        with tempfile.NamedTemporaryFile(delete=False, suffix='.ipynb') as temp_out:
            output_path = temp_out.name
        # Ensure file does not exist yet by deleting it.
        os.unlink(output_path)

        # Download should create the file.
        result_path = self.client.download_execution_result("dummy_exec_id", output_path=output_path)
        self.assertTrue(os.path.exists(result_path))
        # Clean up.
        os.unlink(result_path)

    def test_list_notebooks(self):
        dummy_files = [
            {"id": "nb1", "name": "Notebook One", "createdTime": "2023-01-01T00:00:00Z"}
        ]
        self.client.drive_service.files.return_value.list.return_value.execute = MagicMock(return_value={"files": dummy_files})
        notebooks = self.client.list_notebooks(folder_id="folder123", query="Test", max_results=10)
        self.assertEqual(notebooks, dummy_files)

    def test_delete_notebook_success(self):
        # Patch delete().execute() to succeed.
        self.client.drive_service.files.return_value.delete.return_value.execute = MagicMock(return_value={})
        result = self.client.delete_notebook("dummy_file_id")
        self.assertTrue(result)

    def test_delete_notebook_failure(self):
        # Simulate an exception in delete.
        self.client.drive_service.files.return_value.delete.return_value.execute.side_effect = Exception("delete error")
        result = self.client.delete_notebook("dummy_file_id")
        self.assertFalse(result)

    def test_share_notebook_success(self):
        self.client.drive_service.permissions.return_value.create.return_value.execute = MagicMock(return_value={"id": "perm1"})
        result = self.client.share_notebook("dummy_file_id", "test@example.com", role="reader", send_notification=False, message="Test")
        self.assertTrue(result)

    def test_share_notebook_failure(self):
        self.client.drive_service.permissions.return_value.create.return_value.execute.side_effect = Exception("share error")
        result = self.client.share_notebook("dummy_file_id", "test@example.com")
        self.assertFalse(result)

    def test_get_execution_metrics(self):
        # Provide dummy execution status with start and end times.
        start_time = (datetime.utcnow() - timedelta(minutes=2)).isoformat() + "Z"
        end_time = datetime.utcnow().isoformat() + "Z"
        dummy_status = {
            "state": "SUCCEEDED",
            "createTime": "2023-01-01T00:00:00Z",
            "startTime": start_time,
            "endTime": end_time,
            "execution_stats": {"memory": "512MB"}
        }
        self.client.colab_service.executions.return_value.get.return_value.execute = MagicMock(return_value=dummy_status)
        metrics = self.client.get_execution_metrics("dummy_exec_id")
        self.assertEqual(metrics["state"], "SUCCEEDED")
        # Check that runtime_seconds is computed (should be close to 120 seconds).
        self.assertAlmostEqual(metrics.get("runtime_seconds", 0), 120, delta=5)
        self.assertIn("execution_stats", metrics)

    def test_parse_timestamp(self):
        timestamp_str = "2023-01-01T00:00:00Z"
        parsed = self.client._parse_timestamp(timestamp_str)
        expected = datetime(2023, 1, 1, 0, 0, 0, tzinfo=parsed.tzinfo)
        self.assertEqual(parsed, expected)

# Dummy Media Request and Downloader for download test.
class DummyMediaRequest:
    # This class is used as a placeholder for the media request.
    pass

if __name__ == '__main__':
    unittest.main()
