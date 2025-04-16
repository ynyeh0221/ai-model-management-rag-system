import unittest
from unittest.mock import patch, mock_open, MagicMock

from src.colab_generator.colab_api_client import ColabAPIClient, AuthenticationError

FAKE_CREDS_JSON = '{"type": "service_account", "project_id": "demo"}'

class TestColabAPIClient(unittest.TestCase):

    @patch("src.colab_generator.colab_api_client.build")
    @patch("src.colab_generator.colab_api_client.Credentials")
    @patch("os.path.exists", return_value=True)
    @patch("builtins.open", new_callable=mock_open, read_data=FAKE_CREDS_JSON)
    def test_authentication_with_service_account(self, mock_file, mock_exists, mock_credentials, mock_build):
        # Setup mock
        mock_creds_instance = MagicMock()
        mock_credentials.from_service_account_file.return_value = mock_creds_instance

        client = ColabAPIClient(credentials_path="fake/path/service_account.json", test_mode=False)

        mock_credentials.from_service_account_file.assert_called_once()
        mock_build.assert_any_call("drive", "v3", credentials=mock_creds_instance)
        mock_build.assert_any_call("colab", "v1", credentials=mock_creds_instance)
        self.assertIsNotNone(client.drive_service)
        self.assertIsNotNone(client.colab_service)

    @patch("src.colab_generator.colab_api_client.nbformat.write")
    @patch("src.colab_generator.colab_api_client.nbformat.from_dict")
    @patch("src.colab_generator.colab_api_client.MediaFileUpload")
    @patch("src.colab_generator.colab_api_client.build")
    @patch("src.colab_generator.colab_api_client.Credentials")
    @patch("os.path.exists", return_value=True)
    @patch("builtins.open", new_callable=mock_open, read_data=FAKE_CREDS_JSON)
    def test_create_notebook(self, mock_file, mock_exists, mock_credentials, mock_build, mock_media, mock_from_dict, mock_write):
        # Mocks
        mock_creds = MagicMock()
        mock_credentials.from_service_account_file.return_value = mock_creds

        mock_drive_service = MagicMock()
        mock_files = mock_drive_service.files.return_value
        mock_files.create.return_value.execute.return_value = {'id': 'fake_id'}

        mock_build.side_effect = [mock_drive_service, MagicMock()]  # drive and colab services

        client = ColabAPIClient(credentials_path="fake/path/service.json", test_mode=False)
        notebook_content = {"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 4}

        notebook_id = client.create_notebook(notebook_content, "test_notebook")

        self.assertEqual(notebook_id, "fake_id")
        mock_files.create.assert_called_once()

    @patch("src.colab_generator.colab_api_client.build")
    @patch("src.colab_generator.colab_api_client.Credentials")
    def test_authentication_error(self, mock_credentials, mock_build):
        mock_credentials.from_service_account_file.side_effect = Exception("Invalid credentials")

        with self.assertRaises(AuthenticationError):
            ColabAPIClient(credentials_path="invalid/path.json", test_mode=False)

    @patch("src.colab_generator.colab_api_client.build")
    @patch("src.colab_generator.colab_api_client.Credentials")
    @patch("os.path.exists", return_value=True)
    @patch("builtins.open", new_callable=mock_open, read_data=FAKE_CREDS_JSON)
    def test_execute_notebook_calls_api(self, mock_file, mock_exists, mock_credentials, mock_build):
        mock_creds = MagicMock()
        mock_credentials.from_service_account_file.return_value = mock_creds

        mock_colab_service = MagicMock()
        mock_colab_service.notebooks.return_value.execute.return_value.execute.return_value = {'name': 'execution123'}

        # First build is Drive, second is Colab
        mock_build.side_effect = [MagicMock(), mock_colab_service]

        client = ColabAPIClient(credentials_path="fake/path.json", test_mode=False)
        execution_id = client.execute_notebook(file_id="notebook123")

        self.assertEqual(execution_id, 'execution123')
        mock_colab_service.notebooks.return_value.execute.assert_called_once()

if __name__ == '__main__':
    unittest.main()
