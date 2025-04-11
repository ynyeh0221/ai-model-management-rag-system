import unittest
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock
import sys

# Add the path to the module being tested
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the module to be tested
from src.document_processor.metadata_extractor import MetadataExtractor


class TestMetadataExtractor(unittest.TestCase):
    def setUp(self):
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        self.extractor = MetadataExtractor()

        # Create a test file
        self.test_file_path = os.path.join(self.temp_dir, "test_file.txt")
        with open(self.test_file_path, "w") as f:
            f.write("Test content")

        # Create a config file for testing find_associated_config
        self.config_file_path = os.path.join(self.temp_dir, "config.yaml")
        with open(self.config_file_path, "w") as f:
            f.write("test: value")

    def tearDown(self):
        # Clean up the temporary directory
        shutil.rmtree(self.temp_dir)

    def test_extract_metadata_calls_all_methods(self):
        """Test that extract_metadata calls all the necessary methods"""
        with patch.object(self.extractor, 'extract_git_metadata') as mock_git, \
                patch.object(self.extractor, 'extract_file_metadata') as mock_file, \
                patch.object(self.extractor, 'find_associated_config') as mock_config:
            # Set return values for the mocked methods
            mock_git.return_value = {'test': 'git'}
            mock_file.return_value = {'test': 'file'}
            mock_config.return_value = ['config.yaml']

            # Call the method
            result = self.extractor.extract_metadata(self.test_file_path)

            # Verify all methods were called with the correct arguments
            mock_git.assert_called_once_with(self.test_file_path)
            mock_file.assert_called_once_with(self.test_file_path)
            mock_config.assert_called_once_with(self.test_file_path)

            # Verify the result is correct
            expected = {
                'git': {'test': 'git'},
                'file': {'test': 'file'},
                'associated_configs': ['config.yaml']
            }
            self.assertEqual(result, expected)

    @patch('src.document_processor.metadata_extractor.Repo')
    def test_extract_git_metadata_with_commits(self, mock_repo_class):
        """Test extract_git_metadata when commits exist"""
        # Create mock commits
        commit1 = MagicMock()
        commit1.committed_date = 1000000000  # Some timestamp
        commit2 = MagicMock()
        commit2.committed_date = 1100000000  # Later timestamp

        # Set up mock repo
        mock_repo = MagicMock()
        mock_repo.iter_commits.return_value = [commit2, commit1]  # Newest first
        mock_repo_class.return_value = mock_repo

        result = self.extractor.extract_git_metadata(self.test_file_path)

        # Verify repo was created with correct parameters
        mock_repo_class.assert_called_once()

        # Verify commits were retrieved
        mock_repo.iter_commits.assert_called_once()

        # Verify result contains expected data
        self.assertEqual(result['commit_count'], 2)
        self.assertIsNotNone(result['creation_date'])
        self.assertIsNotNone(result['last_modified_date'])

    @patch('src.document_processor.metadata_extractor.Repo')
    def test_extract_git_metadata_no_commits(self, mock_repo_class):
        """Test extract_git_metadata when no commits exist"""
        # Set up mock repo with no commits
        mock_repo = MagicMock()
        mock_repo.iter_commits.return_value = []
        mock_repo_class.return_value = mock_repo

        result = self.extractor.extract_git_metadata(self.test_file_path)

        # Verify result contains expected data
        self.assertEqual(result['commit_count'], 0)
        self.assertIsNone(result['creation_date'])
        self.assertIsNone(result['last_modified_date'])

    @patch('src.document_processor.metadata_extractor.Repo')
    def test_extract_git_metadata_exception(self, mock_repo_class):
        """Test extract_git_metadata when an exception occurs"""
        # Make the Repo constructor raise an exception
        mock_repo_class.side_effect = Exception("Test exception")

        result = self.extractor.extract_git_metadata(self.test_file_path)

        # Verify result contains expected data
        self.assertEqual(result['commit_count'], 0)
        self.assertIsNone(result['creation_date'])
        self.assertIsNone(result['last_modified_date'])

    def test_extract_file_metadata_success(self):
        """Test extract_file_metadata when the file exists"""
        result = self.extractor.extract_file_metadata(self.test_file_path)

        # Verify result contains expected data
        self.assertIsNotNone(result['size_bytes'])
        self.assertIsNotNone(result['creation_date'])
        self.assertIsNotNone(result['last_modified_date'])
        self.assertEqual(result['file_extension'], '.txt')
        self.assertEqual(result['absolute_path'], os.path.abspath(self.test_file_path))

    def test_extract_file_metadata_exception(self):
        """Test extract_file_metadata when an exception occurs"""
        # Use a non-existent file path
        result = self.extractor.extract_file_metadata(os.path.join(self.temp_dir, "nonexistent.txt"))

        # Verify result contains expected data
        self.assertIsNone(result['size_bytes'])
        self.assertIsNone(result['creation_date'])
        self.assertIsNone(result['last_modified_date'])
        self.assertIsNone(result['file_extension'])
        self.assertIsNone(result['absolute_path'])

    def test_find_associated_config(self):
        """Test find_associated_config"""
        result = self.extractor.find_associated_config(self.test_file_path)

        # Verify result contains the config file we created
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], os.path.abspath(self.config_file_path))

    @patch('src.document_processor.metadata_extractor.glob.glob')
    def test_find_associated_config_with_glob(self, mock_glob):
        """Test find_associated_config with glob patterns"""
        # Set up mock for glob to return additional config files
        extra_config = os.path.join(self.temp_dir, "extra.config.json")
        mock_glob.return_value = [extra_config]

        result = self.extractor.find_associated_config(self.test_file_path)

        # Verify result contains both the direct match and the glob match
        self.assertEqual(len(result), 2)
        self.assertIn(os.path.abspath(self.config_file_path), result)
        self.assertIn(extra_config, result)


if __name__ == '__main__':
    unittest.main()