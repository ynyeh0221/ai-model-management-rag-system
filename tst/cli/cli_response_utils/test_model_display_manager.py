import unittest
from unittest.mock import Mock, patch


class TestModelDisplayManager(unittest.TestCase):
    """Test suite for ModelDisplayManager class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Import the class to test
        from cli.cli_response_utils.model_display_manager import ModelDisplayManager
        self.ModelDisplayManager = ModelDisplayManager

    def test_display_models_pretty_basic(self):
        """Test basic display of models in pretty table format."""
        available_models = [
            {
                'model_id': 'model1',
                'creation_date': '2023-01-01T10:00:00',
                'last_modified_date': '2023-01-02T10:00:00',
                'absolute_path': '/path/to/model1',
                'images_folder': '/images/model1'
            },
            {
                'model_id': 'model2',
                'creation_date': '2023-01-03T10:00:00',
                'last_modified_date': '2023-01-04T10:00:00',
                'absolute_path': '/path/to/model2',
                'images_folder': '/images/model2'
            }
        ]

        with patch('builtins.print') as mock_print:
            self.ModelDisplayManager.display_models_pretty(available_models)

            # Should have called print at least once
            mock_print.assert_called()

            # Get the printed content
            printed_content = str(mock_print.call_args[0][0])

            # Check that model data is present
            self.assertIn('model1', printed_content)
            self.assertIn('model2', printed_content)
            self.assertIn('Rank', printed_content)
            self.assertIn('Model ID', printed_content)

    def test_display_models_pretty_sorting(self):
        """Test that models are sorted by creation date in descending order."""
        available_models = [
            {
                'model_id': 'older_model',
                'creation_date': '2023-01-01T10:00:00',
                'last_modified_date': '2023-01-01T10:00:00',
                'absolute_path': '/path/to/older',
                'images_folder': '/images/older'
            },
            {
                'model_id': 'newer_model',
                'creation_date': '2023-01-03T10:00:00',
                'last_modified_date': '2023-01-03T10:00:00',
                'absolute_path': '/path/to/newer',
                'images_folder': '/images/newer'
            }
        ]

        with patch('builtins.print') as mock_print:
            self.ModelDisplayManager.display_models_pretty(available_models)

            printed_content = str(mock_print.call_args[0][0])

            # Newer model should appear first (rank 1)
            newer_pos = printed_content.find('newer_model')
            older_pos = printed_content.find('older_model')

            self.assertLess(newer_pos, older_pos, "Newer model should appear before older model")

    def test_display_models_pretty_truncation(self):
        """Test that long fields are properly truncated."""
        available_models = [
            {
                'model_id': 'very_long_model_id_that_exceeds_fifty_characters_definitely',
                'creation_date': '2023-01-01T10:00:00',
                'last_modified_date': '2023-01-01T10:00:00',
                'absolute_path': '/very/long/path/that/exceeds/one/hundred/characters' + '/more' * 20,
                'images_folder': '   /path/with/leading/and/trailing/spaces   '
            }
        ]

        with patch('builtins.print') as mock_print:
            self.ModelDisplayManager.display_models_pretty(available_models)

            printed_content = str(mock_print.call_args[0][0])

            # Should contain truncated content (indicated by "...")
            self.assertIn('...', printed_content)

    def test_display_reranked_results_pretty_basic(self):
        """Test basic display of reranked results."""
        reranked_results = [
            {
                'model_id': 'test_model',
                'score': 0.95,
                'distance': 0.05,
                'merged_description': 'Test model description',
                'metadata': {
                    'file': '{"size_bytes": 1048576, "creation_date": "2023-01-01", "last_modified_date": "2023-01-02", "absolute_path": "/test/path"}',
                    'framework': '{"name": "PyTorch", "version": "1.9.0"}',
                    'architecture': '{"type": "CNN", "reason": "Image classification"}',
                    'dataset': '{"name": "ImageNet"}',
                    'training_config': '{"batch_size": 32, "learning_rate": 0.001, "optimizer": "Adam", "epochs": 100, "hardware_used": "GPU"}'
                }
            }
        ]

        with patch('builtins.print') as mock_print:
            self.ModelDisplayManager.display_reranked_results_pretty(reranked_results)

            mock_print.assert_called()
            printed_content = str(mock_print.call_args[0][0])

            # Check for expected content
            self.assertIn('test_model', printed_content)
            self.assertIn('0.950', printed_content)  # Score formatted to 3 decimals
            self.assertIn('PyTorch', printed_content)

    def test_add_result_row_to_table(self):
        """Test adding a result row to table."""
        # Create a mock table
        mock_table = Mock()

        result = {
            'model_id': 'test_model',
            'score': 0.95,
            'distance': 0.05,
            'merged_description': 'Test description',
            'metadata': {
                'file': '{"size_bytes": 2097152}',  # 2MB
                'framework': '{"name": "TensorFlow"}',
                'architecture': '{"type": "RNN"}',
                'dataset': '{"name": "Custom"}',
                'training_config': '{}'
            }
        }

        self.ModelDisplayManager._add_result_row_to_table(mock_table, 0, result)

        # Should have called add_row on the table
        mock_table.add_row.assert_called_once()

        # Get the arguments passed to add_row
        row_args = mock_table.add_row.call_args[0][0]

        # Check some specific values
        self.assertEqual(row_args[0], 1)  # Rank should be indexed + 1
        self.assertEqual(row_args[1], 'test_model')  # Model ID
        self.assertEqual(row_args[2], '0.950')  # Score formatted
        self.assertEqual(row_args[3], '0.050')  # Distance formatted

    def test_parse_nested_json_valid_json(self):
        """Test parsing of valid nested JSON in metadata."""
        result = {
            'metadata': {
                'file': '{"size_bytes": 1024, "path": "/test"}',
                'framework': '{"name": "PyTorch", "version": "1.8"}',
                'invalid_field': 'not_json',  # This won't be processed - only specific fields are
                'architecture': '{"type": "CNN"}'
            }
        }

        parsed = self.ModelDisplayManager._parse_nested_json(result)

        # Valid JSON should be parsed
        self.assertIsInstance(parsed['file'], dict)
        self.assertEqual(parsed['file']['size_bytes'], 1024)

        self.assertIsInstance(parsed['framework'], dict)
        self.assertEqual(parsed['framework']['name'], 'PyTorch')

        # Fields not in the processed list should remain unchanged
        self.assertEqual(parsed['invalid_field'], 'not_json')

        # Architecture should be parsed
        self.assertIsInstance(parsed['architecture'], dict)
        self.assertEqual(parsed['architecture']['type'], 'CNN')

    def test_parse_nested_json_invalid_json(self):
        """Test parsing of invalid JSON in metadata."""
        result = {
            'metadata': {
                'file': '{"invalid": json}',
                'framework': 'not json at all',
                'architecture': '[]'  # Valid JSON but not a dict
            }
        }

        parsed = self.ModelDisplayManager._parse_nested_json(result)

        # All should become empty dicts
        self.assertEqual(parsed['file'], {})
        self.assertEqual(parsed['framework'], {})
        self.assertEqual(parsed['architecture'], {})

    def test_parse_nested_json_no_metadata(self):
        """Test parsing when metadata is missing or not a dict."""
        # Test with no metadata
        result1 = {}
        parsed1 = self.ModelDisplayManager._parse_nested_json(result1)
        self.assertEqual(parsed1, {})

        # Test with metadata as string
        result2 = {'metadata': 'not a dict'}
        parsed2 = self.ModelDisplayManager._parse_nested_json(result2)
        self.assertEqual(parsed2, {})

    def test_extract_file_data_complete(self):
        """Test extracting complete file data."""
        metadata = {
            'file': {
                'size_bytes': 5242880,  # 5MB
                'creation_date': '2023-01-01T10:30:45',
                'last_modified_date': '2023-01-02T15:20:30',
                'absolute_path': '/path/to/model.pth'
            }
        }

        file_data = self.ModelDisplayManager._extract_file_data(metadata)

        self.assertEqual(file_data['size'], '5.0MB')
        self.assertEqual(file_data['created'], '2023-01-01')
        self.assertEqual(file_data['modified'], '2023-01-02')
        self.assertEqual(file_data['path'], '/path/to/model.pth')

    def test_extract_file_data_size_formats(self):
        """Test different file size formats."""
        # Test KB format
        metadata_kb = {'file': {'size_bytes': 2048}}  # 2KB
        file_data_kb = self.ModelDisplayManager._extract_file_data(metadata_kb)
        self.assertEqual(file_data_kb['size'], '2.0KB')

        # Test bytes format
        metadata_bytes = {'file': {'size_bytes': 512}}  # 512 bytes
        file_data_bytes = self.ModelDisplayManager._extract_file_data(metadata_bytes)
        self.assertEqual(file_data_bytes['size'], '512B')

        # Test missing size
        metadata_missing = {'file': {'size_bytes': 'missing'}}
        file_data_missing = self.ModelDisplayManager._extract_file_data(metadata_missing)
        self.assertEqual(file_data_missing['size'], 'missing')

    def test_extract_file_data_missing_fields(self):
        """Test extracting file data with missing fields."""
        metadata = {'file': {}}

        file_data = self.ModelDisplayManager._extract_file_data(metadata)

        self.assertEqual(file_data['size'], 'missing')
        self.assertEqual(file_data['created'], 'missing')
        self.assertEqual(file_data['modified'], 'missing')
        self.assertEqual(file_data['path'], 'missing')

    def test_extract_framework_data_with_version(self):
        """Test extracting framework data with a version."""
        metadata = {
            'framework': {
                'name': 'PyTorch',
                'version': '1.9.0'
            }
        }

        framework = self.ModelDisplayManager._extract_framework_data(metadata)
        self.assertEqual(framework, 'PyTorch 1')

    def test_extract_framework_data_without_version(self):
        """Test extracting framework data without a version."""
        metadata = {
            'framework': {
                'name': 'TensorFlow'
            }
        }

        framework = self.ModelDisplayManager._extract_framework_data(metadata)
        self.assertEqual(framework, 'TensorFlow')

    def test_extract_framework_data_invalid_version(self):
        """Test extracting framework data with an invalid version."""
        test_cases = [
            {'name': 'Keras', 'version': 'missing'},
            {'name': 'Keras', 'version': 'unknown'},
            {'name': 'Keras', 'version': 'unspecified'},
        ]

        for framework_data in test_cases:
            metadata = {'framework': framework_data}
            framework = self.ModelDisplayManager._extract_framework_data(metadata)
            self.assertEqual(framework, 'Keras')

    def test_extract_framework_data_missing(self):
        """Test extracting framework data when missing."""
        metadata = {'framework': {}}

        framework = self.ModelDisplayManager._extract_framework_data(metadata)
        self.assertEqual(framework, 'missing')

    def test_extract_training_data_complete(self):
        """Test extracting complete training configuration data."""
        metadata = {
            'training_config': {
                'batch_size': 64,
                'learning_rate': 0.0001,
                'optimizer': 'SGD',
                'epochs': 200,
                'hardware_used': 'V100'
            }
        }

        training_data = self.ModelDisplayManager._extract_training_data(metadata)

        self.assertEqual(training_data['batch_size'], 64)
        self.assertEqual(training_data['learning_rate'], '1e-04')  # Scientific notation
        self.assertEqual(training_data['optimizer'], 'SGD')
        self.assertEqual(training_data['epochs'], 200)
        self.assertEqual(training_data['hardware'], 'V100')

    def test_extract_training_data_learning_rate_formats(self):
        """Test different learning rate formats."""
        # Small learning rate (should use scientific notation)
        metadata_small = {'training_config': {'learning_rate': 0.0001}}
        training_small = self.ModelDisplayManager._extract_training_data(metadata_small)
        self.assertEqual(training_small['learning_rate'], '1e-04')

        # Regular learning rate (should remain as is)
        metadata_regular = {'training_config': {'learning_rate': 0.01}}
        training_regular = self.ModelDisplayManager._extract_training_data(metadata_regular)
        self.assertEqual(training_regular['learning_rate'], 0.01)

        # Non-numeric learning rate
        metadata_text = {'training_config': {'learning_rate': 'adaptive'}}
        training_text = self.ModelDisplayManager._extract_training_data(metadata_text)
        self.assertEqual(training_text['learning_rate'], 'adaptive')

    def test_extract_training_data_missing_fields(self):
        """Test extracting training data with missing fields."""
        metadata = {'training_config': {}}

        training_data = self.ModelDisplayManager._extract_training_data(metadata)

        self.assertEqual(training_data['batch_size'], 'N/A')
        self.assertEqual(training_data['learning_rate'], 'N/A')
        self.assertEqual(training_data['optimizer'], 'N/A')
        self.assertEqual(training_data['epochs'], 'N/A')
        self.assertEqual(training_data['hardware'], 'N/A')

    def test_score_formatting(self):
        """Test score and distance formatting in result rows."""
        mock_table = Mock()

        # Test with numeric score and distance
        result_numeric = {
            'score': 0.123456,
            'distance': 0.987654,
            'metadata': {}
        }

        self.ModelDisplayManager._add_result_row_to_table(mock_table, 0, result_numeric)
        row_args = mock_table.add_row.call_args[0][0]

        self.assertEqual(row_args[2], '0.123')  # Score to 3 decimals
        self.assertEqual(row_args[3], '0.988')  # Distance to 3 decimals

        # Test with non-numeric values
        mock_table.reset_mock()
        result_text = {
            'score': 'N/A',
            'distance': 'unknown',
            'metadata': {}
        }

        self.ModelDisplayManager._add_result_row_to_table(mock_table, 0, result_text)
        row_args = mock_table.add_row.call_args[0][0]

        self.assertEqual(row_args[2], 'N/A')
        self.assertEqual(row_args[3], 'unknown')

    def test_alternative_field_names(self):
        """Test handling of alternative field names in results."""
        mock_table = Mock()

        # Test alternative score field names
        result_alt = {
            'id': 'alt_model',  # Alternative to model_id
            'similarity': 0.85,  # Alternative to score
            'rank_score': 0.90,  # Another alternative
            'rerank_score': 0.95,  # Yet another alternative
            'metadata': {}
        }

        self.ModelDisplayManager._add_result_row_to_table(mock_table, 0, result_alt)
        row_args = mock_table.add_row.call_args[0][0]

        self.assertEqual(row_args[1], 'alt_model')  # Should use 'id' as model_id
        self.assertEqual(row_args[2], '0.850')  # Should use 'similarity' as score

    def test_description_handling(self):
        """Test handling of merged_description field."""
        mock_table = Mock()

        # Test with N/A description
        result_na = {
            'merged_description': 'N/A',
            'metadata': {}
        }

        self.ModelDisplayManager._add_result_row_to_table(mock_table, 0, result_na)
        row_args = mock_table.add_row.call_args[0][0]

        # N/A should be converted to "missing"
        self.assertEqual(row_args[8], 'missing')

        # Test with normal description
        mock_table.reset_mock()
        result_normal = {
            'merged_description': 'Valid description',
            'metadata': {}
        }

        self.ModelDisplayManager._add_result_row_to_table(mock_table, 0, result_normal)
        row_args = mock_table.add_row.call_args[0][0]

        self.assertEqual(row_args[8], 'Valid description')

    def test_architecture_formatting(self):
        """Test architecture field formatting with type and reason."""
        metadata = {
            'architecture': {
                'type': 'Transformer',
                'reason': 'Natural language processing'
            }
        }

        mock_table = Mock()
        result = {'metadata': metadata}

        self.ModelDisplayManager._add_result_row_to_table(mock_table, 0, result)
        row_args = mock_table.add_row.call_args[0][0]

        # Architecture should be "type\n\nreason"
        expected_arch = 'Transformer\n\nNatural language processing'
        self.assertEqual(row_args[10], expected_arch)

    def test_static_methods(self):
        """Test that all methods are properly defined as static methods."""
        import inspect

        static_methods = [
            'display_models_pretty',
            'display_reranked_results_pretty',
            '_add_result_row_to_table',
            '_parse_nested_json',
            '_extract_file_data',
            '_extract_framework_data',
            '_extract_training_data'
        ]

        for method_name in static_methods:
            method = getattr(self.ModelDisplayManager, method_name)
            sig = inspect.signature(method)
            self.assertNotIn('self', sig.parameters, f"{method_name} should not have 'self' parameter")

    def test_empty_and_edge_cases(self):
        """Test edge cases with empty or minimal data."""
        # Test with an empty model list
        with patch('builtins.print') as mock_print:
            self.ModelDisplayManager.display_models_pretty([])
            mock_print.assert_called()  # Should still print (empty table)

        # Test with empty reranked results
        with patch('builtins.print') as mock_print:
            self.ModelDisplayManager.display_reranked_results_pretty([])
            mock_print.assert_called()  # Should still print (empty table)

    def test_date_truncation(self):
        """Test that long dates are properly truncated."""
        metadata = {
            'file': {
                'creation_date': '2023-01-01T10:30:45.123456Z',
                'last_modified_date': '2023-12-31T23:59:59.999999+00:00'
            }
        }

        file_data = self.ModelDisplayManager._extract_file_data(metadata)

        # Dates should be truncated to YYYY-MM-DD
        self.assertEqual(file_data['created'], '2023-01-01')
        self.assertEqual(file_data['modified'], '2023-12-31')

    def test_integration_with_prettytable(self):
        """Test integration with PrettyTable library."""
        # This test verifies the table configuration works without errors
        available_models = [
            {
                'model_id': 'test_model',
                'creation_date': '2023-01-01T10:00:00',
                'last_modified_date': '2023-01-01T10:00:00',
                'absolute_path': '/test/path',
                'images_folder': '/test/images'
            }
        ]

        # Simply test that the method runs without errors and produces output
        with patch('builtins.print') as mock_print:
            try:
                self.ModelDisplayManager.display_models_pretty(available_models)
                # If we get here, the PrettyTable integration worked
                mock_print.assert_called()

                # Verify the printed content contains expected data
                printed_content = str(mock_print.call_args[0][0])
                self.assertIn('test_model', printed_content)
                self.assertIn('Rank', printed_content)

            except Exception as e:
                self.fail(f"PrettyTable integration failed with error: {e}")


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)