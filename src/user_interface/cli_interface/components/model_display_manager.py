import json
from datetime import datetime

from prettytable import PrettyTable

from user_interface.cli_interface.components.display_utils import DisplayUtils


class ModelDisplayManager:
    """Handles displaying model data in various formats."""

    @staticmethod
    def display_models_pretty(available_models):
        """
        Display models in a nicely formatted table.

        Args:
            available_models (list): List of model dictionaries to display.
        """
        table = PrettyTable()
        table.field_names = ["Rank", "Model ID", "Created", "Last Modified", "Images Folder", "Absolute Path"]

        # Align all columns to the left
        table.align = "l"
        table.align["Rank"] = "c"  # center

        # Sort models by creation date in descending order
        sorted_models = sorted(
            available_models,
            key=lambda m: datetime.fromisoformat(m['creation_date']),
            reverse=True
        )

        for i, model in enumerate(sorted_models):
            model_id = DisplayUtils.truncate_string(model['model_id'], 50)
            created = DisplayUtils.format_timestamp(model['creation_date'])
            modified = DisplayUtils.format_timestamp(model['last_modified_date'])
            absolute_path = DisplayUtils.truncate_string(model['absolute_path'], 100)
            images_folder = DisplayUtils.truncate_string(model['images_folder'].strip(), 120)

            table.add_row([i + 1, model_id, created, modified, images_folder, absolute_path])

        print(table)

    @staticmethod
    def display_reranked_results_pretty(reranked_results):
        """
        Display reranked search results in a nicely formatted table with detailed model metadata.

        Args:
            reranked_results (list): List of reranked search result dictionaries to display.
        """
        from prettytable import PrettyTable, ALL

        table = PrettyTable()
        table.field_names = ["Rank", "Model ID", "Similarity Score", "Similarity Distance", "Size", "Created",
                             "Modified",
                             "Path", "Description", "Framework", "Arch", "Dataset",
                             "Batch", "LR", "Optimizer", "Epochs", "HW"]

        # Align columns
        table.align = "l"  # Default left alignment for all
        table.align["Rank"] = "c"  # center
        table.align["Similarity Score"] = "r"  # right
        table.align["Similarity Distance"] = "r"  # right
        table.align["Size"] = "r"  # right align file size

        # Set max width for columns - optimized for 15-inch laptop
        max_width = {
            "Rank": 4,
            "Model ID": 15,
            "Similarity Score": 6,
            "Similarity Distance": 6,
            "Size": 7,
            "Created": 10,
            "Modified": 10,
            "Path": 15,
            "Description": 20,
            "Framework": 8,
            "Arch": 10,
            "Dataset": 10,
            "Batch": 5,
            "LR": 4,
            "Optimizer": 7,
            "Epochs": 5,
            "HW": 6
        }

        for column in max_width:
            table.max_width[column] = max_width[column]

        # Add horizontal lines and reduce padding for better display
        table.hrules = ALL
        table.padding_width = 1

        for i, result in enumerate(reranked_results):
            ModelDisplayManager._add_result_row_to_table(table, i, result)

        print(table)

    @staticmethod
    def _add_result_row_to_table(table, index, result):
        """Helper method to add a result row to the table."""
        rank = index + 1

        # Parse nested JSON in metadata
        parsed_metadata = ModelDisplayManager._parse_nested_json(result)

        # Extract basic fields
        model_id = result.get('model_id', result.get('id', f'Item {rank}'))

        # Get score from various possible fields
        score = result.get('score', result.get('similarity',
                                               result.get('rank_score', result.get('rerank_score', 'N/A'))))

        # Format score to 3 decimal places if it's a number
        if isinstance(score, (int, float)):
            score = f"{score:.3f}"

        # Get distance
        distance = result.get('distance', 'N/A')

        # Format distance to 3 decimal places if it's a number
        if isinstance(distance, (int, float)):
            distance = f"{distance:.3f}"

        # Extract and format other fields
        file_data = ModelDisplayManager._extract_file_data(parsed_metadata)
        framework = ModelDisplayManager._extract_framework_data(parsed_metadata)
        architecture = parsed_metadata.get('architecture', {}).get('type', 'missing') + "\n\n" + parsed_metadata.get('architecture', {}).get('reason', 'missing')
        dataset = parsed_metadata.get('dataset', {}).get('name', 'missing')
        training_data = ModelDisplayManager._extract_training_data(parsed_metadata)
        description = result.get('merged_description', 'missing')
        if description == "N/A":
            description = "missing"

        # Add row to table
        table.add_row([
            rank, model_id, score, distance,
            file_data['size'], file_data['created'], file_data['modified'],
            file_data['path'], description, framework, architecture, dataset,
            training_data['batch_size'], training_data['learning_rate'],
            training_data['optimizer'], training_data['epochs'], training_data['hardware']
        ])

    @staticmethod
    def _parse_nested_json(result):
        """Helper method to parse nested JSON in metadata."""
        metadata = result.get('metadata', {}) if isinstance(result.get('metadata'), dict) else {}

        # Parse JSON string fields in metadata
        for field in ["file", "framework", "architecture", "dataset", "training_config", "git"]:
            raw_value = metadata.get(field)
            if isinstance(raw_value, str):
                try:
                    parsed = json.loads(raw_value)
                    metadata[field] = parsed if isinstance(parsed, dict) else {}
                except json.JSONDecodeError:
                    metadata[field] = {}

        return metadata

    @staticmethod
    def _extract_file_data(metadata):
        """Extract file data from metadata."""
        file_metadata = metadata.get('file', {})

        # Process file size
        size_bytes = file_metadata.get('size_bytes', 'missing')
        if isinstance(size_bytes, (int, float)):
            size_mb = size_bytes / 1048576  # 1024 * 1024
            if size_mb >= 1:
                file_size = f"{size_mb:.1f}MB"
            else:
                # For small files, show in KB
                size_kb = size_bytes / 1024
                if size_kb >= 1:
                    file_size = f"{size_kb:.1f}KB"
                else:
                    # For very small files, show in bytes
                    file_size = f"{size_bytes}B"
        else:
            file_size = size_bytes  # Keep as "missing" or whatever non-numeric value

        # Process dates
        creation_date = file_metadata.get('creation_date', 'missing')
        if isinstance(creation_date, str) and len(creation_date) > 10:
            creation_date = creation_date[:10]  # Just YYYY-MM-DD

        last_modified = file_metadata.get('last_modified_date', 'missing')
        if isinstance(last_modified, str) and len(last_modified) > 10:
            last_modified = last_modified[:10]  # Just YYYY-MM-DD

        # Extract absolute path
        absolute_path = file_metadata.get('absolute_path', 'missing')

        return {
            'size': file_size,
            'created': creation_date,
            'modified': last_modified,
            'path': absolute_path
        }

    @staticmethod
    def _extract_framework_data(metadata):
        """Extract framework data from metadata."""
        framework_metadata = metadata.get('framework', {})
        framework_name = framework_metadata.get('name', 'missing')
        framework_version = framework_metadata.get('version', '')

        if framework_version and framework_version.lower() not in ['missing', 'unknown', 'unspecified']:
            # Just add major version number
            if '.' in framework_version:
                framework_version = framework_version.split('.')[0]
            return f"{framework_name} {framework_version}"

        return framework_name

    @staticmethod
    def _extract_training_data(metadata):
        """Extract training configuration data from metadata."""
        training_config = metadata.get('training_config', {})

        # Extract batch size
        batch_size = training_config.get('batch_size', 'N/A')

        # Extract and format learning rate
        learning_rate = training_config.get('learning_rate', 'N/A')
        if isinstance(learning_rate, float) and learning_rate < 0.01:
            learning_rate = f"{learning_rate:.0e}"

        # Extract other training parameters
        optimizer = training_config.get('optimizer', 'N/A')
        epochs = training_config.get('epochs', 'N/A')
        hardware = training_config.get('hardware_used', 'N/A')

        return {
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'optimizer': optimizer,
            'epochs': epochs,
            'hardware': hardware
        }