"""
model_id_utils.py

Utilities for generating consistent model identifiers from file paths.
"""

from pathlib import Path


class ModelIdUtils:
    """
    Utility class for generating model IDs from file paths.
    """

    @staticmethod
    def get_model_id(file_path: str):
        """
        Generate a model ID based on the directory name and file stem of the given file path.

        The model ID is constructed by combining the parent directory name and the file stem
        (filename without extension) with an underscore.

        Examples:
            >>> ModelIdUtils.get_model_id("/models/resnet50/config.json")
            "resnet50_config"

        Args:
            file_path (str): The path to the file. Must not be None.

        Returns:
            str: The generated model ID in the format
                 "<parent_directory>_<file_stem>".

        Raises:
            ValueError: If `file_path` is None.
        """
        if file_path is None:
            raise Exception('file_path is None')
        # Prepare model ID
        file_path_obj = Path(file_path)
        folder_name = file_path_obj.parent.name
        file_stem = file_path_obj.stem
        return f"{folder_name}_{file_stem}"