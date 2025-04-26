import datetime
import glob
import os

from git import Repo


class MetadataExtractor:
    def __init__(self):
        # Define common config file patterns to look for
        self.config_patterns = ["config.yaml", "config.yml", "config.json",
                                "settings.yaml", "settings.yml", "settings.json",
                                "config.ini"]

    def extract_metadata(self, file_path):
        """Extract metadata from a file by aggregating Git, file, and configuration information."""
        metadata = {}
        metadata['git'] = self.extract_git_metadata(file_path)
        metadata['file'] = self.extract_file_metadata(file_path)
        metadata['associated_configs'] = self.find_associated_config(file_path)
        return metadata

    def extract_git_metadata(self, file_path):
        """Extract Git metadata for the file.

        Returns a dict with:
          - creation_date: The date of the first commit for the file.
          - last_modified_date: The date of the most recent commit.
          - commit_count: Total number of commits affecting the file.
        """
        git_meta = {}
        try:
            # Locate the repository from the file path (searches parent directories)
            repo = Repo(os.path.dirname(file_path), search_parent_directories=True)
            # List all commits for this file
            commits = list(repo.iter_commits(paths=file_path))
            if commits:
                # Get the oldest commit (for creation date)
                first_commit = commits[-1]
                git_meta['creation_date'] = datetime.datetime.fromtimestamp(first_commit.committed_date).isoformat()
                # Get the most recent commit (for last modified date)
                latest_commit = commits[0]
                git_meta['last_modified_date'] = datetime.datetime.fromtimestamp(latest_commit.committed_date).isoformat()
                # Include count of commits
                git_meta['commit_count'] = len(commits)
            else:
                git_meta['creation_date'] = None
                git_meta['last_modified_date'] = None
                git_meta['commit_count'] = 0
        except Exception as e:
            # If Git data isn't available, leave the fields as None or zero.
            git_meta['creation_date'] = None
            git_meta['last_modified_date'] = None
            git_meta['commit_count'] = 0

        return git_meta

    def extract_file_metadata(self, file_path):
        """Extract basic file metadata including file size, creation time, and modification time."""
        file_meta = {}
        try:
            stat = os.stat(file_path)
            file_meta['size_bytes'] = stat.st_size
            file_meta['creation_date'] = datetime.datetime.fromtimestamp(stat.st_ctime).isoformat()
            file_meta['last_modified_date'] = datetime.datetime.fromtimestamp(stat.st_mtime).isoformat()
            file_meta['file_extension'] = os.path.splitext(file_path)[1]
            file_meta['absolute_path'] = os.path.abspath(file_path)
        except Exception as e:
            file_meta['size_bytes'] = None
            file_meta['creation_date'] = None
            file_meta['last_modified_date'] = None
            file_meta['file_extension'] = None
            file_meta['absolute_path'] = None
        return file_meta

    def find_associated_config(self, file_path):
        """Find associated configuration files located in the same directory as the file.

        Searches for files with common configuration file names.
        Returns a list of absolute paths to found configuration files.
        """
        associated_configs = []
        file_dir = os.path.dirname(file_path)
        # Check each defined pattern in the current directory.
        for pattern in self.config_patterns:
            config_path = os.path.join(file_dir, pattern)
            if os.path.isfile(config_path):
                associated_configs.append(os.path.abspath(config_path))

        # Optionally, search for any *.config.* or *.conf or *.ini files using glob
        glob_patterns = ["*.config.*", "*.conf", "*.ini"]
        for glob_pattern in glob_patterns:
            for config_path in glob.glob(os.path.join(file_dir, glob_pattern)):
                if config_path not in associated_configs:
                    associated_configs.append(os.path.abspath(config_path))

        return associated_configs
