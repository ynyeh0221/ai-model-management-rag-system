"""
MetadataExtractor Workflow ASCII Diagram

┌─────────────────────────────────────────────────────────────────────────────────┐
│                         METADATAEXTRACTOR INITIALIZATION                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│  Define Configuration File Patterns:                                            │
│  • config.yaml, config.yml, config.json                                         │
│  • settings.yaml, settings.yml, settings.json                                   │
│  • config.ini                                                                   │
│  • Glob patterns: *.config.*, *.conf, *.ini                                     │
└─────────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            EXTRACT_METADATA(file_path)                          │
│                                [Main Entry Point]                               │
└─────────────────────────────────────────────────────────────────────────────────┘
                                         │
                         ┌───────────────┼───────────────┐
                         ▼               ▼               ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────────────────────────┐
│ EXTRACT_GIT_    │ │ EXTRACT_FILE_   │ │    FIND_ASSOCIATED_CONFIG()         │
│ METADATA()      │ │ METADATA()      │ │                                     │
├─────────────────┤ ├─────────────────┤ ├─────────────────────────────────────┤
│ Input: file_path│ │ Input: file_path│ │ Input: file_path                    │
│                 │ │                 │ │                                     │
│ ┌─────────────┐ │ │ ┌─────────────┐ │ │ ┌─────────────────────────────────┐ │
│ │   GitPython │ │ │ │  os.stat()  │ │ │ │        Config Search            │ │
│ │  Repository │ │ │ │ File System │ │ │ │                                 │ │
│ │   Analysis  │ │ │ │  Metadata   │ │ │ │ 1. Exact Pattern Matching       │ │
│ └─────────────┘ │ │ └─────────────┘ │ │ │    • config.yaml                │ │
│        │        │ │        │        │ │ │    • settings.json              │ │
│        ▼        │ │        ▼        │ │ │    • config.ini                 │ │
│ ┌─────────────┐ │ │ ┌─────────────┐ │ │ │                                 │ │
│ │ Find Repo   │ │ │ │ Get File    │ │ │ │ 2. Glob Pattern Matching        │ │
│ │ (search     │ │ │ │ Statistics  │ │ │ │    • *.config.*                 │ │
│ │ parents)    │ │ │ │             │ │ │ │    • *.conf                     │ │
│ └─────────────┘ │ │ └─────────────┘ │ │ │    • *.ini                      │ │
│        │        │ │        │        │ │ └─────────────────────────────────┘ │
│        ▼        │ │        ▼        │ └─────────────────────────────────────┘
│ ┌─────────────┐ │ │ ┌─────────────┐ │
│ │ Get Commits │ │ │ │ Extract:    │ │
│ │ for File    │ │ │ │ • Size      │ │
│ │ iter_commits│ │ │ │ • Timestamps│ │
│ │ (paths=...) │ │ │ │ • Extension │ │
│ └─────────────┘ │ │ │ • Abs Path  │ │
│        │        │ │ └─────────────┘ │
│        ▼        │ └─────────────────┘
│ ┌─────────────┐ │
│ │ Process     │ │
│ │ Commit List │ │
│ └─────────────┘ │
└─────────────────┘
         │                │                │
         ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            DETAILED EXTRACTION WORKFLOWS                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────────┐│
│  │                        GIT METADATA EXTRACTION                              ││
│  │  ┌──────────────────────────────────────────────────────────────────────────┤│
│  │  │  try:                                                                    ││
│  │  │    1. Locate Git Repository                                              ││
│  │  │       Repo(dirname(file_path), search_parent_directories=True)           ││
│  │  │                                                                          ││
│  │  │    2. Get All Commits for File                                           ││
│  │  │       commits = list(repo.iter_commits(paths=file_path))                 ││
│  │  │                                                                          ││
│  │  │    3. Process Commit History                                             ││
│  │  │       ┌─────────────────────────────────────────────────────────────┐    ││
│  │  │       │  If commits exist:                                          │    ││
│  │  │       │  • first_commit = commits[-1]  (oldest)                     │    ││
│  │  │       │  • latest_commit = commits[0]  (newest)                     │    ││
│  │  │       │  • creation_date = first_commit.committed_date              │    ││
│  │  │       │  • last_modified_date = latest_commit.committed_date        │    ││
│  │  │       │  • commit_count = len(commits)                              │    ││
│  │  │       └─────────────────────────────────────────────────────────────┘    ││
│  │  │                                                                          ││
│  │  │  except Exception:                                                       ││
│  │  │    • creation_date = None                                                ││
│  │  │    • last_modified_date = None                                           ││
│  │  │    • commit_count = 0                                                    ││
│  │  │                                                                          ││
│  │  │  Output: {                                                               ││
│  │  │    "creation_date": "2023-01-15T10:30:00",                               ││
│  │  │    "last_modified_date": "2024-03-20T15:45:30",                          ││
│  │  │    "commit_count": 15                                                    ││
│  │  │  }                                                                       ││
│  │  └──────────────────────────────────────────────────────────────────────────┘│
│  └──────────────────────────────────────────────────────────────────────────────┘
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────────┐│
│  │                        FILE METADATA EXTRACTION                            │ │
│  │  ┌─────────────────────────────────────────────────────────────────────────┤ │
│  │  │  try:                                                                   │ │
│  │  │    1. Get File Statistics                                               │ │
│  │  │       stat = os.stat(file_path)                                         │ │
│  │  │                                                                         │ │
│  │  │    2. Extract File Properties                                           │ │
│  │  │       ┌─────────────────────────────────────────────────────────────┐   │ │
│  │  │       │  • size_bytes = stat.st_size                                │   │ │
│  │  │       │  • creation_date = datetime.fromtimestamp(stat.st_ctime)    │   │ │
│  │  │       │  • last_modified_date = datetime.fromtimestamp()            │   │ │
│  │  │       │  • file_extension = os.path.splitext(file_path)[1]          │   │ │
│  │  │       │  • absolute_path = os.path.abspath(file_path)               │   │ │
│  │  │       └─────────────────────────────────────────────────────────────┘   │ │
│  │  │                                                                         │ │
│  │  │  except Exception:                                                      │ │
│  │  │    • All fields = None                                                  │ │
│  │  │                                                                         │ │
│  │  │  Output: {                                                              │ │
│  │  │    "size_bytes": 2048576,                                               │ │
│  │  │    "creation_date": "2023-01-15T10:30:00",                              │ │
│  │  │    "last_modified_date": "2024-03-20T15:45:30",                         │ │
│  │  │    "file_extension": ".py",                                             │ │
│  │  │    "absolute_path": "/full/path/to/file.py"                             │ │
│  │  │  }                                                                      │ │
│  │  └─────────────────────────────────────────────────────────────────────────┘ │
│  └─────────────────────────────────────────────────────────────────────────────┘│
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────────┐│
│  │                     CONFIGURATION FILE DISCOVERY                           │ │
│  │  ┌─────────────────────────────────────────────────────────────────────────┤ │
│  │  │  1. Get File Directory                                                  │ │
│  │  │     file_dir = os.path.dirname(file_path)                               │ │
│  │  │                                                                         │ │
│  │  │  2. Exact Pattern Matching                                              │ │
│  │  │     For each pattern in self.config_patterns:                           │ │
│  │  │     ┌─────────────────────────────────────────────────────────────┐     │ │
│  │  │     │  config_path = os.path.join(file_dir, pattern)              │     │ │
│  │  │     │  if os.path.isfile(config_path):                            │     │ │
│  │  │     │    associated_configs.append(os.path.abspath(config_path))  │     │ │
│  │  │     └─────────────────────────────────────────────────────────────┘     │ │
│  │  │                                                                         │ │
│  │  │  3. Glob Pattern Matching                                               │ │
│  │  │     For each glob_pattern in ["*.config.*", "*.conf", "*.ini"]:         │ │
│  │  │     ┌─────────────────────────────────────────────────────────────┐     │ │
│  │  │     │  matches = glob.glob(os.path.join(file_dir, glob_pattern))  │     │ │
│  │  │     │  for match in matches:                                      │     │ │
│  │  │     │    if match not in associated_configs:                      │     │ │
│  │  │     │      associated_configs.append(os.path.abspath(match))      │     │ │
│  │  │     └─────────────────────────────────────────────────────────────┘     │ │
│  │  │                                                                         │ │
│  │  │  Output: [                                                              │ │
│  │  │    "/path/to/config.yaml",                                              │ │
│  │  │    "/path/to/settings.json",                                            │ │
│  │  │    "/path/to/app.config.ini"                                            │ │
│  │  │  ]                                                                      │ │
│  │  └─────────────────────────────────────────────────────────────────────────┘ │
│  └─────────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              FINAL AGGREGATED OUTPUT                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│  {                                                                              │
│    "git": {                              // Git metadata                        │
│      "creation_date": "2023-01-15T10:30:00",                                    │
│      "last_modified_date": "2024-03-20T15:45:30",                               │
│      "commit_count": 15                                                         │
│    },                                                                           │
│    "file": {                             // File system metadata                │
│      "size_bytes": 2048576,                                                     │
│      "creation_date": "2023-01-15T10:30:00",                                    │
│      "last_modified_date": "2024-03-20T15:45:30",                               │
│      "file_extension": ".py",                                                   │
│      "absolute_path": "/full/path/to/file.py"                                   │
│    },                                                                           │
│    "associated_configs": [               // Related configuration files         │
│      "/path/to/config.yaml",                                                    │
│      "/path/to/settings.json",                                                  │
│      "/path/to/app.config.ini"                                                  │
│    ]                                                                            │
│  }                                                                              │
└─────────────────────────────────────────────────────────────────────────────────┘

Key Features:
┌─────────────────────────────────────────────────────────────────────────────────┐
│ • Multi-Source Metadata: Combines Git history, file system, and config data     │
│ • Robust Error Handling: Graceful degradation when data sources unavailable     │
│ • Git Integration: Uses GitPython to extract repository history                 │
│ • Flexible Config Discovery: Multiple patterns for finding configuration files  │
│ • Timestamp Consistency: ISO format timestamps across all metadata sources      │
│ • Path Resolution: Always returns absolute paths for reliable file references   │
└─────────────────────────────────────────────────────────────────────────────────┘

Flow Summary:
1. Initialize with predefined configuration file patterns
2. Extract metadata from three independent sources in parallel:
   - Git repository history (creation, modification, commit count)
   - File system metadata (size, timestamps, extension, path)
   - Associated configuration files (exact and glob pattern matching)
3. Aggregate all metadata into a structured dictionary with error handling
4. Return comprehensive metadata suitable for indexing and analysis
"""
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
        except Exception:
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
        except Exception:
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
