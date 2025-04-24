"""
Refactored implementation of the AI Model Management RAG system UI components.
This refactoring separates concerns into multiple classes for better maintainability.
"""

from datetime import datetime


class DisplayUtils:
    """Utility class for common display functions and constants."""

    # ASCII characters for grayscale representation (from dark to light)
    ASCII_CHARS = '@%#*+=-:. '

    @staticmethod
    def truncate_string(text, max_length=120):
        """Truncate a string if it exceeds the maximum length."""
        if isinstance(text, str) and len(text) > max_length:
            return text[:max_length - 3] + "..."
        return text

    @staticmethod
    def format_timestamp(timestamp, format_str="%Y-%m-%dT%H:%M"):
        """Format a timestamp string."""
        if not timestamp:
            return "Unknown"
        try:
            return datetime.fromisoformat(timestamp).strftime(format_str)
        except (ValueError, TypeError):
            return timestamp