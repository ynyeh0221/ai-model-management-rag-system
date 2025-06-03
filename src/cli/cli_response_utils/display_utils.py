"""Utility helpers for displaying CLI output."""

from datetime import datetime


class DisplayUtils:
    """Utility class for common display functions and constants."""

    # ASCII characters for grayscale representation (from dark to light)
    ASCII_CHARS = '@%#*+=-:. '

    @staticmethod
    def truncate_string(text, max_length: int = 120):
        """Truncate a string if it exceeds ``max_length``."""
        if isinstance(text, str) and len(text) > max_length:
            return text[: max_length - 3] + "..."
        return text

    @staticmethod
    def format_timestamp(timestamp, format_str: str = "%Y-%m-%dT%H:%M"):
        """Return ``timestamp`` formatted according to ``format_str``.

        - If ``timestamp`` is falsy, return ``"Unknown"``.
        - Non-string values are returned unchanged.
        - Strings are parsed with :func:`datetime.fromisoformat` with ``"Z"``
          interpreted as UTC.  If parsing fails, the original value is returned.
        """
        if not timestamp:
            return "Unknown"

        if not isinstance(timestamp, str):
            return timestamp

        try:
            ts = timestamp.replace("Z", "+00:00")
            dt = datetime.fromisoformat(ts)
            return dt.strftime(format_str)
        except (ValueError, TypeError):
            return timestamp

