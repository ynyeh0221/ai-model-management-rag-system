class ThumbnailTable:
    """Custom table class to display ASCII thumbnails."""

    def __init__(self, is_search_result):
        self.rows = []
        # Different headers based on the context
        if is_search_result:
            self.headers = ["#", "ID", "Model ID", "Creation Date", "Last Modified Date", "Epoch", "Thumbnail", "Image Path"]
        else:
            self.headers = ["ID", "Thumbnail", "Image Path"]

    def add_row(self, row_data, ascii_img=None):
        """Add a row to the table with thumbnail data."""
        if ascii_img:
            row_data_with_thumbnail = row_data.copy()
            # Insert thumbnail before the last column (image path)
            row_data_with_thumbnail.insert(len(row_data_with_thumbnail) - 1, ascii_img)
            self.rows.append(row_data_with_thumbnail)
        else:
            # Insert placeholder before the last column
            row_data.insert(len(row_data) - 1, ["Thumbnail not available"])
            self.rows.append(row_data)

    def __str__(self):
        """Render the table as a string."""
        if not self.rows:
            return "No rows to display"

        thumbnail_idx = self.headers.index("Thumbnail")

        # 1) calculate all column widths (excluding thumbnail column)
        col_widths = self._calculate_column_widths(thumbnail_idx)

        # 2) build header line and its separator
        header_line, separator = self._build_header_and_separator(col_widths, thumbnail_idx)

        # 3) build body lines (for each row)
        body_lines = []
        for row in self.rows:
            body_lines.extend(
                self._format_single_row(row, col_widths, thumbnail_idx, separator)
            )

        # join header + separator and all row‐blocks
        return "\n".join([header_line, separator] + body_lines)

    def _calculate_column_widths(self, thumbnail_idx):
        """
        Return a list of maximum widths for every non‐thumbnail column.
        E.g., if headers = ["#", "Name", "Thumbnail", "Path"], and there are 3 rows,
        we compute the width of "#" vs. digits, "Name" vs. names, skip Thumbnail, "Path" vs. paths.
        """
        widths = []
        for i in range(len(self.headers)):
            if i == thumbnail_idx:
                continue

            # find max length among all rows in column i, plus the header length
            max_content_len = max((len(str(row[i])) for row in self.rows), default=0)
            header_len = len(self.headers[i])
            widths.append(max(max_content_len, header_len))
        return widths

    def _build_header_and_separator(self, col_widths, thumbnail_idx):
        """
        Build the header line (with proper left/right alignment) and its underline separator.
        Returns (header_line_str, separator_str).
        """
        parts = []
        non_thumb_col = 0

        for idx, header in enumerate(self.headers):
            if idx == thumbnail_idx:
                # keep the literal header "Thumbnail" in place
                parts.append(header)
            else:
                width = col_widths[non_thumb_col]
                # if it’s the very first column AND it's labeled "#", right‐align
                if idx == 0 and header == "#":
                    parts.append(f"{header:>{width}}")
                else:
                    parts.append(f"{header:<{width}}")
                non_thumb_col += 1

        header_line = " | ".join(parts)
        separator = "-" * len(header_line)
        return header_line, separator

    def _format_single_row(self, row, col_widths, thumbnail_idx, separator):
        """
        Given one `row` (a list), produce a block of lines:
          - First line: all non‐thumbnail columns joined by " | ", followed by " |"
          - Then: each line of the thumbnail (row[thumbnail_idx]) indented under the " |"
          - Then: the final path (or whatever data follows thumbnail) indented the same way
          - Finally: a separator line
        Returns a list of strings.
        """
        # 1) format all non‐thumbnail cells into a list of strings
        formatted_cells = []
        non_thumb_col = 0
        for i, cell in enumerate(row):
            if i == thumbnail_idx:
                continue

            width = col_widths[non_thumb_col]
            # if the first column is "#", right‐align; else left‐align
            if i == 0 and self.headers[0] == "#":
                formatted_cells.append(f"{row[i]:>{width}}")
            else:
                formatted_cells.append(f"{str(row[i]):<{width}}")
            non_thumb_col += 1

        # 2) split the formatted_cells at the index matching thumbnail_idx
        #    (because formatted_cells has one entry per non‐thumbnail column,
        #     so its length == len(self.headers) - 1).
        #    We want "first_part" to contain everything _before_ thumbnail_idx,
        #    and "second_part" to contain everything _after_ thumbnail_idx.
        #
        #    In effect, if thumbnail_idx = 2 and headers = [0,1,2,3],
        #    then non-thumb columns are [0, 1, 3], so formatted_cells = [col0, col1, col3].
        #    We want first_part = formatted_cells[:2], second_part = formatted_cells[2:].
        first_part = " | ".join(formatted_cells[:thumbnail_idx])
        second_part = " | ".join(formatted_cells[thumbnail_idx:])

        lines = []
        # 3) line for "first part" + an orphaned separator pipe at the end
        lines.append(f"{first_part} |")

        # 4) add one line per thumbnail‐text (thumbnail is assumed to be a list of strings)
        indent = " " * (len(first_part) + 2)  # 2 for the "|"
        for thumb_line in row[thumbnail_idx]:
            lines.append(f"{indent}| {thumb_line}")

        # 5) add the final "path" (or other trailing data) under the same indentation
        #    i.e., indent + "|" + second_part
        lines.append(f"{indent}| {second_part}")

        # 6) finally append the separator
        lines.append(separator)

        return lines