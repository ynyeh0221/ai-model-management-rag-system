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

        # Calculate thumbnail index
        thumbnail_idx = self.headers.index("Thumbnail")

        # Calculate column widths for text columns (excluding thumbnail)
        col_widths = [
            max([len(str(row[i])) for row in self.rows] + [len(self.headers[i])])
            for i in range(len(self.headers)) if i != thumbnail_idx
        ]

        # Create header line
        header_parts = []
        col_idx = 0
        for i, header in enumerate(self.headers):
            if i != thumbnail_idx:
                width = col_widths[col_idx]
                # Right align first column if it's a number (#)
                if i == 0 and header == "#":
                    header_parts.append(f"{header:>{width}}")
                else:
                    header_parts.append(f"{header:<{width}}")
                col_idx += 1
            else:
                header_parts.append(header)

        header = " | ".join(header_parts)
        separator = "-" * len(header)

        # Build the table string
        result = [header, separator]

        for row in self.rows:
            # Format each non-thumbnail column
            formatted_row = []
            col_idx = 0
            for i in range(len(row)):
                if i != thumbnail_idx:
                    width = col_widths[col_idx]
                    # Right align first column if it's a number (#)
                    if i == 0 and self.headers[0] == "#":
                        formatted_row.append(f"{row[i]:>{width}}")
                    else:
                        formatted_row.append(f"{str(row[i]):<{width}}")
                    col_idx += 1

            # Calculate where to split the row (before thumbnail)
            first_part = " | ".join(formatted_row[:thumbnail_idx])
            second_part = " | ".join(formatted_row[thumbnail_idx:])

            # Add the row header
            result.append(f"{first_part} |")

            # Add thumbnail lines
            for line in row[thumbnail_idx]:
                result.append(f"{' ' * (len(first_part) + 2)}| {line}")

            # Add the file path
            result.append(f"{' ' * (len(first_part) + 2)}| {second_part}")

            # Add separator
            result.append(separator)

        return "\n".join(result)