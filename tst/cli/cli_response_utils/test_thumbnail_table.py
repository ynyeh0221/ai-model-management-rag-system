import unittest

from src.cli.cli_response_utils.thumbnail_table import ThumbnailTable


class TestThumbnailTable(unittest.TestCase):
    """Test suite for ThumbnailTable class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Import the class to test
        self.ThumbnailTable = ThumbnailTable

    def test_init_search_result_table(self):
        """Test initialization for search result table."""
        table = self.ThumbnailTable(is_search_result=True)

        # Check that rows are initialized as an empty list
        self.assertEqual(table.rows, [])

        # Check search result headers
        expected_headers = ["#", "ID", "Model ID", "Creation Date", "Last Modified Date", "Epoch", "Thumbnail",
                            "Image Path"]
        self.assertEqual(table.headers, expected_headers)
        self.assertEqual(len(table.headers), 8)

    def test_init_regular_table(self):
        """Test initialization for regular (non-search result) table."""
        table = self.ThumbnailTable(is_search_result=False)

        # Check that rows are initialized as an empty list
        self.assertEqual(table.rows, [])

        # Check regular headers
        expected_headers = ["ID", "Thumbnail", "Image Path"]
        self.assertEqual(table.headers, expected_headers)
        self.assertEqual(len(table.headers), 3)

    def test_add_row_with_thumbnail_search_result(self):
        """Test adding a row with thumbnail data to search result table."""
        table = self.ThumbnailTable(is_search_result=True)

        # Test data for the search result (7 fields before thumbnail insertion)
        row_data = [1, "img123", "model1", "2023-01-01", "2023-01-02", "10", "/path/to/image.jpg"]
        ascii_img = ["  ***  ", " ***** ", "*******"]

        table.add_row(row_data, ascii_img)

        # Check that row was added
        self.assertEqual(len(table.rows), 1)

        # Check that thumbnail was inserted before the last column (image path)
        expected_row = [1, "img123", "model1", "2023-01-01", "2023-01-02", "10", ascii_img, "/path/to/image.jpg"]
        self.assertEqual(table.rows[0], expected_row)

    def test_add_row_with_thumbnail_regular_table(self):
        """Test adding a row with thumbnail data to regular table."""
        table = self.ThumbnailTable(is_search_result=False)

        # Test data for regular table (2 fields before thumbnail insertion)
        row_data = ["img123", "/path/to/image.jpg"]
        ascii_img = ["  ***  ", " ***** ", "*******"]

        table.add_row(row_data, ascii_img)

        # Check that row was added
        self.assertEqual(len(table.rows), 1)

        # Check that thumbnail was inserted before the last column (image path)
        expected_row = ["img123", ascii_img, "/path/to/image.jpg"]
        self.assertEqual(table.rows[0], expected_row)

    def test_add_row_without_thumbnail_search_result(self):
        """Test adding a row without thumbnail data to search result table."""
        table = self.ThumbnailTable(is_search_result=True)

        row_data = [1, "img123", "model1", "2023-01-01", "2023-01-02", "10", "/path/to/image.jpg"]

        table.add_row(row_data, ascii_img=None)

        # Check that row was added
        self.assertEqual(len(table.rows), 1)

        # Check that placeholder was inserted before the last column
        expected_row = [1, "img123", "model1", "2023-01-01", "2023-01-02", "10", ["Thumbnail not available"],
                        "/path/to/image.jpg"]
        self.assertEqual(table.rows[0], expected_row)

    def test_add_row_without_thumbnail_regular_table(self):
        """Test adding a row without thumbnail data to regular table."""
        table = self.ThumbnailTable(is_search_result=False)

        row_data = ["img123", "/path/to/image.jpg"]

        table.add_row(row_data, ascii_img=None)

        # Check that row was added
        self.assertEqual(len(table.rows), 1)

        # Check that placeholder was inserted before the last column
        expected_row = ["img123", ["Thumbnail not available"], "/path/to/image.jpg"]
        self.assertEqual(table.rows[0], expected_row)

    def test_add_multiple_rows(self):
        """Test adding multiple rows to the table."""
        table = self.ThumbnailTable(is_search_result=False)

        # Add first row with thumbnail
        row_data1 = ["img1", "/path/to/image1.jpg"]
        ascii_img1 = ["***", "***", "***"]
        table.add_row(row_data1, ascii_img1)

        # Add second row without a thumbnail
        row_data2 = ["img2", "/path/to/image2.jpg"]
        table.add_row(row_data2, ascii_img=None)

        # Check that both rows were added
        self.assertEqual(len(table.rows), 2)

        # Verify first row
        expected_row1 = ["img1", ascii_img1, "/path/to/image1.jpg"]
        self.assertEqual(table.rows[0], expected_row1)

        # Verify the second row
        expected_row2 = ["img2", ["Thumbnail not available"], "/path/to/image2.jpg"]
        self.assertEqual(table.rows[1], expected_row2)

    def test_str_empty_table(self):
        """Test string representation of empty table."""
        table = self.ThumbnailTable(is_search_result=False)

        result = str(table)
        self.assertEqual(result, "No rows to display")

    def test_str_regular_table_with_data(self):
        """Test string representation of regular table with data."""
        table = self.ThumbnailTable(is_search_result=False)

        # Add a row with thumbnail
        row_data = ["img123", "/path/to/image.jpg"]
        ascii_img = ["  ***  ", " ***** ", "*******"]
        table.add_row(row_data, ascii_img)

        result = str(table)

        # Check that result contains expected elements
        self.assertIn("ID", result)
        self.assertIn("Thumbnail", result)
        self.assertIn("Image Path", result)
        self.assertIn("img123", result)
        self.assertIn("/path/to/image.jpg", result)
        self.assertIn("  ***  ", result)
        self.assertIn(" ***** ", result)
        self.assertIn("*******", result)

        # Check that it's a multi-line string
        self.assertIn("\n", result)

        # Check for separators
        self.assertIn("-", result)
        self.assertIn("|", result)

    def test_str_search_result_table_with_data(self):
        """Test string representation of search result table with data."""
        table = self.ThumbnailTable(is_search_result=True)

        # Add a row with thumbnail
        row_data = [1, "img123", "model1", "2023-01-01", "2023-01-02", "10", "/path/to/image.jpg"]
        ascii_img = ["***", "***"]
        table.add_row(row_data, ascii_img)

        result = str(table)

        # Check that result contains expected headers
        search_headers = ["#", "ID", "Model ID", "Creation Date", "Last Modified Date", "Epoch", "Thumbnail",
                          "Image Path"]
        for header in search_headers:
            self.assertIn(header, result)

        # Check that result contains data
        self.assertIn("1", result)
        self.assertIn("img123", result)
        self.assertIn("model1", result)
        self.assertIn("2023-01-01", result)
        self.assertIn("***", result)

    def test_str_table_without_thumbnail(self):
        """Test string representation of table with placeholder thumbnail."""
        table = self.ThumbnailTable(is_search_result=False)

        # Add a row without a thumbnail
        row_data = ["img123", "/path/to/image.jpg"]
        table.add_row(row_data, ascii_img=None)

        result = str(table)

        # Check that placeholder is included
        self.assertIn("Thumbnail not available", result)
        self.assertIn("img123", result)
        self.assertIn("/path/to/image.jpg", result)

    def test_str_multiple_rows(self):
        """Test string representation with multiple rows."""
        table = self.ThumbnailTable(is_search_result=False)

        # Add multiple rows
        table.add_row(["img1", "/path1.jpg"], ["***", "***"])
        table.add_row(["img2", "/path2.jpg"], ["###", "###"])
        table.add_row(["img3", "/path3.jpg"], None)

        result = str(table)

        # Check that all data is present
        self.assertIn("img1", result)
        self.assertIn("img2", result)
        self.assertIn("img3", result)
        self.assertIn("/path1.jpg", result)
        self.assertIn("/path2.jpg", result)
        self.assertIn("/path3.jpg", result)
        self.assertIn("***", result)
        self.assertIn("###", result)
        self.assertIn("Thumbnail not available", result)

        # Check for multiple separators (one after each row)
        separator_count = result.count("----")  # Part of the separator line
        self.assertGreaterEqual(separator_count, 3)  # At least 3 rows worth

    def test_column_width_calculation(self):
        """Test that column widths are calculated correctly."""
        table = self.ThumbnailTable(is_search_result=False)

        # Add rows with different length data
        table.add_row(["a", "/short.jpg"], ["*"])
        table.add_row(["very_long_id_name", "/very/long/path/to/image.jpg"], ["*"])

        result = str(table)

        # The longer values should determine column width
        self.assertIn("very_long_id_name", result)
        self.assertIn("/very/long/path/to/image.jpg", result)

        # Should have proper alignment
        lines = result.split('\n')
        # All non-thumbnail lines should have consistent structure
        header_line = [line for line in lines if "ID" in line and "Thumbnail" in line][0]
        self.assertIn("|", header_line)

    def test_right_alignment_for_numeric_column(self):
        """Test that numeric columns (first column with #) are right-aligned."""
        table = self.ThumbnailTable(is_search_result=True)

        # Add rows with different number lengths
        table.add_row([1, "img1", "model", "date", "date", "epoch", "/path1.jpg"], ["*"])
        table.add_row([100, "img2", "model", "date", "date", "epoch", "/path2.jpg"], ["*"])

        result = str(table)

        # Check that the table renders properly
        self.assertIn("1", result)
        self.assertIn("100", result)

        # The result should contain the data in a tabular format
        self.assertIn("|", result)
        self.assertIn("-", result)

    def test_thumbnail_index_calculation(self):
        """Test that thumbnail index is correctly identified."""
        # Test search result table
        search_table = self.ThumbnailTable(is_search_result=True)
        expected_thumbnail_idx = search_table.headers.index("Thumbnail")
        self.assertEqual(expected_thumbnail_idx, 6)

        # Test regular table
        regular_table = self.ThumbnailTable(is_search_result=False)
        expected_thumbnail_idx = regular_table.headers.index("Thumbnail")
        self.assertEqual(expected_thumbnail_idx, 1)

    def test_add_row_preserves_original_data(self):
        """Test that add_row doesn't modify the original row_data when copying."""
        table = self.ThumbnailTable(is_search_result=False)

        original_row_data = ["img123", "/path/to/image.jpg"]
        original_copy = original_row_data.copy()
        ascii_img = ["***"]

        table.add_row(original_row_data, ascii_img)

        # Original data should be unchanged when ascii_img is provided
        self.assertEqual(original_row_data, original_copy)

    def test_add_row_modifies_data_when_no_thumbnail(self):
        """Test that add_row modifies row_data when no thumbnail is provided."""
        table = self.ThumbnailTable(is_search_result=False)

        row_data = ["img123", "/path/to/image.jpg"]
        original_length = len(row_data)

        table.add_row(row_data, ascii_img=None)

        # row_data should be modified (placeholder inserted)
        self.assertGreater(len(row_data), original_length)
        self.assertIn(["Thumbnail not available"], row_data)

    def test_edge_cases_empty_data(self):
        """Test edge cases with empty or minimal data."""
        table = self.ThumbnailTable(is_search_result=False)

        # Test with minimal data
        table.add_row(["", ""], [""])

        result = str(table)
        self.assertNotEqual(result, "No rows to display")
        self.assertIn("|", result)

    def test_edge_cases_special_characters(self):
        """Test handling of special characters in data."""
        table = self.ThumbnailTable(is_search_result=False)

        # Test with special characters
        row_data = ["img|123", "/path/with spaces/image.jpg"]
        ascii_img = ["  |  ", " ||| ", "  |  "]

        table.add_row(row_data, ascii_img)

        result = str(table)

        # Should handle special characters gracefully
        self.assertIn("img|123", result)
        self.assertIn("/path/with spaces/image.jpg", result)
        self.assertIn(" ||| ", result)

    def test_thumbnail_line_rendering(self):
        """Test that thumbnail lines are rendered with proper indentation."""
        table = self.ThumbnailTable(is_search_result=False)

        row_data = ["img123", "/path/to/image.jpg"]
        ascii_img = ["line1", "line2", "line3"]

        table.add_row(row_data, ascii_img)

        result = str(table)
        lines = result.split('\n')

        # Find thumbnail lines (they should have specific indentation)
        thumbnail_lines = [line for line in lines if "line1" in line or "line2" in line or "line3" in line]

        self.assertEqual(len(thumbnail_lines), 3)

        # Each thumbnail line should have proper indentation and separator
        for line in thumbnail_lines:
            self.assertIn("|", line)


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)