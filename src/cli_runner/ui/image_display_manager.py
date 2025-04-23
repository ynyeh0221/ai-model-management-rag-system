from cli_runner.ui.display_utils import DisplayUtils
from cli_runner.ui.thumbnail_table import ThumbnailTable


class ImageDisplayManager:
    """Handles displaying image data in various formats."""

    def __init__(self):
        # Check if PIL is available for image processing
        try:
            from PIL import Image
            import numpy as np
            self.has_pil = True
        except ImportError:
            print("PIL/Pillow not installed. Thumbnails will not be displayed.")
            self.has_pil = False

    def display_images_with_thumbnails(self, images, is_search_result=False):
        """
        Display images in a formatted table with ASCII thumbnails.

        Args:
            images (list): List of image dictionaries to display.
            is_search_result (bool): Whether the images are from search results.
        """
        if not images:
            print("  No images available")
            return

        # Create and populate the table
        table = self._create_thumbnail_table(images, is_search_result)

        # Print the table
        print(table)

        # Print performance metrics if available
        self._print_performance_metrics(images)

    def _create_thumbnail_table(self, images, is_search_result):
        """Create a table for displaying images with thumbnails."""
        table = ThumbnailTable(is_search_result)

        # Ensure images is always treated as a list
        if isinstance(images, dict) and 'items' in images:
            image_list = images.get('items', [])
        else:
            image_list = images if isinstance(images, list) else []

        for i, image in enumerate(image_list, 1):
            row_data = self._prepare_image_row_data(image, i, is_search_result)

            # Generate ASCII thumbnail
            thumbnail_path = image.get('thumbnail_path', image.get('image_path', ''))
            ascii_img = self._generate_ascii_thumbnail(thumbnail_path)

            table.add_row(row_data, ascii_img)

        return table

    def _prepare_image_row_data(self, image, index, is_search_result):
        """Prepare row data for an image entry."""
        if is_search_result:
            # Handle search result format
            metadata = image.get('metadata', {})
            image_id = image.get('id', 'Unknown')
            model_id = metadata.get('source_model_id', metadata.get('model_id', 'Unknown'))
            prompt = metadata.get('prompt', 'No prompt')
            image_path = image.get('image_path', metadata.get('image_path', 'Not available'))
            row_data = [index, image_id, model_id, prompt, image_path]
        else:
            # Handle list format
            image_id = image.get('id', 'Unknown')
            prompt = image.get('prompt', 'No prompt')
            image_path = image.get('filepath', image.get('image_path', 'No path'))
            row_data = [image_id, prompt, image_path]

        # Truncate long values
        return [DisplayUtils.truncate_string(item) for item in row_data]

    def _generate_ascii_thumbnail(self, thumbnail_path):
        """Generate ASCII art thumbnail from an image file."""
        if not self.has_pil or not thumbnail_path or thumbnail_path in ['No path', 'Not available']:
            return ["Thumbnail not available"]

        try:
            import os
            from PIL import Image
            import numpy as np

            if not os.path.exists(thumbnail_path):
                return ["Image file not found"]

            # Open the image and resize it
            img = Image.open(thumbnail_path)
            img = img.resize((30, 15))
            img = img.convert('L')  # Convert to grayscale

            # Convert pixels to ASCII characters
            pixels = np.array(img)
            ascii_img = []
            for row in pixels:
                ascii_row = ''
                for pixel in row:
                    # Map pixel value to ASCII character
                    index = int(pixel * (len(DisplayUtils.ASCII_CHARS) - 1) / 255)
                    ascii_row += DisplayUtils.ASCII_CHARS[index]
                ascii_img.append(ascii_row)

            return ascii_img
        except Exception as e:
            return [f"Error loading image: {str(e)}"]

    def _print_performance_metrics(self, images):
        """Print performance metrics if available."""
        if isinstance(images, dict) and 'performance' in images:
            performance_data = images.get('performance', {})
            if performance_data:
                print("\nPerformance:")
                for metric, value in performance_data.items():
                    if isinstance(value, (int, float)):
                        print(f"  {metric}: {value:.2f} ms")

    def display_image_search_results(self, search_results):
        """
        Display image search results in a table format with ASCII thumbnails.

        Args:
            search_results (dict): Results from the image search.
        """
        if not search_results.get('success', False):
            print(f"Image search failed: {search_results.get('error', 'Unknown error')}")
            return

        items = search_results.get('items', [])
        if not items:
            print("No images found matching your search criteria.")
            return

        print(f"\nFound {len(items)} images:")
        self.display_images_with_thumbnails(search_results, is_search_result=True)