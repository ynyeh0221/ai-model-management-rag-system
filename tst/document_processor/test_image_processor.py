import unittest
import os
import tempfile
from unittest.mock import patch, MagicMock
from PIL import Image, ExifTags

# Import the class to test
from document_processor.image_processor import ImageProcessor


class TestImageProcessor(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.processor = ImageProcessor()

        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()

        # Create a test image
        self.test_image_path = os.path.join(self.temp_dir.name, "test_image.jpg")
        self.create_test_image(self.test_image_path)

        # Path for thumbnails
        self.thumbnail_path = os.path.join(self.temp_dir.name, "thumbnail.jpg")

    def tearDown(self):
        """Clean up after each test method."""
        self.temp_dir.cleanup()

    def create_test_image(self, path, size=(100, 100)):
        """Helper method to create a test image."""
        img = Image.new('RGB', size, color='white')
        img.save(path)
        return path

    def test_process_image_basic(self):
        """Test basic image processing functionality."""
        result = self.processor.process_image(self.test_image_path)

        # Check that all expected keys are present in the result
        self.assertIn("format", result)
        self.assertIn("mode", result)
        self.assertIn("size", result)
        self.assertIn("metadata", result)
        self.assertIn("exif", result)
        self.assertIn("safety", result)

        # Check basic properties of the test image
        self.assertEqual(result["size"], (100, 100))
        self.assertEqual(result["mode"], "RGB")
        self.assertEqual(result["format"], "JPEG")

    def test_process_image_with_metadata(self):
        """Test processing an image with additional metadata."""
        metadata = {"source": "test", "description": "Test image for unit tests"}
        result = self.processor.process_image(self.test_image_path, metadata)

        # Check that metadata is incorporated
        self.assertEqual(result["metadata"]["source"], "test")
        self.assertEqual(result["metadata"]["description"], "Test image for unit tests")

    def test_process_image_error_handling(self):
        """Test error handling for invalid image paths."""
        with self.assertRaises(ValueError):
            self.processor.process_image("nonexistent_image.jpg")

    def test_generate_thumbnail_from_path(self):
        """Test generating a thumbnail from an image path."""
        thumbnail_path = self.processor.generate_thumbnail(
            self.test_image_path,
            self.thumbnail_path,
            (50, 50)
        )

        # Check that the thumbnail was created
        self.assertTrue(os.path.exists(thumbnail_path))

        # Check the thumbnail size
        with Image.open(thumbnail_path) as img:
            self.assertLessEqual(img.width, 50)
            self.assertLessEqual(img.height, 50)

    def test_generate_thumbnail_from_image(self):
        """Test generating a thumbnail from a PIL Image object."""
        with Image.open(self.test_image_path) as img:
            thumbnail_path = self.processor.generate_thumbnail(
                img,
                self.thumbnail_path,
                (50, 50)
            )

        # Check that the thumbnail was created
        self.assertTrue(os.path.exists(thumbnail_path))

        # Check the thumbnail size
        with Image.open(thumbnail_path) as img:
            self.assertLessEqual(img.width, 50)
            self.assertLessEqual(img.height, 50)

    def test_generate_thumbnail_error_handling(self):
        """Test error handling for invalid inputs to generate_thumbnail."""
        # Test with invalid image path
        with self.assertRaises(ValueError):
            self.processor.generate_thumbnail("nonexistent_image.jpg", self.thumbnail_path)

        # Test with invalid image object
        with self.assertRaises(ValueError):
            self.processor.generate_thumbnail(123, self.thumbnail_path)  # Not a path or Image

    def test_assess_safety(self):
        """Test the safety assessment functionality."""
        safety = self.processor.assess_safety(self.test_image_path)

        # Check that safety scores are present and within expected ranges
        self.assertIn("nsfw_score", safety)
        self.assertIn("violence_score", safety)
        self.assertGreaterEqual(safety["nsfw_score"], 0)
        self.assertLessEqual(safety["nsfw_score"], 1)
        self.assertGreaterEqual(safety["violence_score"], 0)
        self.assertLessEqual(safety["violence_score"], 1)

    def test_extract_exif_data(self):
        """Test EXIF data extraction."""
        # Normal case with no EXIF data
        exif_data = self.processor.extract_exif_data(self.test_image_path)
        self.assertIsInstance(exif_data, dict)

        # Test with mocked EXIF data
        with patch('PIL.Image.open') as mock_open:
            mock_img = MagicMock()
            mock_exif = {
                34665: "Test EXIF data",  # ExifOffset tag
                274: "Orientation"  # Orientation tag
            }
            mock_img._getexif.return_value = mock_exif
            mock_open.return_value.__enter__.return_value = mock_img

            exif_data = self.processor.extract_exif_data("dummy_path.jpg")

            # Check that the EXIF data was extracted and decoded
            # The tag 274 is mapped to 'Orientation' in ExifTags.TAGS
            self.assertIn("Orientation", exif_data)
            # The tag 34665 is mapped to 'ExifOffset' in ExifTags.TAGS
            self.assertIn("ExifOffset", exif_data)
            self.assertEqual(exif_data["ExifOffset"], "Test EXIF data")

    def test_extract_exif_data_error_handling(self):
        """Test error handling during EXIF extraction."""
        # Test with exception during _getexif
        with patch('PIL.Image.open') as mock_open:
            mock_img = MagicMock()
            mock_img._getexif.side_effect = Exception("EXIF error")
            mock_open.return_value.__enter__.return_value = mock_img

            # Should return empty dict without raising an exception
            exif_data = self.processor.extract_exif_data("dummy_path.jpg")
            self.assertEqual(exif_data, {})

    def test_schema_validation(self):
        """Test schema validation if a validator is provided."""
        # Create a mock schema validator
        mock_validator = MagicMock()

        # Create processor with validator
        processor_with_validator = ImageProcessor(schema_validator=mock_validator)

        # Process an image
        processor_with_validator.process_image(self.test_image_path)

        # Check that validate was called with the right schema name
        mock_validator.validate.assert_called_once()
        args, kwargs = mock_validator.validate.call_args
        self.assertEqual(args[1], "generated_image_schema")


if __name__ == '__main__':
    unittest.main()