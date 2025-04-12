import os
import datetime
from git import Repo
from PIL import Image, ExifTags


class ImageProcessor:
    def __init__(self, schema_validator=None):
        self.schema_validator = schema_validator

    def process_image(self, image_path, metadata=None):
        """Process an image and extract metadata."""
        try:
            with Image.open(image_path) as img:
                # Extract basic image properties
                image_info = {
                    "format": img.format,
                    "mode": img.mode,
                    "size": img.size,  # (width, height)
                    "metadata": {"image_path": image_path} if metadata is None else metadata,
                    "creation_date": self._get_creation_date(image_path),
                    "last_modified_date": self._get_last_modified_date(image_path)
                }

                # Extract EXIF data, if available
                exif_data = self.extract_exif_data(image_path)
                image_info["exif"] = exif_data

                # Try to get date taken from EXIF if available
                if "DateTimeOriginal" in exif_data:
                    try:
                        # Parse the EXIF date format (typically "YYYY:MM:DD HH:MM:SS")
                        date_str = exif_data["DateTimeOriginal"]
                        # Convert to ISO format for consistency with other dates
                        parsed_date = datetime.datetime.strptime(date_str, "%Y:%m:%d %H:%M:%S")
                        image_info["date_taken"] = parsed_date.isoformat()
                    except Exception:
                        # If date parsing fails, fall back to file creation date
                        pass

                # Perform safety assessment on the image content
                safety = self.assess_safety(image_path)
                image_info["safety"] = safety

                # Merge additional metadata if provided
                if metadata:
                    image_info.update(metadata)

                # Optionally validate the extracted metadata against a schema
                if self.schema_validator:
                    self.schema_validator.validate(image_info, "generated_image_schema")

                return image_info
        except Exception as e:
            raise ValueError(f"Error processing image {image_path}: {e}")

    def _get_creation_date(self, file_path):
        """Get file creation date, preferring git history if available."""
        try:
            repo = Repo(os.path.dirname(file_path), search_parent_directories=True)
            # Get the earliest commit for the file.
            commits = list(repo.iter_commits(paths=file_path, max_count=1, reverse=True))
            if commits:
                return datetime.datetime.fromtimestamp(commits[0].committed_date).isoformat()
        except Exception:
            pass

        # Fallback: use filesystem creation time
        try:
            stat = os.stat(file_path)
            return datetime.datetime.fromtimestamp(stat.st_ctime).isoformat()
        except Exception:
            return None

    def _get_last_modified_date(self, file_path):
        """Get file last modified date, preferring git history if available."""
        try:
            repo = Repo(os.path.dirname(file_path), search_parent_directories=True)
            commit = next(repo.iter_commits(paths=file_path, max_count=1))
            return datetime.datetime.fromtimestamp(commit.committed_date).isoformat()
        except Exception:
            pass

        # Fallback: use filesystem modification time
        try:
            stat = os.stat(file_path)
            return datetime.datetime.fromtimestamp(stat.st_mtime).isoformat()
        except Exception:
            return None

    def generate_thumbnail(self, image_input, thumbnail_path, size=(128, 128)):
        """
        Generate and save a thumbnail for an image.

        Args:
            image_input (str or PIL.Image.Image): Either a file path to the image or a PIL Image object.
            thumbnail_path (str): The destination file path where the thumbnail should be saved.
            size (tuple): Desired thumbnail size (width, height). Defaults to (128, 128).

        Returns:
            str: The thumbnail_path if successful.

        Raises:
            ValueError: If the image_input is not a valid type.
        """
        # Check if image_input is already a PIL image; otherwise, try to open it as a file.
        if isinstance(image_input, Image.Image):
            image = image_input
        elif isinstance(image_input, (str, os.PathLike)):
            try:
                image = Image.open(image_input)
            except Exception as e:
                raise ValueError(f"Cannot open image from path '{image_input}': {e}")
        else:
            raise ValueError("Invalid image_input type: must be a file path or PIL.Image.Image")

        # Create a copy of the image and generate the thumbnail using LANCZOS resampling.
        thumbnail = image.copy()
        thumbnail.thumbnail(size, Image.Resampling.LANCZOS)

        # Ensure that the directory for the thumbnail exists
        thumb_dir = os.path.dirname(thumbnail_path)
        if thumb_dir and not os.path.exists(thumb_dir):
            os.makedirs(thumb_dir, exist_ok=True)

        # Save the generated thumbnail to the provided thumbnail_path
        thumbnail.save(thumbnail_path)
        return thumbnail_path

    def assess_safety(self, image_path):
        """Perform safety assessment on image content."""
        # Placeholder implementation for safety assessment.
        # In a production system, this might involve a pre-trained NSFW detection model.
        safety_assessment = {
            "nsfw_score": 0.01,       # Simulated low probability of NSFW content
            "violence_score": 0.005   # Simulated low probability of violent content
        }
        return safety_assessment
    
    def extract_exif_data(self, image_path):
        """Extract EXIF data from the image."""
        exif_data = {}
        try:
            with Image.open(image_path) as img:
                info = img._getexif()
                if info:
                    for tag, value in info.items():
                        # Decode tag names using ExifTags.TAGS dictionary, defaulting to tag if unavailable
                        decoded_tag = ExifTags.TAGS.get(tag, tag)
                        exif_data[decoded_tag] = value
        except Exception:
            # If no EXIF data is found or an error occurs, return an empty dict.
            pass
        return exif_data
