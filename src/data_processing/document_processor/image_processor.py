import os
import datetime
from git import Repo
from PIL import Image, ExifTags


class ImageProcessor:
    def __init__(self, schema_validator=None, image_analyzer=None):
        self.schema_validator = schema_validator
        self.image_analyzer = image_analyzer  # Optional component for image content analysis

    def process_image(self, image_path, metadata=None):
        """Process an image and extract metadata."""
        try:
            with Image.open(image_path) as img:
                # Extract basic image properties
                image_dimensions = img.size  # This is a tuple (width, height)
                # Convert tuple to list for proper JSON serialization
                image_size = list(image_dimensions)

                image_info = {
                    "format": img.format,
                    "mode": img.mode,
                    "size": image_size,  # Now this is a list [width, height]
                    "image_path": image_path,
                    "thumbnail_path": None,  # Will be set later if needed
                    "exif": self.extract_exif_data(image_path)
                }

                # Extract dates and format them properly
                creation_date = self._get_creation_date(image_path)
                last_modified_date = self._get_last_modified_date(image_path)

                # Parse dates into components
                dates = {
                    "creation_date": creation_date,
                    "created_at": creation_date,  # Duplicate field for compatibility
                    "last_modified_date": last_modified_date,
                    "date_taken": None
                }

                # Add month and year information
                if creation_date:
                    try:
                        dt = datetime.datetime.fromisoformat(creation_date)
                        dates["created_month"] = dt.strftime("%m")
                        dates["created_year"] = dt.strftime("%Y")
                        # Keep old field names for backward compatibility
                        dates["creation_month"] = dt.strftime("%m")
                        dates["creation_year"] = dt.strftime("%Y")
                    except ValueError:
                        pass

                if last_modified_date:
                    try:
                        dt = datetime.datetime.fromisoformat(last_modified_date)
                        dates["last_modified_month"] = dt.strftime("%m")
                        dates["last_modified_year"] = dt.strftime("%Y")
                    except ValueError:
                        pass

                # Try to get date taken from EXIF if available
                if "DateTimeOriginal" in image_info["exif"]:
                    try:
                        # Parse the EXIF date format (typically "YYYY:MM:DD HH:MM:SS")
                        date_str = image_info["exif"]["DateTimeOriginal"]
                        # Convert to ISO format for consistency with other dates
                        parsed_date = datetime.datetime.strptime(date_str, "%Y:%m:%d %H:%M:%S")
                        dates["date_taken"] = parsed_date.isoformat()
                    except Exception:
                        # If date parsing fails, fall back to file creation date
                        pass

                # Add extracted dates to image info
                image_info["dates"] = dates

                # Initialize image content information with the new structure
                image_info["image_content"] = {
                    "subject_type": None,
                    "subject_details": {},
                    "scene_type": None,
                    "style": None,
                    "tags": [],
                    "colors": [],
                    "objects": []
                }

                # Analyze image content if image analyzer is available
                if self.image_analyzer:
                    content_analysis = self.image_analyzer.analyze_image(img)
                    if content_analysis:
                        # Update image_content with analysis results
                        for key, value in content_analysis.items():
                            if key == "description":
                                image_info["description"] = value
                            elif key in image_info["image_content"]:
                                image_info["image_content"][key] = value

                # Extract epoch information from the filename or metadata
                # Format could be: model_name_epoch_X.png or similar
                filename = os.path.basename(image_path)
                try:
                    # Try to extract epoch from filename
                    # Common patterns include: something_epoch_42.png or something_e42.png
                    parts = os.path.splitext(filename)[0].split('_')
                    for i, part in enumerate(parts):
                        if part.lower() == "epoch" and i + 1 < len(parts):
                            image_info["epoch"] = int(parts[i + 1])
                            break
                        elif part.lower().startswith('e') and part[1:].isdigit():
                            image_info["epoch"] = int(part[1:])
                            break
                except (ValueError, IndexError):
                    # If we can't extract epoch from filename, set to null
                    image_info["epoch"] = None

                # Merge additional metadata if provided
                if metadata:
                    image_info.update(metadata)

                # Generate a default ID if not provided (needed by schema)
                if "id" not in image_info:
                    # Create an ID from the filename
                    base_name = os.path.splitext(os.path.basename(image_path))[0]
                    image_info["id"] = f"image_{base_name}"

                # Optionally validate the extracted metadata against a schema
                if self.schema_validator:
                    self.schema_validator.validate(image_info, "generated_image_schema")

                return {"metadata": image_info}
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