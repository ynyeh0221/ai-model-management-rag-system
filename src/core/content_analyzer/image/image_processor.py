import datetime
import os

from PIL import Image, ExifTags
from git import Repo


class ImageProcessor:
    def __init__(self, schema_validator=None, image_analyzer=None):
        self.schema_validator = schema_validator
        self.image_analyzer = image_analyzer  # Optional component for image content analysis

    def process_image(self, image_path, metadata=None):
        """Process an image and extract metadata."""
        try:
            with Image.open(image_path) as img:
                image_info = self._build_basic_info(img, image_path)
                dates = self._build_dates(image_path, image_info["exif"])
                image_info["dates"] = dates

                image_info["image_content"] = self._init_content_struct()
                if self.image_analyzer:
                    self._populate_content(image_info, img)

                image_info["epoch"] = self._extract_epoch_from_filename(image_path)
                if metadata:
                    image_info.update(metadata)
                if "id" not in image_info:
                    image_info["id"] = self._generate_default_id(image_path)

                if self.schema_validator:
                    self.schema_validator.validate(image_info, "generated_image_schema")

                return {"metadata": image_info}

        except Exception as e:
            raise ValueError(f"Error processing image {image_path}: {e}")

    def _build_basic_info(self, img, image_path):
        """Extract basic properties (format, mode, size, EXIF)."""
        width, height = img.size
        return {
            "format": img.format,
            "mode": img.mode,
            "size": [width, height],
            "image_path": image_path,
            "thumbnail_path": None,
            "exif": self.extract_exif_data(image_path),
        }

    def _build_dates(self, image_path, exif_data):
        """
        Extract creation, last modified, and date taken,
        and derive month/year fields.
        """
        creation = self._get_creation_date(image_path)
        modified = self._get_last_modified_date(image_path)
        dates = {
            "creation_date": creation,
            "created_at": creation,
            "last_modified_date": modified,
            "date_taken": None,
        }

        # Add month/year for creation
        self._add_month_year(dates, "creation", creation)
        # Add month/year for last modified
        self._add_month_year(dates, "last_modified", modified)

        # Try to pull date_taken from EXIF DateTimeOriginal
        dto = exif_data.get("DateTimeOriginal")
        if dto:
            try:
                parsed = datetime.datetime.strptime(dto, "%Y:%m:%d %H:%M:%S")
                dates["date_taken"] = parsed.isoformat()
            except Exception:
                pass

        return dates

    def _add_month_year(self, dates_dict, prefix, iso_str):
        """Given a date string in ISO format, add prefix_month and prefix_year."""
        if not iso_str:
            return
        try:
            dt = datetime.datetime.fromisoformat(iso_str)
            dates_dict[f"{prefix}_month"] = dt.strftime("%m")
            dates_dict[f"{prefix}_year"] = dt.strftime("%Y")
            # For backward compatibility: mirror 'creation' into 'created'
            if prefix == "creation":
                dates_dict["created_month"] = dates_dict["creation_month"]
                dates_dict["created_year"] = dates_dict["creation_year"]
        except ValueError:
            pass

    def _init_content_struct(self):
        """Create the empty image_content structure."""
        return {
            "subject_type": None,
            "subject_details": {},
            "scene_type": None,
            "style": None,
            "tags": [],
            "colors": [],
            "objects": [],
        }

    def _populate_content(self, image_info, img):
        """
        Run the image analyzer and insert description or other fields
        into image_info and image_content.
        """
        analysis = self.image_analyzer.analyze_image(img)
        if not analysis:
            return

        for key, value in analysis.items():
            if key == "description":
                image_info["description"] = value
            elif key in image_info["image_content"]:
                image_info["image_content"][key] = value

    def _extract_epoch_from_filename(self, image_path):
        """
        Look for patterns like: <name>_epoch_<number>.* or <name>_e<number>.*
        Return integer epoch or None.
        """
        filename = os.path.splitext(os.path.basename(image_path))[0]
        parts = filename.split("_")
        for i, part in enumerate(parts):
            lower = part.lower()
            if lower == "epoch" and i + 1 < len(parts):
                try:
                    return int(parts[i + 1])
                except (ValueError, IndexError):
                    return None
            if lower.startswith("e") and lower[1:].isdigit():
                try:
                    return int(lower[1:])
                except ValueError:
                    return None
        return None

    def _generate_default_id(self, image_path):
        """Create a fallback ID based on filename (without extension)."""
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        return f"image_{base_name}"

    def extract_exif_data(self, image_path):
        """
        Return a dictionary of EXIF tags (including 'DateTimeOriginal' if available).
        """
        exif_data_named = {}
        try:
            with Image.open(image_path) as img:
                raw_exif = img._getexif()
                if raw_exif:
                    for tag_id, value in raw_exif.items():
                        tag = ExifTags.TAGS.get(tag_id, tag_id)
                        exif_data_named[tag] = value
        except Exception:
            pass
        return exif_data_named

    def _get_creation_date(self, image_path):
        """
        Return creation date as ISO-formatted string, or None on failure.
        """
        try:
            ts = os.path.getctime(image_path)
            return datetime.datetime.fromtimestamp(ts).isoformat()
        except Exception:
            return None

    def _get_last_modified_date(self, image_path):
        """
        Return last modified date as ISO-formatted string, or None on failure.
        """
        try:
            ts = os.path.getmtime(image_path)
            return datetime.datetime.fromtimestamp(ts).isoformat()
        except Exception:
            return None

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