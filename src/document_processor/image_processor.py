import os
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
                    "size": img.size  # (width, height)
                }
                
                # Extract EXIF data, if available
                exif_data = self.extract_exif_data(image_path)
                image_info["exif"] = exif_data

                # Perform safety assessment on the image content
                safety = self.assess_safety(image_path)
                image_info["safety"] = safety

                # Merge additional metadata if provided
                if metadata:
                    image_info.update(metadata)
                
                # Optionally validate the extracted metadata against a schema
                if self.schema_validator:
                    self.schema_validator.validate(image_info)
                
                return image_info
        except Exception as e:
            raise ValueError(f"Error processing image {image_path}: {e}")
    
    def generate_thumbnail(self, image_path, output_path=None, size=(128, 128)):
        """Generate a thumbnail for the image."""
        try:
            with Image.open(image_path) as img:
                # Create a copy for generating thumbnail, preserving original image
                thumbnail = img.copy()
                thumbnail.thumbnail(size)
                # Determine output path if not provided
                if not output_path:
                    base, ext = os.path.splitext(image_path)
                    output_path = f"{base}_thumbnail{ext}"
                thumbnail.save(output_path)
                return output_path
        except Exception as e:
            raise ValueError(f"Error generating thumbnail for {image_path}: {e}")
    
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

