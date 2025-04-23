import logging
import os
import tempfile
from typing import List, Tuple, Union

import numpy as np
import open_clip
import torch
from PIL import Image
from torchvision import transforms


class ImageEmbedder:
    """
    Generates vector embeddings for images using Open-CLIP models directly.
    Supports both global image embeddings and tiled embeddings for region-based search.
    Also provides text embedding functionality to enable text-to-image search.
    """

    def __init__(self, model_name="ViT-B-32", pretrained="laion2b_s34b_b79k", device=None, target_dim=384):
        """
        Initialize the ImageEmbedder with the specified model.

        Args:
            model_name (str): The CLIP model architecture to use (default: "ViT-B-32")
            pretrained (str): The pre-trained weights to use (default: "laion2b_s34b_b79k")
            device (str, optional): Device to run the model on ('cuda' or 'cpu').
                                   If None, will use CUDA if available.
            target_dim (int): Target dimension for embeddings (default: 384)
                             Used to ensure compatibility with Chroma collections
        """
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device or ("mps" if torch.backends.mps.is_available() else
                                 ("cuda" if torch.cuda.is_available() else "cpu"))
        self.target_dim = target_dim
        self.logger = logging.getLogger(__name__)
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the CLIP model for image embedding."""
        try:
            # Load the model and preprocessing transform
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                self.model_name,
                pretrained=self.pretrained
            )

            # Move model to the appropriate device
            self.model = self.model.to(self.device)

            # Set model to evaluation mode
            self.model.eval()

            # Get embedding dimension from the model
            self.embedding_dim = self.model.visual.output_dim

            # Load the tokenizer
            self.tokenizer = open_clip.get_tokenizer(self.model_name)

            self.logger.info(f"CLIP model initialized: {self.model_name}")
            self.logger.info(f"Original embedding dimension: {self.embedding_dim}")
            self.logger.info(f"Target embedding dimension: {self.target_dim}")
            self.logger.info(f"Running on device: {self.device}")

        except Exception as e:
            self.logger.error(f"Failed to initialize CLIP model: {e}")
            raise RuntimeError(f"Failed to initialize CLIP model: {e}")

    def _resize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """
        Resize an embedding vector to match the target dimension.

        Args:
            embedding (np.ndarray): Original embedding vector

        Returns:
            np.ndarray: Resized embedding vector with target dimension
        """
        original_dim = embedding.shape[0]

        if original_dim == self.target_dim:
            # No resizing needed
            return embedding

        elif original_dim > self.target_dim:
            # Truncate the embedding (keep the first target_dim elements)
            return embedding[:self.target_dim]

        else:
            # Pad the embedding with zeros
            resized = np.zeros(self.target_dim)
            resized[:original_dim] = embedding

            # Re-normalize after padding to maintain unit length
            norm = np.linalg.norm(resized)
            if norm > 0:  # Avoid division by zero
                resized = resized / norm

            return resized

    async def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a text query, making it compatible with image embeddings.
        Async method for compatibility with SearchDispatcher.

        Args:
            text (str): The text query to embed

        Returns:
            np.ndarray: L2-normalized embedding vector in the same space as image embeddings
        """
        return self._embed_text_sync(text)

    def _embed_text_sync(self, text: str) -> np.ndarray:
        """
        Synchronous implementation of text embedding generation.

        Args:
            text (str): The text query to embed

        Returns:
            np.ndarray: L2-normalized embedding vector in the same space as image embeddings,
                        resized to match target dimension if needed
        """
        try:
            # Tokenize the text
            tokens = self.tokenizer([text]).to(self.device)

            # Generate text embedding
            with torch.no_grad():
                text_features = self.model.encode_text(tokens)

                # L2 normalize the embedding
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                # Convert to numpy array and flatten
                embedding = text_features.cpu().numpy().flatten()

            # Resize the embedding to match the target dimension if needed
            return self._resize_embedding(embedding)
        except Exception as e:
            self.logger.error(f"Error generating text embedding: {e}")
            # Return a zero vector as fallback
            return np.zeros(self.target_dim)

    def _load_and_preprocess_image(self, image_path_or_array: Union[str, np.ndarray]) -> torch.Tensor:
        """
        Load and preprocess an image for embedding.

        Args:
            image_path_or_array: Path to the image file or numpy array

        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        try:
            if isinstance(image_path_or_array, np.ndarray):
                # Convert numpy array to PIL Image
                image = Image.fromarray(image_path_or_array.astype('uint8'))
            elif isinstance(image_path_or_array, str):
                if not os.path.exists(image_path_or_array):
                    raise FileNotFoundError(f"Image not found: {image_path_or_array}")
                # Open image and convert to RGB (in case it's grayscale or has alpha channel)
                image = Image.open(image_path_or_array).convert('RGB')
            else:
                raise ValueError(f"Unsupported image input type: {type(image_path_or_array)}")

            # Apply CLIP preprocessing
            preprocessed = self.preprocess(image).unsqueeze(0).to(self.device)
            return preprocessed

        except Exception as e:
            self.logger.error(f"Error preprocessing image: {e}")
            raise ValueError(f"Error preprocessing image: {e}")

    def embed_image(self, image_path_or_array: Union[str, np.ndarray]) -> np.ndarray:
        """
        Generate embedding for a single image.

        Args:
            image_path_or_array: Path to the image file or numpy array

        Returns:
            np.ndarray: L2-normalized embedding vector, resized to match target dimension
        """
        try:
            # Load and preprocess the image
            preprocessed = self._load_and_preprocess_image(image_path_or_array)

            # Generate embedding
            with torch.no_grad():
                image_features = self.model.encode_image(preprocessed)

                # L2 normalize the embedding
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                # Convert to numpy array and flatten
                embedding = image_features.cpu().numpy().flatten()

            # Resize the embedding to match the target dimension
            return self._resize_embedding(embedding)
        except Exception as e:
            self.logger.error(f"Error embedding image {image_path_or_array}: {e}")
            # Return a zero vector as fallback
            return np.zeros(self.target_dim)

    def embed_batch(self, image_paths: List[str]) -> np.ndarray:
        """
        Generate embeddings for a batch of images.

        Args:
            image_paths (List[str]): List of paths to image files

        Returns:
            np.ndarray: Array of L2-normalized embedding vectors, shape (n_images, target_dim)
        """
        if not image_paths:
            return np.array([])

        # Preprocess all images
        batch = []
        valid_paths = []

        for path in image_paths:
            try:
                preprocessed = self._load_and_preprocess_image(path)
                batch.append(preprocessed)
                valid_paths.append(path)
            except Exception as e:
                self.logger.warning(f"Warning: Skipping image {path}: {e}")

        if not batch:
            return np.array([])

        # Concatenate all preprocessed images
        batch_tensor = torch.cat(batch, dim=0)

        # Generate embeddings for the batch
        with torch.no_grad():
            image_features = self.model.encode_image(batch_tensor)

            # L2 normalize the embeddings
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Convert to numpy array
            embeddings = image_features.cpu().numpy()

        # Resize each embedding to match the target dimension
        resized_embeddings = np.zeros((embeddings.shape[0], self.target_dim))
        for i, embedding in enumerate(embeddings):
            resized_embeddings[i] = self._resize_embedding(embedding)

        return resized_embeddings

    async def generate_embedding(self, image_data: bytes) -> np.ndarray:
        """
        Generate embedding for an image provided as binary data.
        Async method for compatibility with SearchDispatcher.

        Args:
            image_data (bytes): Binary image data

        Returns:
            np.ndarray: L2-normalized embedding vector, resized to match target dimension
        """
        try:
            # Create a temporary file to save the image
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                temp_file.write(image_data)
                temp_path = temp_file.name

            try:
                # Process the image
                return self.embed_image(temp_path)
            finally:
                # Clean up the temporary file
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        except Exception as e:
            self.logger.error(f"Error generating embedding from image data: {e}")
            # Return a zero vector as fallback
            return np.zeros(self.target_dim)

    def embed_image_tiled(self, image_path: str, tile_config: dict = None) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Generate tiled embeddings for an image.

        Args:
            image_path (str): Path to the image file
            tile_config (dict, optional): Configuration for tiling. If None, default values will be used.

        Returns:
            Tuple[np.ndarray, Tuple[int, int]]:
                - Array of embeddings for each tile
                - Tile grid dimensions
        """
        # Set default tile config if not provided
        if tile_config is None:
            tile_config = {
                "tile_size": 224,
                "overlap": 32
            }

        # Extract tile parameters
        tile_size = tile_config.get("tile_size", 224)
        overlap = tile_config.get("overlap", 32)

        # Call the embed_tiles method with the appropriate parameters
        return self.embed_tiles(image_path, tile_size, overlap)

    def embed_tiles(self, image_path: str, tile_size: int = 224, overlap: int = 32) -> Tuple[
        np.ndarray, Tuple[int, int]]:
        """
        Generate embeddings for tiles of an image for region-based search.

        Args:
            image_path (str): Path to the image file
            tile_size (int): Size of the tiles in pixels (default: 224)
            overlap (int): Overlap between adjacent tiles in pixels (default: 32)

        Returns:
            Tuple[np.ndarray, Tuple[int, int]]:
                - Array of embeddings for each tile, shape (n_tiles, embedding_dim)
                - Tuple of (n_tiles_width, n_tiles_height) indicating the grid shape
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Open the image
        image = Image.open(image_path).convert('RGB')
        width, height = image.size

        # Calculate the effective stride (tile_size - overlap)
        stride = tile_size - overlap

        # Calculate number of tiles in each dimension
        n_tiles_width = max(1, (width - tile_size) // stride + 1)
        n_tiles_height = max(1, (height - tile_size) // stride + 1)

        # Prepare transformation for tiles
        # Note: We don't use self.preprocess here because we're manually
        # creating tiles and need consistent processing after that
        tile_transform = transforms.Compose([
            transforms.Resize(224, interpolation=Image.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)
            )
        ])

        # Extract and process tiles
        tiles = []
        for y in range(n_tiles_height):
            for x in range(n_tiles_width):
                # Calculate tile coordinates
                left = x * stride
                upper = y * stride
                right = min(left + tile_size, width)
                lower = min(upper + tile_size, height)

                # Extract tile
                tile = image.crop((left, upper, right, lower))

                # Process tile
                processed_tile = tile_transform(tile).unsqueeze(0).to(self.device)
                tiles.append(processed_tile)

        # Concatenate all tiles
        if not tiles:
            return np.array([]), (0, 0)

        tiles_tensor = torch.cat(tiles, dim=0)

        # Generate embeddings for all tiles
        with torch.no_grad():
            tile_features = self.model.encode_image(tiles_tensor)

            # L2 normalize the embeddings
            tile_features = tile_features / tile_features.norm(dim=-1, keepdim=True)

            # Convert to numpy array
            tile_embeddings = tile_features.cpu().numpy()

        return tile_embeddings, (n_tiles_width, n_tiles_height)

    def create_tile_config(self, image_path: str, tile_size: int = 224, overlap: int = 32) -> dict:
        """
        Create tile configuration metadata for an image.

        Args:
            image_path (str): Path to the image file
            tile_size (int): Size of the tiles in pixels
            overlap (int): Overlap between adjacent tiles in pixels

        Returns:
            dict: Tile configuration metadata
        """
        image = Image.open(image_path).convert('RGB')
        width, height = image.size

        # Calculate the effective stride
        stride = tile_size - overlap

        # Calculate number of tiles in each dimension
        n_tiles_width = max(1, (width - tile_size) // stride + 1)
        n_tiles_height = max(1, (height - tile_size) // stride + 1)

        return {
            "tile_size": tile_size,
            "overlap": overlap,
            "image_dimensions": {
                "width": width,
                "height": height
            },
            "grid_dimensions": {
                "width": n_tiles_width,
                "height": n_tiles_height
            },
            "total_tiles": n_tiles_width * n_tiles_height
        }

    def __call__(self, input: List[str]) -> List[List[float]]:
        """
        Make ImageEmbedder compatible with ChromaDB's embedding_function interface.

        Args:
            input: List of image paths to embed

        Returns:
            List of embedding vectors (List[List[float]])
        """
        if not input:
            return []

        try:
            embeddings = self.embed_batch(input)
            return embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings
        except Exception as e:
            self.logger.error(f"Error in ImageEmbedder __call__: {e}")
            return [[0.0] * self.target_dim for _ in input]

    def name(self) -> str:
        return f"ImageEmbedder::OpenCLIP::{self.model_name}"