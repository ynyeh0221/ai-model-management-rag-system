# src/vector_db_manager/image_embedder.py

import os
import torch
import numpy as np
from PIL import Image
import open_clip
from torchvision import transforms
from typing import List, Union, Tuple, Optional


class ImageEmbedder:
    """
    Generates vector embeddings for images using Open-CLIP models.
    Supports both global image embeddings and tiled embeddings for region-based search.
    """

    def __init__(self, model_name="ViT-B/32", pretrained="openai", device=None):
        """
        Initialize the ImageEmbedder with the specified model.
        
        Args:
            model_name (str): The CLIP model architecture to use (default: "ViT-B/32")
            pretrained (str): The pre-trained weights to use (default: "openai")
            device (str, optional): Device to run the model on ('cuda' or 'cpu'). 
                                    If None, will use CUDA if available.
        """
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
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
            
            # Get embedding dimension
            # For ViT-B/32, should be 512 as per the design doc
            self.embedding_dim = self.model.visual.output_dim
            
            print(f"CLIP model initialized: {self.model_name}")
            print(f"Embedding dimension: {self.embedding_dim}")
            print(f"Running on device: {self.device}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize CLIP model: {e}")
    
    def _load_and_preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Load and preprocess an image for embedding.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        try:
            # Open image and convert to RGB (in case it's grayscale or has alpha channel)
            image = Image.open(image_path).convert('RGB')
            
            # Apply CLIP preprocessing
            preprocessed = self.preprocess(image).unsqueeze(0).to(self.device)
            
            return preprocessed
        except Exception as e:
            raise ValueError(f"Error preprocessing image {image_path}: {e}")
    
    def embed_image(self, image_path: str) -> np.ndarray:
        """
        Generate embedding for a single image.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            np.ndarray: L2-normalized embedding vector (512-dimensional for ViT-B/32)
        """
        # Load and preprocess the image
        preprocessed = self._load_and_preprocess_image(image_path)
        
        # Generate embedding
        with torch.no_grad():
            image_features = self.model.encode_image(preprocessed)
            
            # L2 normalize the embedding as specified in the design doc
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Convert to numpy array and flatten
            embedding = image_features.cpu().numpy().flatten()
            
        return embedding
    
    def embed_batch(self, image_paths: List[str]) -> np.ndarray:
        """
        Generate embeddings for a batch of images.
        
        Args:
            image_paths (List[str]): List of paths to image files
            
        Returns:
            np.ndarray: Array of L2-normalized embedding vectors, shape (n_images, embedding_dim)
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
                print(f"Warning: Skipping image {path}: {e}")
        
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
            
        return embeddings
    
    def embed_tiles(self, image_path: str, tile_size: int = 224, overlap: int = 32) -> Tuple[np.ndarray, Tuple[int, int]]:
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
