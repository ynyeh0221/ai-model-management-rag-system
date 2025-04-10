import unittest
import numpy as np
import torch
from unittest.mock import patch, MagicMock
from PIL import Image
from io import BytesIO

# Import the ImageEmbedder class
from src.vector_db_manager.image_embedder import ImageEmbedder

# Test constants
TEST_MODEL_NAME = "ViT-L-14-336"
TEST_PRETRAINED = "openai"
TEST_DEVICE = "cpu"  # Using CPU for consistency in tests
TEST_TARGET_DIM = 384
TEST_EMBEDDING_DIM = 768  # Original embedding dimension from the model


class TestImageEmbedder(unittest.TestCase):

    def setUp(self):
        # Create mock embedding
        embedding = np.random.rand(TEST_EMBEDDING_DIM)
        self.mock_embedding = embedding / np.linalg.norm(embedding)

        # Create mock image with proper mocking setup
        self.mock_image = MagicMock()
        self.mock_image.convert.return_value = self.mock_image  # Return self for chaining

        # Create mock image bytes
        real_img = Image.new('RGB', (10, 10), color='red')
        img_io = BytesIO()
        real_img.save(img_io, format='PNG')
        self.mock_image_bytes = img_io.getvalue()

        # Set up proper mock clip model
        self.mock_clip_model = MagicMock()
        self.mock_clip_model.visual.output_dim = TEST_EMBEDDING_DIM
        self.mock_clip_model.encode_image.return_value = torch.ones((1, TEST_EMBEDDING_DIM), device=TEST_DEVICE)
        self.mock_clip_model.encode_text.return_value = torch.ones((1, TEST_EMBEDDING_DIM), device=TEST_DEVICE)

        # Set up preprocess mock
        def preprocess_mock(image):
            return torch.ones((3, 224, 224), device=TEST_DEVICE)

        self.mock_preprocess = MagicMock()
        self.mock_preprocess.side_effect = preprocess_mock

    @patch('open_clip.create_model_and_transforms')
    @patch('open_clip.tokenize')
    def test_initialization(self, mock_tokenize, mock_create_model):
        # Set up the mocks
        mock_create_model.return_value = (self.mock_clip_model, None, self.mock_preprocess)
        mock_tokenize.return_value = torch.ones((1, 77), dtype=torch.long, device=TEST_DEVICE)

        # Configure the model.to() mock chain
        to_mock = MagicMock()
        to_mock.visual.output_dim = TEST_EMBEDDING_DIM
        self.mock_clip_model.to.return_value = to_mock

        # Initialize the ImageEmbedder
        embedder = ImageEmbedder(
            model_name=TEST_MODEL_NAME,
            pretrained=TEST_PRETRAINED,
            device=TEST_DEVICE,
            target_dim=TEST_TARGET_DIM
        )

        # Check if the model was initialized correctly
        self.assertEqual(embedder.model_name, TEST_MODEL_NAME)
        self.assertEqual(embedder.pretrained, TEST_PRETRAINED)
        self.assertEqual(embedder.device, TEST_DEVICE)
        self.assertEqual(embedder.target_dim, TEST_TARGET_DIM)

        # For model attributes, we could either:
        # 1. Compare the specific value, which might be brittle if implementation changes
        # 2. Verify it has an appropriate type/value (more robust approach)

        # Alternative 1: Simply verify it has the expected value
        self.assertEqual(embedder.embedding_dim, TEST_EMBEDDING_DIM)

        # Alternative 2: More flexible approach - just make sure it's not a mock object
        # self.assertIsInstance(embedder.embedding_dim, int)

        # Check if create_model_and_transforms was called with the correct parameters
        mock_create_model.assert_called_once_with(TEST_MODEL_NAME, pretrained=TEST_PRETRAINED)

    @patch('open_clip.create_model_and_transforms')
    @patch('open_clip.tokenize')
    def test_resize_embedding(self, mock_tokenize, mock_create_model):
        # Set up the mocks
        mock_create_model.return_value = (self.mock_clip_model, None, self.mock_preprocess)
        mock_tokenize.return_value = torch.ones((1, 77), dtype=torch.long, device=TEST_DEVICE)

        # Initialize the ImageEmbedder
        embedder = ImageEmbedder(
            target_dim=TEST_TARGET_DIM,
            device=TEST_DEVICE
        )

        # Test case 1: Resizing a larger embedding
        large_embedding = np.ones(TEST_EMBEDDING_DIM) / np.sqrt(TEST_EMBEDDING_DIM)  # Normalized
        resized = embedder._resize_embedding(large_embedding)
        self.assertEqual(resized.shape, (TEST_TARGET_DIM,))
        self.assertTrue(np.allclose(resized[:TEST_TARGET_DIM], large_embedding[:TEST_TARGET_DIM]))

        # Test case 2: Resizing a smaller embedding
        small_size = 100
        small_embedding = np.ones(small_size) / np.sqrt(small_size)  # Normalized
        resized = embedder._resize_embedding(small_embedding)
        self.assertEqual(resized.shape, (TEST_TARGET_DIM,))
        self.assertTrue(np.allclose(resized[:small_size], small_embedding))

        # Test case 3: Resizing an embedding that is exactly target_dim
        exact_embedding = np.ones(TEST_TARGET_DIM) / np.sqrt(TEST_TARGET_DIM)  # Normalized
        resized = embedder._resize_embedding(exact_embedding)
        self.assertEqual(resized.shape, (TEST_TARGET_DIM,))
        self.assertTrue(np.allclose(resized, exact_embedding))

    @patch('open_clip.create_model_and_transforms')
    @patch('open_clip.tokenize')
    def test_generate_text_embedding(self, mock_tokenize, mock_create_model):
        # Set up the mocks
        mock_create_model.return_value = (self.mock_clip_model, None, self.mock_preprocess)
        mock_tokenize.return_value = torch.ones((1, 77), dtype=torch.long, device=TEST_DEVICE)

        # Initialize the ImageEmbedder
        embedder = ImageEmbedder(device=TEST_DEVICE)

        # Mock the _generate_text_embedding_sync method
        embedder._generate_text_embedding_sync = MagicMock()
        embedder._generate_text_embedding_sync.return_value = np.ones(TEST_TARGET_DIM)

        # Test the async generate_text_embedding method
        import asyncio
        result = asyncio.run(embedder.generate_text_embedding("test query"))

        # Check if _generate_text_embedding_sync was called with the correct parameters
        embedder._generate_text_embedding_sync.assert_called_once_with("test query")
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (TEST_TARGET_DIM,))

    @patch('open_clip.create_model_and_transforms')
    @patch('open_clip.tokenize')
    def test_generate_text_embedding_sync(self, mock_tokenize, mock_create_model):
        # Set up the mocks
        mock_create_model.return_value = (self.mock_clip_model, None, self.mock_preprocess)

        # Mock the tokenizer to return tensor on the right device
        token_tensor = torch.ones((1, 77), dtype=torch.long, device=TEST_DEVICE)
        mock_tokenize.return_value = token_tensor

        # Initialize the ImageEmbedder
        embedder = ImageEmbedder(device=TEST_DEVICE)

        # Mock the model's encode_text method
        embedder.model.encode_text = MagicMock()
        # Create a mock tensor with the original dimension on the correct device
        mock_text_features = torch.ones((1, TEST_EMBEDDING_DIM), device=TEST_DEVICE)
        # Mock the norm method to return a tensor on the correct device
        mock_text_features.norm = MagicMock(return_value=torch.ones((1, 1), device=TEST_DEVICE))
        embedder.model.encode_text.return_value = mock_text_features

        # Mock the _resize_embedding method
        embedder._resize_embedding = MagicMock()
        embedder._resize_embedding.return_value = np.ones(TEST_TARGET_DIM)

        # Test the _generate_text_embedding_sync method
        result = embedder._generate_text_embedding_sync("test query")

        # Check if the tokenizer was called with the correct parameters
        mock_tokenize.assert_called_once_with(["test query"])

        # Check if the model's encode_text method was called with the token tensor
        embedder.model.encode_text.assert_called_once()

        # Compare device-agnostic tensor data instead of comparing devices
        self.assertTrue(
            np.array_equal(
                embedder.model.encode_text.call_args[0][0].cpu().numpy(),
                token_tensor.cpu().numpy()
            )
        )

        # Check if _resize_embedding was called
        self.assertTrue(embedder._resize_embedding.called)

        # Check the result
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (TEST_TARGET_DIM,))

    @patch('open_clip.create_model_and_transforms')
    @patch('open_clip.tokenize')
    @patch('os.path.exists')
    @patch('PIL.Image.open')
    def test_load_and_preprocess_image(self, mock_image_open, mock_exists, mock_tokenize, mock_create_model):
        # Set up the mocks
        mock_create_model.return_value = (self.mock_clip_model, None, self.mock_preprocess)
        mock_tokenize.return_value = torch.ones((1, 77), dtype=torch.long, device=TEST_DEVICE)
        mock_exists.return_value = True

        # Create a proper mock for the image
        mock_image = MagicMock()
        mock_image.convert.return_value = mock_image
        mock_image_open.return_value = mock_image

        # Initialize the ImageEmbedder
        embedder = ImageEmbedder(device=TEST_DEVICE)

        # Create a proper mock for the preprocess method
        mock_preprocess_result = torch.ones((3, 224, 224), device=TEST_DEVICE)
        mock_preprocess = MagicMock(return_value=mock_preprocess_result)
        embedder.preprocess = mock_preprocess

        # Test the _load_and_preprocess_image method
        result = embedder._load_and_preprocess_image("test_image.jpg")

        # Check if os.path.exists was called with the correct parameters
        mock_exists.assert_called_once_with("test_image.jpg")

        # Check if PIL.Image.open was called with the correct parameters
        mock_image_open.assert_called_once_with("test_image.jpg")

        # Check if the image was converted to RGB
        mock_image.convert.assert_called_once_with('RGB')

        # Check if preprocess was called with the image
        mock_preprocess.assert_called_once_with(mock_image)

        # Test error case when file does not exist
        mock_exists.return_value = False
        with self.assertRaises(FileNotFoundError):
            embedder._load_and_preprocess_image("nonexistent_image.jpg")

    @patch('open_clip.create_model_and_transforms')
    @patch('open_clip.tokenize')
    @patch('os.path.exists')
    def test_embed_image(self, mock_exists, mock_tokenize, mock_create_model):
        # Set up the mocks
        mock_create_model.return_value = (self.mock_clip_model, None, self.mock_preprocess)
        mock_tokenize.return_value = torch.ones((1, 77), dtype=torch.long, device=TEST_DEVICE)
        mock_exists.return_value = True

        # Initialize the ImageEmbedder
        embedder = ImageEmbedder(device=TEST_DEVICE)

        # Mock the _load_and_preprocess_image method
        embedder._load_and_preprocess_image = MagicMock()
        mock_preprocessed = torch.ones((1, 3, 224, 224), device=TEST_DEVICE)
        embedder._load_and_preprocess_image.return_value = mock_preprocessed

        # Mock the model's encode_image method
        embedder.model.encode_image = MagicMock()
        # Create a mock tensor with the original dimension
        mock_image_features = torch.ones((1, TEST_EMBEDDING_DIM), device=TEST_DEVICE)
        # Mock the norm method to return the same tensor
        mock_image_features.norm = MagicMock(return_value=torch.ones((1, 1), device=TEST_DEVICE))
        embedder.model.encode_image.return_value = mock_image_features

        # Mock the _resize_embedding method
        embedder._resize_embedding = MagicMock()
        embedder._resize_embedding.return_value = np.ones(TEST_TARGET_DIM)

        # Test the embed_image method
        result = embedder.embed_image("test_image.jpg")

        # Check if _load_and_preprocess_image was called with the correct parameters
        embedder._load_and_preprocess_image.assert_called_once_with("test_image.jpg")

        # Check if the model's encode_image method was called with the preprocessed image
        embedder.model.encode_image.assert_called_once()

        # Check if _resize_embedding was called
        self.assertTrue(embedder._resize_embedding.called)

        # Check the result
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (TEST_TARGET_DIM,))

    @patch('open_clip.create_model_and_transforms')
    @patch('open_clip.tokenize')
    def test_embed_batch(self, mock_tokenize, mock_create_model):
        # Set up the mocks
        mock_create_model.return_value = (self.mock_clip_model, None, self.mock_preprocess)
        mock_tokenize.return_value = torch.ones((1, 77), dtype=torch.long, device=TEST_DEVICE)

        # Initialize the ImageEmbedder
        embedder = ImageEmbedder(device=TEST_DEVICE)

        # Mock the _load_and_preprocess_image method
        embedder._load_and_preprocess_image = MagicMock()
        mock_preprocessed = torch.ones((1, 3, 224, 224), device=TEST_DEVICE)
        embedder._load_and_preprocess_image.return_value = mock_preprocessed

        # Mock the model's encode_image method
        embedder.model.encode_image = MagicMock()
        # Create a mock tensor with the original dimension for a batch of 3 images
        mock_image_features = torch.ones((3, TEST_EMBEDDING_DIM), device=TEST_DEVICE)
        # Mock the norm method to return the same tensor
        mock_image_features.norm = MagicMock(return_value=torch.ones((3, 1), device=TEST_DEVICE))
        embedder.model.encode_image.return_value = mock_image_features

        # Mock the _resize_embedding method
        embedder._resize_embedding = MagicMock()
        embedder._resize_embedding.return_value = np.ones(TEST_TARGET_DIM)

        # Test the embed_batch method with a list of image paths
        result = embedder.embed_batch(["image1.jpg", "image2.jpg", "image3.jpg"])

        # Check if _load_and_preprocess_image was called for each image
        self.assertEqual(embedder._load_and_preprocess_image.call_count, 3)

        # Check if the model's encode_image method was called once with the batch
        embedder.model.encode_image.assert_called_once()

        # Check if _resize_embedding was called for each image
        self.assertEqual(embedder._resize_embedding.call_count, 3)

        # Check the result
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (3, TEST_TARGET_DIM))

        # Test with empty list
        result = embedder.embed_batch([])
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (0,))

        # Test error handling for invalid images
        embedder._load_and_preprocess_image.side_effect = Exception("Test error")
        result = embedder.embed_batch(["invalid1.jpg", "invalid2.jpg"])
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (0,))

    @patch('open_clip.create_model_and_transforms')
    @patch('open_clip.tokenize')
    def test_embed_image_tiled(self, mock_tokenize, mock_create_model):
        # Set up the mocks
        mock_create_model.return_value = (self.mock_clip_model, None, self.mock_preprocess)
        mock_tokenize.return_value = torch.ones((1, 77), dtype=torch.long, device=TEST_DEVICE)

        # Initialize the ImageEmbedder
        embedder = ImageEmbedder(device=TEST_DEVICE)

        # Mock the embed_tiles method
        embedder.embed_tiles = MagicMock()
        mock_tiles_result = (np.ones((4, TEST_EMBEDDING_DIM)), (2, 2))  # 2x2 grid of tiles
        embedder.embed_tiles.return_value = mock_tiles_result

        # Test the embed_image_tiled method with default config
        result = embedder.embed_image_tiled("test_image.jpg")

        # Check if embed_tiles was called with the correct parameters
        embedder.embed_tiles.assert_called_once_with("test_image.jpg", 224, 32)

        # Check the result
        self.assertIsInstance(result, tuple)
        self.assertIsInstance(result[0], np.ndarray)
        self.assertEqual(result[1], (2, 2))

        # Test with custom tile config
        embedder.embed_tiles.reset_mock()
        custom_config = {
            "tile_size": 112,
            "overlap": 16
        }
        result = embedder.embed_image_tiled("test_image.jpg", custom_config)

        # Check if embed_tiles was called with the correct parameters
        embedder.embed_tiles.assert_called_once_with("test_image.jpg", 112, 16)

    @patch('open_clip.create_model_and_transforms')
    @patch('open_clip.tokenize')
    def test_generate_embedding(self, mock_tokenize, mock_create_model):
        # Set up the mocks
        mock_create_model.return_value = (self.mock_clip_model, None, self.mock_preprocess)
        mock_tokenize.return_value = torch.ones((1, 77), dtype=torch.long, device=TEST_DEVICE)

        # Initialize the ImageEmbedder
        embedder = ImageEmbedder(device=TEST_DEVICE)

        # Mock the preprocess method
        mock_preprocess_result = torch.ones((3, 224, 224), device=TEST_DEVICE)
        embedder.preprocess = MagicMock(return_value=mock_preprocess_result)

        # Mock the model's encode_image method
        embedder.model.encode_image = MagicMock()
        # Create a mock tensor with the original dimension
        mock_image_features = torch.ones((1, TEST_EMBEDDING_DIM), device=TEST_DEVICE)
        # Mock the norm method to return the same tensor
        mock_image_features.norm = MagicMock(return_value=torch.ones((1, 1), device=TEST_DEVICE))
        embedder.model.encode_image.return_value = mock_image_features

        # Mock the _resize_embedding method
        embedder._resize_embedding = MagicMock()
        embedder._resize_embedding.return_value = np.ones(TEST_TARGET_DIM)

        # Mock PIL.Image.open
        with patch('PIL.Image.open') as mock_image_open:
            # Create a proper mock for the image
            mock_image = MagicMock()
            mock_image.convert.return_value = mock_image
            mock_image_open.return_value = mock_image

            # Test the generate_embedding method
            import asyncio
            result = asyncio.run(embedder.generate_embedding(self.mock_image_bytes))

            # Check if PIL.Image.open was called
            mock_image_open.assert_called_once()

            # Check if the image was converted to RGB
            mock_image.convert.assert_called_once_with('RGB')

            # Check if preprocess was called with the image
            embedder.preprocess.assert_called_once_with(mock_image)

            # Check if the model's encode_image method was called
            embedder.model.encode_image.assert_called_once()

            # Check if _resize_embedding was called
            embedder._resize_embedding.assert_called_once()

            # Check the result
            self.assertIsInstance(result, np.ndarray)
            self.assertEqual(result.shape, (TEST_TARGET_DIM,))

    @patch('open_clip.create_model_and_transforms')
    @patch('open_clip.tokenize')
    @patch('os.path.exists')
    @patch('PIL.Image.open')
    def test_embed_tiles(self, mock_image_open, mock_exists, mock_tokenize, mock_create_model):
        # Set up the mocks
        mock_create_model.return_value = (self.mock_clip_model, None, self.mock_preprocess)
        mock_tokenize.return_value = torch.ones((1, 77), dtype=torch.long, device=TEST_DEVICE)
        mock_exists.return_value = True

        # Create a proper mock for the image with size attribute
        mock_image = MagicMock()
        mock_image.size = (100, 100)  # Set size as attribute
        mock_image.convert.return_value = mock_image
        mock_image.crop.return_value = mock_image
        mock_image_open.return_value = mock_image

        # Initialize the ImageEmbedder
        embedder = ImageEmbedder(device=TEST_DEVICE)

        # Mock transforms.Compose and the resulting transform
        with patch('torchvision.transforms.Compose') as mock_compose:
            mock_transform = MagicMock()
            mock_transform.return_value = torch.ones((3, 224, 224), device=TEST_DEVICE)
            mock_compose.return_value = mock_transform

            # Mock the model's encode_image method
            embedder.model.encode_image = MagicMock()
            # Create a mock tensor with the original dimension for 1 tile
            mock_tile_features = torch.ones((1, TEST_EMBEDDING_DIM), device=TEST_DEVICE)
            # Mock the norm method to return the same tensor
            mock_tile_features.norm = MagicMock(return_value=torch.ones((1, 1), device=TEST_DEVICE))
            embedder.model.encode_image.return_value = mock_tile_features

            # Test the embed_tiles method
            result, grid_dims = embedder.embed_tiles("test_image.jpg")

            # Check if os.path.exists was called with the correct parameters
            mock_exists.assert_called_once_with("test_image.jpg")

            # Check if PIL.Image.open was called with the correct parameters
            mock_image_open.assert_called_once_with("test_image.jpg")

            # Check if the image was converted to RGB
            mock_image.convert.assert_called_once_with('RGB')

            # Check if image.crop was called for each tile
            # With a 100x100 image, tile_size=224, and overlap=32, we should have 1 tile
            self.assertEqual(mock_image.crop.call_count, 1)

            # Check if the model's encode_image method was called once for all tiles
            embedder.model.encode_image.assert_called_once()

            # Check the result
            self.assertIsInstance(result, np.ndarray)
            self.assertEqual(grid_dims, (1, 1))  # Should be a 1x1 grid for a small image

            # Test error case when file does not exist
            mock_exists.return_value = False
            with self.assertRaises(FileNotFoundError):
                embedder.embed_tiles("nonexistent_image.jpg")

    @patch('open_clip.create_model_and_transforms')
    @patch('open_clip.tokenize')
    @patch('os.path.exists')
    @patch('PIL.Image.open')
    def test_create_tile_config(self, mock_image_open, mock_exists, mock_tokenize, mock_create_model):
        # Set up the mocks
        mock_create_model.return_value = (self.mock_clip_model, None, self.mock_preprocess)
        mock_tokenize.return_value = torch.ones((1, 77), dtype=torch.long, device=TEST_DEVICE)
        mock_exists.return_value = True

        # Create a proper mock for the image with size attribute
        mock_image = MagicMock()
        mock_image.size = (500, 300)  # Set size as attribute
        mock_image.convert.return_value = mock_image
        mock_image_open.return_value = mock_image

        # Initialize the ImageEmbedder
        embedder = ImageEmbedder(device=TEST_DEVICE)

        # Test the create_tile_config method
        config = embedder.create_tile_config("test_image.jpg", tile_size=224, overlap=32)

        # Check if PIL.Image.open was called with the correct parameters
        mock_image_open.assert_called_once_with("test_image.jpg")

        # Check if the image was converted to RGB
        mock_image.convert.assert_called_once_with('RGB')

        # Check the config
        self.assertEqual(config["tile_size"], 224)
        self.assertEqual(config["overlap"], 32)
        self.assertEqual(config["image_dimensions"]["width"], 500)
        self.assertEqual(config["image_dimensions"]["height"], 300)

        # Calculate expected grid dimensions
        stride = 224 - 32  # tile_size - overlap
        expected_width = max(1, (500 - 224) // stride + 1)
        expected_height = max(1, (300 - 224) // stride + 1)

        self.assertEqual(config["grid_dimensions"]["width"], expected_width)
        self.assertEqual(config["grid_dimensions"]["height"], expected_height)
        self.assertEqual(config["total_tiles"], expected_width * expected_height)


if __name__ == "__main__":
    unittest.main()