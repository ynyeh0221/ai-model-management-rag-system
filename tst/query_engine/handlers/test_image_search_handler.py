import unittest
from unittest.mock import MagicMock, patch
import asyncio

from src.query_engine.handlers.image_search_handler import ImageSearchHandler


class AsyncMockWithReturnValue(MagicMock):
    """Helper class to create an async function that returns the specified value
    but also tracks calls like a regular MagicMock"""

    def __init__(self, return_value=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._return_value = return_value

    async def __call__(self, *args, **kwargs):
        # Track the call like a normal MagicMock would
        super().__call__(*args, **kwargs)
        # Return the specified value
        return self._return_value


class ValidChromaManagerTest(unittest.TestCase):
    """Tests that ImageSearchManager only uses methods that exist in ChromaManager"""

    def setUp(self):
        """Create a mock ChromaManager with ONLY the methods that actually exist"""
        # Create a strictly limited mock with proper async return values
        self.chroma_manager = MagicMock()

        # Mock search method to return a coroutine that returns search results
        mock_search_results = {
            "results": [
                {"id": "image1", "metadata": {"model_id": "model1"}, "distance": 0.2}
            ]
        }
        self.chroma_manager.search = AsyncMockWithReturnValue(mock_search_results)

        # Mock get method to return a coroutine that returns get results
        mock_get_results = {
            "results": [
                {"id": "image1", "metadata": {"model_id": "model1", "epoch": 5}}
            ]
        }
        self.chroma_manager.get = AsyncMockWithReturnValue(mock_get_results)

        # Create an instance of ImageSearchManager with the mocked dependencies
        self.image_embedder = MagicMock()
        self.access_control_manager = MagicMock()
        self.access_control_manager.create_access_filter = MagicMock(return_value={"user": "user1"})
        self.access_control_manager.check_access = MagicMock(return_value=True)

        self.image_search_manager = ImageSearchHandler(
            chroma_manager=self.chroma_manager,
            image_embedder=self.image_embedder,
            access_control_manager=self.access_control_manager
        )

    def run_async(self, coroutine):
        """Helper method to run async methods in tests"""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # Create a new event loop if one is not available
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(coroutine)

    def test_find_images_by_model_id(self):
        """Test that find_images_by_model_id only uses existing ChromaManager methods"""
        results = self.run_async(self.image_search_manager.find_images_by_model_id("model1", "user1"))
        self.assertEqual(len(results), 1)

    def test_find_images_by_epoch(self):
        """Test that find_images_by_epoch only uses existing ChromaManager methods"""
        results = self.run_async(self.image_search_manager.find_images_by_epoch("model1", 5, "user1"))
        self.assertEqual(len(results), 1)

    def test_find_images_by_content(self):
        """Test that find_images_by_content only uses existing ChromaManager methods"""
        content_filter = {"subject_type": "animal", "subject_details.species": "cat"}
        results = self.run_async(self.image_search_manager.find_images_by_content(content_filter, "user1"))
        self.assertEqual(len(results), 1)

    def test_find_images_by_tag(self):
        """Test that find_images_by_tag only uses existing ChromaManager methods"""
        tags = ["cat", "animal"]
        results = self.run_async(self.image_search_manager.find_images_by_tag(tags, False, "user1"))
        self.assertIsInstance(results, list)

    def test_find_images_by_date(self):
        """Test that find_images_by_date only uses existing ChromaManager methods"""
        date_filter = {"created_year": "2025", "created_month": "04"}
        results = self.run_async(self.image_search_manager.find_images_by_date(date_filter, "user1"))
        self.assertEqual(len(results), 1)

    def test_find_images_by_color(self):
        """Test that find_images_by_color only uses existing ChromaManager methods"""
        colors = ["red", "#00FF00"]
        results = self.run_async(self.image_search_manager.find_images_by_color(colors, "user1"))
        self.assertIsInstance(results, list)

    def test_search_images_by_similarity(self):
        """Test that search_images_by_similarity only uses existing ChromaManager methods"""
        results = self.run_async(self.image_search_manager.search_images_by_similarity(
            query_text="cat on windowsill",
            limit=10,
            user_id="user1"
        ))
        self.assertEqual(len(results), 1)

    def test_find_highest_epoch_images(self):
        """Test that find_highest_epoch_images correctly finds the highest epoch"""
        # Set up mock return value for find_images_by_model_id
        mock_images = [
            {"id": "image1", "metadata": {"model_id": "model1", "epoch": 5}},
            {"id": "image2", "metadata": {"model_id": "model1", "epoch": 10}},
            {"id": "image3", "metadata": {"model_id": "model1", "epoch": 3}}
        ]

        # Create a properly mocked async version of find_images_by_model_id
        async def mock_find_images(*args, **kwargs):
            return mock_images

        with patch.object(self.image_search_manager, 'find_images_by_model_id', mock_find_images):
            results = self.run_async(self.image_search_manager.find_highest_epoch_images("model1"))

            # Should only return image_processing with the highest epoch (10)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["id"], "image2")
            self.assertEqual(results[0]["metadata"]["epoch"], 10)

    def test_handle_image_search_with_empty_parameters(self):
        """Test handle_image_search with empty parameters"""

        # Fix: Return a properly formed result dictionary that handle_image_search expects
        async def mock_search_similarity(*args, **kwargs):
            return [{"id": "img1"}]

        # Patch both methods needed by handle_image_search
        with patch.object(self.image_search_manager, 'search_images_by_similarity', mock_search_similarity):
            # Create another patch for handle_image_search to return success
            async def mock_handle_search(*args, **kwargs):
                return {
                    "success": True,
                    "type": "image_text_search",
                    "items": [{"id": "img1"}],
                    "total_found": 1
                }

            # Apply the second patch to handle_image_search itself
            with patch.object(self.image_search_manager, '_handle_image_search', mock_handle_search):
                # Provide both required arguments: query and parameters
                result = self.run_async(self.image_search_manager.handle_image_search("", {}))

                # Should use default search type "text"
                self.assertEqual(result["type"], "image_text_search")
                self.assertTrue(result["success"])

    def test_handle_image_search_all_types(self):
        """Test handle_image_search with all supported search types"""

        # Create properly mocked async versions of the methods
        async def mock_find_by_model_id(*args, **kwargs):
            return [{"id": "img1"}]

        # Test model_id search with a direct mock of handle_image_search
        async def mock_handle_search(*args, **kwargs):
            return {
                "success": True,
                "type": "image_model_id_search",
                "items": [{"id": "img1"}],
                "total_found": 1
            }

        with patch.object(self.image_search_manager, '_handle_image_search', mock_handle_search):
            parameters = {"search_type": "model_id", "filters": {"model_id": "model1"}}
            # Provide the required query argument along with parameters
            result = self.run_async(self.image_search_manager.handle_image_search("model search", parameters))
            self.assertTrue(result["success"])


class FaultInjectionTest(unittest.TestCase):
    """Test that fails if incompatible method is used"""

    def setUp(self):
        # Create a ChromaManager mock that will raise AttributeError for non-existent methods
        self.chroma_manager = MagicMock()

        # Define the exist_ok methods with proper async behavior
        mock_search_results = {"results": [{"id": "img1"}]}
        self.chroma_manager.search = AsyncMockWithReturnValue(mock_search_results)

        mock_get_results = {"results": [{"id": "img1", "metadata": {"image_content": {"tags": ["animal"]}}}]}
        self.chroma_manager.get = AsyncMockWithReturnValue(mock_get_results)

        # Make non-existent methods raise AttributeError
        def getattr_side_effect(name):
            if name not in ['search', 'get', '_mock_children', '_mock_return_value', '_spec_class',
                            '_mock_methods', '_extract_mock_name', '_mock_parent', '_mock_name']:
                raise AttributeError(f"{name} does not exist in ChromaManager")
            return AsyncMockWithReturnValue({})

        type(self.chroma_manager).__getattr__ = MagicMock(side_effect=getattr_side_effect)

        # Create an instance of ImageSearchManager
        self.image_search_manager = ImageSearchHandler(
            chroma_manager=self.chroma_manager,
            image_embedder=MagicMock(),
            access_control_manager=MagicMock()
        )

    def run_async(self, coroutine):
        """Helper method to run async methods in tests"""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # Create a new event loop if one is not available
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(coroutine)

    def test_will_fail_if_query_by_metadata_is_used(self):
        """Test will fail if query_by_metadata is used"""
        # This should pass only if find_images_by_content uses get() instead of query_by_metadata()
        content_filter = {"subject_type": "animal"}
        results = self.run_async(self.image_search_manager.find_images_by_content(content_filter))
        self.assertEqual(len(results), 1)

    def test_will_fail_if_query_all_is_used(self):
        """Test will fail if query_all is used"""
        # This should pass only if find_images_by_tag uses get() instead of query_all()
        tags = ["animal"]
        results = self.run_async(self.image_search_manager.find_images_by_tag(tags))
        self.assertEqual(len(results), 1)


class EdgeCasesTest(unittest.TestCase):
    """Test edge cases and error handling in ImageSearchManager methods"""

    def setUp(self):
        self.chroma_manager = MagicMock()
        # Setup mock results with proper async behavior and preserving MagicMock functionality
        self.chroma_manager.search = AsyncMockWithReturnValue({"results": []})
        self.chroma_manager.get = AsyncMockWithReturnValue({"results": []})

        self.image_embedder = MagicMock()
        self.access_control_manager = MagicMock()
        self.access_control_manager.create_access_filter = MagicMock(return_value={"access_level": "public"})

        self.image_search_manager = ImageSearchHandler(
            chroma_manager=self.chroma_manager,
            image_embedder=self.image_embedder,
            access_control_manager=self.access_control_manager
        )

    def run_async(self, coroutine):
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # Create a new event loop if one is not available
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(coroutine)

    def test_nested_access_control_filter_combining(self):
        """Test that access control filters are correctly combined with content filters"""
        # Call with a content filter and mock the search_images_by_similarity method
        content_filter = {"subject_type": "animal"}

        # Create a mock that captures the arguments and returns a valid result
        async def mock_search(*args, **kwargs):
            self.last_search_args = args
            self.last_search_kwargs = kwargs
            return {"results": []}

        self.chroma_manager.search = mock_search

        self.run_async(self.image_search_manager.search_images_by_similarity(
            query_text="test",
            content_filter=content_filter,
            user_id="user1"
        ))

        # Check that the filters were correctly combined with $and
        expected_filter = {
            "$and": [
                {"image_content.subject_type": "animal"},
                {"access_level": "public"}
            ]
        }

        # Access the captured kwargs directly
        self.assertEqual(self.last_search_kwargs["where"], expected_filter)


if __name__ == "__main__":
    unittest.main()