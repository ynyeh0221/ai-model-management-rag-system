import json
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.core.vector_db.chroma_manager import ChromaManager


# --- Enhanced Fake Classes for Testing ---

class DummyEmbeddingFunction:
    def __init__(self, model_name):
        self.model_name = model_name

    def __call__(self, texts):
        # For simplicity, for each input text returns a fixed-dimension vector
        return [[1.0] * 5 for _ in texts]

    async def embed_text(self, text):
        # For image embedder's embed_text method
        return [1.0] * 5


class DummyCollection:
    def __init__(self, name, embedding_function, metadata=None):
        self.name = name
        self.embedding_function = embedding_function
        self.metadata = metadata or {}
        self.docs = {}  # Store documents as a dict keyed by doc id

    def upsert(self, ids, documents=None, embeddings=None, metadatas=None, images=None):
        # Simulate adding documents with support for images
        for i, doc_id in enumerate(ids):
            self.docs[doc_id] = {
                "content": documents[i] if documents else "",
                "metadata": metadatas[i] if metadatas else {},
                "embedding": embeddings[i] if embeddings else None,
                "image": images[i] if images else None
            }

    def update(self, **update_args):
        ids = update_args.get("ids", [])
        for idx, doc_id in enumerate(ids):
            if doc_id in self.docs:
                if "documents" in update_args and update_args["documents"]:
                    self.docs[doc_id]["content"] = update_args["documents"][idx]
                if "metadatas" in update_args and update_args["metadatas"]:
                    self.docs[doc_id]["metadata"] = update_args["metadatas"][idx]
                if "embeddings" in update_args and update_args["embeddings"]:
                    self.docs[doc_id]["embedding"] = update_args["embeddings"][idx]

    def delete(self, ids):
        for doc_id in ids:
            self.docs.pop(doc_id, None)

    def get(self, ids=None, where=None, limit=None, offset=None, include=None):
        docs_list = list(self.docs.items())
        if ids is not None:
            docs_list = [(doc_id, self.docs[doc_id]) for doc_id in ids if doc_id in self.docs]

        # Apply offset and limit
        if offset:
            docs_list = docs_list[offset:]
        if limit:
            docs_list = docs_list[:limit]

        ids_list = [[doc_id for (doc_id, _) in docs_list]]
        documents_list = [[doc["content"] for (_, doc) in docs_list]]
        metadatas_list = [[doc["metadata"] for (_, doc) in docs_list]]
        embeddings_list = [[doc.get("embedding") for (_, doc) in docs_list]]

        return {
            "ids": ids_list,
            "documents": documents_list,
            "metadatas": metadatas_list,
            "embeddings": embeddings_list
        }

    def query(self, query_embeddings, n_results, include, where=None):
        result = self.get()
        result["distances"] = [[0.0 for _ in result["ids"][0]]] if result["ids"][0] else [[]]
        return result

    def count(self):
        return len(self.docs)


class FailingDummyCollection(DummyCollection):
    """A collection that fails operations for testing error handling"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.should_fail = False

    def upsert(self, *args, **kwargs):
        if self.should_fail:
            raise Exception("Simulated upsert failure")
        return super().upsert(*args, **kwargs)

    def get(self, *args, **kwargs):
        if self.should_fail:
            raise Exception("Simulated get failure")
        return super().get(*args, **kwargs)


class DummyPersistentClient:
    def __init__(self, path, settings):
        self.path = path
        self.settings = settings
        self.collections = {}
        self.should_fail_create = False

    def get_collection(self, name, embedding_function):
        if name in self.collections:
            return self.collections[name]
        else:
            raise ValueError("Collection does not exist")

    def create_collection(self, name, embedding_function, metadata):
        if self.should_fail_create:
            raise Exception("Simulated collection creation failure")
        collection = DummyCollection(name, embedding_function, metadata)
        self.collections[name] = collection
        return collection


# --- Extended Unit Test Class ---

class ExtendedTestChromaManager(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

        # Patch os.makedirs
        self.makedirs_patcher = patch("src.core.vector_db.chroma_manager.os.makedirs")
        self.mock_makedirs = self.makedirs_patcher.start()

        # Patch chromadb.PersistentClient
        self.client_patcher = patch(
            "src.core.vector_db.chroma_manager.chromadb.PersistentClient",
            side_effect=lambda path, settings: DummyPersistentClient(path, settings)
        )
        self.mock_client = self.client_patcher.start()

        self.text_embedder = DummyEmbeddingFunction("dummy-text")
        self.image_embedder = DummyEmbeddingFunction("dummy-image")

        self.manager = ChromaManager(
            text_embedder=self.text_embedder,
            image_embedder=self.image_embedder,
            persist_directory=self.temp_dir
        )

        # Override _run_in_executor for synchronous execution
        async def immediate(func, *args, **kwargs):
            return func(*args, **kwargs)

        self.manager._run_in_executor = immediate

    def tearDown(self):
        self.makedirs_patcher.stop()
        self.client_patcher.stop()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # Test initialization error handling
    async def test_initialize_client_error(self):
        """Test error handling during client initialization"""
        with patch("src.core.vector_db.chroma_manager.chromadb.PersistentClient") as mock_client:
            mock_client.side_effect = Exception("Database connection failed")

            with self.assertRaises(Exception):
                ChromaManager(
                    text_embedder=self.text_embedder,
                    image_embedder=self.image_embedder,
                    persist_directory="/invalid/path"
                )

    async def test_initialize_default_collections_error(self):
        """Test error handling during default collections initialization"""
        # Create manager with failing client
        failing_client = DummyPersistentClient(self.temp_dir, {})
        failing_client.should_fail_create = True

        with patch("src.core.vector_db.chroma_manager.chromadb.PersistentClient", return_value=failing_client):
            with self.assertRaises(Exception):
                ChromaManager(
                    text_embedder=self.text_embedder,
                    image_embedder=self.image_embedder,
                    persist_directory=self.temp_dir
                )

    async def test_get_collection_image_type(self):
        """Test getting collection for image types"""
        collection = self.manager.get_collection("generated_images")
        self.assertIsInstance(collection, DummyCollection)
        self.assertEqual(collection.name, "generated_images")

    async def test_get_collection_error_handling(self):
        """Test error handling in get_collection"""
        with patch.object(self.manager, '_get_or_create_collection') as mock_get:
            mock_get.side_effect = Exception("Collection error")

            with self.assertRaises(Exception):
                self.manager.get_collection("failing_collection")

    # Test add_document edge cases
    async def test_add_document_empty_document(self):
        """Test adding empty document"""
        with self.assertRaises(ValueError):
            await self.manager.add_document({})

    async def test_add_document_with_numpy_embedding(self):
        """Test adding document with numpy array embedding"""
        document = {
            "id": "doc_numpy",
            "content": "Test content",
            "metadata": {"test": "value"}
        }
        embedding = np.array([1.0, 2.0, 3.0])

        doc_id = await self.manager.add_document(
            document,
            embed_content=embedding,
            collection_name="model_script_processing"
        )
        self.assertEqual(doc_id, "doc_numpy")

    async def test_add_document_no_embedding(self):
        """Test adding document without embedding"""
        document = {
            "id": "doc_no_embed",
            "content": "Test content",
            "metadata": {"test": "value"}
        }

        doc_id = await self.manager.add_document(
            document,
            embed_content=False,
            collection_name="model_script_processing"
        )
        self.assertEqual(doc_id, "doc_no_embed")

    async def test_add_document_generated_images_missing_path(self):
        """Test adding document to generated_images without image_path"""
        document = {
            "id": "img_doc_fail",
            "content": "",
            "metadata": {}
        }

        with self.assertRaises(ValueError):
            await self.manager.add_document(
                document,
                collection_name="generated_images"
            )

    async def test_add_document_error_during_upsert(self):
        """Test error handling during document upsert"""
        # Create a failing collection
        failing_collection = FailingDummyCollection("test", self.text_embedder)
        failing_collection.should_fail = True
        self.manager.collections["failing_collection"] = failing_collection

        document = {
            "id": "doc_fail",
            "content": "Test content",
            "metadata": {"test": "value"}
        }

        with self.assertRaises(Exception):
            await self.manager.add_document(document, collection_name="failing_collection")

    # Test metadata flattening edge cases
    async def test_flatten_metadata_complex_types(self):
        """Test flattening of complex metadata types"""
        metadata = {
            "string_val": "test",
            "int_val": 42,
            "float_val": 3.14,
            "bool_val": True,
            "none_val": None,
            "dict_val": {"nested": "value"},
            "list_val": [1, 2, 3],
            "custom_obj": object()
        }

        flattened = self.manager._flatten_metadata(metadata)

        self.assertEqual(flattened["string_val"], "test")
        self.assertEqual(flattened["int_val"], 42)
        self.assertEqual(flattened["float_val"], 3.14)
        self.assertTrue(flattened["bool_val"])
        self.assertEqual(flattened["none_val"], "")
        self.assertEqual(flattened["dict_val"], json.dumps({"nested": "value"}))
        self.assertEqual(flattened["list_val"], json.dumps([1, 2, 3]))
        self.assertIsInstance(flattened["custom_obj"], str)

    async def test_flatten_metadata_non_dict(self):
        """Test flattening non-dictionary metadata"""
        result = self.manager._flatten_metadata("not a dict")
        self.assertEqual(result, {})

        result = self.manager._flatten_metadata(None)
        self.assertEqual(result, {})

    # Test search edge cases
    async def test_search_with_non_string_query(self):
        """Test search with non-string query"""
        # Add a document first
        document = {"id": "doc1", "content": "test", "metadata": {"key": "value"}}
        await self.manager.add_document(document, collection_name="model_script_processing")

        # Search with dict query (triggers get() path)
        results = await self.manager.search(
            query={"metadata": "filter"},
            collection_name="model_script_processing"
        )
        self.assertIn("results", results)

    async def test_search_with_image_collection(self):
        """Test search in image collection"""
        results = await self.manager.search(
            "test query",
            collection_name="generated_images"
        )
        self.assertIn("results", results)

    async def test_search_with_where_filter(self):
        """Test search with where filter"""
        document = {"id": "doc1", "content": "test", "metadata": {"category": "A"}}
        await self.manager.add_document(document, collection_name="model_script_processing")

        results = await self.manager.search(
            "test",
            collection_name="model_script_processing",
            where={"category": {"$eq": "A"}}
        )
        self.assertIn("results", results)

    async def test_search_with_user_id(self):
        """Test search with user ID for access control"""
        document = {"id": "doc1", "content": "test", "metadata": {"owner": "user123"}}
        await self.manager.add_document(document, collection_name="model_script_processing")

        results = await self.manager.search(
            "test",
            collection_name="model_script_processing",
            user_id="user123",
            where={"owner": "user123"}
        )
        self.assertIn("results", results)

    async def test_search_error_handling(self):
        """Test error handling in search"""
        failing_collection = FailingDummyCollection("test", self.text_embedder)
        failing_collection.should_fail = True
        self.manager.collections["failing_search"] = failing_collection

        with self.assertRaises(Exception):
            await self.manager.search("test", collection_name="failing_search")

    # Test get operations edge cases
    async def test_get_with_offset_and_limit(self):
        """Test get with offset and limit parameters"""
        documents = [
            {"id": f"doc{i}", "content": f"content{i}", "metadata": {}}
            for i in range(5)
        ]
        await self.manager.add_documents(documents, collection_name="model_script_processing")

        results = await self.manager.get(
            collection_name="model_script_processing",
            limit=2,
            offset=1
        )
        self.assertIn("results", results)
        self.assertLessEqual(len(results["results"]), 2)

    async def test_get_with_include_parameter(self):
        """Test get with specific include parameter"""
        document = {"id": "doc1", "content": "test", "metadata": {"key": "value"}}
        await self.manager.add_document(document, collection_name="model_script_processing")

        results = await self.manager.get(
            collection_name="model_script_processing",
            ids=["doc1"],
            include=["metadatas"]
        )

        self.assertIn("results", results)
        self.assertIn("metadata", results["results"][0])

    async def test_get_document_not_found(self):
        """Test getting non-existent document"""
        result = await self.manager.get_document(
            "nonexistent",
            collection_name="model_script_processing"
        )
        self.assertIsNone(result)

    # Test update operations edge cases
    async def test_update_document_not_found(self):
        """Test updating non-existent document"""
        update_data = {"content": "new content", "metadata": {}}

        success = await self.manager.update_document(
            "nonexistent",
            update_data,
            collection_name="model_script_processing"
        )
        self.assertFalse(success)

    async def test_update_document_with_user_access_control(self):
        """Test updating document with user access control"""
        # This will fail because document doesn't exist for the user
        update_data = {"content": "new content", "metadata": {}}

        success = await self.manager.update_document(
            "doc1",
            update_data,
            collection_name="model_script_processing",
            user_id="user123"
        )
        self.assertFalse(success)

    async def test_update_document_invalid_metadata(self):
        """Test updating document with invalid metadata"""
        document = {"id": "doc1", "content": "test", "metadata": {}}
        await self.manager.add_document(document, collection_name="model_script_processing")

        update_data = {"content": "new content", "metadata": "invalid"}

        success = await self.manager.update_document(
            "doc1",
            update_data,
            collection_name="model_script_processing"
        )
        self.assertTrue(success)  # Should convert invalid metadata to {}

    # Test delete operations edge cases
    async def test_delete_document_not_found(self):
        """Test deleting non-existent document"""
        success = await self.manager.delete_document(
            "nonexistent",
            collection_name="model_script_processing"
        )
        self.assertTrue(success)  # Should not fail for non-existent documents

    async def test_delete_document_with_user_access_control(self):
        """Test deleting document with user access control"""
        success = await self.manager.delete_document(
            "doc1",
            collection_name="model_script_processing",
            user_id="user123"
        )
        self.assertFalse(success)  # Should fail because document doesn't exist for user

    async def test_delete_documents_no_matches(self):
        """Test deleting documents with no matches"""
        deleted_count = await self.manager.delete_documents(
            where={"nonexistent": "value"},
            collection_name="model_script_processing"
        )
        self.assertEqual(deleted_count, 0)

    async def test_delete_documents_with_user_access_control(self):
        """Test batch delete with user access control"""
        deleted_count = await self.manager.delete_documents(
            where={"test": "value"},
            collection_name="model_script_processing",
            user_id="user123"
        )
        self.assertEqual(deleted_count, 0)

    # Test utility methods
    async def test_check_content_exists_various_types(self):
        """Test _check_content_exists with various content types"""
        # String content
        self.assertTrue(self.manager._check_content_exists("content"))
        self.assertFalse(self.manager._check_content_exists(""))
        self.assertFalse(self.manager._check_content_exists("   "))

        # Bytes content
        self.assertTrue(self.manager._check_content_exists(b"content"))
        self.assertFalse(self.manager._check_content_exists(b""))

        # Numpy array
        self.assertTrue(self.manager._check_content_exists(np.array([1, 2, 3])))
        self.assertFalse(self.manager._check_content_exists(np.array([])))

        # List
        self.assertTrue(self.manager._check_content_exists([1, 2, 3]))
        self.assertFalse(self.manager._check_content_exists([]))

        # Other types
        self.assertTrue(self.manager._check_content_exists(42))
        self.assertFalse(self.manager._check_content_exists(0))
        self.assertFalse(self.manager._check_content_exists(None))

    async def test_normalize_embed_flag(self):
        """Test _normalize_embed_flag with various inputs"""
        self.assertTrue(self.manager._normalize_embed_flag(True))
        self.assertFalse(self.manager._normalize_embed_flag(False))
        self.assertTrue(self.manager._normalize_embed_flag(np.array([1, 2, 3])))
        self.assertFalse(self.manager._normalize_embed_flag(np.array([0, 0, 0])))
        self.assertTrue(self.manager._normalize_embed_flag("truthy"))
        self.assertFalse(self.manager._normalize_embed_flag(""))

    async def test_get_collection_stats(self):
        """Test getting collection statistics"""
        # Add some documents
        documents = [
            {"id": f"doc{i}", "content": f"content{i}", "metadata": {}}
            for i in range(3)
        ]
        await self.manager.add_documents(documents, collection_name="model_script_processing")

        stats = await self.manager.get_collection_stats("model_script_processing")

        self.assertIn("name", stats)
        self.assertIn("count", stats)
        self.assertIn("description", stats)
        self.assertIn("embedding_function", stats)
        self.assertEqual(stats["count"], 3)

    async def test_get_collection_stats_image_collection(self):
        """Test getting stats for image collection"""
        stats = await self.manager.get_collection_stats("generated_images")
        self.assertEqual(stats["embedding_function"], "image_embedder")

    # Test access control edge cases
    async def test_apply_access_control_empty_where(self):
        """Test applying access control with empty where clause"""
        result = self.manager._apply_access_control({}, "user123")
        self.assertIn("$or", result)

    async def test_apply_access_control_none_where(self):
        """Test applying access control with None where clause"""
        result = self.manager._apply_access_control(None, "user123")
        self.assertIn("$or", result)

    # Test batch operations edge cases
    async def test_add_documents_empty_list(self):
        """Test adding empty list of documents"""
        result = await self.manager.add_documents([], collection_name="model_script_processing")
        self.assertEqual(result, [])

    async def test_add_documents_with_image_collection(self):
        """Test batch adding to image collection"""
        documents = [
            {"id": "img1", "content": "", "metadata": {}},
            {"id": "img2", "content": "", "metadata": {}}
        ]

        result = await self.manager.add_documents(documents, collection_name="generated_images")
        self.assertEqual(len(result), 2)

    # Test result processing edge cases
    async def test_process_search_results_empty(self):
        """Test processing empty search results"""
        empty_results = {"ids": [[]], "documents": [[]], "metadatas": [[]], "embeddings": [[]]}
        processed = self.manager._process_search_results(empty_results, ["metadatas", "documents"])
        self.assertEqual(processed["results"], [])

    async def test_process_search_results_with_distances(self):
        """Test processing search results with distances"""
        results = {
            "ids": [["doc1"]],
            "documents": [["content1"]],
            "metadatas": [[{"key": "value"}]],
            "distances": [[0.5]]
        }
        processed = self.manager._process_search_results(results, ["metadatas", "documents", "distances"])
        self.assertIn("distance", processed["results"][0])
        self.assertEqual(processed["results"][0]["distance"], 0.5)

    async def test_flatten_list_edge_cases(self):
        """Test _flatten_list with various inputs"""
        # Nested list
        result = self.manager._flatten_list([[1, 2, 3]])
        self.assertEqual(result, [1, 2, 3])

        # Regular list
        result = self.manager._flatten_list([1, 2, 3])
        self.assertEqual(result, [1, 2, 3])

        # Non-list
        result = self.manager._flatten_list("not a list")
        self.assertEqual(result, [])

        # Empty nested list
        result = self.manager._flatten_list([[]])
        self.assertEqual(result, [])

    async def test_count_documents_with_filters(self):
        """Test counting documents with where filters"""
        documents = [
            {"id": "doc1", "content": "test1", "metadata": {"category": "A"}},
            {"id": "doc2", "content": "test2", "metadata": {"category": "B"}}
        ]
        await self.manager.add_documents(documents, collection_name="model_script_processing")

        count = await self.manager.count_documents(
            collection_name="model_script_processing",
            where={"category": "A"}
        )
        self.assertEqual(count, 2)  # Our dummy implementation returns all

    async def test_count_documents_with_user_access_control(self):
        """Test counting documents with user access control"""
        count = await self.manager.count_documents(
            collection_name="model_script_processing",
            where={"test": "value"},
            user_id="user123"
        )
        self.assertEqual(count, 0)

    # Test embedding generation edge cases
    async def test_generate_embeddings_with_existing_array(self):
        """Test embedding generation when embed_content is already an array"""
        embedding_array = np.array([1.0, 2.0, 3.0])
        result = await self.manager._generate_embeddings("content", "model_script_processing", embedding_array)
        self.assertEqual(result, [embedding_array])

    async def test_generate_embeddings_no_content_non_image(self):
        """Test embedding generation with no content for non-image collection"""
        result = await self.manager._generate_embeddings("", "model_script_processing", True)
        self.assertIsNone(result)

    async def test_generate_embeddings_image_collection_with_file(self):
        """Test embedding generation for image collection with file path"""
        with patch("os.path.exists", return_value=True):
            result = await self.manager._generate_embeddings("", "generated_images", "/path/to/image.jpg")
            self.assertIsNotNone(result)

    async def test_generate_embeddings_image_collection_without_file(self):
        """Test embedding generation for image collection without file"""
        result = await self.manager._generate_embeddings("", "generated_images", True)
        self.assertIsNotNone(result)  # Should return random embedding


if __name__ == "__main__":
    unittest.main()