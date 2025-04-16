import unittest

from src.query_engine.search_dispatcher import SearchDispatcher, QueryIntent


# --- Dummy Dependencies --- #

class DummyChromaManager:
    async def search(self, collection_name, **kwargs):
        if collection_name == "model_scripts_chunks":
            return {
                "results": [
                    {
                        "id": "chunk1",
                        "score": 0.92,
                        "metadata": {
                            "model_id": "model1",
                            "metadata_doc_id": "meta1"
                        },
                        "document": "sample text chunk"
                    }
                ]
            }
        elif collection_name == "generated_images":
            return {
                "results": [
                    {
                        "id": "img1",
                        "score": 0.80,
                        "metadata": {"image_path": "img1.png", "thumbnail_path": "thumb1.png"}
                    }
                ]
            }
        return {"results": []}

    async def get(self, collection_name, **kwargs):
        if collection_name == "model_scripts_metadata":
            return {
                "results": [
                    {
                        "id": "meta1",
                        "metadata": {
                            "model_id": "model1",
                            "version": "1.0",
                            "creation_date": "2023-01-01T00:00:00Z"
                        }
                    }
                ]
            }
        if collection_name == "model_scripts":
            return {
                "results": [
                    {
                        "id": "model1",
                        "metadata": {
                            "version": "1.0",
                            "creation_date": "2023-01-01T00:00:00Z"
                        }
                    }
                ]
            }
        return {"results": []}

class DummyTextEmbedder:
    def embed_text(self, query):
        return [0.1] * 5

class DummyImageEmbedder:
    async def generate_embedding(self, image_data):
        return [0.2] * 5

    async def generate_text_embedding(self, query):
        return [0.3] * 5

class DummyAccessControlManager:
    def apply_access_filters(self, parameters, user_id):
        parameters["access_applied"] = True
        return parameters

class DummyAnalytics:
    def __init__(self):
        self.logged_metrics = []
        self.updated_query_status = []

    def log_performance_metrics(self, query_id, total_time_ms, search_time_ms, embedding_time_ms=None):
        self.logged_metrics.append({
            "query_id": query_id,
            "total_time_ms": total_time_ms,
            "search_time_ms": search_time_ms,
            "embedding_time_ms": embedding_time_ms,
        })

    def update_query_status(self, query_id, status):
        self.updated_query_status.append({
            "query_id": query_id,
            "status": status
        })


# --- Unit Tests --- #

class TestSearchDispatcher(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.chroma_manager = DummyChromaManager()
        self.text_embedder = DummyTextEmbedder()
        self.image_embedder = DummyImageEmbedder()
        self.access_control_manager = DummyAccessControlManager()
        self.analytics = DummyAnalytics()

        self.dispatcher = SearchDispatcher(
            chroma_manager=self.chroma_manager,
            text_embedder=self.text_embedder,
            image_embedder=self.image_embedder,
            access_control_manager=self.access_control_manager,
            analytics=self.analytics
        )
        self.dispatcher._run_in_executor = lambda func, *args, **kwargs: func(*args, **kwargs)

    async def test_dispatch_text_search(self):
        query = "find information about AI models"
        parameters = {"limit": 5, "query_id": "q1", "user_id": "user1"}
        result = await self.dispatcher.dispatch(query, "retrieval", parameters, user_id="user1")

        self.assertTrue(result.get("success"))
        self.assertEqual(result.get("type"), "text_search")
        self.assertIn("items", result)
        self.assertGreaterEqual(len(result["items"]), 1)
        self.assertIn("metadata", result)
        self.assertEqual(result["metadata"]["intent"], QueryIntent.RETRIEVAL.value)
        self.assertIn("result_count", result["metadata"])
        self.assertNotIn("user_id", result["metadata"]["parameters"])
        self.assertNotIn("query_id", result["metadata"]["parameters"])

# Add the other tests if needed...

if __name__ == "__main__":
    unittest.main()
