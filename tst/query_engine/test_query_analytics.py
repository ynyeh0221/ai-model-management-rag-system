import os
import time
import json
import sqlite3
import datetime
import tempfile
import unittest
import pandas as pd

from src.query_engine.query_analytics import QueryAnalytics

class TestQueryAnalytics(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory and SQLite database file for testing.
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_analytics.db")
        self.analytics = QueryAnalytics(db_path=self.db_path)

    def tearDown(self):
        # Cleanup the temporary directory.
        self.temp_dir.cleanup()

    def test_log_query(self):
        query_text = "What is AI?"
        intent = "retrieval"
        parameters = {"param1": "value1"}
        user_id = "test_user"
        processing_time_ms = 150

        query_id = self.analytics.log_query(query_text, intent, parameters, user_id, processing_time_ms)
        self.assertIsInstance(query_id, str)

        # Verify the record exists in the database.
        with self.analytics._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM queries WHERE query_id = ?", (query_id,))
            row = cursor.fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row["query_text"], query_text)
            self.assertEqual(row["intent"], intent)
            self.assertEqual(json.loads(row["parameters"]), parameters)
            self.assertEqual(row["user_id"], user_id)
            self.assertEqual(row["processing_time_ms"], processing_time_ms)
            self.assertEqual(row["status"], "pending")

    def test_update_query_status(self):
        query_id = self.analytics.log_query("dummy", "retrieval", {})
        updated = self.analytics.update_query_status(query_id, "success")
        self.assertTrue(updated)
        with self.analytics._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT status FROM queries WHERE query_id = ?", (query_id,))
            row = cursor.fetchone()
            self.assertEqual(row["status"], "success")

    def test_log_result(self):
        # Log a query first.
        query_id = self.analytics.log_query("dummy", "retrieval", {})
        results = ["result1", "result2", "result3"]
        selected_result = "result1"
        response_time_ms = 200

        result_id = self.analytics.log_result(query_id, results, selected_result, response_time_ms)
        self.assertIsInstance(result_id, str)

        # Verify entry in query_results.
        with self.analytics._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM query_results WHERE result_id = ?", (result_id,))
            row = cursor.fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row["query_id"], query_id)
            self.assertEqual(row["results_count"], len(results))
            self.assertEqual(row["selected_result"], selected_result)
            self.assertEqual(row["response_time_ms"], response_time_ms)
            # Also check that the query status was updated to 'success'.
            cursor.execute("SELECT status FROM queries WHERE query_id = ?", (query_id,))
            row2 = cursor.fetchone()
            self.assertEqual(row2["status"], "success")

    def test_log_performance_metrics(self):
        query_id = self.analytics.log_query("dummy", "retrieval", {})
        total_time_ms = 500
        embedding_time_ms = 100
        search_time_ms = 200
        ranking_time_ms = 50
        memory_usage_mb = 123.45

        metric_id = self.analytics.log_performance_metrics(query_id, total_time_ms, embedding_time_ms,
                                                           search_time_ms, ranking_time_ms, memory_usage_mb)
        self.assertIsInstance(metric_id, str)
        with self.analytics._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM performance_metrics WHERE metric_id = ?", (metric_id,))
            row = cursor.fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row["query_id"], query_id)
            self.assertEqual(row["total_time_ms"], total_time_ms)
            self.assertEqual(row["embedding_time_ms"], embedding_time_ms)
            self.assertEqual(row["search_time_ms"], search_time_ms)
            self.assertEqual(row["ranking_time_ms"], ranking_time_ms)
            self.assertAlmostEqual(row["memory_usage_mb"], memory_usage_mb)

    def test_log_user_feedback(self):
        query_id = self.analytics.log_query("dummy", "retrieval", {})
        rating = 4
        comments = "Good result"
        feedback_id = self.analytics.log_user_feedback(query_id, rating, comments)
        self.assertIsInstance(feedback_id, str)
        with self.analytics._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM user_feedback WHERE feedback_id = ?", (feedback_id,))
            row = cursor.fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row["query_id"], query_id)
            self.assertEqual(row["rating"], rating)
            self.assertEqual(row["comments"], comments)

    def test_get_query_distribution(self):
        now = int(time.time())
        # Log two queries.
        q1 = self.analytics.log_query("query1", "retrieval", {})
        q2 = self.analytics.log_query("query2", "comparison", {})
        # Update one query to success.
        self.analytics.update_query_status(q1, "success")

        distribution = self.analytics.get_query_distribution(time_period="day", start_time=now - 1000, end_time=now + 1000)
        self.assertIn("total_queries", distribution)
        self.assertIn("intent_distribution", distribution)
        self.assertIn("success_rate", distribution)
        self.assertIn("avg_processing_time_ms", distribution)
        # Expect exactly 2 queries logged.
        self.assertEqual(distribution["total_queries"], 2)
        # With one success, success_rate should be 0.5.
        self.assertAlmostEqual(distribution["success_rate"], 0.5)

    def test_get_performance_metrics(self):
        now = int(time.time())
        q1 = self.analytics.log_query("query1", "retrieval", {})
        self.analytics.log_performance_metrics(q1,
