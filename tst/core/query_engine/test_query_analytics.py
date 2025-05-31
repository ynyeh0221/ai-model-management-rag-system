import os
import sqlite3
import tempfile
import time
import unittest

from src.core.query_engine.query_analytics import QueryAnalytics


class TestQueryAnalytics(unittest.TestCase):
    def setUp(self):
        # Create a temporary file for the SQLite database
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.close()
        self.db_path = tmp.name
        self.analytics = QueryAnalytics(db_path=self.db_path)

    def tearDown(self):
        # Remove the database file if it exists
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        # Clean up any exported files
        for fname in os.listdir('.'):
            if fname.startswith("query_analytics_") and (fname.endswith(".csv") or fname.endswith(".json")):
                try:
                    os.remove(fname)
                except OSError:
                    pass

    def test_log_query_and_update_status(self):
        # Log a new query
        query_id = self.analytics.log_query(
            query_text="SELECT * FROM users",
            intent="fetch_users",
            parameters={"limit": 10},
            user_id="user123",
            processing_time_ms=50
        )
        self.assertIsInstance(query_id, str)

        # Verify the query exists in the database with status 'pending'
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM queries WHERE query_id = ?", (query_id,))
        row = cursor.fetchone()
        conn.close()
        self.assertIsNotNone(row)
        self.assertEqual(row["query_text"], "SELECT * FROM users")
        self.assertEqual(row["intent"], "fetch_users")
        self.assertEqual(row["status"], "pending")

        # Update status to 'success'
        updated = self.analytics.update_query_status(query_id, "success")
        self.assertTrue(updated)

        # Verify status has been updated
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT status FROM queries WHERE query_id = ?", (query_id,))
        row = cursor.fetchone()
        conn.close()
        self.assertEqual(row["status"], "success")

    def test_log_result_and_status_change(self):
        # First, log a query
        query_id = self.analytics.log_query(
            query_text="FIND items WHERE price < 100",
            intent="search_items",
            parameters={"max_price": 100},
            user_id="user456",
            processing_time_ms=30
        )

        # Log a result (simulate a list of 3 hits)
        result_id = self.analytics.log_result(
            query_id=query_id,
            results=[{"id": 1}, {"id": 2}, {"id": 3}],
            selected_result="item_2",
            response_time_ms=120
        )
        self.assertIsInstance(result_id, str)

        # Verify result is in query_results table
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM query_results WHERE result_id = ?", (result_id,))
        result_row = cursor.fetchone()
        self.assertIsNotNone(result_row)
        self.assertEqual(result_row["query_id"], query_id)
        self.assertEqual(result_row["results_count"], 3)
        self.assertEqual(result_row["selected_result"], "item_2")

        # Verify that the original query status changed to 'success'
        cursor.execute("SELECT status FROM queries WHERE query_id = ?", (query_id,))
        status_row = cursor.fetchone()
        conn.close()
        self.assertEqual(status_row["status"], "success")

    def test_log_performance_metrics(self):
        # Log a query first
        query_id = self.analytics.log_query(
            query_text="CALCULATE stats",
            intent="compute_stats",
            parameters={},
            user_id=None,
            processing_time_ms=200
        )

        # Log performance metrics
        metric_id = self.analytics.log_performance_metrics(
            query_id=query_id,
            total_time_ms=200,
            embedding_time_ms=20,
            search_time_ms=100,
            ranking_time_ms=80,
            memory_usage_mb=15.5
        )
        self.assertIsInstance(metric_id, str)

        # Verify entry in performance_metrics table
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM performance_metrics WHERE metric_id = ?", (metric_id,))
        perf_row = cursor.fetchone()
        conn.close()
        self.assertIsNotNone(perf_row)
        self.assertEqual(perf_row["query_id"], query_id)
        self.assertEqual(perf_row["total_time_ms"], 200)
        self.assertEqual(perf_row["embedding_time_ms"], 20)
        self.assertEqual(perf_row["search_time_ms"], 100)
        self.assertEqual(perf_row["ranking_time_ms"], 80)
        self.assertAlmostEqual(perf_row["memory_usage_mb"], 15.5, places=3)

    def test_log_user_feedback(self):
        # Log a query
        query_id = self.analytics.log_query(
            query_text="GET user profile",
            intent="fetch_profile",
            parameters={"user_id": "user789"},
            user_id="user789",
            processing_time_ms=40
        )

        # Log feedback
        feedback_id = self.analytics.log_user_feedback(
            query_id=query_id,
            rating=4,
            comments="Useful result"
        )
        self.assertIsInstance(feedback_id, str)

        # Verify entry in user_feedback table
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM user_feedback WHERE feedback_id = ?", (feedback_id,))
        feedback_row = cursor.fetchone()
        conn.close()
        self.assertIsNotNone(feedback_row)
        self.assertEqual(feedback_row["query_id"], query_id)
        self.assertEqual(feedback_row["rating"], 4)
        self.assertEqual(feedback_row["comments"], "Useful result")

    def test_get_query_distribution_and_performance_metrics(self):
        # Log several queries with different intents
        q1 = self.analytics.log_query(
            query_text="A",
            intent="intent_a",
            parameters={},
            user_id=None,
            processing_time_ms=10
        )
        time.sleep(1)
        q2 = self.analytics.log_query(
            query_text="B",
            intent="intent_b",
            parameters={},
            user_id=None,
            processing_time_ms=20
        )
        # Immediately update statuses so they count as 'success'
        self.analytics.update_query_status(q1, "success")
        self.analytics.update_query_status(q2, "success")

        # Log performance metrics for each
        self.analytics.log_performance_metrics(
            query_id=q1,
            total_time_ms=10,
            embedding_time_ms=2,
            search_time_ms=4,
            ranking_time_ms=4,
            memory_usage_mb=5.0
        )
        self.analytics.log_performance_metrics(
            query_id=q2,
            total_time_ms=20,
            embedding_time_ms=3,
            search_time_ms=10,
            ranking_time_ms=7,
            memory_usage_mb=6.0
        )

        # Retrieve distribution and metrics over the last 5 seconds
        now_ts = int(time.time())
        start_ts = now_ts - 5
        dist = self.analytics.get_query_distribution(time_period="day", start_time=start_ts, end_time=now_ts)
        self.assertIn("intent_distribution", dist)
        self.assertGreaterEqual(dist["total_queries"], 2)
        self.assertGreaterEqual(dist["success_rate"], 1.0)  # both marked 'success'

        perf = self.analytics.get_performance_metrics(time_period="day", start_time=start_ts, end_time=now_ts)
        self.assertIn("performance_metrics", perf)
        self.assertIn("avg_total_time", perf["performance_metrics"])
        self.assertIn("percentile_95_response_time", perf["performance_metrics"])
        self.assertGreaterEqual(perf["performance_metrics"]["avg_total_time"], 10)

    def test_export_query_data_creates_file(self):
        # Log a single query to ensure there's something to export
        qid = self.analytics.log_query(
            query_text="EXPORT TEST",
            intent="export_intent",
            parameters={},
            user_id="export_user",
            processing_time_ms=15
        )
        # Export to CSV
        export_path = self.analytics.export_query_data(format="csv")
        self.assertTrue(os.path.exists(export_path))
        self.assertTrue(export_path.endswith(".csv"))
        # Check that file is non-empty
        self.assertGreater(os.path.getsize(export_path), 0)

        # Export to JSON
        export_path_json = self.analytics.export_query_data(format="json")
        self.assertTrue(os.path.exists(export_path_json))
        self.assertTrue(export_path_json.endswith(".json"))
        self.assertGreater(os.path.getsize(export_path_json), 0)

    def test_cleanup_old_data_deletes_records(self):
        # Log a query with an artificially old timestamp
        old_query_id = self.analytics.log_query(
            query_text="OLD DATA",
            intent="old_intent",
            parameters={},
            user_id="old_user",
            processing_time_ms=5
        )
        # Manually set its timestamp to far in the past
        cutoff = int(time.time()) - (90 * 86400) - 1  # older than 90 days
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("UPDATE queries SET timestamp = ? WHERE query_id = ?", (cutoff, old_query_id))
        conn.commit()
        conn.close()

        # Now cleanup with retention_days = 90 (so cutoff is current_time - 90d)
        deleted = self.analytics.cleanup_old_data(retention_days=90)
        self.assertGreaterEqual(deleted, 1)

        # Verify that the old query is gone
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM queries WHERE query_id = ?", (old_query_id,))
        row = cursor.fetchone()
        conn.close()
        self.assertIsNone(row)


if __name__ == "__main__":
    unittest.main()
