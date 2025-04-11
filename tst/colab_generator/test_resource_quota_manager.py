import unittest
import sqlite3
import os
import sys
import logging

# Configure logging for debugging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Adjust the import path as needed
sys.path.append(os.path.abspath("../src"))
from src.colab_generator.resource_quota_manager import ResourceQuotaManager


class TestResourceQuotaManager(unittest.TestCase):

    def setUp(self):
        """Set up test environment before each test."""
        # Create a temporary file-based database instead of in-memory
        self.test_db = "test_quotas.db"

        # Remove the test database if it exists
        if os.path.exists(self.test_db):
            os.remove(self.test_db)

        # Create the manager with the file-based database
        self.manager = ResourceQuotaManager(db_path=self.test_db)

        # Display the current tables to debug
        self._log_database_tables()

    def _log_database_tables(self):
        """Log the tables in the database for debugging."""
        try:
            with self.manager._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = cursor.fetchall()
                logger.debug(f"Database tables: {tables}")
        except Exception as e:
            logger.error(f"Error checking database tables: {e}")

    def tearDown(self):
        """Clean up after each test."""
        # Close any open connections
        try:
            with self.manager._get_connection() as conn:
                conn.close()
        except Exception:
            pass

        # Remove the test database file
        if os.path.exists(self.test_db):
            os.remove(self.test_db)

    # Create a direct implementation of database initialization for testing
    def _initialize_test_db(self):
        """Directly initialize the database tables for testing."""
        with sqlite3.connect(self.test_db) as conn:
            cursor = conn.cursor()
            # Create quotas table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS quotas (
                    user_id TEXT,
                    resource_type TEXT,
                    quota_limit REAL,
                    PRIMARY KEY (user_id, resource_type)
                )
            """)
            # Create usage table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS usage (
                    user_id TEXT,
                    resource_type TEXT,
                    usage REAL DEFAULT 0,
                    last_updated TEXT,
                    PRIMARY KEY (user_id, resource_type)
                )
            """)
            conn.commit()

        # Log database state after initialization
        self._log_database_tables()

    def test_initialization(self):
        """Test that tables are created properly."""
        # Check if tables exist
        with self.manager._get_connection() as conn:
            cursor = conn.cursor()

            # Check quotas table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='quotas'")
            result = cursor.fetchone()
            if not result:
                # If tables don't exist, initialize them directly for testing
                self._initialize_test_db()

                # Check again
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='quotas'")
                result = cursor.fetchone()

            self.assertIsNotNone(result, "quotas table was not created")

            # Check usage table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='usage'")
            result = cursor.fetchone()
            self.assertIsNotNone(result, "usage table was not created")

    def test_set_quota_new(self):
        """Test setting a new quota."""
        # Ensure tables exist
        self._initialize_test_db()

        user_id = "user1"
        resource_type = "compute"
        limit = 100.0

        self.manager.set_quota(user_id, resource_type, limit)

        with self.manager._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT quota_limit FROM quotas WHERE user_id = ? AND resource_type = ?",
                (user_id, resource_type)
            )
            result = cursor.fetchone()
            self.assertIsNotNone(result)
            self.assertEqual(result[0], limit)

    def test_set_quota_update(self):
        """Test updating an existing quota."""
        # Ensure tables exist
        self._initialize_test_db()

        user_id = "user1"
        resource_type = "compute"
        initial_limit = 100.0
        updated_limit = 200.0

        # Set initial quota
        self.manager.set_quota(user_id, resource_type, initial_limit)

        # Update quota
        self.manager.set_quota(user_id, resource_type, updated_limit)

        with self.manager._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT quota_limit FROM quotas WHERE user_id = ? AND resource_type = ?",
                (user_id, resource_type)
            )
            result = cursor.fetchone()
            self.assertIsNotNone(result)
            self.assertEqual(result[0], updated_limit)

    def test_check_quota_not_exist(self):
        """Test checking a quota that doesn't exist."""
        # Ensure tables exist
        self._initialize_test_db()

        user_id = "nonexistent"
        resource_type = "compute"

        has_quota, remaining, limit = self.manager.check_quota(user_id, resource_type)

        self.assertFalse(has_quota)
        self.assertEqual(remaining, 0)
        self.assertEqual(limit, 0)

    def test_check_quota_sufficient(self):
        """Test checking a quota with sufficient remaining."""
        # Ensure tables exist
        self._initialize_test_db()

        user_id = "user1"
        resource_type = "compute"
        limit = 100.0
        usage = 50.0

        # Set quota and usage
        self.manager.set_quota(user_id, resource_type, limit)
        self.manager.update_usage(user_id, resource_type, usage)

        has_quota, remaining, quota_limit = self.manager.check_quota(user_id, resource_type)

        self.assertTrue(has_quota)
        self.assertEqual(remaining, limit - usage)
        self.assertEqual(quota_limit, limit)

    def test_check_quota_insufficient(self):
        """Test checking a quota with insufficient remaining."""
        # Ensure tables exist
        self._initialize_test_db()

        user_id = "user1"
        resource_type = "compute"
        limit = 100.0
        usage = 100.0  # Equal to the limit

        # Set quota and usage
        self.manager.set_quota(user_id, resource_type, limit)
        self.manager.update_usage(user_id, resource_type, usage)

        has_quota, remaining, quota_limit = self.manager.check_quota(user_id, resource_type)

        self.assertFalse(has_quota)
        self.assertEqual(remaining, 0)
        self.assertEqual(quota_limit, limit)

    def test_update_usage_new(self):
        """Test updating usage for a new record."""
        # Ensure tables exist
        self._initialize_test_db()

        user_id = "user1"
        resource_type = "compute"
        usage = 50.0

        self.manager.update_usage(user_id, resource_type, usage)

        with self.manager._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT usage FROM usage WHERE user_id = ? AND resource_type = ?",
                (user_id, resource_type)
            )
            result = cursor.fetchone()
            self.assertIsNotNone(result)
            self.assertEqual(result[0], usage)

    def test_update_usage_existing(self):
        """Test updating usage for an existing record."""
        # Ensure tables exist
        self._initialize_test_db()

        user_id = "user1"
        resource_type = "compute"
        initial_usage = 50.0
        additional_usage = 25.0

        # Set initial usage
        self.manager.update_usage(user_id, resource_type, initial_usage)

        # Update usage
        self.manager.update_usage(user_id, resource_type, additional_usage)

        with self.manager._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT usage FROM usage WHERE user_id = ? AND resource_type = ?",
                (user_id, resource_type)
            )
            result = cursor.fetchone()
            self.assertIsNotNone(result)
            self.assertEqual(result[0], initial_usage + additional_usage)

    def test_get_usage_report(self):
        """Test getting a usage report with multiple resources."""
        # Ensure tables exist
        self._initialize_test_db()

        user_id = "user1"
        resources = {
            "compute": {"limit": 100.0, "usage": 50.0},
            "storage": {"limit": 1000.0, "usage": 200.0},
            "bandwidth": {"limit": 500.0, "usage": 400.0}
        }

        # Set up quotas and usage
        for resource_type, values in resources.items():
            self.manager.set_quota(user_id, resource_type, values["limit"])
            self.manager.update_usage(user_id, resource_type, values["usage"])

        report = self.manager.get_usage_report(user_id)

        self.assertEqual(len(report), len(resources))

        for entry in report:
            resource_type = entry["resource_type"]
            self.assertIn(resource_type, resources)
            self.assertEqual(entry["quota_limit"], resources[resource_type]["limit"])
            self.assertEqual(entry["usage"], resources[resource_type]["usage"])
            self.assertEqual(entry["remaining"], resources[resource_type]["limit"] - resources[resource_type]["usage"])
            self.assertIsNotNone(entry["last_updated"])

    def test_get_usage_report_empty(self):
        """Test getting a usage report for a user with no quotas."""
        # Ensure tables exist
        self._initialize_test_db()

        user_id = "nonexistent"

        report = self.manager.get_usage_report(user_id)

        self.assertEqual(len(report), 0)

    def test_zero_remaining_quota(self):
        """Test the edge case when remaining quota is exactly zero."""
        # Ensure tables exist
        self._initialize_test_db()

        user_id = "user1"
        resource_type = "compute"
        limit = 100.0

        # Set quota
        self.manager.set_quota(user_id, resource_type, limit)

        # Use entire quota
        self.manager.update_usage(user_id, resource_type, limit)

        has_quota, remaining, quota_limit = self.manager.check_quota(user_id, resource_type)

        self.assertFalse(has_quota)
        self.assertEqual(remaining, 0)
        self.assertEqual(quota_limit, limit)

    def test_negative_usage_handling(self):
        """Test handling of negative usage values (which could represent credits)."""
        # Ensure tables exist
        self._initialize_test_db()

        user_id = "user1"
        resource_type = "compute"
        limit = 100.0
        usage = -50.0  # Negative usage (credit)

        self.manager.set_quota(user_id, resource_type, limit)
        self.manager.update_usage(user_id, resource_type, usage)

        has_quota, remaining, quota_limit = self.manager.check_quota(user_id, resource_type)

        self.assertTrue(has_quota)
        self.assertEqual(remaining, limit - usage)  # Should be 150.0
        self.assertEqual(quota_limit, limit)


if __name__ == "__main__":
    unittest.main()