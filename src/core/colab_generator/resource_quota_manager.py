import sqlite3
from datetime import datetime


class ResourceQuotaManager:
    def __init__(self, db_path="./quotas.db"):
        self.db_path = db_path
        self._initialize_db()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _initialize_db(self):
        """Initialize the quotas and usage tables."""
        with self._get_connection() as conn:
            c = conn.cursor()
            c.execute("""
                CREATE TABLE IF NOT EXISTS quotas (
                    user_id TEXT,
                    resource_type TEXT,
                    quota_limit REAL,
                    PRIMARY KEY (user_id, resource_type)
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS usage (
                    user_id TEXT,
                    resource_type TEXT,
                    usage REAL DEFAULT 0,
                    last_updated TEXT,
                    PRIMARY KEY (user_id, resource_type)
                )
            """)
            conn.commit()

    def check_quota(self, user_id, resource_type):
        """Check if the user has available quota for the resource."""
        with self._get_connection() as conn:
            c = conn.cursor()
            c.execute("""
                SELECT q.quota_limit, COALESCE(u.usage, 0)
                FROM quotas q
                LEFT JOIN usage u ON q.user_id = u.user_id AND q.resource_type = u.resource_type
                WHERE q.user_id = ? AND q.resource_type = ?
            """, (user_id, resource_type))
            row = c.fetchone()
            if not row:
                return False, 0, 0  # No quota set

            quota_limit, current_usage = row
            remaining = quota_limit - current_usage
            return remaining > 0, remaining, quota_limit

    def update_usage(self, user_id, resource_type, amount):
        """Update a user's usage by increasing the usage count."""
        now = datetime.utcnow().isoformat()

        with self._get_connection() as conn:
            c = conn.cursor()
            # Check if usage record exists
            c.execute("""
                SELECT usage FROM usage WHERE user_id = ? AND resource_type = ?
            """, (user_id, resource_type))
            row = c.fetchone()

            if row:
                new_usage = row[0] + amount
                c.execute("""
                    UPDATE usage SET usage = ?, last_updated = ? 
                    WHERE user_id = ? AND resource_type = ?
                """, (new_usage, now, user_id, resource_type))
            else:
                c.execute("""
                    INSERT INTO usage (user_id, resource_type, usage, last_updated)
                    VALUES (?, ?, ?, ?)
                """, (user_id, resource_type, amount, now))

            conn.commit()

    def set_quota(self, user_id, resource_type, limit):
        """Set or update a quota limit for the user and resource."""
        with self._get_connection() as conn:
            c = conn.cursor()
            c.execute("""
                INSERT INTO quotas (user_id, resource_type, quota_limit)
                VALUES (?, ?, ?)
                ON CONFLICT(user_id, resource_type) DO UPDATE SET quota_limit = excluded.quota_limit
            """, (user_id, resource_type, limit))
            conn.commit()

    def get_usage_report(self, user_id):
        """Get a detailed usage report for a user."""
        with self._get_connection() as conn:
            c = conn.cursor()
            c.execute("""
                SELECT q.resource_type, q.quota_limit, COALESCE(u.usage, 0), u.last_updated
                FROM quotas q
                LEFT JOIN usage u ON q.user_id = u.user_id AND q.resource_type = u.resource_type
                WHERE q.user_id = ?
            """, (user_id,))
            rows = c.fetchall()

            return [
                {
                    "resource_type": row[0],
                    "quota_limit": row[1],
                    "usage": row[2],
                    "remaining": max(row[1] - row[2], 0),
                    "last_updated": row[3]
                }
                for row in rows
            ]
