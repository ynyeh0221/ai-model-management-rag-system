# src/query_engine/query_analytics.py

import sqlite3
import json
import time
import datetime
import uuid
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd
import matplotlib.pyplot as plt
from contextlib import contextmanager

class QueryAnalytics:
    """
    Analytics collector for query engine monitoring.
    Tracks query patterns, performance metrics, and user interactions.
    """
    
    def __init__(self, db_path="./analytics.db"):
        """
        Initialize the QueryAnalytics with a database connection.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._initialize_db()
    
    @contextmanager
    def _get_connection(self):
        """
        Context manager for database connections.
        Ensures connections are properly closed even if exceptions occur.
        
        Yields:
            sqlite3.Connection: The database connection
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # Return rows as dict-like objects
            yield conn
        except sqlite3.Error as e:
            self.logger.error(f"Database error: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()
    
    def _initialize_db(self):
        """
        Initialize the analytics database with necessary tables.
        Creates tables for queries, results, and performance metrics.
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Create queries table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS queries (
                    query_id TEXT PRIMARY KEY,
                    query_text TEXT NOT NULL,
                    intent TEXT NOT NULL,
                    parameters TEXT,
                    user_id TEXT,
                    timestamp INTEGER NOT NULL,
                    processing_time_ms INTEGER,
                    status TEXT DEFAULT 'pending'
                )
                ''')
                
                # Create results table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS query_results (
                    result_id TEXT PRIMARY KEY,
                    query_id TEXT NOT NULL,
                    results_count INTEGER NOT NULL,
                    selected_result TEXT,
                    timestamp INTEGER NOT NULL,
                    response_time_ms INTEGER,
                    FOREIGN KEY (query_id) REFERENCES queries (query_id)
                )
                ''')
                
                # Create performance metrics table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    metric_id TEXT PRIMARY KEY,
                    query_id TEXT NOT NULL,
                    embedding_time_ms INTEGER,
                    search_time_ms INTEGER,
                    ranking_time_ms INTEGER,
                    total_time_ms INTEGER NOT NULL,
                    memory_usage_mb REAL,
                    timestamp INTEGER NOT NULL,
                    FOREIGN KEY (query_id) REFERENCES queries (query_id)
                )
                ''')
                
                # Create user feedback table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_feedback (
                    feedback_id TEXT PRIMARY KEY,
                    query_id TEXT NOT NULL,
                    rating INTEGER,
                    comments TEXT,
                    timestamp INTEGER NOT NULL,
                    FOREIGN KEY (query_id) REFERENCES queries (query_id)
                )
                ''')
                
                conn.commit()
                self.logger.info("Analytics database initialized successfully")
                
        except sqlite3.Error as e:
            self.logger.error(f"Database initialization error: {e}")
            raise
    
    def log_query(self, query_text: str, intent: str, parameters: Dict[str, Any], 
                  user_id: Optional[str] = None, processing_time_ms: Optional[int] = None) -> str:
        """
        Log a query to the analytics database.
        
        Args:
            query_text: The raw query text
            intent: The classified intent
            parameters: Dictionary of extracted parameters
            user_id: Optional user identifier
            processing_time_ms: Processing time in milliseconds
            
        Returns:
            str: The generated query_id
        """
        query_id = str(uuid.uuid4())
        timestamp = int(time.time())
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Convert parameters dict to JSON string
                parameters_json = json.dumps(parameters)
                
                cursor.execute('''
                INSERT INTO queries 
                (query_id, query_text, intent, parameters, user_id, timestamp, processing_time_ms, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (query_id, query_text, intent, parameters_json, user_id, timestamp, 
                      processing_time_ms, 'pending'))
                
                conn.commit()
                self.logger.debug(f"Logged query with ID: {query_id}")
                
                return query_id
                
        except sqlite3.Error as e:
            self.logger.error(f"Error logging query: {e}")
            raise
    
    def update_query_status(self, query_id: str, status: str) -> bool:
        """
        Update the status of a query.
        
        Args:
            query_id: The query identifier
            status: The new status (success, failed, etc.)
            
        Returns:
            bool: Success status
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                UPDATE queries SET status = ? WHERE query_id = ?
                ''', (status, query_id))
                
                conn.commit()
                return cursor.rowcount > 0
                
        except sqlite3.Error as e:
            self.logger.error(f"Error updating query status: {e}")
            return False
    
    def log_result(self, query_id: str, results: Union[List[Any], int], 
                   selected_result: Optional[str] = None, 
                   response_time_ms: Optional[int] = None) -> str:
        """
        Log results for a query.
        
        Args:
            query_id: The query identifier
            results: The query results or count of results
            selected_result: Optional identifier of the selected/clicked result
            response_time_ms: Response time in milliseconds
            
        Returns:
            str: The generated result_id
        """
        result_id = str(uuid.uuid4())
        timestamp = int(time.time())
        
        # Handle results as either a list or a count
        results_count = len(results) if isinstance(results, list) else results
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                INSERT INTO query_results 
                (result_id, query_id, results_count, selected_result, timestamp, response_time_ms)
                VALUES (?, ?, ?, ?, ?, ?)
                ''', (result_id, query_id, results_count, selected_result, timestamp, response_time_ms))
                
                # Update query status to success
                cursor.execute('''
                UPDATE queries SET status = ? WHERE query_id = ?
                ''', ('success', query_id))
                
                conn.commit()
                self.logger.debug(f"Logged results for query ID: {query_id}")
                
                return result_id
                
        except sqlite3.Error as e:
            self.logger.error(f"Error logging query results: {e}")
            raise
    
    def log_performance_metrics(self, query_id: str, total_time_ms: int, 
                               embedding_time_ms: Optional[int] = None,
                               search_time_ms: Optional[int] = None,
                               ranking_time_ms: Optional[int] = None,
                               memory_usage_mb: Optional[float] = None) -> str:
        """
        Log performance metrics for a query.
        
        Args:
            query_id: The query identifier
            total_time_ms: Total processing time in milliseconds
            embedding_time_ms: Time spent on embedding generation
            search_time_ms: Time spent on search
            ranking_time_ms: Time spent on result ranking
            memory_usage_mb: Memory usage in megabytes
            
        Returns:
            str: The generated metric_id
        """
        metric_id = str(uuid.uuid4())
        timestamp = int(time.time())
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                INSERT INTO performance_metrics 
                (metric_id, query_id, embedding_time_ms, search_time_ms, ranking_time_ms, 
                 total_time_ms, memory_usage_mb, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (metric_id, query_id, embedding_time_ms, search_time_ms, ranking_time_ms, 
                      total_time_ms, memory_usage_mb, timestamp))
                
                conn.commit()
                self.logger.debug(f"Logged performance metrics for query ID: {query_id}")
                
                return metric_id
                
        except sqlite3.Error as e:
            self.logger.error(f"Error logging performance metrics: {e}")
            raise
    
    def log_user_feedback(self, query_id: str, rating: int, comments: Optional[str] = None) -> str:
        """
        Log user feedback for a query.
        
        Args:
            query_id: The query identifier
            rating: User rating (1-5)
            comments: Optional user comments
            
        Returns:
            str: The generated feedback_id
        """
        feedback_id = str(uuid.uuid4())
        timestamp = int(time.time())
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                INSERT INTO user_feedback 
                (feedback_id, query_id, rating, comments, timestamp)
                VALUES (?, ?, ?, ?, ?)
                ''', (feedback_id, query_id, rating, comments, timestamp))
                
                conn.commit()
                self.logger.debug(f"Logged user feedback for query ID: {query_id}")
                
                return feedback_id
                
        except sqlite3.Error as e:
            self.logger.error(f"Error logging user feedback: {e}")
            raise
    
    def get_query_distribution(self, time_period: str = "day", 
                              start_time: Optional[int] = None, 
                              end_time: Optional[int] = None) -> Dict[str, Any]:
        """
        Get the distribution of queries by intent.
        
        Args:
            time_period: Time period for analysis ('hour', 'day', 'week', 'month')
            start_time: Optional start timestamp
            end_time: Optional end timestamp
            
        Returns:
            Dict containing query intent distribution data
        """
        # Calculate time range if not explicitly provided
        now = int(time.time())
        if end_time is None:
            end_time = now
            
        if start_time is None:
            if time_period == "hour":
                start_time = now - 3600
            elif time_period == "day":
                start_time = now - 86400
            elif time_period == "week":
                start_time = now - 604800
            elif time_period == "month":
                start_time = now - 2592000
            else:
                start_time = now - 86400  # Default to 1 day
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Get query count by intent
                cursor.execute('''
                SELECT intent, COUNT(*) as count
                FROM queries
                WHERE timestamp BETWEEN ? AND ?
                GROUP BY intent
                ORDER BY count DESC
                ''', (start_time, end_time))
                
                intent_distribution = {}
                for row in cursor.fetchall():
                    intent_distribution[row['intent']] = row['count']
                
                # Get total query count
                cursor.execute('''
                SELECT COUNT(*) as total
                FROM queries
                WHERE timestamp BETWEEN ? AND ?
                ''', (start_time, end_time))
                
                total_queries = cursor.fetchone()['total']
                
                # Get success rate
                cursor.execute('''
                SELECT status, COUNT(*) as count
                FROM queries
                WHERE timestamp BETWEEN ? AND ?
                GROUP BY status
                ''', (start_time, end_time))
                
                status_counts = {row['status']: row['count'] for row in cursor.fetchall()}
                success_count = status_counts.get('success', 0)
                success_rate = success_count / total_queries if total_queries > 0 else 0
                
                # Get average processing time
                cursor.execute('''
                SELECT AVG(processing_time_ms) as avg_time
                FROM queries
                WHERE timestamp BETWEEN ? AND ? AND processing_time_ms IS NOT NULL
                ''', (start_time, end_time))
                
                avg_processing_time = cursor.fetchone()['avg_time']
                
                # Format date range for display
                start_dt = datetime.datetime.fromtimestamp(start_time)
                end_dt = datetime.datetime.fromtimestamp(end_time)
                date_range = f"{start_dt.strftime('%Y-%m-%d %H:%M')} to {end_dt.strftime('%Y-%m-%d %H:%M')}"
                
                return {
                    "time_period": time_period,
                    "date_range": date_range,
                    "total_queries": total_queries,
                    "intent_distribution": intent_distribution,
                    "success_rate": success_rate,
                    "avg_processing_time_ms": avg_processing_time
                }
                
        except sqlite3.Error as e:
            self.logger.error(f"Error getting query distribution: {e}")
            raise
    
    def get_performance_metrics(self, time_period: str = "day", 
                               start_time: Optional[int] = None, 
                               end_time: Optional[int] = None) -> Dict[str, Any]:
        """
        Get performance metrics for queries.
        
        Args:
            time_period: Time period for analysis ('hour', 'day', 'week', 'month')
            start_time: Optional start timestamp
            end_time: Optional end timestamp
            
        Returns:
            Dict containing query performance metrics
        """
        # Calculate time range if not explicitly provided
        now = int(time.time())
        if end_time is None:
            end_time = now
            
        if start_time is None:
            if time_period == "hour":
                start_time = now - 3600
            elif time_period == "day":
                start_time = now - 86400
            elif time_period == "week":
                start_time = now - 604800
            elif time_period == "month":
                start_time = now - 2592000
            else:
                start_time = now - 86400  # Default to 1 day
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Get performance metrics
                cursor.execute('''
                SELECT 
                    AVG(total_time_ms) as avg_total_time,
                    MIN(total_time_ms) as min_total_time,
                    MAX(total_time_ms) as max_total_time,
                    AVG(embedding_time_ms) as avg_embedding_time,
                    AVG(search_time_ms) as avg_search_time,
                    AVG(ranking_time_ms) as avg_ranking_time,
                    AVG(memory_usage_mb) as avg_memory_usage
                FROM performance_metrics
                WHERE timestamp BETWEEN ? AND ?
                ''', (start_time, end_time))
                
                perf_metrics = dict(cursor.fetchone())
                
                # Get 95th percentile response time
                cursor.execute('''
                SELECT total_time_ms
                FROM performance_metrics
                WHERE timestamp BETWEEN ? AND ?
                ORDER BY total_time_ms
                ''', (start_time, end_time))
                
                response_times = [row['total_time_ms'] for row in cursor.fetchall()]
                if response_times:
                    percentile_95 = sorted(response_times)[int(len(response_times) * 0.95)]
                    perf_metrics['percentile_95_response_time'] = percentile_95
                else:
                    perf_metrics['percentile_95_response_time'] = None
                
                # Get average results count
                cursor.execute('''
                SELECT AVG(results_count) as avg_results_count
                FROM query_results
                WHERE timestamp BETWEEN ? AND ?
                ''', (start_time, end_time))
                
                avg_results = cursor.fetchone()['avg_results_count']
                
                # Get performance breakdown by intent
                cursor.execute('''
                SELECT q.intent, AVG(p.total_time_ms) as avg_time
                FROM performance_metrics p
                JOIN queries q ON p.query_id = q.query_id
                WHERE p.timestamp BETWEEN ? AND ?
                GROUP BY q.intent
                ''', (start_time, end_time))
                
                intent_performance = {row['intent']: row['avg_time'] for row in cursor.fetchall()}
                
                # Format date range for display
                start_dt = datetime.datetime.fromtimestamp(start_time)
                end_dt = datetime.datetime.fromtimestamp(end_time)
                date_range = f"{start_dt.strftime('%Y-%m-%d %H:%M')} to {end_dt.strftime('%Y-%m-%d %H:%M')}"
                
                return {
                    "time_period": time_period,
                    "date_range": date_range,
                    "performance_metrics": perf_metrics,
                    "avg_results_count": avg_results,
                    "intent_performance": intent_performance
                }
                
        except sqlite3.Error as e:
            self.logger.error(f"Error getting performance metrics: {e}")
            raise
    
    def generate_performance_report(self, time_period: str = "day") -> Dict[str, Any]:
        """
        Generate a comprehensive performance report.
        
        Args:
            time_period: Time period for analysis ('hour', 'day', 'week', 'month')
            
        Returns:
            Dict containing report data and visualizations
        """
        # Get query distribution
        distribution = self.get_query_distribution(time_period)
        
        # Get performance metrics
        metrics = self.get_performance_metrics(time_period)
        
        # Get hourly query volume
        now = int(time.time())
        if time_period == "hour":
            lookback = now - 3600
            interval = "5 minutes"
        elif time_period == "day":
            lookback = now - 86400
            interval = "1 hour"
        elif time_period == "week":
            lookback = now - 604800
            interval = "1 day"
        else:  # month
            lookback = now - 2592000
            interval = "1 day"
        
        try:
            with self._get_connection() as conn:
                # Use pandas for time series analysis
                df = pd.read_sql_query('''
                SELECT timestamp, intent, status, processing_time_ms
                FROM queries
                WHERE timestamp >= ?
                ''', conn, params=(lookback,))
                
                # Convert timestamp to datetime
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
                
                # Set datetime as index
                df.set_index('datetime', inplace=True)
                
                # Resample by appropriate interval
                query_volume = df.resample(interval[0:2]).count()['timestamp']
                
                # Get success rate over time
                success_rate = df.resample(interval[0:2]).apply(
                    lambda x: (x['status'] == 'success').mean() if len(x) > 0 else 0
                )
                
                # Get top queries
                cursor = conn.cursor()
                cursor.execute('''
                SELECT query_text, COUNT(*) as count
                FROM queries
                WHERE timestamp >= ?
                GROUP BY query_text
                ORDER BY count DESC
                LIMIT 10
                ''', (lookback,))
                
                top_queries = [(row['query_text'], row['count']) for row in cursor.fetchall()]
                
                # Get average rating
                cursor.execute('''
                SELECT AVG(rating) as avg_rating
                FROM user_feedback
                WHERE timestamp >= ?
                ''', (lookback,))
                
                avg_rating = cursor.fetchone()['avg_rating']
                
                return {
                    "time_period": time_period,
                    "date_range": distribution["date_range"],
                    "total_queries": distribution["total_queries"],
                    "success_rate": distribution["success_rate"],
                    "avg_processing_time_ms": distribution["avg_processing_time_ms"],
                    "intent_distribution": distribution["intent_distribution"],
                    "performance_metrics": metrics["performance_metrics"],
                    "query_volume_over_time": query_volume.to_dict(),
                    "success_rate_over_time": success_rate.to_dict(),
                    "top_queries": top_queries,
                    "avg_user_rating": avg_rating
                }
                
        except (sqlite3.Error, pd.errors.DatabaseError) as e:
            self.logger.error(f"Error generating performance report: {e}")
            raise
    
    def export_query_data(self, start_time: Optional[int] = None, 
                         end_time: Optional[int] = None,
                         format: str = "csv") -> str:
        """
        Export query data for further analysis.
        
        Args:
            start_time: Optional start timestamp
            end_time: Optional end timestamp
            format: Export format ('csv' or 'json')
            
        Returns:
            str: Path to the exported file
        """
        now = int(time.time())
        if end_time is None:
            end_time = now
        if start_time is None:
            start_time = now - 604800  # Default to 1 week
        
        try:
            with self._get_connection() as conn:
                # Use pandas to simplify export
                df = pd.read_sql_query('''
                SELECT q.query_id, q.query_text, q.intent, q.parameters, q.user_id,
                       q.timestamp, q.processing_time_ms, q.status,
                       r.results_count, r.selected_result, r.response_time_ms,
                       p.embedding_time_ms, p.search_time_ms, p.ranking_time_ms, 
                       p.total_time_ms, p.memory_usage_mb,
                       f.rating, f.comments
                FROM queries q
                LEFT JOIN query_results r ON q.query_id = r.query_id
                LEFT JOIN performance_metrics p ON q.query_id = p.query_id
                LEFT JOIN user_feedback f ON q.query_id = f.query_id
                WHERE q.timestamp BETWEEN ? AND ?
                ''', conn, params=(start_time, end_time))
                
                # Convert timestamp to datetime
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
                
                # Generate export filename
                start_dt = datetime.datetime.fromtimestamp(start_time)
                end_dt = datetime.datetime.fromtimestamp(end_time)
                date_range = f"{start_dt.strftime('%Y%m%d')}_to_{end_dt.strftime('%Y%m%d')}"
                export_path = f"./query_analytics_{date_range}.{format}"
                
                # Export to selected format
                if format.lower() == 'csv':
                    df.to_csv(export_path, index=False)
                elif format.lower() == 'json':
                    df.to_json(export_path, orient='records')
                else:
                    raise ValueError(f"Unsupported export format: {format}")
                
                return export_path
                
        except (sqlite3.Error, pd.errors.DatabaseError) as e:
            self.logger.error(f"Error exporting query data: {e}")
            raise
    
    def cleanup_old_data(self, retention_days: int = 90) -> int:
        """
        Clean up old analytics data.
        
        Args:
            retention_days: Number of days to retain data
            
        Returns:
            int: Number of records deleted
        """
        cutoff_time = int(time.time()) - (retention_days * 86400)
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Delete from dependent tables first
                cursor.execute('''
                DELETE FROM user_feedback
                WHERE timestamp < ?
                ''', (cutoff_time,))
                feedback_deleted = cursor.rowcount
                
                cursor.execute('''
                DELETE FROM performance_metrics
                WHERE timestamp < ?
                ''', (cutoff_time,))
                metrics_deleted = cursor.rowcount
                
                cursor.execute('''
                DELETE FROM query_results
                WHERE timestamp < ?
                ''', (cutoff_time,))
                results_deleted = cursor.rowcount
                
                # Finally delete from queries table
                cursor.execute('''
                DELETE FROM queries
                WHERE timestamp < ?
                ''', (cutoff_time,))
                queries_deleted = cursor.rowcount
                
                conn.commit()
                
                total_deleted = feedback_deleted + metrics_deleted + results_deleted + queries_deleted
                self.logger.info(f"Cleaned up {total_deleted} old analytics records")
                
                return total_deleted
                
        except sqlite3.Error as e:
            self.logger.error(f"Error cleaning up old data: {e}")
            raise
            
