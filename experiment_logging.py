"""
Experiment logging module for the Speak2Data system.
Tracks all experiments in a SQLite database for research purposes.
"""
from typing import Dict, List, Any, Optional
import sqlite3
import json
from datetime import datetime
from pathlib import Path
import pandas as pd

from config import config
from utils import get_timestamp


class ExperimentLogger:
    """Logger for tracking experiments."""
    
    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize experiment logger.
        
        Args:
            db_path: Path to SQLite database (defaults to config.experiments_db)
        """
        self.db_path = db_path or config.experiments_db
        self._initialize_db()
    
    def _initialize_db(self):
        """Create the experiments table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                query TEXT NOT NULL,
                task_type TEXT,
                task_understanding TEXT,
                sql_query TEXT,
                dataset_info TEXT,
                model_name TEXT,
                metrics TEXT,
                config TEXT,
                error TEXT,
                success INTEGER
            )
        """)
        
        conn.commit()
        conn.close()
    
    def log_experiment(
        self,
        query: str,
        task_type: Optional[str] = None,
        task_understanding: Optional[Dict[str, Any]] = None,
        sql_query: Optional[str] = None,
        dataset_info: Optional[Dict[str, Any]] = None,
        model_name: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None,
        config_dict: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        success: bool = True
    ) -> int:
        """
        Log an experiment to the database.
        
        Args:
            query: Natural language query
            task_type: Inferred task type
            task_understanding: Full task understanding dict
            sql_query: Generated SQL query
            dataset_info: Information about the dataset
            model_name: Name of the best model
            metrics: Model metrics
            config_dict: Configuration used
            error: Error message if any
            success: Whether the experiment succeeded
            
        Returns:
            Experiment ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO experiments (
                timestamp, query, task_type, task_understanding,
                sql_query, dataset_info, model_name, metrics,
                config, error, success
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            get_timestamp(),
            query,
            task_type,
            json.dumps(task_understanding) if task_understanding else None,
            sql_query,
            json.dumps(dataset_info) if dataset_info else None,
            model_name,
            json.dumps(metrics) if metrics else None,
            json.dumps(config_dict) if config_dict else None,
            error,
            1 if success else 0
        ))
        
        experiment_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return experiment_id
    
    def get_experiment(self, experiment_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve an experiment by ID.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            Experiment dictionary or None
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM experiments WHERE id = ?
        """, (experiment_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        columns = [
            'id', 'timestamp', 'query', 'task_type', 'task_understanding',
            'sql_query', 'dataset_info', 'model_name', 'metrics',
            'config', 'error', 'success'
        ]
        
        experiment = dict(zip(columns, row))
        
        # Parse JSON fields
        for field in ['task_understanding', 'dataset_info', 'metrics', 'config']:
            if experiment[field]:
                try:
                    experiment[field] = json.loads(experiment[field])
                except:
                    pass
        
        return experiment
    
    def get_recent_experiments(self, limit: int = 10) -> pd.DataFrame:
        """
        Get recent experiments.
        
        Args:
            limit: Maximum number of experiments to retrieve
            
        Returns:
            DataFrame with experiment data
        """
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT 
                id, timestamp, query, task_type, model_name,
                success
            FROM experiments
            ORDER BY timestamp DESC
            LIMIT ?
        """
        
        df = pd.read_sql_query(query, conn, params=(limit,))
        conn.close()
        
        return df
    
    def get_experiments_by_task_type(self, task_type: str) -> pd.DataFrame:
        """
        Get all experiments of a specific task type.
        
        Args:
            task_type: Task type to filter by
            
        Returns:
            DataFrame with experiment data
        """
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT *
            FROM experiments
            WHERE task_type = ?
            ORDER BY timestamp DESC
        """
        
        df = pd.read_sql_query(query, conn, params=(task_type,))
        conn.close()
        
        return df
    
    def get_experiment_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about all experiments.
        
        Returns:
            Dictionary with statistics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total experiments
        cursor.execute("SELECT COUNT(*) FROM experiments")
        total = cursor.fetchone()[0]
        
        # Successful experiments
        cursor.execute("SELECT COUNT(*) FROM experiments WHERE success = 1")
        successful = cursor.fetchone()[0]
        
        # Experiments by task type
        cursor.execute("""
            SELECT task_type, COUNT(*) as count
            FROM experiments
            GROUP BY task_type
            ORDER BY count DESC
        """)
        by_task_type = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Most common models
        cursor.execute("""
            SELECT model_name, COUNT(*) as count
            FROM experiments
            WHERE model_name IS NOT NULL
            GROUP BY model_name
            ORDER BY count DESC
            LIMIT 5
        """)
        top_models = {row[0]: row[1] for row in cursor.fetchall()}
        
        conn.close()
        
        return {
            "total_experiments": total,
            "successful_experiments": successful,
            "success_rate": successful / total if total > 0 else 0,
            "by_task_type": by_task_type,
            "top_models": top_models
        }
    
    def export_experiments(self, output_path: Path, format: str = "csv") -> None:
        """
        Export all experiments to a file.
        
        Args:
            output_path: Path to save the file
            format: Export format ("csv" or "json")
        """
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM experiments", conn)
        conn.close()
        
        if format == "csv":
            df.to_csv(output_path, index=False)
        elif format == "json":
            df.to_json(output_path, orient="records", indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def clear_experiments(self, confirm: bool = False) -> None:
        """
        Clear all experiments from the database.
        
        Args:
            confirm: Must be True to confirm deletion
        """
        if not confirm:
            raise ValueError("Must confirm deletion by setting confirm=True")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM experiments")
        conn.commit()
        conn.close()


# Global logger instance
experiment_logger = ExperimentLogger()
