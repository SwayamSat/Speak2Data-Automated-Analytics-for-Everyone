"""
Database manager for the Speak2Data system.
Handles connections to various data sources (SQLite, CSV, Excel, Parquet)
and provides safe SQL query execution.
"""
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine, inspect, text, MetaData, Table
from sqlalchemy.engine import Engine
import tempfile
import shutil

from utils import clean_sql_query, validate_dataframe, sample_dataframe, infer_column_type
from config import config


class DatabaseManager:
    """Manages database connections and query execution."""
    
    def __init__(self, source: Optional[Union[str, Path]] = None):
        """
        Initialize database manager.
        
        Args:
            source: Can be:
                - Database URL (e.g., 'sqlite:///path/to/db.db', 'postgresql://...')
                - Path to SQLite file
                - Path to CSV/Excel/Parquet file
                - None (will be set later)
        """
        self.engine: Optional[Engine] = None
        self.source_type: Optional[str] = None
        self.source_path: Optional[Path] = None
        self.temp_db_path: Optional[Path] = None
        
        if source:
            self.connect(source)
    
    def connect(self, source: Union[str, Path]) -> None:
        """
        Connect to a data source.
        
        Args:
            source: Database URL or file path
        """
        source_str = str(source)
        
        # Check if it's a database URL
        if any(source_str.startswith(prefix) for prefix in ['sqlite:///', 'postgresql://', 'mysql://', 'mssql://']):
            self.engine = create_engine(source_str)
            self.source_type = "database_url"
            self.source_path = None
        else:
            # It's a file path
            source_path = Path(source)
            if not source_path.exists():
                raise FileNotFoundError(f"File not found: {source_path}")
            
            self.source_path = source_path
            suffix = source_path.suffix.lower()
            
            if suffix in ['.db', '.sqlite', '.sqlite3']:
                # Direct SQLite connection
                self.engine = create_engine(f'sqlite:///{source_path}')
                self.source_type = "sqlite"
            elif suffix in ['.csv', '.xlsx', '.xls', '.parquet']:
                # Load into temporary SQLite database
                self._load_file_to_temp_db(source_path)
                self.source_type = f"file_{suffix[1:]}"
            else:
                raise ValueError(f"Unsupported file type: {suffix}")
    
    def _load_file_to_temp_db(self, file_path: Path) -> None:
        """
        Load a data file (CSV, Excel, Parquet) into a temporary SQLite database.
        
        Args:
            file_path: Path to the data file
        """
        # Read the file
        suffix = file_path.suffix.lower()
        
        if suffix == '.csv':
            df = pd.read_csv(file_path)
        elif suffix in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        elif suffix == '.parquet':
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
        
        # Create temporary SQLite database
        self.temp_db_path = config.temp_db_dir / f"temp_{file_path.stem}.db"
        self.engine = create_engine(f'sqlite:///{self.temp_db_path}')
        
        # Use the file stem as the table name (cleaned)
        table_name = self._clean_table_name(file_path.stem)
        
        # Write DataFrame to SQLite
        df.to_sql(table_name, self.engine, if_exists='replace', index=False)
    
    def _clean_table_name(self, name: str) -> str:
        """Clean table name to be SQL-compatible."""
        import re
        # Replace non-alphanumeric characters with underscore
        name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        # Ensure it starts with a letter
        if name[0].isdigit():
            name = 'table_' + name
        return name.lower()
    
    def get_tables(self) -> List[str]:
        """
        Get list of table names in the database.
        
        Returns:
            List of table names
        """
        if not self.engine:
            raise RuntimeError("No database connection established")
        
        inspector = inspect(self.engine)
        return inspector.get_table_names()
    
    def get_schema(self, table_name: Optional[str] = None) -> Dict[str, List[Dict[str, str]]]:
        """
        Get schema information for tables.
        
        Args:
            table_name: Specific table to get schema for, or None for all tables
            
        Returns:
            Dict mapping table names to lists of column info dicts
        """
        if not self.engine:
            raise RuntimeError("No database connection established")
        
        inspector = inspect(self.engine)
        schema = {}
        
        tables = [table_name] if table_name else self.get_tables()
        
        for table in tables:
            columns = inspector.get_columns(table)
            schema[table] = [
                {
                    "name": col["name"],
                    "type": str(col["type"]),
                    "nullable": col.get("nullable", True),
                    "default": str(col.get("default")) if col.get("default") else None
                }
                for col in columns
            ]
        
        return schema
    
    def run_query(self, sql: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Execute a SQL query and return results as DataFrame.
        
        Args:
            sql: SQL query string (SELECT only)
            params: Optional parameters for parameterized queries
            
        Returns:
            Query results as DataFrame
            
        Raises:
            ValueError: If query is not read-only
            RuntimeError: If no database connection
        """
        if not self.engine:
            raise RuntimeError("No database connection established")
        
        # Validate and clean SQL
        sql = clean_sql_query(sql)
        
        # Execute query
        try:
            with self.engine.connect() as conn:
                df = pd.read_sql(text(sql), conn, params=params)
            
            # Sample if too large
            if len(df) > config.sample_threshold:
                original_rows = len(df)
                df = sample_dataframe(df, config.max_query_rows, config.random_state)
                print(f"Warning: Sampled {len(df)} rows from {original_rows} total rows")
            
            return df
        
        except Exception as e:
            raise RuntimeError(f"Query execution failed: {str(e)}\nSQL: {sql}")
    
    def get_table_sample(self, table_name: str, n: int = 5) -> pd.DataFrame:
        """
        Get a sample of rows from a table.
        
        Args:
            table_name: Name of the table
            n: Number of rows to retrieve
            
        Returns:
            Sample DataFrame
        """
        sql = f"SELECT * FROM {table_name} LIMIT {n}"
        return self.run_query(sql)
    
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """
        Get comprehensive information about a table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Dict with table information
        """
        # Get row count
        count_sql = f"SELECT COUNT(*) as count FROM {table_name}"
        row_count = self.run_query(count_sql)['count'].iloc[0]
        
        # Get schema
        schema = self.get_schema(table_name)[table_name]
        
        # Get sample
        sample = self.get_table_sample(table_name, 100)
        
        # Infer semantic types
        semantic_types = {col: infer_column_type(sample[col]) for col in sample.columns}
        
        return {
            "name": table_name,
            "row_count": int(row_count),
            "column_count": len(schema),
            "columns": schema,
            "semantic_types": semantic_types
        }
    
    def close(self) -> None:
        """Close database connection and clean up temporary files."""
        if self.engine:
            self.engine.dispose()
            self.engine = None
        
        # Clean up temporary database if it exists
        if self.temp_db_path and self.temp_db_path.exists():
            try:
                self.temp_db_path.unlink()
            except Exception as e:
                print(f"Warning: Could not delete temporary database: {e}")
    
    def __del__(self):
        """Cleanup on deletion."""
        self.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
