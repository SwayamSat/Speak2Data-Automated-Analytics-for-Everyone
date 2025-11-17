"""
Utility functions for the Speak2Data system.
Includes schema printing, type inference, safe casting, and helpers.
"""
from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime
import re


def infer_column_type(series: pd.Series) -> str:
    """
    Infer the semantic type of a pandas Series.
    
    Returns:
        One of: "numeric", "categorical", "datetime", "text", "boolean"
    """
    # Handle boolean
    if series.dtype == bool or set(series.dropna().unique()).issubset({0, 1, True, False}):
        return "boolean"
    
    # Handle datetime
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    
    # Try to parse as datetime if object type
    if series.dtype == object:
        try:
            pd.to_datetime(series.dropna().head(100))
            return "datetime"
        except:
            pass
    
    # Handle numeric
    if pd.api.types.is_numeric_dtype(series):
        # Check if it's actually categorical (few unique values relative to total)
        unique_ratio = series.nunique() / len(series)
        if unique_ratio < 0.05 and series.nunique() < 20:
            return "categorical"
        return "numeric"
    
    # Handle categorical vs text
    if series.dtype == object or pd.api.types.is_categorical_dtype(series):
        unique_ratio = series.nunique() / len(series)
        avg_length = series.astype(str).str.len().mean()
        
        # If few unique values and short strings, likely categorical
        if unique_ratio < 0.5 and avg_length < 50:
            return "categorical"
        else:
            return "text"
    
    return "text"


def get_column_statistics(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Get basic statistics for each column in a DataFrame.
    
    Returns:
        Dictionary mapping column names to their statistics.
    """
    stats = {}
    
    for col in df.columns:
        col_stats = {
            "dtype": str(df[col].dtype),
            "inferred_type": infer_column_type(df[col]),
            "missing_count": int(df[col].isna().sum()),
            "missing_percent": float(df[col].isna().sum() / len(df) * 100),
            "unique_count": int(df[col].nunique())
        }
        
        if col_stats["inferred_type"] == "numeric":
            col_stats.update({
                "min": float(df[col].min()) if pd.notna(df[col].min()) else None,
                "max": float(df[col].max()) if pd.notna(df[col].max()) else None,
                "mean": float(df[col].mean()) if pd.notna(df[col].mean()) else None,
                "median": float(df[col].median()) if pd.notna(df[col].median()) else None,
                "std": float(df[col].std()) if pd.notna(df[col].std()) else None
            })
        elif col_stats["inferred_type"] == "categorical":
            value_counts = df[col].value_counts().head(5)
            col_stats["top_values"] = {str(k): int(v) for k, v in value_counts.items()}
        
        stats[col] = col_stats
    
    return stats


def format_schema_for_llm(schema_info: Dict[str, List[Dict[str, str]]]) -> str:
    """
    Format database schema information for LLM consumption.
    
    Args:
        schema_info: Dict mapping table names to lists of column info dicts
        
    Returns:
        Formatted string representation of the schema
    """
    lines = ["Database Schema:\n"]
    
    for table_name, columns in schema_info.items():
        lines.append(f"Table: {table_name}")
        lines.append("Columns:")
        for col in columns:
            col_name = col.get("name", "unknown")
            col_type = col.get("type", "unknown")
            lines.append(f"  - {col_name} ({col_type})")
        lines.append("")
    
    return "\n".join(lines)


def format_dataframe_sample_for_llm(df: pd.DataFrame, max_rows: int = 5) -> str:
    """
    Format a DataFrame sample for LLM consumption.
    
    Args:
        df: DataFrame to format
        max_rows: Maximum number of rows to include
        
    Returns:
        Formatted string representation
    """
    lines = [f"Data sample ({len(df)} total rows):\n"]
    lines.append(df.head(max_rows).to_string())
    lines.append("\n\nColumn types:")
    for col in df.columns:
        lines.append(f"  - {col}: {infer_column_type(df[col])}")
    
    return "\n".join(lines)


def safe_cast_to_numeric(series: pd.Series) -> pd.Series:
    """
    Safely cast a Series to numeric, returning NaN for non-convertible values.
    """
    return pd.to_numeric(series, errors='coerce')


def clean_sql_query(sql: str) -> str:
    """
    Clean and validate SQL query.
    - Remove comments
    - Strip whitespace
    - Basic validation for read-only operations
    """
    # Remove SQL comments
    sql = re.sub(r'--.*$', '', sql, flags=re.MULTILINE)
    sql = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL)
    
    # Strip and normalize whitespace
    sql = ' '.join(sql.split())
    sql = sql.strip()
    
    # Ensure it ends with semicolon (optional, but tidy)
    if not sql.endswith(';'):
        sql += ';'
    
    # Basic validation: check for dangerous keywords
    dangerous_keywords = ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER', 'TRUNCATE']
    sql_upper = sql.upper()
    
    for keyword in dangerous_keywords:
        if keyword in sql_upper:
            raise ValueError(f"SQL query contains forbidden keyword: {keyword}")
    
    return sql


def truncate_text(text: str, max_length: int = 1000) -> str:
    """Truncate text to maximum length."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def get_timestamp() -> str:
    """Get current timestamp as ISO string."""
    return datetime.now().isoformat()


def parse_json_from_llm_response(response: str) -> Dict[str, Any]:
    """
    Parse JSON from LLM response, handling markdown code blocks.
    
    Args:
        response: LLM response that may contain JSON in markdown blocks
        
    Returns:
        Parsed JSON as dictionary
    """
    import json
    
    # Try to extract JSON from markdown code blocks
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
    if json_match:
        response = json_match.group(1)
    
    # Try to parse directly
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        # Try to find the first { and last } and extract that
        start = response.find('{')
        end = response.rfind('}')
        if start != -1 and end != -1:
            try:
                return json.loads(response[start:end+1])
            except:
                pass
    
    raise ValueError(f"Could not parse JSON from response: {truncate_text(response)}")


def validate_dataframe(df: pd.DataFrame, min_rows: int = 1) -> bool:
    """
    Validate that a DataFrame is suitable for analysis.
    
    Args:
        df: DataFrame to validate
        min_rows: Minimum required rows
        
    Returns:
        True if valid
        
    Raises:
        ValueError with descriptive message if invalid
    """
    if df is None:
        raise ValueError("DataFrame is None")
    
    if len(df) < min_rows:
        raise ValueError(f"DataFrame has {len(df)} rows, minimum required is {min_rows}")
    
    if len(df.columns) == 0:
        raise ValueError("DataFrame has no columns")
    
    return True


def sample_dataframe(df: pd.DataFrame, max_rows: int, random_state: int = 42) -> pd.DataFrame:
    """
    Sample a DataFrame if it exceeds max_rows.
    
    Args:
        df: DataFrame to sample
        max_rows: Maximum rows to keep
        random_state: Random seed for reproducibility
        
    Returns:
        Sampled or original DataFrame
    """
    if len(df) <= max_rows:
        return df
    
    return df.sample(n=max_rows, random_state=random_state)
