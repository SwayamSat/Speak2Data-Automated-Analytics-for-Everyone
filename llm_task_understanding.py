"""
Task understanding module using LLM.
Infers the task type and extracts relevant information from natural language queries.
"""
from typing import Dict, List, Any, Optional
from pathlib import Path

from llm_client import create_llm_client, LLMClient
from utils import format_schema_for_llm
from config import config


def load_prompt_template(prompt_name: str) -> str:
    """
    Load a prompt template from the prompts directory.
    
    Args:
        prompt_name: Name of the prompt file (without .txt extension)
        
    Returns:
        Prompt template content
    """
    prompt_path = config.prompts_dir / f"{prompt_name}.txt"
    if prompt_path.exists():
        return prompt_path.read_text(encoding='utf-8')
    else:
        # Return default if file doesn't exist
        return get_default_task_understanding_prompt()


def get_default_task_understanding_prompt() -> str:
    """Get the default task understanding prompt."""
    return """You are a data analyst assistant helping to understand user queries about databases.

Given a natural language query and database schema, analyze the query and return a structured JSON response with the following fields:

{
  "task_type": "<one of: descriptive_analytics, aggregation, classification, regression, time_series_forecast, clustering, comparison, correlation_analysis>",
  "explanation": "<brief explanation of what the user wants>",
  "relevant_tables": ["<table1>", "<table2>", ...],
  "target_column": "<column name for supervised learning tasks, or null>",
  "feature_columns": ["<col1>", "<col2>", ...],
  "filters": "<SQL WHERE conditions if any, or null>",
  "grouping": ["<columns to group by>", ...],
  "time_column": "<datetime column for time series, or null>",
  "aggregations": {"<column>": "<function like SUM, AVG, COUNT>", ...}
}

Task type definitions:
- descriptive_analytics: Show data, basic statistics, distributions
- aggregation: Group by and aggregate (SUM, COUNT, AVG, etc.)
- classification: Predict categorical target variable
- regression: Predict numeric target variable
- time_series_forecast: Predict future values based on time series
- clustering: Group similar records without predefined labels
- comparison: Compare groups or segments
- correlation_analysis: Analyze relationships between variables

Important:
- Be specific about which tables and columns are relevant
- For ML tasks (classification, regression, clustering), identify potential target and feature columns
- Extract any filtering conditions from the query
- If the query is ambiguous, make reasonable assumptions based on the schema

Database Schema:
{schema}

User Query:
{query}

Return only valid JSON."""


def infer_task(
    query: str,
    schema_info: Dict[str, List[Dict[str, str]]],
    llm_client: Optional[LLMClient] = None
) -> Dict[str, Any]:
    """
    Infer the task type and extract structured information from a natural language query.
    
    Args:
        query: Natural language query from the user
        schema_info: Database schema information
        llm_client: Optional LLM client (will create default if not provided)
        
    Returns:
        Dictionary with task understanding information
    """
    if llm_client is None:
        llm_client = create_llm_client()
    
    # Load prompt template
    prompt_template = load_prompt_template("task_understanding_prompt")
    
    # Format schema
    schema_str = format_schema_for_llm(schema_info)
    
    # Fill in template
    prompt = prompt_template.replace("{schema}", schema_str).replace("{query}", query)
    
    # Get structured response from LLM
    try:
        response = llm_client.structured_complete(prompt, temperature=0.3)
        
        # Validate required fields
        required_fields = ["task_type", "explanation", "relevant_tables"]
        for field in required_fields:
            if field not in response:
                response[field] = None
        
        # Ensure task_type is valid
        valid_task_types = [
            "descriptive_analytics",
            "aggregation",
            "classification",
            "regression",
            "time_series_forecast",
            "clustering",
            "comparison",
            "correlation_analysis"
        ]
        
        if response["task_type"] not in valid_task_types:
            # Try to map to a valid type
            response["task_type"] = "descriptive_analytics"
        
        return response
    
    except Exception as e:
        # Fallback response
        print(f"Warning: Task understanding failed: {e}")
        return {
            "task_type": "descriptive_analytics",
            "explanation": f"Failed to understand task: {str(e)}",
            "relevant_tables": list(schema_info.keys()),
            "target_column": None,
            "feature_columns": [],
            "filters": None,
            "grouping": [],
            "time_column": None,
            "aggregations": {}
        }


def refine_task_understanding(
    task_info: Dict[str, Any],
    available_columns: List[str],
    data_sample: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Refine task understanding based on actual available columns and data.
    
    Args:
        task_info: Initial task understanding
        available_columns: List of actual column names in the dataset
        data_sample: Optional sample of the data
        
    Returns:
        Refined task understanding
    """
    import pandas as pd
    
    # Filter feature columns to only include those that exist
    if task_info.get("feature_columns"):
        task_info["feature_columns"] = [
            col for col in task_info["feature_columns"]
            if col in available_columns
        ]
    
    # Validate target column exists
    if task_info.get("target_column") and task_info["target_column"] not in available_columns:
        # Try to find a similar column name
        target_lower = task_info["target_column"].lower()
        for col in available_columns:
            if target_lower in col.lower() or col.lower() in target_lower:
                task_info["target_column"] = col
                break
        else:
            task_info["target_column"] = None
    
    # Validate time column exists
    if task_info.get("time_column") and task_info["time_column"] not in available_columns:
        # Try to find datetime columns
        if data_sample is not None and isinstance(data_sample, pd.DataFrame):
            datetime_cols = data_sample.select_dtypes(include=['datetime64']).columns.tolist()
            if not datetime_cols:
                # Check for columns with date/time in name
                datetime_cols = [col for col in available_columns if any(
                    keyword in col.lower() for keyword in ['date', 'time', 'year', 'month']
                )]
            if datetime_cols:
                task_info["time_column"] = datetime_cols[0]
            else:
                task_info["time_column"] = None
        else:
            task_info["time_column"] = None
    
    return task_info
