"""
SQL generation module using LLM.
Generates safe, read-only SQL queries from task understanding and schema.
"""
from typing import Dict, List, Any, Optional
from pathlib import Path

from llm_client import create_llm_client, LLMClient
from utils import format_schema_for_llm, clean_sql_query
from config import config


def load_sql_prompt_template(prompt_name: str = "sql_generation_prompt") -> str:
    """
    Load SQL generation prompt template.
    
    Args:
        prompt_name: Name of the prompt file
        
    Returns:
        Prompt template content
    """
    prompt_path = config.prompts_dir / f"{prompt_name}.txt"
    if prompt_path.exists():
        return prompt_path.read_text(encoding='utf-8')
    else:
        return get_default_sql_generation_prompt()


def get_default_sql_generation_prompt() -> str:
    """Get the default SQL generation prompt."""
    return """You are an expert SQL developer. Generate a safe, read-only SQL query based on the task requirements.

CRITICAL REQUIREMENTS:
1. Generate ONLY SELECT queries - no INSERT, UPDATE, DELETE, DROP, CREATE, ALTER, or TRUNCATE
2. Use standard SQL compatible with SQLite, PostgreSQL, and MySQL
3. Use proper table and column names from the schema
4. Include appropriate WHERE, GROUP BY, ORDER BY, and LIMIT clauses as needed
5. Return ONLY the SQL query, no explanations or markdown

Database Schema:
{schema}

Task Information:
{task_info}

Generate a SQL query that:
- Retrieves data from the relevant tables: {tables}
- Includes these columns: {columns}
- Applies filters: {filters}
- Groups by: {grouping}
- Applies aggregations: {aggregations}

Return ONLY the SQL query."""


def generate_sql(
    task_info: Dict[str, Any],
    schema_info: Dict[str, List[Dict[str, str]]],
    llm_client: Optional[LLMClient] = None
) -> str:
    """
    Generate SQL query from task understanding and schema.
    
    Args:
        task_info: Task understanding dictionary from llm_task_understanding
        schema_info: Database schema information
        llm_client: Optional LLM client
        
    Returns:
        SQL query string
    """
    if llm_client is None:
        llm_client = create_llm_client()
    
    # Load prompt template
    prompt_template = load_sql_prompt_template()
    
    # Format schema
    schema_str = format_schema_for_llm(schema_info)
    
    # Extract task details
    tables = task_info.get("relevant_tables", [])
    
    # Determine columns to select
    columns = []
    if task_info.get("target_column"):
        columns.append(task_info["target_column"])
    if task_info.get("feature_columns"):
        columns.extend(task_info["feature_columns"])
    if task_info.get("time_column"):
        columns.append(task_info["time_column"])
    
    # If no columns specified, use all
    if not columns:
        columns = ["*"]
    
    filters = task_info.get("filters", "None")
    grouping = task_info.get("grouping", [])
    aggregations = task_info.get("aggregations", {})
    
    # Fill in template
    prompt = prompt_template.replace("{schema}", schema_str)
    prompt = prompt.replace("{task_info}", str(task_info))
    prompt = prompt.replace("{tables}", ", ".join(tables) if tables else "all")
    prompt = prompt.replace("{columns}", ", ".join(columns))
    prompt = prompt.replace("{filters}", str(filters))
    prompt = prompt.replace("{grouping}", ", ".join(grouping) if grouping else "None")
    prompt = prompt.replace("{aggregations}", str(aggregations))
    
    # Generate SQL
    try:
        sql = llm_client.complete(prompt, temperature=0.2)
        
        # Clean up the response
        sql = sql.strip()
        
        # Remove markdown code blocks if present
        if sql.startswith("```sql"):
            sql = sql[6:]
        elif sql.startswith("```"):
            sql = sql[3:]
        if sql.endswith("```"):
            sql = sql[:-3]
        
        sql = sql.strip()
        
        # Validate SQL
        sql = clean_sql_query(sql)
        
        return sql
    
    except Exception as e:
        raise RuntimeError(f"SQL generation failed: {str(e)}")


def fix_sql_error(
    sql: str,
    error_message: str,
    schema_info: Dict[str, List[Dict[str, str]]],
    llm_client: Optional[LLMClient] = None,
    max_attempts: int = 3
) -> str:
    """
    Attempt to fix a SQL query that caused an error.
    
    Args:
        sql: The SQL query that failed
        error_message: The error message from the database
        schema_info: Database schema information
        llm_client: Optional LLM client
        max_attempts: Maximum number of fix attempts
        
    Returns:
        Fixed SQL query
    """
    if llm_client is None:
        llm_client = create_llm_client()
    
    schema_str = format_schema_for_llm(schema_info)
    
    prompt = f"""The following SQL query produced an error. Fix the query to make it work.

Database Schema:
{schema_str}

Failed SQL Query:
{sql}

Error Message:
{error_message}

Requirements:
1. Keep the query's intent the same
2. Use only tables and columns from the schema
3. Generate ONLY SELECT queries
4. Return ONLY the corrected SQL query

Corrected SQL Query:"""
    
    try:
        fixed_sql = llm_client.complete(prompt, temperature=0.2)
        
        # Clean up
        fixed_sql = fixed_sql.strip()
        if fixed_sql.startswith("```sql"):
            fixed_sql = fixed_sql[6:]
        elif fixed_sql.startswith("```"):
            fixed_sql = fixed_sql[3:]
        if fixed_sql.endswith("```"):
            fixed_sql = fixed_sql[:-3]
        fixed_sql = fixed_sql.strip()
        
        # Validate
        fixed_sql = clean_sql_query(fixed_sql)
        
        return fixed_sql
    
    except Exception as e:
        raise RuntimeError(f"SQL fix failed: {str(e)}")


def generate_sql_with_retry(
    task_info: Dict[str, Any],
    schema_info: Dict[str, List[Dict[str, str]]],
    db_manager,
    llm_client: Optional[LLMClient] = None,
    max_retries: int = 3
) -> tuple[str, Optional[str]]:
    """
    Generate SQL with automatic error recovery.
    
    Args:
        task_info: Task understanding
        schema_info: Database schema
        db_manager: DatabaseManager instance to test queries
        llm_client: Optional LLM client
        max_retries: Maximum retry attempts
        
    Returns:
        Tuple of (successful_sql, error_message or None)
    """
    if llm_client is None:
        llm_client = create_llm_client()
    
    # Generate initial SQL
    sql = generate_sql(task_info, schema_info, llm_client)
    
    for attempt in range(max_retries):
        try:
            # Test the query
            _ = db_manager.run_query(sql)
            # Success!
            return sql, None
        
        except Exception as e:
            error_message = str(e)
            
            if attempt < max_retries - 1:
                # Try to fix
                print(f"SQL error (attempt {attempt + 1}): {error_message}")
                print("Attempting to fix...")
                sql = fix_sql_error(sql, error_message, schema_info, llm_client)
            else:
                # Final attempt failed
                return sql, error_message
    
    return sql, "Max retries exceeded"
