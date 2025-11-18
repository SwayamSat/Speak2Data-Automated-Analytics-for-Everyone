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
3. Use EXACT table and column names from the schema (case-sensitive)
4. When selecting columns, use the actual column names from the schema
5. If columns list shows "*", select all columns using SELECT * FROM table_name
6. Include appropriate WHERE, GROUP BY, ORDER BY clauses as needed
7. Add LIMIT clause to prevent returning too many rows (default LIMIT 10000)
8. Return ONLY the SQL query, no explanations or markdown

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

IMPORTANT:
- Verify all column names exist in the schema
- If aggregations are specified, use GROUP BY for non-aggregated columns
- Ensure the query returns actual data, not NULL values
- Order results logically (by time column if available)

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
    
    # If no tables specified, get the first table from schema
    if not tables and schema_info:
        tables = list(schema_info.keys())[:1]
    
    # Determine columns to select
    columns = []
    
    # Add grouping columns first (they must be in SELECT if using GROUP BY)
    grouping = task_info.get("grouping", [])
    if grouping:
        columns.extend(grouping)
    
    # Add aggregation columns
    aggregations = task_info.get("aggregations", {})
    if aggregations:
        for col, agg_func in aggregations.items():
            if col not in columns:
                columns.append(f"{agg_func}({col})")
    
    # Add time column if specified
    if task_info.get("time_column") and task_info["time_column"] not in columns:
        columns.append(task_info["time_column"])
    
    # Add target column if specified
    if task_info.get("target_column") and task_info["target_column"] not in columns:
        columns.append(task_info["target_column"])
    
    # Add feature columns
    if task_info.get("feature_columns"):
        for col in task_info["feature_columns"]:
            if col not in columns and not any(f"({col})" in c for c in columns):
                columns.append(col)
    
    # If still no columns specified, use all columns from the first table
    if not columns:
        if tables and tables[0] in schema_info:
            table_columns = [col["name"] for col in schema_info[tables[0]]]
            columns = table_columns if table_columns else ["*"]
        else:
            columns = ["*"]
    
    filters = task_info.get("filters", "None")
    
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


def validate_sql_columns(sql: str, schema_info: Dict[str, List[Dict[str, str]]]) -> tuple[bool, str]:
    """
    Validate that SQL query references valid columns from the schema.
    
    Args:
        sql: SQL query to validate
        schema_info: Database schema information
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    import re
    
    # Get all column names from schema
    all_columns = set()
    for table_name, columns in schema_info.items():
        for col in columns:
            all_columns.add(col["name"].lower())
    
    # Extract column references from SQL (simplified)
    # Look for SELECT columns
    select_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql, re.IGNORECASE | re.DOTALL)
    if select_match:
        select_clause = select_match.group(1)
        
        # Skip if SELECT *
        if '*' in select_clause:
            return True, ""
        
        # Extract column names (basic parsing)
        # This won't catch all cases but helps with common issues
        columns_in_query = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', select_clause)
        
        # Check if at least some columns are valid
        valid_cols = [col for col in columns_in_query if col.lower() in all_columns]
        
        if not valid_cols and columns_in_query:
            # None of the columns match - might be a problem
            return False, f"No valid columns found in SELECT clause. Available columns: {', '.join(sorted(all_columns))}"
    
    return True, ""


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
    
    # Validate column references before execution
    is_valid, validation_error = validate_sql_columns(sql, schema_info)
    if not is_valid:
        print(f"SQL validation warning: {validation_error}")
        # Try to fix using the validation error
        try:
            sql = fix_sql_error(sql, validation_error, schema_info, llm_client)
        except Exception as e:
            print(f"Could not fix validation error: {e}")
    
    for attempt in range(max_retries):
        try:
            # Test the query
            result = db_manager.run_query(sql)
            
            # Check if query returned empty or all-null data
            if len(result) == 0:
                raise RuntimeError("Query returned no rows. The data might not exist or filters are too restrictive.")
            
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
