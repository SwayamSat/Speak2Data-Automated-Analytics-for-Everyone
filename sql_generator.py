"""
SQL Generator module for Speak2Data platform.
Handles SQL query generation and optimization based on parsed natural language queries.
"""

import re
from typing import Dict, List, Any, Optional
from nlp_module import NLPProcessor

# Import SQL validator if available
try:
    from sql_validator import SQLValidator
except ImportError:
    SQLValidator = None


class SQLGenerator:
    """Generates SQL queries from parsed natural language queries."""
    
    def __init__(self, nlp_processor: NLPProcessor):
        """Initialize SQL generator with NLP processor.
        
        Args:
            nlp_processor: Instance of NLPProcessor for query parsing
        """
        self.nlp_processor = nlp_processor
        self.schema_info = nlp_processor.schema_info
    
    def generate_query(self, user_query: str) -> Dict[str, Any]:
        """Generate SQL query from natural language input.
        
        Args:
            user_query: Natural language query from user
            
        Returns:
            Dictionary containing SQL query and metadata
        """
        try:
            # Parse the query using NLP processor
            parsed_query = self.nlp_processor.parse_query(user_query)
            
            # Generate SQL query
            sql_query = self.nlp_processor.generate_sql_query(parsed_query)
            
            # Validate and optimize the query
            optimized_query = self._optimize_query(sql_query, parsed_query)
            
            # Validate and fix query against actual schema
            if SQLValidator:
                validator = SQLValidator(self.schema_info)
                fixed_query, validation_error = validator.validate_and_fix_query(optimized_query)
                if validation_error:
                    # Try to use fixed query even if there's a validation error
                    optimized_query = fixed_query
                else:
                    optimized_query = fixed_query
            
            return {
                "original_query": user_query,
                "parsed_query": parsed_query,
                "sql_query": optimized_query,
                "query_type": parsed_query.get("task_type", "sql_query"),
                "tables_used": self._extract_tables_from_sql(optimized_query),
                "columns_used": self._extract_columns_from_sql(optimized_query),
                "is_valid": self._validate_sql_syntax(optimized_query)
            }
            
        except Exception as e:
            return {
                "original_query": user_query,
                "error": str(e),
                "sql_query": None,
                "is_valid": False
            }
    
    def _optimize_query(self, sql_query: str, parsed_query: Dict[str, Any]) -> str:
        """Optimize SQL query for better performance and readability.
        
        Args:
            sql_query: Generated SQL query
            parsed_query: Parsed query information
            
        Returns:
            Optimized SQL query
        """
        # Basic query optimization
        optimized = sql_query.strip()
        
        # Remove trailing semicolons
        optimized = optimized.rstrip(';')
        
        # Ensure proper formatting
        optimized = re.sub(r'\s+', ' ', optimized)
        optimized = re.sub(r'\s*,\s*', ', ', optimized)
        optimized = re.sub(r'\s*=\s*', ' = ', optimized)
        optimized = re.sub(r'\s*>\s*=\s*', ' >= ', optimized)
        optimized = re.sub(r'\s*<\s*=\s*', ' <= ', optimized)
        optimized = re.sub(r'\s*!=\s*', ' != ', optimized)
        
        # Add LIMIT if not present and query might return many rows
        if 'LIMIT' not in optimized.upper() and parsed_query.get("intent") == "data_retrieval":
            # Make sure LIMIT is added before any semicolon
            if ';' in optimized:
                optimized = optimized.replace(';', ' LIMIT 1000;')
            else:
                optimized += " LIMIT 1000"
        
        # Ensure proper JOIN syntax
        optimized = self._fix_join_syntax(optimized)
        
        return optimized
    
    def _fix_join_syntax(self, sql_query: str) -> str:
        """Fix common JOIN syntax issues in generated SQL.
        
        Args:
            sql_query: SQL query to fix
            
        Returns:
            SQL query with fixed JOIN syntax
        """
        # Fix implicit joins
        query = sql_query
        
        # Build table relationships dynamically based on schema
        table_relationships = {}
        tables = self.schema_info.get("tables", {})
        
        # Look for common foreign key patterns (table_id columns)
        for table_name, columns in tables.items():
            for other_table_name, other_columns in tables.items():
                if table_name != other_table_name:
                    # Check for common FK patterns
                    fk_patterns = [
                        (f"{other_table_name}_id", f"{table_name}.{other_table_name}_id = {other_table_name}.{other_table_name.split('_')[0]}_id"),
                        (f"{other_table_name.split('_')[0]}_id", f"{table_name}.{other_table_name.split('_')[0]}_id = {other_table_name}.{other_table_name.split('_')[0]}_id"),
                        (f"{table_name}_id", f"{table_name}.{table_name}_id = {other_table_name}.{table_name}_id"),
                    ]
                    
                    for fk_col, join_condition in fk_patterns:
                        if fk_col in columns:
                            key = (table_name, other_table_name)
                            if key not in table_relationships:
                                table_relationships[key] = join_condition
                            break
        
        # Also check for common patterns like customer_id, product_id, etc.
        common_ids = ['customer_id', 'product_id', 'order_id', 'patient_id', 'doctor_id', 'account_id', 'loan_id', 'appointment_id', 'transaction_id']
        for table_name, columns in tables.items():
            for other_table_name, other_columns in tables.items():
                if table_name != other_table_name:
                    for common_id in common_ids:
                        if common_id in columns and common_id in other_columns:
                            key = (table_name, other_table_name)
                            if key not in table_relationships:
                                # Try to find the primary key pattern
                                pk_col = f"{other_table_name.split('_')[0]}_id" if other_table_name.endswith('s') else f"{other_table_name}_id"
                                if pk_col in other_columns or any(c.endswith('_id') for c in other_columns):
                                    join_condition = f"{table_name}.{common_id} = {other_table_name}.{common_id}"
                                    table_relationships[key] = join_condition
                                    break
        
        # Check if query has multiple tables but no explicit JOINs
        tables_in_query = []
        for table in tables.keys():
            if table.lower() in query.lower():
                tables_in_query.append(table)
        
        if len(tables_in_query) > 1 and 'JOIN' not in query.upper():
            # Add explicit JOINs if relationships found
            if table_relationships:
                query = self._add_explicit_joins(query, tables_in_query, table_relationships)
        
        return query
    
    def _add_explicit_joins(self, query: str, tables: List[str], relationships: Dict) -> str:
        """Add explicit JOIN clauses to SQL query.
        
        Args:
            query: SQL query
            tables: List of tables in query
            relationships: Dictionary of table relationships
            
        Returns:
            SQL query with explicit JOINs
        """
        # This is a simplified implementation
        # In a production system, you'd want more sophisticated JOIN logic
        
        if len(tables) >= 2:
            # Find the first table (usually in FROM clause)
            from_match = re.search(r'FROM\s+(\w+)', query, re.IGNORECASE)
            if from_match:
                base_table = from_match.group(1)
                
                # Add JOINs for other tables
                for table in tables:
                    if table != base_table:
                        relationship_key = (base_table, table) if (base_table, table) in relationships else (table, base_table)
                        if relationship_key in relationships:
                            join_condition = relationships[relationship_key]
                            query = query.replace(f"FROM {base_table}", f"FROM {base_table} JOIN {table} ON {join_condition}")
        
        return query
    
    def _extract_tables_from_sql(self, sql_query: str) -> List[str]:
        """Extract table names from SQL query.
        
        Args:
            sql_query: SQL query string
            
        Returns:
            List of table names used in query
        """
        tables = []
        
        # Find tables in FROM clause
        from_match = re.search(r'FROM\s+(\w+)', sql_query, re.IGNORECASE)
        if from_match:
            tables.append(from_match.group(1))
        
        # Find tables in JOIN clauses
        join_matches = re.findall(r'JOIN\s+(\w+)', sql_query, re.IGNORECASE)
        tables.extend(join_matches)
        
        return list(set(tables))  # Remove duplicates
    
    def _extract_columns_from_sql(self, sql_query: str) -> List[str]:
        """Extract column names from SQL query.
        
        Args:
            sql_query: SQL query string
            
        Returns:
            List of column names used in query
        """
        columns = []
        
        # Find columns in SELECT clause
        select_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql_query, re.IGNORECASE | re.DOTALL)
        if select_match:
            select_clause = select_match.group(1)
            # Split by comma and clean up
            column_list = [col.strip().split()[0] for col in select_clause.split(',')]
            columns.extend(column_list)
        
        # Find columns in WHERE clause
        where_match = re.search(r'WHERE\s+(.*?)(?:\s+GROUP\s+BY|\s+ORDER\s+BY|\s+LIMIT|$)', sql_query, re.IGNORECASE | re.DOTALL)
        if where_match:
            where_clause = where_match.group(1)
            # Extract column names from conditions
            column_matches = re.findall(r'(\w+\.\w+|\w+)\s*[=<>!]', where_clause)
            columns.extend(column_matches)
        
        return list(set(columns))  # Remove duplicates
    
    def _validate_sql_syntax(self, sql_query: str) -> bool:
        """Basic SQL syntax validation.
        
        Args:
            sql_query: SQL query to validate
            
        Returns:
            True if syntax appears valid, False otherwise
        """
        try:
            # Basic checks
            query_upper = sql_query.upper().strip()
            
            # Must start with SELECT
            if not query_upper.startswith('SELECT'):
                return False
            
            # Must have FROM clause
            if 'FROM' not in query_upper:
                return False
            
            # Check for balanced parentheses
            if query_upper.count('(') != query_upper.count(')'):
                return False
            
            # Check for basic SQL keywords
            required_keywords = ['SELECT', 'FROM']
            for keyword in required_keywords:
                if keyword not in query_upper:
                    return False
            
            # Check for common SQL injection patterns (basic security)
            dangerous_patterns = ['DROP', 'DELETE', 'INSERT', 'UPDATE', 'ALTER', 'CREATE', 'TRUNCATE']
            for pattern in dangerous_patterns:
                if pattern in query_upper:
                    return False
            
            return True
            
        except Exception:
            return False
    
    def get_query_explanation(self, sql_query: str) -> str:
        """Generate human-readable explanation of SQL query.
        
        Args:
            sql_query: SQL query to explain
            
        Returns:
            Human-readable explanation
        """
        try:
            # Extract basic information
            tables = self._extract_tables_from_sql(sql_query)
            columns = self._extract_columns_from_sql(sql_query)
            
            explanation_parts = []
            
            # Explain what data is being retrieved
            if tables:
                explanation_parts.append(f"This query retrieves data from {', '.join(tables)} table(s).")
            
            if columns:
                explanation_parts.append(f"It selects the following columns: {', '.join(columns[:5])}{'...' if len(columns) > 5 else ''}.")
            
            # Check for aggregations
            if any(func in sql_query.upper() for func in ['SUM', 'COUNT', 'AVG', 'MAX', 'MIN']):
                explanation_parts.append("The query includes aggregation functions to summarize the data.")
            
            # Check for grouping
            if 'GROUP BY' in sql_query.upper():
                explanation_parts.append("Results are grouped by specified categories.")
            
            # Check for ordering
            if 'ORDER BY' in sql_query.upper():
                explanation_parts.append("Results are sorted in a specific order.")
            
            # Check for filtering
            if 'WHERE' in sql_query.upper():
                explanation_parts.append("The query applies filters to narrow down the results.")
            
            return " ".join(explanation_parts) if explanation_parts else "This query retrieves data from the database."
            
        except Exception as e:
            return f"Query explanation unavailable: {str(e)}"
    
    def suggest_query_improvements(self, sql_query: str) -> List[str]:
        """Suggest improvements for SQL query.
        
        Args:
            sql_query: SQL query to analyze
            
        Returns:
            List of improvement suggestions
        """
        suggestions = []
        
        try:
            query_upper = sql_query.upper()
            
            # Check for missing LIMIT
            if 'LIMIT' not in query_upper and 'SELECT' in query_upper:
                suggestions.append("Consider adding a LIMIT clause to prevent returning too many rows.")
            
            # Check for missing WHERE clause on large tables
            if 'WHERE' not in query_upper and any(table in query_upper for table in ['SALES', 'ORDERS']):
                suggestions.append("Consider adding a WHERE clause to filter results, especially for large tables.")
            
            # Check for missing indexes (simplified check)
            if 'JOIN' in query_upper and 'WHERE' not in query_upper:
                suggestions.append("Consider adding WHERE conditions to improve JOIN performance.")
            
            # Check for SELECT *
            if 'SELECT *' in query_upper:
                suggestions.append("Consider selecting specific columns instead of using SELECT * for better performance.")
            
            return suggestions
            
        except Exception:
            return ["Unable to analyze query for improvements."]
