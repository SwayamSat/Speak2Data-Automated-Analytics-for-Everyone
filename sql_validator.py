"""
SQL Validator module to validate and fix SQL queries against database schema.
"""

import re
from typing import Dict, List, Optional


class SQLValidator:
    """Validates and fixes SQL queries against database schema."""
    
    def __init__(self, schema_info: Dict[str, List[str]]):
        """Initialize validator with database schema.
        
        Args:
            schema_info: Dictionary mapping table names to their column lists
        """
        self.schema_info = schema_info
        self.tables = {name.lower(): name for name in schema_info.get("tables", {}).keys()}
        self.table_columns = {}
        for table_name, columns in schema_info.get("tables", {}).items():
            self.table_columns[table_name.lower()] = {
                col.lower(): col for col in columns
            }
    
    def validate_and_fix_query(self, sql_query: str) -> tuple[str, Optional[str]]:
        """Validate SQL query and fix table/column references.
        
        Args:
            sql_query: SQL query to validate and fix
            
        Returns:
            Tuple of (fixed_query, error_message)
        """
        try:
            fixed_query = sql_query
            
            # Extract table names from query
            from_match = re.search(r'FROM\s+(\w+)', fixed_query, re.IGNORECASE)
            join_matches = re.findall(r'JOIN\s+(\w+)', fixed_query, re.IGNORECASE)
            
            tables_in_query = []
            if from_match:
                tables_in_query.append(from_match.group(1))
            tables_in_query.extend(join_matches)
            
            # Fix table names - ensure they exist in schema
            for table in tables_in_query:
                table_lower = table.lower()
                if table_lower in self.tables:
                    # Table exists, use correct case
                    correct_table = self.tables[table_lower]
                    if table != correct_table:
                        fixed_query = re.sub(
                            rf'\b{table}\b',
                            correct_table,
                            fixed_query,
                            flags=re.IGNORECASE
                        )
                else:
                    # Table doesn't exist - find closest match
                    closest = self._find_closest_table(table)
                    if closest:
                        fixed_query = re.sub(
                            rf'\b{table}\b',
                            closest,
                            fixed_query,
                            flags=re.IGNORECASE
                        )
                    else:
                        return sql_query, f"Table '{table}' doesn't exist. Available tables: {', '.join(self.tables.values())}"
            
            # Fix column references in SELECT clause
            fixed_query = self._fix_columns_in_select(fixed_query)
            
            # Fix column references in WHERE clause
            fixed_query = self._fix_columns_in_where(fixed_query)
            
            # Fix column references in GROUP BY, ORDER BY
            fixed_query = self._fix_columns_in_clauses(fixed_query)
            
            return fixed_query, None
            
        except Exception as e:
            return sql_query, f"Validation error: {str(e)}"
    
    def _find_closest_table(self, table_name: str) -> Optional[str]:
        """Find closest matching table name from schema.
        
        Args:
            table_name: Table name to match
            
        Returns:
            Closest matching table name or None
        """
        table_lower = table_name.lower()
        
        # Exact match (case insensitive)
        if table_lower in self.tables:
            return self.tables[table_lower]
        
        # Partial match
        for schema_table in self.tables.keys():
            if table_lower in schema_table or schema_table in table_lower:
                return self.tables[schema_table]
        
        return None
    
    def _find_closest_column(self, column_name: str, table_name: Optional[str] = None) -> Optional[str]:
        """Find closest matching column name from schema.
        
        Args:
            column_name: Column name to match
            table_name: Optional table name to limit search
            
        Returns:
            Closest matching column name or None
        """
        column_lower = column_name.lower()
        
        if table_name:
            table_lower = table_name.lower()
            if table_lower in self.table_columns:
                cols = self.table_columns[table_lower]
                if column_lower in cols:
                    return cols[column_lower]
                
                # Partial match
                for col_key, col_val in cols.items():
                    if column_lower in col_key or col_key in column_lower:
                        return col_val
        else:
            # Search all tables
            for table_cols in self.table_columns.values():
                if column_lower in table_cols:
                    return table_cols[column_lower]
                
                # Partial match
                for col_key, col_val in table_cols.items():
                    if column_lower in col_key or col_key in column_lower:
                        return col_val
        
        return None
    
    def _fix_columns_in_select(self, query: str) -> str:
        """Fix column references in SELECT clause.
        
        Args:
            query: SQL query
            
        Returns:
            Fixed SQL query
        """
        # Find SELECT clause
        select_match = re.search(r'SELECT\s+(.*?)\s+FROM', query, re.IGNORECASE | re.DOTALL)
        if not select_match:
            return query
        
        select_clause = select_match.group(1)
        original_select = select_clause
        
        # Handle table.column references
        def fix_column_ref(match):
            table = match.group(1)
            column = match.group(2)
            table_lower = table.lower()
            
            if table_lower in self.table_columns:
                col_fixed = self._find_closest_column(column, table)
                if col_fixed:
                    return f"{table}.{col_fixed}"
            return match.group(0)
        
        select_clause = re.sub(r'(\w+)\.(\w+)', fix_column_ref, select_clause)
        
        # Handle standalone columns (without table prefix)
        def fix_standalone_column(match):
            column = match.group(0)
            # Skip if it's an alias or function
            if any(x in column.upper() for x in ['AS', 'COUNT', 'SUM', 'AVG', 'MAX', 'MIN']):
                return column
            
            col_fixed = self._find_closest_column(column)
            return col_fixed if col_fixed else column
        
        # Fix columns that aren't part of functions
        words = re.findall(r'\b\w+\b', select_clause)
        for word in words:
            if word.upper() not in ['AS', 'COUNT', 'SUM', 'AVG', 'MAX', 'MIN', 'SELECT', 'DISTINCT']:
                if not word.isdigit():
                    fixed = self._find_closest_column(word)
                    if fixed and fixed != word:
                        select_clause = re.sub(rf'\b{word}\b', fixed, select_clause, count=1)
        
        if select_clause != original_select:
            query = query.replace(original_select, select_clause)
        
        return query
    
    def _fix_columns_in_where(self, query: str) -> str:
        """Fix column references in WHERE clause.
        
        Args:
            query: SQL query
            
        Returns:
            Fixed SQL query
        """
        where_match = re.search(r'WHERE\s+(.*?)(?:\s+(?:GROUP|ORDER|LIMIT|$))', query, re.IGNORECASE | re.DOTALL)
        if not where_match:
            return query
        
        where_clause = where_match.group(1)
        original_where = where_clause
        
        # Fix table.column references in WHERE
        def fix_where_column(match):
            table = match.group(1)
            column = match.group(2)
            table_lower = table.lower()
            
            if table_lower in self.table_columns:
                col_fixed = self._find_closest_column(column, table)
                if col_fixed:
                    return f"{table}.{col_fixed}"
            return match.group(0)
        
        where_clause = re.sub(r'(\w+)\.(\w+)', fix_where_column, where_clause)
        
        if where_clause != original_where:
            query = query.replace(original_where, where_clause)
        
        return query
    
    def _fix_columns_in_clauses(self, query: str) -> str:
        """Fix column references in GROUP BY, ORDER BY clauses.
        
        Args:
            query: SQL query
            
        Returns:
            Fixed SQL query
        """
        # Fix GROUP BY
        group_by_match = re.search(r'GROUP BY\s+(\w+)', query, re.IGNORECASE)
        if group_by_match:
            column = group_by_match.group(1)
            fixed = self._find_closest_column(column)
            if fixed and fixed != column:
                query = re.sub(
                    rf'GROUP BY\s+{column}\b',
                    f'GROUP BY {fixed}',
                    query,
                    flags=re.IGNORECASE
                )
        
        # Fix ORDER BY
        order_by_match = re.search(r'ORDER BY\s+(\w+)', query, re.IGNORECASE)
        if order_by_match:
            column = order_by_match.group(1)
            fixed = self._find_closest_column(column)
            if fixed and fixed != column:
                query = re.sub(
                    rf'ORDER BY\s+{column}\b',
                    f'ORDER BY {fixed}',
                    query,
                    flags=re.IGNORECASE
                )
        
        return query

