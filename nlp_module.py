"""
NLP module for Speak2Data platform.
Handles natural language processing using Google Gemini Pro API.
"""

import google.generativeai as genai
import json
import os
import pandas as pd
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class NLPProcessor:
    """Handles natural language processing using Google Gemini Pro."""
    
    def __init__(self):
        """Initialize the NLP processor with Gemini Pro."""
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Database schema for context
        self.schema_info = {
            "tables": {
                "customers": ["customer_id", "name", "email", "phone", "city", "state", "registration_date", "customer_segment"],
                "products": ["product_id", "name", "category", "subcategory", "price", "cost", "supplier", "launch_date"],
                "orders": ["order_id", "customer_id", "order_date", "total_amount", "status", "shipping_city", "shipping_state"],
                "order_items": ["order_item_id", "order_id", "product_id", "quantity", "unit_price", "total_price"],
                "sales": ["sale_id", "product_id", "customer_id", "sale_date", "quantity", "unit_price", "total_amount", "region", "sales_rep"]
            }
        }
    
    def parse_query(self, user_query: str) -> Dict[str, Any]:
        """Parse user query and extract intent, entities, and task type.
        
        Args:
            user_query: Natural language query from user
            
        Returns:
            Dictionary containing parsed query information
        """
        # Validate input
        if not user_query or not user_query.strip():
            return self._fallback_parse("empty query")
        
        prompt = f"""
        You are an expert business analyst and SQL query generator. Analyze the following business question and extract key information.

        Database Schema:
        {json.dumps(self.schema_info, indent=2)}

        User Query: "{user_query}"

        Please analyze this query and return a JSON response with the following structure:
        {{
            "intent": "data_retrieval|analytics|prediction|classification|clustering|trend_analysis",
            "entities": {{
                "tables": ["list of relevant table names"],
                "columns": ["list of relevant column names"],
                "filters": ["list of filter conditions mentioned"],
                "aggregations": ["list of aggregation functions mentioned (SUM, COUNT, AVG, etc.)"],
                "time_periods": ["list of time periods mentioned"],
                "categories": ["list of categories or groups mentioned"]
            }},
            "task_type": "sql_query|ml_analysis|both",
            "sql_requirements": {{
                "select_columns": ["list of columns to select"],
                "from_tables": ["list of tables to query"],
                "where_conditions": ["list of WHERE conditions"],
                "group_by": ["list of columns to group by"],
                "order_by": ["list of columns to order by"],
                "limit": "number if specified"
            }},
            "ml_requirements": {{
                "target_variable": "column name for prediction target",
                "features": ["list of feature columns"],
                "problem_type": "regression|classification|clustering|time_series",
                "objective": "description of what to predict or analyze"
            }},
            "visualization_suggestions": ["list of suggested chart types"],
            "explanation": "brief explanation of what the user is asking for"
        }}

        Guidelines:
        - If the query asks for data retrieval, set task_type to "sql_query"
        - If the query asks for predictions, forecasting, or ML analysis, set task_type to "ml_analysis"
        - If the query needs both data retrieval and ML analysis, set task_type to "both"
        - Be specific about table and column names based on the schema
        - Extract time periods, filters, and aggregations accurately
        - For ML tasks, identify the target variable and relevant features
        - Suggest appropriate visualizations based on the query intent

        Return only the JSON response, no additional text.
        """
        
        try:
            response = self.model.generate_content(prompt)
            # Extract JSON from response
            response_text = response.text.strip()
            
            # Clean up the response to extract JSON
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            
            parsed_result = json.loads(response_text)
            return parsed_result
            
        except json.JSONDecodeError as e:
            # Fallback parsing if JSON is malformed
            return self._fallback_parse(user_query)
        except Exception as e:
            # If API fails, use fallback
            print(f"Warning: Gemini API error: {str(e)}")
            return self._fallback_parse(user_query)
    
    def _fallback_parse(self, user_query: str) -> Dict[str, Any]:
        """Fallback parsing method if Gemini Pro fails.
        
        Args:
            user_query: Natural language query from user
            
        Returns:
            Basic parsed query information
        """
        query_lower = user_query.lower()
        
        # Determine intent
        if any(word in query_lower for word in ['predict', 'forecast', 'classify', 'cluster', 'trend']):
            intent = 'prediction'
            task_type = 'ml_analysis'
        elif any(word in query_lower for word in ['show', 'list', 'get', 'find', 'display', 'what', 'how many', 'count']):
            intent = 'data_retrieval'
            task_type = 'sql_query'
        else:
            intent = 'analytics'
            task_type = 'sql_query'  # Default to SQL query
        
        # Extract basic entities
        entities = {
            "tables": [],
            "columns": [],
            "filters": [],
            "aggregations": [],
            "time_periods": [],
            "categories": []
        }
        
        # Simple keyword matching for tables
        if any(word in query_lower for word in ['customer', 'customers']):
            entities["tables"].append("customers")
        if any(word in query_lower for word in ['product', 'products']):
            entities["tables"].append("products")
        if any(word in query_lower for word in ['order', 'orders']):
            entities["tables"].append("orders")
        if any(word in query_lower for word in ['sale', 'sales']):
            entities["tables"].append("sales")
        
        # If no tables detected, default to sales (most common)
        if not entities["tables"]:
            entities["tables"].append("sales")
        
        # Extract aggregations
        if any(word in query_lower for word in ['total', 'sum', 'add']):
            entities["aggregations"].append("SUM")
        if any(word in query_lower for word in ['count', 'number', 'how many']):
            entities["aggregations"].append("COUNT")
        if any(word in query_lower for word in ['average', 'avg', 'mean']):
            entities["aggregations"].append("AVG")
        
        # Extract columns based on common business terms
        if any(word in query_lower for word in ['amount', 'price', 'cost', 'revenue', 'sales']):
            entities["columns"].append("total_amount")
        if any(word in query_lower for word in ['category', 'type', 'kind']):
            entities["columns"].append("category")
        if any(word in query_lower for word in ['name', 'title']):
            entities["columns"].append("name")
        if any(word in query_lower for word in ['date', 'time', 'when']):
            entities["columns"].append("sale_date")
        
        return {
            "intent": intent,
            "entities": entities,
            "task_type": task_type,
            "sql_requirements": {
                "select_columns": entities["columns"] if entities["columns"] else ["*"],
                "from_tables": entities["tables"],
                "where_conditions": [],
                "group_by": [],
                "order_by": [],
                "limit": None
            },
            "ml_requirements": {
                "target_variable": None,
                "features": [],
                "problem_type": "regression",
                "objective": "Analyze data patterns"
            },
            "visualization_suggestions": ["bar_chart", "line_chart"],
            "explanation": f"User is asking for {intent} related to business data"
        }
    
    def generate_sql_query(self, parsed_query: Dict[str, Any]) -> str:
        """Generate SQL query based on parsed query information.
        
        Args:
            parsed_query: Parsed query information from parse_query
            
        Returns:
            SQL query string
        """
        try:
            response = self.model.generate_content(f"""
            You are an expert SQL query generator. Based on the following parsed query information, generate a proper SQL query.

            Parsed Query Information:
            {json.dumps(parsed_query, indent=2)}

            Database Schema:
            {json.dumps(self.schema_info, indent=2)}

            Generate a SQL query that:
            1. Uses the correct table and column names from the schema
            2. Implements proper JOINs where needed
            3. Includes appropriate WHERE conditions
            4. Uses correct aggregation functions
            5. Orders results appropriately
            6. Limits results if specified

            Return only the SQL query, no additional text or explanations.
            """)
            
            sql_query = response.text.strip()
            
            # Clean up the SQL query
            if sql_query.startswith('```sql'):
                sql_query = sql_query[6:]
            if sql_query.endswith('```'):
                sql_query = sql_query[:-3]
            
            return sql_query.strip()
            
        except Exception as e:
            # Fallback to simple SQL generation
            print(f"Warning: Gemini API error in SQL generation: {str(e)}")
            return self._generate_simple_sql(parsed_query)
    
    def _generate_simple_sql(self, parsed_query: Dict[str, Any]) -> str:
        """Generate simple SQL query as fallback.
        
        Args:
            parsed_query: Parsed query information
            
        Returns:
            Simple SQL query string
        """
        try:
            # Get basic information
            tables = parsed_query.get("sql_requirements", {}).get("from_tables", ["sales"])
            columns = parsed_query.get("sql_requirements", {}).get("select_columns", ["*"])
            aggregations = parsed_query.get("entities", {}).get("aggregations", [])
            
            # Use first table if multiple
            table = tables[0] if tables else "sales"
            
            # Build SELECT clause
            if aggregations and columns and columns != ["*"]:
                if "SUM" in aggregations and "total_amount" in columns:
                    select_clause = "SUM(total_amount) AS total_sales"
                elif "COUNT" in aggregations:
                    select_clause = "COUNT(*) AS count"
                elif "AVG" in aggregations and "total_amount" in columns:
                    select_clause = "AVG(total_amount) AS avg_sales"
                else:
                    select_clause = ", ".join(columns) if columns != ["*"] else "*"
            else:
                select_clause = ", ".join(columns) if columns != ["*"] else "*"
            
            # Build basic query
            sql = f"SELECT {select_clause} FROM {table}"
            
            # Add GROUP BY if aggregations
            if aggregations and "category" in columns:
                sql += " GROUP BY category"
            
            # Add ORDER BY
            if "total_amount" in columns or "total_sales" in select_clause:
                sql += " ORDER BY total_amount DESC"
            elif "count" in select_clause.lower():
                sql += " ORDER BY count DESC"
            
            # Add LIMIT
            sql += " LIMIT 100"
            
            return sql
            
        except Exception as e:
            # Ultimate fallback
            return "SELECT * FROM sales LIMIT 100"
    
    def explain_results(self, query: str, results: pd.DataFrame, query_type: str = "sql") -> str:
        """Generate natural language explanation of query results.
        
        Args:
            query: Original user query
            results: DataFrame containing query results
            query_type: Type of query ("sql" or "ml")
            
        Returns:
            Natural language explanation of results
        """
        # Convert DataFrame to summary for context
        results_summary = {
            "row_count": len(results),
            "columns": list(results.columns),
            "sample_data": results.head(3).to_dict('records') if len(results) > 0 else []
        }
        
        # Get basic stats for numeric columns
        numeric_stats = {}
        for col in results.columns:
            if pd.api.types.is_numeric_dtype(results[col]):
                numeric_stats[col] = {
                    "min": float(results[col].min()),
                    "max": float(results[col].max()),
                    "mean": float(results[col].mean()),
                    "sum": float(results[col].sum()) if len(results) > 0 else 0
                }
        
        prompt = f"""
        You are a business analyst explaining query results to a non-technical user.
        
        Original Query: "{query}"
        Query Type: {query_type}
        Results: {len(results)} rows, {len(results.columns)} columns
        Columns: {', '.join(results.columns)}
        Numeric Stats: {json.dumps(numeric_stats, indent=2)}
        
        Provide a SHORT, business-friendly explanation (1-2 sentences max):
        - What the data shows
        - Key insight or pattern
        - Business implication
        
        Be concise and focus on actionable insights.
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"API Error in explanation: {str(e)}")
            # Generate a comprehensive fallback explanation
            if len(results) > 0:
                numeric_cols = [col for col in results.columns if pd.api.types.is_numeric_dtype(results[col])]
                categorical_cols = [col for col in results.columns if not pd.api.types.is_numeric_dtype(results[col])]
                
                explanation = f"**📊 Data Analysis Results:**\n\n"
                explanation += f"• **Records Found:** {len(results):,} rows\n"
                explanation += f"• **Data Columns:** {len(results.columns)} fields\n"
                
                if numeric_cols:
                    # Calculate key statistics
                    total_sum = sum(results[col].sum() for col in numeric_cols)
                    max_col = max(numeric_cols, key=lambda col: results[col].sum())
                    max_value = results[max_col].sum()
                    
                    explanation += f"• **Key Metrics:** Total value of {total_sum:,.0f} across {len(numeric_cols)} numeric columns\n"
                    explanation += f"• **Top Performer:** {max_col} leads with {max_value:,.0f}\n"
                    
                    if len(numeric_cols) > 1:
                        avg_values = {col: results[col].mean() for col in numeric_cols}
                        explanation += f"• **Averages:** {', '.join([f'{col}: {avg_values[col]:,.0f}' for col in list(avg_values.keys())[:2]])}\n"
                
                if categorical_cols:
                    explanation += f"• **Categories:** {', '.join(categorical_cols[:3])}\n"
                
                explanation += f"\n**💡 Insight:** This data shows {len(results)} records with {len(numeric_cols)} numeric and {len(categorical_cols)} categorical fields, providing comprehensive business intelligence."
                
                return explanation
            else:
                return "**❌ No Results:** No data found matching your search criteria. Try adjusting your query or filters."
    
    def suggest_follow_up_questions(self, original_query: str, results: pd.DataFrame) -> List[str]:
        """Suggest follow-up questions based on the original query and results.
        
        Args:
            original_query: Original user query
            results: DataFrame containing query results
            
        Returns:
            List of suggested follow-up questions
        """
        # Get column info for better suggestions
        numeric_cols = [col for col in results.columns if pd.api.types.is_numeric_dtype(results[col])]
        categorical_cols = [col for col in results.columns if not pd.api.types.is_numeric_dtype(results[col])]
        
        prompt = f"""
        Based on this query and results, suggest 3-5 SHORT, actionable follow-up questions:
        
        Original Query: "{original_query}"
        Data: {len(results)} rows, columns: {', '.join(results.columns)}
        Numeric columns: {', '.join(numeric_cols)}
        Text columns: {', '.join(categorical_cols)}
        
        Suggest questions that:
        - Drill down into specific data segments
        - Compare different categories or time periods  
        - Ask for trends, predictions, or deeper analysis
        - Explore relationships between columns
        
        Keep questions SHORT and business-focused. Return as simple list, one per line.
        """
        
        try:
            response = self.model.generate_content(prompt)
            questions = [q.strip() for q in response.text.strip().split('\n') if q.strip()]
            return questions[:5]  # Limit to 5 questions
        except Exception as e:
            print(f"API Error in follow-up questions: {str(e)}")
            # Generate context-aware follow-up questions based on the data
            questions = []
            
            if numeric_cols:
                questions.append(f"What are the top performing {numeric_cols[0]} categories?")
                if len(numeric_cols) > 1:
                    questions.append(f"How do {numeric_cols[0]} and {numeric_cols[1]} compare?")
                questions.append(f"What's the average {numeric_cols[0]} per record?")
            
            if categorical_cols:
                questions.append(f"Show me breakdown by {categorical_cols[0]}")
                if len(categorical_cols) > 1:
                    questions.append(f"Compare {categorical_cols[0]} vs {categorical_cols[1]}")
            
            # Add general questions
            questions.extend([
                "What trends do you see in this data?",
                "Are there any outliers or unusual patterns?",
                "What predictions can we make?"
            ])
            
            return questions[:5]  # Return up to 5 questions
