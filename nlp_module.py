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
import warnings
import logging

# Suppress gRPC/ALTS warnings
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GRPC_TRACE'] = ''
os.environ['GLOG_minloglevel'] = '2'

# Suppress warnings
warnings.filterwarnings('ignore')
logging.getLogger('google').setLevel(logging.ERROR)
logging.getLogger('grpc').setLevel(logging.ERROR)

# Load environment variables
load_dotenv()


class NLPProcessor:
    """Handles natural language processing using Google Gemini Pro."""
    
    def __init__(self, schema_info: Optional[Dict[str, Any]] = None):
        """Initialize the NLP processor with Gemini Pro.
        
        Args:
            schema_info: Optional database schema information. If not provided, uses default schema.
        """
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        genai.configure(api_key=api_key)
        
        # List available models and use the first one that works
        try:
            # Get available models
            available_models = [model.name.split('/')[-1] for model in genai.list_models()]
            
            # Prefer these models in order (avoid experimental models)
            preferred_models = ['gemini-pro', 'gemini-1.5-pro']
            
            # Find first available preferred model
            model_name = None
            for pref_model in preferred_models:
                if pref_model in available_models:
                    model_name = pref_model
                    break
            
            # If no preferred model found, use first available model
            if model_name is None and available_models:
                model_name = available_models[0]
            
            # If still no model, try gemini-pro anyway (might work)
            if model_name is None:
                model_name = 'gemini-pro'
            
            self.model = genai.GenerativeModel(model_name)
            self.model_name = model_name
            
        except Exception as e:
            # Fallback: try common model names directly
            fallback_models = ['gemini-pro', 'gemini-1.5-pro']
            self.model = None
            self.model_name = None
            
            for model_name in fallback_models:
                try:
                    self.model = genai.GenerativeModel(model_name)
                    self.model_name = model_name
                    break
                except Exception:
                    continue
            
            # If all models failed, raise error
            if self.model is None:
                raise ValueError(
                    f"Failed to initialize Gemini model. Error: {str(e)}\n"
                    "Please check:\n"
                    "1. Your GEMINI_API_KEY is correct\n"
                    "2. Your API key has access to Gemini models\n"
                    "3. Check available models at: https://ai.google.dev/models/gemini"
                )
        
        # Database schema for context - use provided schema or default
        if schema_info:
            self.schema_info = schema_info
        else:
            # Default schema for backward compatibility
            self.schema_info = {
                "tables": {
                    "customers": ["customer_id", "name", "email", "phone", "city", "state", "registration_date", "customer_segment"],
                    "products": ["product_id", "name", "category", "subcategory", "price", "cost", "supplier", "launch_date"],
                    "orders": ["order_id", "customer_id", "order_date", "total_amount", "status", "shipping_city", "shipping_state"],
                    "order_items": ["order_item_id", "order_id", "product_id", "quantity", "unit_price", "total_price"],
                    "sales": ["sale_id", "product_id", "customer_id", "sale_date", "quantity", "unit_price", "total_amount", "region", "sales_rep"]
                }
            }
    
    def _try_generate_content(self, prompt: str, max_retries: int = 3) -> Any:
        """Try to generate content, with rate limiting and model fallback on error.
        
        Args:
            prompt: Prompt to send to the model
            max_retries: Maximum number of retries
            
        Returns:
            Model response
            
        Raises:
            Exception: If all retries fail or quota is exceeded
        """
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                return self.model.generate_content(prompt)
            except Exception as e:
                error_str = str(e).lower()
                last_error = e
                
                # Check for quota/rate limit errors (429)
                if '429' in error_str or 'quota' in error_str or 'rate limit' in error_str or 'exceeded' in error_str:
                    # Extract retry delay if available
                    retry_delay = 60  # Default 60 seconds
                    if 'retry in' in error_str or 'retry_delay' in error_str:
                        try:
                            # Try to extract seconds from error message
                            import re
                            delay_match = re.search(r'retry in (\d+\.?\d*)\s*s', error_str)
                            if delay_match:
                                retry_delay = int(float(delay_match.group(1))) + 1
                        except:
                            pass
                    
                    # Raise a user-friendly error with retry info
                    raise Exception(
                        f"⚠️ API Quota Exceeded: You've reached the free tier limit for Gemini API.\n\n"
                        f"Please wait {retry_delay} seconds and try again, or:\n"
                        f"1. Check your usage at: https://ai.dev/usage?tab=rate-limit\n"
                        f"2. Upgrade your plan at: https://ai.google.dev/gemini-api/docs/rate-limits\n"
                        f"3. The app will use fallback suggestions until the quota resets."
                    )
                
                # Check if it's a model not found error
                if '404' in error_str or 'not found' in error_str or 'not supported' in error_str:
                    # Try to find an available model
                    if attempt < max_retries:
                        try:
                            # List available models
                            available_models = [model.name.split('/')[-1] for model in genai.list_models()]
                            
                            # Try preferred models in order (avoid experimental models)
                            preferred_models = ['gemini-pro', 'gemini-1.5-pro']
                            new_model_name = None
                            
                            for pref_model in preferred_models:
                                if pref_model in available_models and pref_model != self.model_name:
                                    new_model_name = pref_model
                                    break
                            
                            # If no preferred model available, try first available that's not experimental
                            if new_model_name is None and available_models:
                                for model_name in available_models:
                                    if (model_name != self.model_name and 
                                        'gemini' in model_name.lower() and 
                                        'exp' not in model_name.lower() and
                                        'experimental' not in model_name.lower()):
                                        new_model_name = model_name
                                        break
                            
                            if new_model_name:
                                try:
                                    self.model = genai.GenerativeModel(new_model_name)
                                    self.model_name = new_model_name
                                    time.sleep(1)  # Brief delay before retry
                                    continue  # Retry with new model
                                except Exception:
                                    pass
                        except Exception:
                            pass
                
                # For other errors, add a small delay before retry
                if attempt < max_retries:
                    time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
                    continue
                
                # If all retries exhausted, raise the error
                if attempt == max_retries:
                    raise last_error
        
        # Should never reach here, but just in case
        return self.model.generate_content(prompt)
    
    def update_schema(self, schema_info: Dict[str, Any]):
        """Update the database schema information.
        
        Args:
            schema_info: Dictionary containing table names as keys and column lists as values
        """
        # Convert schema dict to the format expected by schema_info
        if isinstance(schema_info, dict):
            # Check if it's already in the correct format
            if "tables" in schema_info:
                self.schema_info = schema_info
            else:
                # Convert simple dict format to schema_info format
                self.schema_info = {"tables": schema_info}
    
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
        
        # Format schema clearly for the AI
        schema_text = "\nDatabase Schema (EXACT tables and columns available):\n"
        for table_name, columns in self.schema_info.get("tables", {}).items():
            schema_text += f"  Table: {table_name}\n"
            schema_text += f"    Columns: {', '.join(columns)}\n"
        
        prompt = f"""
        You are an expert business analyst and SQL query generator. Analyze the following business question and extract key information.

        {schema_text}

        CRITICAL: The tables and columns listed above are the ONLY ones that exist in the database.
        Do NOT suggest or reference tables/columns that are NOT in the schema above.

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
        - CRITICAL: Only use table and column names that EXIST in the Database Schema above
        - Do NOT create or assume columns/tables that don't exist in the schema
        - Match user's intent to the closest matching tables/columns from the schema
        - Extract time periods, filters, and aggregations accurately
        - For ML tasks, identify the target variable and relevant features from the schema
        - Suggest appropriate visualizations based on the query intent

        IMPORTANT: If the user's question mentions tables/columns that don't exist in the schema, 
        find the closest matching tables/columns from the schema instead.

        Return only the JSON response, no additional text.
        """
        
        try:
            response = self._try_generate_content(prompt)
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
            # Format schema in a clear way for the AI
            schema_text = "Database Schema:\n"
            for table_name, columns in self.schema_info.get("tables", {}).items():
                schema_text += f"  - Table: {table_name}\n"
                schema_text += f"    Columns: {', '.join(columns)}\n"
            
            # Create a comprehensive schema reference for the AI
            schema_details = []
            for table_name, columns in self.schema_info.get("tables", {}).items():
                schema_details.append(f"Table '{table_name}' has columns: {', '.join(columns)}")
            
            schema_reference = "\n".join(schema_details)
            
            prompt_text = f"""
            You are an expert SQL query generator. Generate a SQLite query for the following database.

            DATABASE SCHEMA (ONLY THESE TABLES AND COLUMNS EXIST):
            {schema_reference}

            Parsed Query Information:
            {json.dumps(parsed_query, indent=2)}

            ABSOLUTELY CRITICAL RULES - FOLLOW THESE EXACTLY:
            1. ONLY use table names that appear in the DATABASE SCHEMA above
            2. ONLY use column names that appear in the corresponding table's column list above
            3. DO NOT use column names that are NOT listed in the schema
            4. DO NOT assume any columns exist beyond what's in the schema
            5. If the parsed query mentions a column that doesn't exist, find the closest match from the actual columns
            6. If the parsed query mentions a table that doesn't exist, find the closest match from the actual tables
            7. Use proper SQLite syntax
            8. Only JOIN tables that exist in the schema
            9. Always add LIMIT 100 to prevent large result sets

            STEP BY STEP PROCESS:
            1. Identify which table(s) from the schema match the user's intent
            2. Identify which column(s) from those table(s) match what the user wants
            3. Build a SELECT query using ONLY those tables and columns
            4. Add WHERE, GROUP BY, ORDER BY using ONLY columns from the schema
            5. Add JOINs only between tables that exist in the schema

            Generate the SQL query NOW using ONLY the tables and columns from the DATABASE SCHEMA above.

            Return ONLY the SQL query text, nothing else.
            """
            
            response = self._try_generate_content(prompt_text)
            sql_query = response.text.strip()
            
            # Clean up the SQL query - remove markdown formatting
            if sql_query.startswith('```sql'):
                sql_query = sql_query[6:]
            elif sql_query.startswith('```'):
                sql_query = sql_query[3:]
            if sql_query.endswith('```'):
                sql_query = sql_query[:-3]
            
            # Remove any explanatory text before or after SQL
            # Look for SQL keywords to find where the actual query starts
            sql_keywords = ['SELECT', 'WITH']
            query_start = -1
            for keyword in sql_keywords:
                idx = sql_query.upper().find(keyword)
                if idx != -1:
                    if query_start == -1 or idx < query_start:
                        query_start = idx
            
            if query_start > 0:
                sql_query = sql_query[query_start:]
            
            # Remove trailing text after semicolon
            if ';' in sql_query:
                sql_query = sql_query[:sql_query.index(';') + 1]
            
            return sql_query.strip()
            
        except Exception as e:
            # Fallback to simple SQL generation
            print(f"Warning: Gemini API error in SQL generation: {str(e)}")
            return self._generate_simple_sql(parsed_query)
    
    def _generate_simple_sql(self, parsed_query: Dict[str, Any]) -> str:
        """Generate simple SQL query as fallback using actual schema.
        
        Args:
            parsed_query: Parsed query information
            
        Returns:
            Simple SQL query string
        """
        try:
            # Get schema tables
            schema_tables = self.schema_info.get("tables", {})
            if not schema_tables:
                return "SELECT 1 LIMIT 1"  # Empty fallback
            
            # Get basic information
            tables = parsed_query.get("sql_requirements", {}).get("from_tables", [])
            columns = parsed_query.get("sql_requirements", {}).get("select_columns", [])
            aggregations = parsed_query.get("entities", {}).get("aggregations", [])
            
            # Find first available table from schema
            table = None
            if tables:
                for t in tables:
                    if t.lower() in [name.lower() for name in schema_tables.keys()]:
                        table = t
                        break
            
            # If no matching table, use first table from schema
            if not table:
                table = list(schema_tables.keys())[0]
            
            # Get columns for this table
            table_columns = schema_tables.get(table, [])
            if not table_columns:
                return f"SELECT * FROM {table} LIMIT 100"
            
            # Build SELECT clause using actual columns
            if aggregations and columns and columns != ["*"]:
                # Find columns that actually exist in the table
                valid_columns = [col for col in columns if col.lower() in [c.lower() for c in table_columns]]
                
                if "SUM" in aggregations:
                    # Find amount/total/price columns
                    amount_cols = [c for c in table_columns if any(x in c.lower() for x in ['amount', 'total', 'price', 'cost', 'value'])]
                    if amount_cols:
                        select_clause = f"SUM({amount_cols[0]}) AS total"
                    elif valid_columns:
                        select_clause = f"SUM({valid_columns[0]}) AS total"
                    else:
                        select_clause = "COUNT(*) AS count"
                elif "COUNT" in aggregations:
                    select_clause = "COUNT(*) AS count"
                elif "AVG" in aggregations:
                    # Find numeric columns
                    amount_cols = [c for c in table_columns if any(x in c.lower() for x in ['amount', 'total', 'price', 'cost', 'value'])]
                    if amount_cols:
                        select_clause = f"AVG({amount_cols[0]}) AS avg"
                    elif valid_columns:
                        select_clause = f"AVG({valid_columns[0]}) AS avg"
                    else:
                        select_clause = "COUNT(*) AS count"
                else:
                    select_clause = ", ".join(valid_columns[:5]) if valid_columns else "*"
            else:
                # Select first few columns or all
                if columns and columns != ["*"]:
                    valid_columns = [col for col in columns if col.lower() in [c.lower() for c in table_columns]]
                    select_clause = ", ".join(valid_columns[:5]) if valid_columns else ", ".join(table_columns[:5])
                else:
                    select_clause = ", ".join(table_columns[:5])  # Limit to 5 columns
            
            # Build basic query
            sql = f"SELECT {select_clause} FROM {table}"
            
            # Add GROUP BY if aggregations and we have category/type columns
            if aggregations:
                category_cols = [c for c in table_columns if any(x in c.lower() for x in ['category', 'type', 'status', 'segment', 'class'])]
                if category_cols:
                    sql += f" GROUP BY {category_cols[0]}"
            
            # Add ORDER BY using actual columns
            if "total" in select_clause.lower():
                amount_cols = [c for c in table_columns if any(x in c.lower() for x in ['amount', 'total', 'price', 'value'])]
                if amount_cols:
                    sql += f" ORDER BY {amount_cols[0]} DESC"
            elif "count" in select_clause.lower():
                sql += " ORDER BY count DESC"
            elif "avg" in select_clause.lower():
                amount_cols = [c for c in table_columns if any(x in c.lower() for x in ['amount', 'total', 'price', 'value'])]
                if amount_cols:
                    sql += f" ORDER BY avg DESC"
            
            # Add LIMIT
            sql += " LIMIT 100"
            
            return sql
            
        except Exception as e:
            # Ultimate fallback - use first table from schema
            schema_tables = self.schema_info.get("tables", {})
            if schema_tables:
                first_table = list(schema_tables.keys())[0]
                return f"SELECT * FROM {first_table} LIMIT 100"
            else:
                return "SELECT 1 LIMIT 1"
    
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
            response = self._try_generate_content(prompt)
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
            response = self._try_generate_content(prompt)
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
    
    def generate_query_suggestions(self, schema: Dict[str, List[str]], num_suggestions: int = 6) -> List[str]:
        """Generate AI-based query suggestions based on database schema using Gemini API.
        
        Args:
            schema: Dictionary mapping table names to their column lists
            num_suggestions: Number of suggestions to generate
            
        Returns:
            List of suggested natural language queries
        """
        try:
            # Format schema information clearly for the AI
            schema_details = []
            for table, columns in schema.items():
                schema_details.append(f"Table '{table}' with columns: {', '.join(columns)}")
            
            schema_text = "\n".join(schema_details)
            
            prompt = f"""
            You are a database query suggestion expert. Based on the EXACT database schema below, generate {num_suggestions} specific, relevant, and actionable business questions that users can ask about this database.

            DATABASE SCHEMA (These are the ONLY tables and columns available):
            {schema_text}

            Generate {num_suggestions} questions that:
            1. Use ONLY the tables and columns listed in the schema above
            2. Are specific to the actual data structure (don't assume columns that aren't in the schema)
            3. Are varied in type:
               - Questions asking for aggregations (totals, counts, averages)
               - Questions asking for top/best/worst performing items
               - Questions asking for trends or time-based analysis
               - Questions asking for comparisons or breakdowns
               - Questions asking for specific data lookups
            4. Are natural language questions (how a user would ask them)
            5. Are actionable and business-relevant

            IMPORTANT:
            - Only reference tables that exist: {', '.join(list(schema.keys()))}
            - Only reference columns that exist in those tables
            - Make questions specific to this database structure
            - Make each question unique and useful

            Return ONLY the {num_suggestions} questions, one per line, without numbering, bullets, or any other formatting.
            """
            
            response = self._try_generate_content(prompt)
            response_text = response.text.strip()
            
            # Parse questions from response
            questions = []
            for line in response_text.split('\n'):
                line = line.strip()
                if not line:
                    continue
                
                # Remove common prefixes (numbers, bullets, dashes)
                line = line.lstrip('0123456789.-•* ').strip()
                # Remove quotes if present
                line = line.strip('"\'')
                
                # Remove markdown formatting
                if line.startswith('```'):
                    continue
                
                # Only keep substantial questions
                if len(line) > 15 and not line.startswith('Note') and not line.startswith('Important'):
                    questions.append(line)
            
            # Clean up questions (remove numbering, bullets, etc.)
            cleaned_questions = []
            for q in questions:
                # Additional cleanup
                q = q.lstrip('0123456789.-•* ').strip()
                q = q.strip('"\'')
                if q and len(q) > 10 and q[0].isupper():  # Only keep substantial questions starting with capital
                    cleaned_questions.append(q)
            
            # Ensure we have enough suggestions
            if len(cleaned_questions) < num_suggestions:
                # Generate fallback suggestions based on schema
                fallback_suggestions = self._generate_fallback_suggestions(schema, num_suggestions - len(cleaned_questions))
                cleaned_questions.extend(fallback_suggestions)
            
            return cleaned_questions[:num_suggestions]
            
        except Exception as e:
            error_str = str(e).lower()
            # Check if it's a quota error
            if '429' in str(e) or 'quota' in error_str or 'rate limit' in error_str or 'exceeded' in error_str:
                # Log the error but don't print to console (user will see friendly message)
                print(f"⚠️ API Quota Exceeded - Using fallback suggestions")
                # Still return fallback suggestions so the app continues to work
                return self._generate_fallback_suggestions(schema, num_suggestions)
            else:
                print(f"API Error generating query suggestions: {str(e)}")
                # Fallback to schema-based suggestions
                return self._generate_fallback_suggestions(schema, num_suggestions)
    
    def _generate_fallback_suggestions(self, schema: Dict[str, List[str]], num_suggestions: int = 6) -> List[str]:
        """Generate fallback query suggestions based on schema analysis.
        
        Args:
            schema: Dictionary mapping table names to their column lists
            num_suggestions: Number of suggestions to generate
            
        Returns:
            List of suggested queries
        """
        suggestions = []
        
        # Analyze schema to generate relevant suggestions
        tables = list(schema.keys())
        
        for table in tables[:3]:  # Use first 3 tables
            columns = schema[table]
            
            # Find numeric columns
            numeric_cols = [col for col in columns if any(term in col.lower() for term in ['amount', 'price', 'cost', 'quantity', 'total', 'count', 'id'])]
            
            # Find categorical columns
            categorical_cols = [col for col in columns if any(term in col.lower() for term in ['category', 'type', 'status', 'segment', 'name', 'city', 'state', 'region'])]
            
            # Find date columns
            date_cols = [col for col in columns if 'date' in col.lower() or 'time' in col.lower()]
            
            # Generate suggestions based on column types
            if numeric_cols and categorical_cols:
                suggestions.append(f"Show me the total {numeric_cols[0]} by {categorical_cols[0]}")
            
            if categorical_cols and len(categorical_cols) > 1:
                suggestions.append(f"What are the top 10 items by {categorical_cols[0]}?")
            
            if date_cols and numeric_cols:
                suggestions.append(f"Show me {numeric_cols[0]} trends over time")
            
            if len(tables) > 1:
                suggestions.append(f"Compare data across {', '.join(tables[:2])}")
            
            if numeric_cols:
                suggestions.append(f"What is the average {numeric_cols[0]}?")
            
            suggestions.append(f"Analyze {table} data patterns")
        
        # Ensure we have enough suggestions
        if len(suggestions) < num_suggestions:
            suggestions.extend([
                "What are the top performing items?",
                "Show me data breakdown by category",
                "What trends can you identify?",
                "Compare performance across different segments"
            ])
        
        return suggestions[:num_suggestions]