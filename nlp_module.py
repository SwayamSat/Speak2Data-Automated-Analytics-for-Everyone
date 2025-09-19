import google.generativeai as genai
import json
import re
import pandas as pd
from typing import Dict, Any, Optional

class NLPProcessor:
    def __init__(self):
        self.client = None
        self.model_name = "gemini-1.5-flash"  # Updated model name
        self.model = None
    
    def set_api_key(self, api_key: str):
        """Set Google AI API key and configure client"""
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(self.model_name)
            self.client = True
        except Exception as e:
            print(f"Error setting up Gemini API: {e}")
            self.client = None
    
    def set_model(self, model_name: str):
        """Set the Gemini model to use"""
        self.model_name = model_name
        if self.client:
            self.model = genai.GenerativeModel(model_name)
    
    def test_connection(self) -> bool:
        """Test connection to Gemini API"""
        if not self.model:
            return False
        
        try:
            response = self.model.generate_content("Hello, respond with 'OK' if you can hear me.")
            return "OK" in response.text or "ok" in response.text.lower()
        except Exception as e:
            print(f"Connection test failed: {e}")
            return False
    
    def parse_query(self, natural_query: str, temperature: float = 0.1) -> Optional[Dict[str, Any]]:
        """Parse natural language query into structured format using Gemini"""
        if not self.model:
            raise ValueError("Gemini API not configured")
        
        prompt = f"""
        Analyze this business query and return ONLY a valid JSON object with these exact fields:

        Query to analyze: "{natural_query}"

        IMPORTANT: Respond with ONLY the JSON object, no explanations, no markdown, no extra text.

        Required JSON format:
        {{
            "query_type": "data_retrieval",
            "entities": ["table_names_mentioned"],
            "conditions": ["any_filters_mentioned"],
            "metrics": ["calculations_needed"],
            "requires_ml": false,
            "visualization_type": "bar",
            "intent": "brief description",
            "complexity": "simple"
        }}

        Choose query_type from: "data_retrieval", "analysis", "prediction", "aggregation"
        Choose visualization_type from: "bar", "line", "pie", "scatter", "histogram"
        Choose complexity from: "simple", "medium", "complex"
        Set requires_ml to true only if prediction/forecasting is explicitly requested.
        """
        
        try:
            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                top_p=0.8,
                top_k=40,
                max_output_tokens=1024,
            )
            
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            content = response.text.strip()
            print(f"Raw Gemini response: {content}")  # Debug output
            
            # Remove markdown formatting if present
            content = re.sub(r'```json\s*', '', content)
            content = re.sub(r'```\s*', '', content)
            content = content.strip()
            
            # Parse JSON
            parsed_data = json.loads(content)
            return parsed_data
        
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Raw response: {content}")
            # Try to extract JSON from response more robustly
            try:
                # Look for JSON-like structure
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    parsed_data = json.loads(json_str)
                    return parsed_data
                else:
                    # If no JSON found, create a basic response
                    return {
                        "query_type": "data_retrieval",
                        "entities": [],
                        "conditions": [],
                        "metrics": [],
                        "requires_ml": False,
                        "visualization_type": "bar",
                        "intent": "Data retrieval request",
                        "complexity": "simple"
                    }
            except Exception as fallback_error:
                print(f"Fallback parsing failed: {fallback_error}")
                # Return basic default response
                return {
                    "query_type": "data_retrieval",
                    "entities": [],
                    "conditions": [],
                    "metrics": [],
                    "requires_ml": False,
                    "visualization_type": "bar", 
                    "intent": "Data retrieval request",
                    "complexity": "simple"
                }
        except Exception as e:
            print(f"Error parsing query with Gemini: {e}")
            # Return basic default response instead of None
            return {
                "query_type": "data_retrieval",
                "entities": [],
                "conditions": [],
                "metrics": [],
                "requires_ml": False,
                "visualization_type": "bar",
                "intent": "Data retrieval request", 
                "complexity": "simple"
            }
    
    def generate_sql(self, natural_query: str, database_schema: str, 
                    max_tokens: int = 2048, temperature: float = 0.1) -> Optional[str]:
        """Generate SQL query from natural language using Gemini"""
        if not self.model:
            raise ValueError("Gemini API not configured")
        
        prompt = f"""
        You are an expert SQL developer. Given this database schema:
        
        {database_schema}
        
        Generate a valid SQL query for this request: "{natural_query}"
        
        Rules:
        - Return ONLY the SQL query without any explanation or markdown
        - Use proper SQL syntax for SQLite
        - Include appropriate JOINs if multiple tables are needed
        - Use aggregate functions (COUNT, SUM, AVG) when appropriate
        - Handle date/time queries properly
        - Use LIMIT if the query might return too many results
        - Ensure column names and table names match the schema exactly
        
        SQL Query:
        """
        
        try:
            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                top_p=0.8,
                top_k=40,
                max_output_tokens=max_tokens,
            )
            
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            sql_query = response.text.strip()
            
            # Remove markdown formatting if present - FIXED REGEX
            sql_query = re.sub(r'```sql\s*', '', sql_query)
            sql_query = re.sub(r'```\s*', '', sql_query)
            
            # Remove any extra text before or after the SQL
            lines = sql_query.split('\n')
            sql_lines = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('--') and not line.startswith('#'):
                    sql_lines.append(line)
            
            sql_query = ' '.join(sql_lines)
            
            # Basic validation
            if not any(keyword in sql_query.upper() for keyword in ['SELECT', 'INSERT', 'UPDATE', 'DELETE']):
                return None
                
            return sql_query
        
        except Exception as e:
            print(f"Error generating SQL with Gemini: {e}")
            return None
    
    def generate_insights(self, original_query: str, results_df: pd.DataFrame) -> str:
        """Generate natural language insights from query results using Gemini"""
        if not self.model or results_df.empty:
            return "No insights available."
        
        # Create a summary of the results
        try:
            summary_info = {
                "total_rows": len(results_df),
                "columns": list(results_df.columns),
                "sample_data": results_df.head(3).to_dict('records') if len(results_df) > 0 else [],
                "numeric_summary": results_df.describe().to_dict() if len(results_df.select_dtypes(include=['number']).columns) > 0 else {}
            }
            
            prompt = f"""
            Analyze these query results and provide business insights in 2-3 sentences.
            
            Original query: "{original_query}"
            
            Results summary:
            - Total rows: {summary_info['total_rows']}
            - Columns: {', '.join(summary_info['columns'])}
            - Sample data: {str(summary_info['sample_data'])[:500]}
            
            Provide clear, business-friendly insights about what these results reveal. Focus on:
            - Key patterns or trends
            - Notable findings
            - Business implications
            
            Keep it concise and actionable.
            """
            
            generation_config = genai.types.GenerationConfig(
                temperature=0.3,
                top_p=0.8,
                max_output_tokens=512,
            )
            
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            return response.text.strip()
            
        except Exception as e:
            print(f"Error generating insights: {e}")
            return "Results processed successfully."
    
    def explain_sql(self, sql_query: str) -> str:
        """Explain what an SQL query does in plain English"""
        if not self.model:
            return "SQL explanation not available."
        
        prompt = f"""
        Explain this SQL query in simple, business-friendly language:
        
        {sql_query}
        
        Describe what this query does in 1-2 sentences, focusing on:
        - What data is being retrieved
        - Any calculations or filters applied
        - The business purpose
        
        Keep it simple and avoid technical jargon.
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Error explaining SQL: {e}"
    
    def suggest_follow_up_queries(self, original_query: str, results_df: pd.DataFrame) -> list:
        """Suggest follow-up queries based on current results"""
        if not self.model or results_df.empty:
            return []
        
        columns = ', '.join(results_df.columns)
        
        prompt = f"""
        Based on this query: "{original_query}"
        And these result columns: {columns}
        
        Suggest 3 relevant follow-up questions a business user might ask.
        Return as a simple list, one question per line.
        
        Examples:
        - What are the trends over time?
        - How does this compare to other categories?
        - What are the top performers?
        """
        
        try:
            response = self.model.generate_content(prompt)
            suggestions = [line.strip().lstrip('- ') for line in response.text.split('\n') 
                          if line.strip() and not line.strip().startswith('Example')]
            return suggestions[:3]  # Return max 3 suggestions
        except Exception as e:
            print(f"Error generating suggestions: {e}")
            return []
