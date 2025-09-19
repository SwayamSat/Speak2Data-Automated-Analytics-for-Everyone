import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import re

def clean_sql_query(sql_query: str) -> str:
    """Clean and format SQL query"""
    # Remove extra whitespace
    sql_query = re.sub(r'\s+', ' ', sql_query.strip())
    
    # Remove markdown formatting
    sql_query = re.sub(r'```sql\s*', '', sql_query)
    sql_query = re.sub(r'```\n?', '', sql_query)
    
    # Remove common LLM prefixes
    sql_query = re.sub(r'^(SQL Query:|Query:|Here\'s the SQL query:|The SQL query is:)', '', sql_query).strip()
    
    return sql_query

def validate_dataframe(df: pd.DataFrame) -> bool:
    """Validate if DataFrame is suitable for analysis"""
    if df is None or df.empty:
        return False
    
    if len(df) < 2:
        return False
    
    return True

def detect_data_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Detect and categorize column data types"""
    result = {
        'numeric': [],
        'categorical': [],
        'datetime': [],
        'text': [],
        'boolean': []
    }
    
    for col in df.columns:
        dtype = str(df[col].dtype)
        
        if 'int' in dtype or 'float' in dtype:
            result['numeric'].append(col)
        elif 'datetime' in dtype:
            result['datetime'].append(col)
        elif 'bool' in dtype:
            result['boolean'].append(col)
        elif df[col].nunique() < len(df) * 0.5 and df[col].nunique() <= 20:  # Less than 50% unique values and max 20 categories
            result['categorical'].append(col)
        else:
            result['text'].append(col)
    
    return result

def suggest_visualizations(df: pd.DataFrame) -> List[str]:
    """Suggest appropriate visualizations based on data types"""
    suggestions = []
    data_types = detect_data_types(df)
    
    # Bar chart for categorical data
    if data_types['categorical'] and data_types['numeric']:
        suggestions.append('bar_chart')
    
    # Line chart for time series
    if data_types['datetime'] and data_types['numeric']:
        suggestions.append('line_chart')
    
    # Histogram for numeric data
    if len(data_types['numeric']) > 0:
        suggestions.append('histogram')
    
    # Scatter plot for two numeric variables
    if len(data_types['numeric']) >= 2:
        suggestions.append('scatter_plot')
    
    # Pie chart for categorical data
    if data_types['categorical']:
        suggestions.append('pie_chart')
    
    # Box plot for numeric data with categories
    if data_types['categorical'] and data_types['numeric']:
        suggestions.append('box_plot')
    
    return suggestions

def format_number(num: float) -> str:
    """Format numbers for display"""
    if pd.isna(num):
        return "N/A"
    
    if abs(num) >= 1_000_000_000:
        return f"{num/1_000_000_000:.1f}B"
    elif abs(num) >= 1_000_000:
        return f"{num/1_000_000:.1f}M"
    elif abs(num) >= 1_000:
        return f"{num/1_000:.1f}K"
    else:
        return f"{num:.2f}"

def extract_insights(df: pd.DataFrame) -> List[str]:
    """Extract basic insights from DataFrame"""
    insights = []
    
    if df.empty:
        return insights
    
    # Basic statistics
    insights.append(f"Dataset contains {len(df)} rows and {len(df.columns)} columns")
    
    # Missing values
    missing_data = df.isnull().sum()
    if missing_data.sum() > 0:
        insights.append(f"Found {missing_data.sum()} missing values across {(missing_data > 0).sum()} columns")
    
    # Numeric insights
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        for col in numeric_cols:
            max_val = df[col].max()
            min_val = df[col].min()
            mean_val = df[col].mean()
            
            insights.append(f"{col}: Range {format_number(min_val)} to {format_number(max_val)}, average {format_number(mean_val)}")
    
    # Categorical insights
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        for col in categorical_cols:
            unique_count = df[col].nunique()
            most_common = df[col].mode().iloc[0] if not df[col].mode().empty else "N/A"
            insights.append(f"{col}: {unique_count} unique values, most common: {most_common}")
    
    return insights

def parse_date_column(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Parse and format date column"""
    try:
        df[date_col] = pd.to_datetime(df[date_col])
        return df
    except:
        return df

def handle_missing_values(df: pd.DataFrame, strategy: str = 'drop') -> pd.DataFrame:
    """Handle missing values in DataFrame"""
    if strategy == 'drop':
        return df.dropna()
    elif strategy == 'fill_mean':
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        return df
    elif strategy == 'fill_mode':
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            mode_value = df[col].mode()
            fill_value = mode_value.iloc[0] if not mode_value.empty else 'Unknown'
            df[col] = df[col].fillna(fill_value)
        return df
    
    return df

def validate_sql_query(query: str) -> Dict[str, Any]:
    """Validate SQL query for safety and correctness"""
    result = {
        'is_valid': True,
        'is_safe': True,
        'warnings': [],
        'errors': []
    }
    
    query_upper = query.upper().strip()
    
    # Check for dangerous operations
    dangerous_operations = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE', 'TRUNCATE']
    for op in dangerous_operations:
        if op in query_upper:
            result['is_safe'] = False
            result['errors'].append(f"Potentially dangerous operation: {op}")
    
    # Check for basic SQL structure
    if not any(keyword in query_upper for keyword in ['SELECT', 'WITH']):
        result['is_valid'] = False
        result['errors'].append("Query must contain SELECT statement")
    
    # Check for common SQL injection patterns
    injection_patterns = ["';", "--", "/*", "*/", "xp_", "sp_"]
    for pattern in injection_patterns:
        if pattern in query:
            result['is_safe'] = False
            result['errors'].append(f"Potential SQL injection pattern: {pattern}")
    
    return result

def clean_gemini_response(response_text: str) -> str:
    """Clean Gemini API response text"""
    # Remove markdown formatting
    text = re.sub(r'```json\s*', '', response_text)
    text = re.sub(r'```sql\s*', '', text)
    text = re.sub(r'```\n?', '', text)
    
    # Remove common prefixes
    prefixes = [
        'Here is the', 'Here\'s the', 'The answer is', 'Based on',
        'According to', 'SQL Query:', 'Query:', 'Result:'
    ]
    
    for prefix in prefixes:
        if text.strip().startswith(prefix):
            text = text.replace(prefix, '', 1).strip()
    
    return text.strip()

def create_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Create comprehensive data summary for LLM context"""
    summary = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'data_types': detect_data_types(df)
    }
    
    # Add sample data (first few rows)
    if not df.empty:
        summary['sample_data'] = df.head(3).to_dict('records')
    
    # Add numeric summaries
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        summary['numeric_summary'] = df[numeric_cols].describe().to_dict()
    
    return summary

def estimate_query_complexity(query: str) -> str:
    """Estimate the complexity of a natural language query"""
    query_lower = query.lower()
    
    complexity_indicators = {
        'simple': ['show', 'list', 'get', 'find'],
        'medium': ['group by', 'order by', 'aggregate', 'sum', 'count', 'average'],
        'complex': ['predict', 'forecast', 'analyze', 'trend', 'correlation', 'join', 'multiple tables']
    }
    
    for complexity, indicators in complexity_indicators.items():
        if any(indicator in query_lower for indicator in indicators):
            return complexity
    
    return 'simple'

def format_ml_results(results: Dict[str, Any]) -> str:
    """Format ML results for display"""
    if not results:
        return "No ML results available"
    
    formatted = []
    
    if 'model_performance' in results:
        perf = results['model_performance']
        formatted.append(f"Model: {perf.get('best_model', 'Unknown')}")
        formatted.append(f"Accuracy: {perf.get('accuracy', 0):.2%}")
    
    if 'predictions' in results:
        pred_df = results['predictions']
        formatted.append(f"Generated {len(pred_df)} predictions")
    
    return "; ".join(formatted)
