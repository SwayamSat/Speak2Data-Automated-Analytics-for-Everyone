"""
Utility functions for Speak2Data platform.
Handles data processing, visualization, and common operations.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from typing import Dict, List, Any, Optional, Tuple
import json
from datetime import datetime, timedelta


class DataProcessor:
    """Handles data processing and transformation operations."""
    
    @staticmethod
    def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare DataFrame for analysis.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        # Make a copy to avoid modifying original
        clean_df = df.copy()
        
        # Convert date columns
        date_columns = clean_df.select_dtypes(include=['object']).columns
        for col in date_columns:
            if clean_df[col].dtype == 'object':
                try:
                    clean_df[col] = pd.to_datetime(clean_df[col], errors='ignore')
                except:
                    pass
        
        # Handle missing values
        numeric_columns = clean_df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if clean_df[col].isnull().any():
                clean_df[col].fillna(clean_df[col].median(), inplace=True)
        
        categorical_columns = clean_df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if clean_df[col].isnull().any():
                clean_df[col].fillna(clean_df[col].mode()[0], inplace=True)
        
        return clean_df
    
    @staticmethod
    def detect_data_types(df: pd.DataFrame) -> Dict[str, str]:
        """Detect and categorize data types.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary mapping column names to data types
        """
        data_types = {}
        
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                data_types[col] = 'numeric'
            elif df[col].dtype == 'object':
                # Check if it's actually a date
                try:
                    pd.to_datetime(df[col].head(10))
                    data_types[col] = 'date'
                except:
                    data_types[col] = 'categorical'
            else:
                data_types[col] = 'other'
        
        return data_types
    
    @staticmethod
    def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
        """Get comprehensive data summary.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary containing data summary
        """
        summary = {
            "shape": df.shape,
            "columns": list(df.columns),
            "data_types": DataProcessor.detect_data_types(df),
            "missing_values": df.isnull().sum().to_dict(),
            "missing_percentage": (df.isnull().sum() / len(df) * 100).to_dict(),
            "numeric_summary": df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {},
            "categorical_summary": {}
        }
        
        # Categorical summary
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            summary["categorical_summary"][col] = {
                "unique_values": df[col].nunique(),
                "most_common": df[col].mode().iloc[0] if not df[col].mode().empty else None,
                "value_counts": df[col].value_counts().head(10).to_dict()
            }
        
        return summary


class VisualizationGenerator:
    """Generates various types of visualizations for data analysis."""
    
    @staticmethod
    def create_bar_chart(df: pd.DataFrame, x_col: str, y_col: str, 
                        title: str = "Bar Chart", color_col: str = None) -> go.Figure:
        """Create a bar chart.
        
        Args:
            df: DataFrame containing data
            x_col: Column name for x-axis
            y_col: Column name for y-axis
            title: Chart title
            color_col: Column name for color grouping
            
        Returns:
            Plotly figure object
        """
        fig = px.bar(df, x=x_col, y=y_col, color=color_col, title=title)
        fig.update_layout(
            xaxis_tickangle=-45,
            height=500,
            showlegend=True
        )
        return fig
    
    @staticmethod
    def create_line_chart(df: pd.DataFrame, x_col: str, y_col: str, 
                         title: str = "Line Chart", color_col: str = None) -> go.Figure:
        """Create a line chart.
        
        Args:
            df: DataFrame containing data
            x_col: Column name for x-axis
            y_col: Column name for y-axis
            title: Chart title
            color_col: Column name for color grouping
            
        Returns:
            Plotly figure object
        """
        fig = px.line(df, x=x_col, y=y_col, color=color_col, title=title)
        fig.update_layout(
            xaxis_tickangle=-45,
            height=500,
            showlegend=True
        )
        return fig
    
    @staticmethod
    def create_scatter_plot(df: pd.DataFrame, x_col: str, y_col: str, 
                           title: str = "Scatter Plot", color_col: str = None) -> go.Figure:
        """Create a scatter plot.
        
        Args:
            df: DataFrame containing data
            x_col: Column name for x-axis
            y_col: Column name for y-axis
            title: Chart title
            color_col: Column name for color grouping
            
        Returns:
            Plotly figure object
        """
        fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title=title)
        fig.update_layout(
            height=500,
            showlegend=True
        )
        return fig
    
    @staticmethod
    def create_histogram(df: pd.DataFrame, col: str, title: str = "Histogram", 
                        bins: int = 30) -> go.Figure:
        """Create a histogram.
        
        Args:
            df: DataFrame containing data
            col: Column name for histogram
            title: Chart title
            bins: Number of bins
            
        Returns:
            Plotly figure object
        """
        try:
            # Check if column exists and has data
            if col not in df.columns or df[col].isnull().all():
                raise ValueError(f"Column {col} not found or has no data")
            
            # Remove null values for histogram
            clean_data = df[col].dropna()
            if clean_data.empty:
                raise ValueError(f"Column {col} has no valid data after removing nulls")
            
            fig = px.histogram(clean_data, x=col, nbins=bins, title=title)
            fig.update_layout(height=500)
            return fig
        except Exception as e:
            # Return a simple error figure
            fig = go.Figure()
            fig.add_annotation(text=f"Error creating histogram: {str(e)}", 
                             xref="paper", yref="paper", x=0.5, y=0.5, 
                             showarrow=False, font_size=16)
            fig.update_layout(height=500, title=title)
            return fig
    
    @staticmethod
    def create_pie_chart(df: pd.DataFrame, col: str, title: str = "Pie Chart") -> go.Figure:
        """Create a pie chart.
        
        Args:
            df: DataFrame containing data
            col: Column name for pie chart
            title: Chart title
            
        Returns:
            Plotly figure object
        """
        try:
            # Check if column exists and has data
            if col not in df.columns or df[col].isnull().all():
                raise ValueError(f"Column {col} not found or has no data")
            
            # Remove null values
            clean_data = df[col].dropna()
            if clean_data.empty:
                raise ValueError(f"Column {col} has no valid data after removing nulls")
            
            value_counts = clean_data.value_counts()
            if len(value_counts) == 0:
                raise ValueError(f"Column {col} has no unique values")
            
            fig = px.pie(values=value_counts.values, names=value_counts.index, title=title)
            fig.update_layout(height=500)
            return fig
        except Exception as e:
            # Return a simple error figure
            fig = go.Figure()
            fig.add_annotation(text=f"Error creating pie chart: {str(e)}", 
                             xref="paper", yref="paper", x=0.5, y=0.5, 
                             showarrow=False, font_size=16)
            fig.update_layout(height=500, title=title)
            return fig
    
    @staticmethod
    def create_heatmap(df: pd.DataFrame, title: str = "Correlation Heatmap") -> go.Figure:
        """Create a correlation heatmap.
        
        Args:
            df: DataFrame containing numeric data
            title: Chart title
            
        Returns:
            Plotly figure object
        """
        try:
            numeric_df = df.select_dtypes(include=[np.number])
            
            if numeric_df.empty:
                raise ValueError("No numeric columns found for correlation heatmap")
            
            if len(numeric_df.columns) < 2:
                raise ValueError("Need at least 2 numeric columns for correlation heatmap")
            
            # Remove columns with all NaN values
            numeric_df = numeric_df.dropna(axis=1, how='all')
            
            if numeric_df.empty:
                raise ValueError("No valid numeric data for correlation heatmap")
            
            corr_matrix = numeric_df.corr()
            
            # Check if correlation matrix is valid
            if corr_matrix.isnull().all().all():
                raise ValueError("Correlation matrix contains only NaN values")
            
            fig = px.imshow(corr_matrix, 
                           text_auto=True, 
                           aspect="auto",
                           title=title)
            fig.update_layout(height=500)
            return fig
        except Exception as e:
            # Return a simple error figure
            fig = go.Figure()
            fig.add_annotation(text=f"Error creating heatmap: {str(e)}", 
                             xref="paper", yref="paper", x=0.5, y=0.5, 
                             showarrow=False, font_size=16)
            fig.update_layout(height=500, title=title)
            return fig
    
    @staticmethod
    def create_dashboard(df: pd.DataFrame, chart_configs: List[Dict[str, Any]]) -> go.Figure:
        """Create a multi-panel dashboard.
        
        Args:
            df: DataFrame containing data
            chart_configs: List of chart configuration dictionaries
            
        Returns:
            Plotly figure object with subplots
        """
        n_charts = len(chart_configs)
        rows = (n_charts + 1) // 2
        cols = 2 if n_charts > 1 else 1
        
        fig = make_subplots(
            rows=rows, 
            cols=cols,
            subplot_titles=[config.get('title', f'Chart {i+1}') for i, config in enumerate(chart_configs)]
        )
        
        for i, config in enumerate(chart_configs):
            row = (i // cols) + 1
            col = (i % cols) + 1
            
            chart_type = config.get('type', 'bar')
            x_col = config.get('x_col')
            y_col = config.get('y_col')
            color_col = config.get('color_col')
            
            if chart_type == 'bar':
                trace = go.Bar(x=df[x_col], y=df[y_col], name=config.get('title', f'Chart {i+1}'))
            elif chart_type == 'line':
                trace = go.Scatter(x=df[x_col], y=df[y_col], mode='lines', name=config.get('title', f'Chart {i+1}'))
            elif chart_type == 'scatter':
                trace = go.Scatter(x=df[x_col], y=df[y_col], mode='markers', name=config.get('title', f'Chart {i+1}'))
            else:
                continue
            
            fig.add_trace(trace, row=row, col=col)
        
        fig.update_layout(height=300 * rows, showlegend=False)
        return fig
    
    @staticmethod
    def auto_visualize(df: pd.DataFrame, target_col: str = None) -> List[go.Figure]:
        """Automatically generate appropriate visualizations for data.
        
        Args:
            df: DataFrame containing data
            target_col: Target column for analysis
            
        Returns:
            List of Plotly figure objects
        """
        figures = []
        
        try:
            # Check if DataFrame is empty
            if df.empty:
                return figures
            
            data_types = DataProcessor.detect_data_types(df)
            
            # Numeric columns
            numeric_cols = [col for col, dtype in data_types.items() if dtype == 'numeric']
            categorical_cols = [col for col, dtype in data_types.items() if dtype == 'categorical']
            
            # Create histograms for numeric columns
            for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                try:
                    if not df[col].isnull().all():  # Check if column has valid data
                        fig = VisualizationGenerator.create_histogram(df, col, f"Distribution of {col}")
                        figures.append(fig)
                except Exception as e:
                    print(f"Warning: Could not create histogram for {col}: {e}")
                    continue
            
            # Create bar charts for categorical columns
            for col in categorical_cols[:3]:  # Limit to first 3 categorical columns
                try:
                    if df[col].nunique() <= 20 and df[col].nunique() > 1:  # Only for columns with reasonable number of categories
                        fig = VisualizationGenerator.create_pie_chart(df, col, f"Distribution of {col}")
                        figures.append(fig)
                except Exception as e:
                    print(f"Warning: Could not create pie chart for {col}: {e}")
                    continue
            
            # Create correlation heatmap if multiple numeric columns
            if len(numeric_cols) > 1:
                try:
                    fig = VisualizationGenerator.create_heatmap(df, "Correlation Matrix")
                    figures.append(fig)
                except Exception as e:
                    print(f"Warning: Could not create heatmap: {e}")
            
            # Create scatter plot if target column is numeric
            if target_col and target_col in numeric_cols and len(numeric_cols) > 1:
                try:
                    other_numeric = [col for col in numeric_cols if col != target_col]
                    if other_numeric:
                        fig = VisualizationGenerator.create_scatter_plot(
                            df, other_numeric[0], target_col, 
                            f"{other_numeric[0]} vs {target_col}"
                        )
                        figures.append(fig)
                except Exception as e:
                    print(f"Warning: Could not create scatter plot: {e}")
        
        except Exception as e:
            print(f"Warning: Error in auto_visualize: {e}")
        
        return figures


class StreamlitHelpers:
    """Helper functions for Streamlit UI components."""
    
    @staticmethod
    def display_dataframe(df: pd.DataFrame, max_rows: int = 1000):
        """Display DataFrame in Streamlit with formatting.
        
        Args:
            df: DataFrame to display
            max_rows: Maximum number of rows to display
        """
        if len(df) > max_rows:
            st.warning(f"Showing first {max_rows} rows out of {len(df)} total rows")
            df_display = df.head(max_rows)
        else:
            df_display = df
        
        st.dataframe(df_display, use_container_width=True)
    
    @staticmethod
    def display_metrics(metrics: Dict[str, float], title: str = "Model Metrics"):
        """Display model metrics in Streamlit.
        
        Args:
            metrics: Dictionary containing metrics
            title: Title for the metrics section
        """
        st.subheader(title)
        
        cols = st.columns(len(metrics))
        for i, (metric_name, metric_value) in enumerate(metrics.items()):
            with cols[i]:
                st.metric(
                    label=metric_name.replace('_', ' ').title(),
                    value=f"{metric_value:.4f}" if isinstance(metric_value, (int, float)) else str(metric_value)
                )
    
    @staticmethod
    def display_feature_importance(feature_importance: Dict[str, float], 
                                 title: str = "Feature Importance"):
        """Display feature importance in Streamlit.
        
        Args:
            feature_importance: Dictionary mapping features to importance scores
            title: Title for the feature importance section
        """
        if not feature_importance:
            st.info("Feature importance not available for this model")
            return
        
        st.subheader(title)
        
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        # Create bar chart
        features, importances = zip(*sorted_features[:10])  # Top 10 features
        
        fig = go.Figure(data=[
            go.Bar(x=list(features), y=list(importances))
        ])
        
        fig.update_layout(
            title="Top 10 Most Important Features",
            xaxis_title="Features",
            yaxis_title="Importance Score",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def create_sidebar_filters(df: pd.DataFrame) -> Dict[str, Any]:
        """Create sidebar filters for DataFrame.
        
        Args:
            df: DataFrame to create filters for
            
        Returns:
            Dictionary containing filter values
        """
        filters = {}
        
        with st.sidebar:
            st.header("Filters")
            
            # Date filters
            date_columns = df.select_dtypes(include=['datetime64']).columns
            for col in date_columns:
                if not df[col].isnull().all():
                    min_date = df[col].min()
                    max_date = df[col].max()
                    
                    date_range = st.date_input(
                        f"Filter by {col}",
                        value=(min_date, max_date),
                        min_value=min_date,
                        max_value=max_date
                    )
                    
                    if len(date_range) == 2:
                        filters[col] = date_range
            
            # Categorical filters
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if df[col].nunique() <= 20:  # Only for columns with reasonable number of categories
                    unique_values = df[col].unique()
                    selected_values = st.multiselect(
                        f"Filter by {col}",
                        options=unique_values,
                        default=unique_values
                    )
                    filters[col] = selected_values
            
            # Numeric filters
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                min_val = float(df[col].min())
                max_val = float(df[col].max())
                
                # Handle case where min_val equals max_val
                if min_val == max_val:
                    # Use a small range around the single value
                    range_val = abs(min_val) if min_val != 0 else 1.0
                    adjusted_min = min_val - range_val * 0.1
                    adjusted_max = max_val + range_val * 0.1
                    
                    st.info(f"Column '{col}' has constant value: {min_val:.2f}")
                    range_values = (min_val, max_val)
                else:
                    range_values = st.slider(
                        f"Filter by {col}",
                        min_value=min_val,
                        max_value=max_val,
                        value=(min_val, max_val)
                    )
                filters[col] = range_values
        
        return filters
    
    @staticmethod
    def apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """Apply filters to DataFrame.
        
        Args:
            df: Original DataFrame
            filters: Dictionary containing filter values
            
        Returns:
            Filtered DataFrame
        """
        filtered_df = df.copy()
        
        for col, filter_value in filters.items():
            if col not in filtered_df.columns:
                continue
            
            if isinstance(filter_value, tuple) and len(filter_value) == 2:
                # Date or numeric range filter
                if filtered_df[col].dtype == 'datetime64[ns]':
                    filtered_df = filtered_df[
                        (filtered_df[col] >= pd.to_datetime(filter_value[0])) &
                        (filtered_df[col] <= pd.to_datetime(filter_value[1]))
                    ]
                else:
                    filtered_df = filtered_df[
                        (filtered_df[col] >= filter_value[0]) &
                        (filtered_df[col] <= filter_value[1])
                    ]
            elif isinstance(filter_value, list):
                # Categorical filter
                filtered_df = filtered_df[filtered_df[col].isin(filter_value)]
        
        return filtered_df


class ErrorHandler:
    """Handles errors and provides user-friendly messages."""
    
    @staticmethod
    def handle_database_error(error: Exception) -> str:
        """Handle database-related errors.
        
        Args:
            error: Exception object
            
        Returns:
            User-friendly error message
        """
        error_str = str(error).lower()
        
        if "syntax error" in error_str:
            return "There's a syntax error in the generated SQL query. Please try rephrasing your question."
        elif "no such table" in error_str:
            return "The requested table doesn't exist in the database. Please check your question."
        elif "no such column" in error_str:
            return "The requested column doesn't exist. Please check your question."
        elif "permission denied" in error_str:
            return "You don't have permission to access this data."
        else:
            return f"Database error: {str(error)}"
    
    @staticmethod
    def handle_ml_error(error: Exception) -> str:
        """Handle machine learning-related errors.
        
        Args:
            error: Exception object
            
        Returns:
            User-friendly error message
        """
        error_str = str(error).lower()
        
        if "not enough data" in error_str:
            return "Not enough data available for machine learning analysis. Please try with a larger dataset."
        elif "target variable" in error_str:
            return "Unable to identify a suitable target variable for prediction. Please specify what you want to predict."
        elif "feature" in error_str and "empty" in error_str:
            return "No suitable features found for analysis. Please check your data."
        else:
            return f"Machine learning error: {str(error)}"
    
    @staticmethod
    def handle_nlp_error(error: Exception) -> str:
        """Handle NLP-related errors.
        
        Args:
            error: Exception object
            
        Returns:
            User-friendly error message
        """
        error_str = str(error).lower()
        
        if "api key" in error_str:
            return "API key not configured. Please check your environment settings."
        elif "quota" in error_str:
            return "API quota exceeded. Please try again later."
        elif "network" in error_str:
            return "Network error. Please check your internet connection."
        else:
            return f"Natural language processing error: {str(error)}"
