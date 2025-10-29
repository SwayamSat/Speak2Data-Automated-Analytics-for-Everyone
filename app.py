"""
Speak2Data: Automated Analytics for Everyone
Main Streamlit application for natural language data analysis.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import sys

# Import our custom modules
from db_module import DatabaseManager
from nlp_module import NLPProcessor
from sql_generator import SQLGenerator
from ml_pipeline_simple import SimpleMLPipeline as MLPipeline
from utils import DataProcessor, VisualizationGenerator, StreamlitHelpers, ErrorHandler

# Page configuration
st.set_page_config(
    page_title="Speak2Data",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 3rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    
    /* Modern ML Section Styling */
    .ml-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        margin: 20px 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .ml-header {
        color: white;
        text-align: center;
        margin: 0;
        font-size: 1.8rem;
        font-weight: 600;
    }
    
    .ml-subheader {
        color: #f0f0f0;
        text-align: center;
        margin: 10px 0;
        font-size: 1rem;
        opacity: 0.9;
    }
    
    /* Enhanced metric cards */
    .metric-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    
    /* Modern button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 10px 30px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Enhanced info boxes */
    .stInfo {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-left: 4px solid #2196f3;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Success message styling */
    .stSuccess {
        background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%);
        border-left: 4px solid #4caf50;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Error message styling */
    .stError {
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
        border-left: 4px solid #f44336;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Warning message styling */
    .stWarning {
        background: linear-gradient(135deg, #fff3e0 0%, #ffcc02 100%);
        border-left: 4px solid #ff9800;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'db_manager' not in st.session_state:
    st.session_state.db_manager = None
if 'nlp_processor' not in st.session_state:
    st.session_state.nlp_processor = None
if 'sql_generator' not in st.session_state:
    st.session_state.sql_generator = None
if 'ml_pipeline' not in st.session_state:
    st.session_state.ml_pipeline = None
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'current_results' not in st.session_state:
    st.session_state.current_results = None

def initialize_components():
    """Initialize all components of the application."""
    try:
        # Initialize database manager
        if st.session_state.db_manager is None:
            with st.spinner("Initializing database..."):
                st.session_state.db_manager = DatabaseManager()
        
        # Initialize NLP processor
        if st.session_state.nlp_processor is None:
            with st.spinner("Initializing NLP processor..."):
                st.session_state.nlp_processor = NLPProcessor()
        
        # Initialize SQL generator
        if st.session_state.sql_generator is None:
            st.session_state.sql_generator = SQLGenerator(st.session_state.nlp_processor)
        
        # Initialize ML pipeline
        if st.session_state.ml_pipeline is None:
            st.session_state.ml_pipeline = MLPipeline()
        
        return True
    except Exception as e:
        st.error(f"Failed to initialize components: {str(e)}")
        return False

def display_header():
    """Display the application header."""
    st.markdown('<h1 class="main-header">📊 Speak2Data</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Ask questions in natural language and get instant data insights</p>', unsafe_allow_html=True)

def display_sidebar():
    """Display the sidebar with options and information."""
    with st.sidebar:
        st.header("🔧 Options")
        
        # Database information
        if st.session_state.db_manager:
            st.subheader("📊 Database Info")
            schema = st.session_state.db_manager.get_table_schema()
            for table, columns in schema.items():
                with st.expander(f"Table: {table}"):
                    st.write(f"Columns: {', '.join(columns)}")
        
        # Sample queries
        st.subheader("💡 Sample Queries")
        sample_queries = [
            "Show me the total sales by product category",
            "What are the top 10 customers by order value?",
            "Predict customer churn based on order history",
            "Show me sales trends over the last 6 months",
            "Which products are most profitable?",
            "Cluster customers based on their purchase behavior"
        ]
        
        for i, query in enumerate(sample_queries):
            if st.button(f"💬 {query}", key=f"sample_{i}"):
                st.session_state.sample_query = query
                st.rerun()
        
        # Query history
        if st.session_state.query_history:
            st.subheader("📝 Recent Queries")
            for i, query in enumerate(st.session_state.query_history[-5:]):
                if st.button(f"🔄 {query[:50]}...", key=f"history_{i}"):
                    st.session_state.sample_query = query
                    st.rerun()

def generate_basic_explanation(results_df: pd.DataFrame) -> str:
    """Generate a basic explanation when API is unavailable."""
    if results_df.empty:
        return "No data found matching your criteria."
    
    # Get basic stats
    row_count = len(results_df)
    col_count = len(results_df.columns)
    
    # Find numeric columns for analysis
    numeric_cols = [col for col in results_df.columns if pd.api.types.is_numeric_dtype(results_df[col])]
    
    if numeric_cols:
        # Calculate totals and averages for numeric columns
        total_values = {}
        avg_values = {}
        for col in numeric_cols:
            total_values[col] = results_df[col].sum()
            avg_values[col] = results_df[col].mean()
        
        # Find the column with highest total
        max_col = max(total_values.keys(), key=lambda k: total_values[k])
        
        explanation = f"**Data Summary:** Found {row_count:,} records with {col_count} columns. "
        explanation += f"The {max_col} column shows the highest total value of {total_values[max_col]:,.0f}, "
        explanation += f"with an average of {avg_values[max_col]:,.0f} per record. "
        
        if len(numeric_cols) > 1:
            other_cols = [col for col in numeric_cols if col != max_col]
            explanation += f"Other key metrics include {', '.join(other_cols[:2])}."
    else:
        # For non-numeric data
        explanation = f"**Data Summary:** Retrieved {row_count:,} records with {col_count} columns: {', '.join(results_df.columns[:3])}"
        if col_count > 3:
            explanation += f" and {col_count-3} more columns."
    
    return explanation

def process_query(user_query: str):
    """Process user query and generate results."""
    try:
        # Validate input
        if not user_query or not user_query.strip():
            st.warning("Please enter a valid question.")
            return
        
        # Store the user query for explanation
        st.session_state.last_query = user_query.strip()
        
        # Clear sample query after processing
        if 'sample_query' in st.session_state:
            st.session_state.sample_query = None
        
        # Add to query history
        if user_query not in st.session_state.query_history:
            st.session_state.query_history.append(user_query)
        
        # Generate SQL query
        with st.spinner("Analyzing your question..."):
            try:
                query_result = st.session_state.sql_generator.generate_query(user_query)
            except Exception as e:
                st.error(f"❌ Query Analysis Error: {str(e)}")
                return
        
        # Debug information
        with st.expander("🔍 Debug Information", expanded=False):
            st.write("**Parsed Query:**")
            st.json(query_result.get("parsed_query", {}))
            st.write("**Generated SQL:**")
            st.code(query_result.get("sql_query", "No SQL generated"))
            st.write("**Is Valid:**")
            st.write(query_result.get("is_valid", False))
        
        if not query_result.get("is_valid", False):
            st.error(f"❌ {query_result.get('error', 'Invalid query generated')}")
            # Try to show what was generated anyway
            if query_result.get("sql_query"):
                st.info("Generated SQL (may have issues):")
                st.code(query_result["sql_query"], language="sql")
            return
        
        # Display query information
        st.subheader("🔍 Query Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Generated SQL:**")
            st.code(query_result["sql_query"], language="sql")
        
        with col2:
            st.write("**Query Type:**")
            st.info(query_result["query_type"])
            
            st.write("**Tables Used:**")
            st.info(", ".join(query_result["tables_used"]))
        
        # Execute SQL query
        with st.spinner("Executing query..."):
            try:
                results_df = st.session_state.db_manager.execute_query(query_result["sql_query"])
                st.session_state.current_results = results_df
            except Exception as e:
                st.error(f"❌ Database Error: {ErrorHandler.handle_database_error(e)}")
                return
        
        # Check if results are empty
        if results_df.empty:
            st.info("No results found for your query.")
            st.session_state.current_results = None
            return
        
        # Store results in session state for display in main function
        st.success(f"✅ Query executed successfully! Found {len(results_df)} rows.")
        st.rerun()  # Refresh to show results in main function
    
    except Exception as e:
        st.error(f"❌ Processing Error: {str(e)}")

def main():
    """Main application function."""
    # Display header
    display_header()
    
    # Initialize components
    if not initialize_components():
        st.stop()
    
    # Display sidebar
    display_sidebar()
    
    # Main content area
    st.subheader("💬 Ask Your Question")
    
    # Check for sample query from sidebar
    if 'sample_query' in st.session_state and st.session_state.sample_query:
        user_query = st.text_area(
            "Enter your business question in natural language:",
            value=st.session_state.sample_query,
            height=100,
            help="Examples: 'Show me sales by category', 'Predict customer churn', 'What are the top products?'"
        )
        # Don't clear the sample query here - let it be processed first
    else:
        user_query = st.text_area(
            "Enter your business question in natural language:",
            height=100,
            help="Examples: 'Show me sales by category', 'Predict customer churn', 'What are the top products?'"
        )
    
    # Process query button
    if st.button("🚀 Analyze Data", type="primary"):
        if user_query.strip():
            process_query(user_query.strip())
        else:
            st.warning("Please enter a question to analyze.")
    
    # Display current results if available
    if st.session_state.current_results is not None:
        results_df = st.session_state.current_results
        
        # Display data summary
        st.subheader("📊 Results Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", len(results_df))
        with col2:
            st.metric("Total Columns", len(results_df.columns))
        with col3:
            st.metric("Memory Usage", f"{results_df.memory_usage(deep=True).sum() / 1024:.1f} KB")
        
        # Display data table
        st.subheader("📋 Data Table")
        try:
            # Add filters
            filters = StreamlitHelpers.create_sidebar_filters(results_df)
            filtered_results = StreamlitHelpers.apply_filters(results_df, filters)
            
            # Display filtered results
            if not filtered_results.equals(results_df):
                st.info(f"Showing {len(filtered_results)} filtered results out of {len(results_df)} total")
            
            StreamlitHelpers.display_dataframe(filtered_results)
        except Exception as e:
            st.warning(f"Could not apply filters: {str(e)}")
            st.info("Displaying unfiltered results:")
            StreamlitHelpers.display_dataframe(results_df)
        
        # Generate visualizations
        st.subheader("📈 Visualizations")
        try:
            # Auto-generate visualizations
            figures = VisualizationGenerator.auto_visualize(results_df)
            
            if figures:
                for i, fig in enumerate(figures):
                    try:
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Add some spacing between charts
                        if i < len(figures) - 1:
                            st.markdown("---")
                    except Exception as chart_error:
                        st.warning(f"Could not display chart {i+1}: {str(chart_error)}")
            else:
                st.info("No suitable visualizations could be generated for this data.")
        
        except Exception as e:
            st.warning(f"Could not generate visualizations: {str(e)}")
        
        # Generate explanation
        st.subheader("💡 Explanation")
        try:
            # Get the actual user query from session state or use a default
            user_query = st.session_state.get('last_query', 'data analysis')
            explanation = st.session_state.nlp_processor.explain_results(
                user_query, results_df, "sql"
            )
            st.write(explanation)
        except Exception as e:
            st.warning(f"Could not generate explanation: {str(e)}")
            # Provide a basic explanation as fallback
            st.write(generate_basic_explanation(results_df))
        
        # Suggest follow-up questions
        st.subheader("🤔 Follow-up Questions")
        try:
            follow_up_questions = st.session_state.nlp_processor.suggest_follow_up_questions(
                "User query", results_df
            )
            
            for i, question in enumerate(follow_up_questions):
                if st.button(f"💬 {question}", key=f"followup_{i}"):
                    st.session_state.sample_query = question
                    st.rerun()
        except Exception as e:
            st.warning(f"Could not generate follow-up questions: {str(e)}")
        
        # ML Analysis section
        if len(results_df) > 10:  # Only show ML options for larger datasets
            st.markdown("---")
            st.markdown("### 🤖 Machine Learning Analysis")
            
            # Create a modern container for ML section
            with st.container():
                st.markdown("""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            padding: 20px; border-radius: 10px; margin: 10px 0;">
                    <h3 style="color: white; margin: 0; text-align: center;">🔬 Advanced Analytics</h3>
                    <p style="color: #f0f0f0; text-align: center; margin: 10px 0;">Unlock insights with machine learning</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Target column selection with modern styling
                col1, col2 = st.columns([2, 1])
                with col1:
                    target_column = st.selectbox(
                        "🎯 Select Target Variable",
                        options=results_df.columns.tolist(),
                        help="Choose the column you want to predict or analyze",
                        key="ml_target_select"
                    )
                
                with col2:
                    st.markdown("### 📊 Data Overview")
                    st.metric("Samples", len(results_df))
                    st.metric("Features", len(results_df.columns) - 1)
                
                if target_column:
                    # Data type and analysis info
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        data_type = "Numeric" if pd.api.types.is_numeric_dtype(results_df[target_column]) else "Categorical"
                        st.info(f"**Data Type:** {data_type}")
                    with col2:
                        unique_vals = results_df[target_column].nunique()
                        st.info(f"**Unique Values:** {unique_vals}")
                    with col3:
                        missing_vals = results_df[target_column].isnull().sum()
                        st.info(f"**Missing Values:** {missing_vals}")
                    
                    # ML Analysis button with modern styling
                    if st.button("🚀 Run Machine Learning Analysis", type="primary", use_container_width=True):
                        with st.spinner("🔬 Analyzing data and training model..."):
                            try:
                                # Analyze data
                                analysis = st.session_state.ml_pipeline.analyze_data(results_df, target_column)
                                
                                # Display analysis in modern cards
                                st.markdown("### 📈 Data Analysis Results")
                                
                                # Analysis metrics in cards
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Problem Type", analysis.get("problem_type", "Unknown").title())
                                with col2:
                                    st.metric("Numeric Columns", len(analysis.get("numeric_columns", [])))
                                with col3:
                                    st.metric("Categorical Columns", len(analysis.get("categorical_columns", [])))
                                with col4:
                                    missing_pct = (results_df.isnull().sum().sum() / (len(results_df) * len(results_df.columns))) * 100
                                    st.metric("Missing Data %", f"{missing_pct:.1f}%")
                                
                                # Data quality recommendations
                                if analysis.get("recommendations"):
                                    st.markdown("### 💡 Data Quality Recommendations")
                                    for rec in analysis["recommendations"]:
                                        st.warning(f"⚠️ {rec}")
                                
                                # Prepare data
                                features_df, target_series = st.session_state.ml_pipeline.prepare_data(
                                    results_df, target_column
                                )
                                
                                # Train model
                                problem_type = analysis.get("problem_type", "regression")
                                ml_results = st.session_state.ml_pipeline.train_model(
                                    features_df, target_series, problem_type
                                )
                                
                                if ml_results.get("training_successful", False):
                                    st.success("✅ Model trained successfully!")
                                    
                                    # Display results in modern format
                                    st.markdown("### 🎯 Model Performance")
                                    
                                    # Metrics in a nice layout
                                    metrics = ml_results.get("metrics", {})
                                    if metrics:
                                        col1, col2, col3, col4 = st.columns(4)
                                        metric_cols = list(metrics.keys())
                                        
                                        for i, (col, metric_key) in enumerate(zip([col1, col2, col3, col4], metric_cols[:4])):
                                            with col:
                                                value = metrics[metric_key]
                                                if isinstance(value, float):
                                                    st.metric(metric_key.replace("_", " ").title(), f"{value:.4f}")
                                                else:
                                                    st.metric(metric_key.replace("_", " ").title(), value)
                                    
                                    # Model summary in simple text format
                                    st.markdown("### 📋 Model Summary")
                                    
                                    summary = st.session_state.ml_pipeline.get_model_summary()
                                    
                                    # Create a modern info box for model summary
                                    st.markdown("""
                                    <div style="background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); 
                                                padding: 20px; border-radius: 10px; margin: 10px 0; 
                                                border-left: 4px solid #667eea;">
                                        <h4 style="color: #2c3e50; margin-top: 0;">🔬 Model Details</h4>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.markdown("**🎯 Model Configuration:**")
                                        st.write(f"• **Problem Type:** {summary.get('problem_type', 'Unknown').title()}")
                                        st.write(f"• **Model Type:** {summary.get('model_type', 'Unknown').replace('_', ' ').title()}")
                                        st.write(f"• **Target Column:** {summary.get('target_column', 'Unknown')}")
                                        
                                    with col2:
                                        st.markdown("**📊 Training Data:**")
                                        st.write(f"• **Training Samples:** {len(features_df):,}")
                                        st.write(f"• **Features Count:** {len(features_df.columns)}")
                                        st.write(f"• **Feature Columns:** {', '.join(summary.get('feature_columns', []))}")
                                    
                                    # Additional model information
                                    if summary.get('metrics'):
                                        st.markdown("**📈 Model Performance:**")
                                        metrics = summary.get('metrics', {})
                                        for metric_name, metric_value in metrics.items():
                                            if isinstance(metric_value, float):
                                                st.write(f"• **{metric_name.replace('_', ' ').title()}:** {metric_value:.4f}")
                                            else:
                                                st.write(f"• **{metric_name.replace('_', ' ').title()}:** {metric_value}")
                                    
                                    # Predictions visualization if available
                                    predictions = ml_results.get("predictions", [])
                                    if len(predictions) > 0:
                                        st.markdown("### 📊 Predictions Visualization")
                                        
                                        # Create a DataFrame for visualization
                                        try:
                                            # Ensure we have the same length for both arrays
                                            min_len = min(len(target_series), len(predictions))
                                            actual_vals = target_series.iloc[:min_len] if hasattr(target_series, 'iloc') else target_series[:min_len]
                                            pred_vals = predictions[:min_len]
                                            
                                            pred_df = pd.DataFrame({
                                                'Actual': actual_vals,
                                                'Predicted': pred_vals
                                            })
                                        except Exception as e:
                                            st.warning(f"Could not create predictions DataFrame: {str(e)}")
                                            pred_df = pd.DataFrame()
                                        
                                        # Scatter plot for regression
                                        if problem_type == "regression" and not pred_df.empty:
                                            try:
                                                import plotly.express as px
                                                import plotly.graph_objects as go
                                                
                                                fig = px.scatter(
                                                    pred_df, 
                                                    x='Actual', 
                                                    y='Predicted',
                                                    title="Actual vs Predicted Values",
                                                    labels={'Actual': 'Actual Values', 'Predicted': 'Predicted Values'}
                                                )
                                                
                                                # Add perfect prediction line
                                                min_val = float(pred_df['Actual'].min())
                                                max_val = float(pred_df['Actual'].max())
                                                fig.add_trace(go.Scatter(
                                                    x=[min_val, max_val],
                                                    y=[min_val, max_val],
                                                    mode='lines',
                                                    name='Perfect Prediction',
                                                    line=dict(dash='dash', color='red')
                                                ))
                                                
                                                fig.update_layout(showlegend=True)
                                                st.plotly_chart(fig, use_container_width=True)
                                            except Exception as e:
                                                st.warning(f"Could not create scatter plot: {str(e)}")
                                        
                                        # Confusion matrix for classification
                                        elif problem_type == "classification":
                                            try:
                                                from sklearn.metrics import confusion_matrix
                                                import plotly.express as px
                                                import plotly.graph_objects as go
                                                
                                                # Ensure we have the same length for both arrays
                                                min_len = min(len(target_series), len(predictions))
                                                y_true = target_series.iloc[:min_len] if hasattr(target_series, 'iloc') else target_series[:min_len]
                                                y_pred = predictions[:min_len]
                                                
                                                cm = confusion_matrix(y_true, y_pred)
                                                fig = px.imshow(
                                                    cm, 
                                                    text_auto=True,
                                                    title="Confusion Matrix",
                                                    labels=dict(x="Predicted", y="Actual")
                                                )
                                                st.plotly_chart(fig, use_container_width=True)
                                            except Exception as e:
                                                st.warning(f"Could not create confusion matrix: {str(e)}")
                                    
                                    # Feature importance if available
                                    if hasattr(st.session_state.ml_pipeline, 'model') and st.session_state.ml_pipeline.model:
                                        model = st.session_state.ml_pipeline.model
                                        if hasattr(model, 'feature_importances_'):
                                            try:
                                                import plotly.express as px
                                                
                                                st.markdown("### 🔍 Feature Importance")
                                                importance_df = pd.DataFrame({
                                                    'Feature': features_df.columns,
                                                    'Importance': model.feature_importances_
                                                }).sort_values('Importance', ascending=True)
                                                
                                                fig = px.bar(
                                                    importance_df, 
                                                    x='Importance', 
                                                    y='Feature',
                                                    orientation='h',
                                                    title="Feature Importance",
                                                    color='Importance',
                                                    color_continuous_scale='Viridis'
                                                )
                                                st.plotly_chart(fig, use_container_width=True)
                                            except Exception as e:
                                                st.warning(f"Could not create feature importance chart: {str(e)}")
                                
                                else:
                                    st.error(f"❌ ML Analysis failed: {ml_results.get('error', 'Unknown error')}")
                                    
                            except Exception as e:
                                st.error(f"❌ Error during ML analysis: {str(e)}")
                
                # Tips section with modern styling
                st.markdown("### 💡 Analysis Tips")
                tip_col1, tip_col2, tip_col3 = st.columns(3)
                
                with tip_col1:
                    st.markdown("""
                    **🎯 Target Selection:**
                    - Choose numeric columns for regression
                    - Choose categorical columns for classification
                    - Ensure sufficient data quality
                    """)
                
                with tip_col2:
                    st.markdown("""
                    **📊 Data Quality:**
                    - More data = better models
                    - Clean missing values first
                    - Check for outliers
                    """)
                
                with tip_col3:
                    st.markdown("""
                    **🔬 Model Types:**
                    - Regression: Predicting numbers
                    - Classification: Predicting categories
                    - Clustering: Finding patterns
                    """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Speak2Data** - Powered by Google Gemini Pro | Built with Streamlit | "
        "Data analysis made simple for everyone"
    )

if __name__ == "__main__":
    main()
