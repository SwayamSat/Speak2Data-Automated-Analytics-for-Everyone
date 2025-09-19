import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from nlp_module import NLPProcessor
from db_module import DatabaseManager
import json
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="NLP to SQL Converter",
    page_icon="⚡",
    layout="wide"
)

# Initialize components
@st.cache_resource
def initialize_components():
    nlp_processor = NLPProcessor()
    db_manager = DatabaseManager()
    return nlp_processor, db_manager

def create_automatic_visualizations(df):
    """Create automatic visualizations based on data types"""
    visualizations = []
    
    if df.empty:
        return visualizations
    
    # Get numeric and categorical columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Bar chart for categorical data
    if len(categorical_cols) > 0 and len(numeric_cols) > 0:
        cat_col = categorical_cols[0]
        num_col = numeric_cols[0]
        
        if len(df[cat_col].unique()) <= 20:  # Avoid too many categories
            agg_df = df.groupby(cat_col)[num_col].sum().reset_index()
            fig = px.bar(agg_df, x=cat_col, y=num_col, 
                        title=f"{num_col} by {cat_col}",
                        color=num_col,
                        color_continuous_scale="viridis")
            fig.update_layout(height=400)
            visualizations.append(fig)
    
    # Line chart for time series data
    date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    if len(date_cols) > 0 and len(numeric_cols) > 0:
        date_col = date_cols[0]
        num_col = numeric_cols[0]
        
        fig = px.line(df, x=date_col, y=num_col, 
                     title=f"{num_col} over time",
                     markers=True)
        fig.update_layout(height=400)
        visualizations.append(fig)
    
    # Histogram for numeric data
    if len(numeric_cols) > 0:
        num_col = numeric_cols[0]
        fig = px.histogram(df, x=num_col, 
                          title=f"Distribution of {num_col}",
                          color_discrete_sequence=['#1f77b4'])
        fig.update_layout(height=400)
        visualizations.append(fig)
    
    # Pie chart for categorical data with counts
    if len(categorical_cols) > 0:
        cat_col = categorical_cols[0]
        if len(df[cat_col].unique()) <= 10:  # Limit pie chart categories
            value_counts = df[cat_col].value_counts()
            fig = px.pie(values=value_counts.values, names=value_counts.index,
                        title=f"Distribution of {cat_col}")
            fig.update_layout(height=400)
            visualizations.append(fig)
    
    # Scatter plot for two numeric columns
    if len(numeric_cols) >= 2:
        x_col = numeric_cols[0]
        y_col = numeric_cols[1]
        fig = px.scatter(df, x=x_col, y=y_col,
                        title=f"{y_col} vs {x_col}")
        fig.update_layout(height=400)
        visualizations.append(fig)
    
    return visualizations

def main():
    st.title("Natural Language to SQL Converter")
    st.markdown("Convert natural language queries into SQL and retrieve data with automatic visualizations using Google's Gemini AI")
    
    # Load API key from environment
    api_key = os.getenv("GOOGLE_API_KEY")
    
    # Initialize components
    nlp_processor, db_manager = initialize_components()
    
    # Set up API key and model
    if api_key:
        nlp_processor.set_api_key(api_key)
        nlp_processor.set_model("gemini-1.5-flash")
    
    # Sidebar for database management
    with st.sidebar:
        st.header("Database Management")
        
        # Database initialization
        if st.button("Initialize Sample Database", key="init_db"):
            with st.spinner("Setting up sample database..."):
                db_manager.initialize_database()
                st.success("Sample database initialized!")
                st.rerun()
        
        # Show available tables
        tables = db_manager.get_table_names()
        if tables:
            st.subheader("Available Tables")
            for table in tables:
                st.write(f"• {table}")
                
                # Show sample data for each table
                with st.expander(f"View {table} sample"):
                    sample_df = db_manager.get_table_sample(table, 3)
                    if sample_df is not None:
                        st.dataframe(sample_df, use_container_width=True)
        
        # API Status
        st.subheader("API Status")
        if api_key:
            try:
                test_result = nlp_processor.test_connection()
                if test_result:
                    st.success("Gemini API Connected")
                else:
                    st.error("Gemini API Connection Failed")
            except Exception as e:
                st.error(f"API Error: {str(e)}")
        else:
            st.error("GOOGLE_API_KEY not found in .env file")
            st.info("Please add your Google AI API key to the .env file")
    
    # Main interface
    if not api_key:
        st.error("API Key Required: Please add your GOOGLE_API_KEY to the .env file to continue.")
        st.info("Get your API key from: https://makersuite.google.com/app/apikey")
        return
    
    # Check if database has tables
    if not tables:
        st.warning("Database not initialized: Please click 'Initialize Sample Database' in the sidebar to start.")
        return
    
    st.header("Natural Language Query Interface")
    
    # Example queries
    example_queries = [
        "Show me all customers from New York",
        "What are the total sales by month?",
        "Find customers with the highest purchase amounts",
        "Show product performance by category",
        "Which products have the best profit margins?",
        "Show customer distribution by state",
        "Find top 5 selling products",
        "Show order trends over time"
    ]
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_example = st.selectbox("Choose an example query:", 
                                      ["Custom query..."] + example_queries)
        
        if selected_example != "Custom query...":
            user_query = st.text_area("Your Query:", value=selected_example, height=100)
        else:
            user_query = st.text_area("Your Query:", 
                                    placeholder="e.g., Show me sales trends for the last quarter",
                                    height=100)
    
    with col2:
        st.subheader("Settings")
        temperature = st.slider("AI Creativity", 0.0, 1.0, 0.1, 
                               help="Higher values make output more creative")
        max_tokens = st.number_input("Max Response Length", 100, 4000, 1500,
                                   help="Maximum length of AI response")
    
    # Process query button
    if st.button("Convert & Execute", type="primary", use_container_width=True):
        if not user_query.strip():
            st.error("Please enter a query")
            return
        
        with st.spinner("Processing your query with Gemini AI..."):
            try:
                # Step 1: Parse natural language query
                st.subheader("1. Query Analysis")
                parsed_result = nlp_processor.parse_query(user_query, temperature=temperature)
                
                if parsed_result:
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Query Type", parsed_result.get('query_type', 'Unknown'))
                        st.metric("Complexity", parsed_result.get('complexity', 'Unknown'))
                    with col_b:
                        st.metric("Visualization", parsed_result.get('visualization_type', 'Auto'))
                        st.metric("Intent", parsed_result.get('intent', 'Data retrieval')[:30] + "...")
                    
                    with st.expander("View Detailed Analysis"):
                        st.json(parsed_result)
                    
                    # Step 2: Generate and execute SQL
                    st.subheader("2. SQL Generation & Execution")
                    sql_query = nlp_processor.generate_sql(user_query, db_manager.get_schema(), 
                                                          max_tokens=max_tokens, temperature=temperature)
                    
                    if sql_query:
                        st.code(sql_query, language="sql")
                        
                        # Explain SQL query
                        explanation = nlp_processor.explain_sql(sql_query)
                        st.info(f"Query Explanation: {explanation}")
                        
                        # Execute SQL
                        results_df = db_manager.execute_query(sql_query)
                        
                        if results_df is not None and not results_df.empty:
                            # Step 3: Display results
                            st.subheader("3. Query Results")
                            st.dataframe(results_df, use_container_width=True)
                            
                            # Show summary statistics
                            col_x, col_y = st.columns(2)
                            with col_x:
                                st.metric("Total Rows", len(results_df))
                            with col_y:
                                st.metric("Total Columns", len(results_df.columns))
                            
                            # Step 4: Generate insights
                            st.subheader("4. AI Insights")
                            insights = nlp_processor.generate_insights(user_query, results_df)
                            if insights:
                                st.info(f"Key Insights: {insights}")
                            
                            # Step 5: Automatic visualizations
                            st.subheader("5. Automatic Visualizations")
                            visualizations = create_automatic_visualizations(results_df)
                            
                            if visualizations:
                                for i, viz in enumerate(visualizations):
                                    st.plotly_chart(viz, use_container_width=True, key=f"viz_{i}")
                            else:
                                st.info("No suitable visualizations could be generated for this data.")
                            
                            # Suggest follow-up queries
                            st.subheader("Suggested Follow-up Queries")
                            suggestions = nlp_processor.suggest_follow_up_queries(user_query, results_df)
                            if suggestions:
                                for suggestion in suggestions:
                                    if st.button(f"• {suggestion}", key=f"suggest_{suggestion[:20]}"):
                                        st.rerun()
                        
                        else:
                            st.warning("Query executed successfully but returned no results")
                    else:
                        st.error("Failed to generate SQL query. Please try rephrasing your question.")
                        
                    # Since parse_query now always returns a valid result, we don't need this check
                    # The parsing will continue with default values if needed
            
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")
                with st.expander("Debug Information"):
                    st.write(f"Error Type: {type(e).__name__}")
                    st.write(f"Error Message: {str(e)}")
                    st.write(f"Model: gemini-1.5-flash")
                    st.write(f"Temperature: {temperature}")

    # Footer
    st.markdown("---")
    st.markdown("Project Status: Phase 1 Complete - NLP to SQL Conversion with Automatic Visualizations")
    st.markdown("Technology Stack: Streamlit • Google Gemini AI • SQLite • Plotly")

if __name__ == "__main__":
    main()