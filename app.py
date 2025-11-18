"""
Speak2Data: Natural Language to Automated Analytics and ML
Main Streamlit application orchestrating the complete pipeline.
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import traceback
from typing import Optional, Dict, Any

# Import project modules
from config import config
from db_manager import DatabaseManager
from llm_client import create_llm_client
from llm_task_understanding import infer_task, refine_task_understanding
from llm_sql_generator import generate_sql_with_retry
from data_preprocessing import preprocess_data
from ml_pipeline import run_pipeline
from visualization import create_visualizations
from experiment_logging import experiment_logger
from utils import get_column_statistics, format_schema_for_llm, truncate_text

# Page configuration
st.set_page_config(
    page_title="Speak2Data: NL to Analytics & ML",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session state initialization
if 'db_manager' not in st.session_state:
    st.session_state.db_manager = None
if 'schema_info' not in st.session_state:
    st.session_state.schema_info = None
if 'query_result' not in st.session_state:
    st.session_state.query_result = None
if 'llm_client' not in st.session_state:
    st.session_state.llm_client = None


def initialize_llm_client():
    """Initialize LLM client with API key from config/session state."""
    try:
        if st.session_state.llm_client is None:
            # Get API key from session state (loaded from .env) or config
            provider = st.session_state.get('llm_provider', config.llm_provider)
            api_key = st.session_state.get('api_key', config.get_llm_api_key())
            
            if not api_key:
                return None
            
            st.session_state.llm_client = create_llm_client(provider=provider, api_key=api_key)
        
        return st.session_state.llm_client
    
    except Exception as e:
        st.error(f"Failed to initialize LLM client: {e}")
        return None


def sidebar_data_source():
    """Sidebar section for data source configuration."""
    st.sidebar.header("📁 Data Source")
    
    source_type = st.sidebar.radio(
        "Choose data source:",
        ["Upload File", "Database URL"],
        key="source_type"
    )
    
    if source_type == "Upload File":
        uploaded_file = st.sidebar.file_uploader(
            "Upload data file",
            type=["csv", "xlsx", "xls", "db", "sqlite", "parquet"],
            help="Upload a CSV, Excel, SQLite, or Parquet file"
        )
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            temp_path = config.temp_db_dir / uploaded_file.name
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            try:
                st.session_state.db_manager = DatabaseManager(temp_path)
                st.sidebar.success(f"✅ Loaded: {uploaded_file.name}")
                
                # Get schema
                st.session_state.schema_info = st.session_state.db_manager.get_schema()
                
            except Exception as e:
                st.sidebar.error(f"Error loading file: {e}")
    
    else:
        db_url = st.sidebar.text_input(
            "Database URL",
            placeholder="sqlite:///path/to/db.db",
            help="Enter database connection string"
        )
        
        if st.sidebar.button("Connect"):
            if db_url:
                try:
                    st.session_state.db_manager = DatabaseManager(db_url)
                    st.sidebar.success("✅ Connected to database")
                    
                    # Get schema
                    st.session_state.schema_info = st.session_state.db_manager.get_schema()
                
                except Exception as e:
                    st.sidebar.error(f"Connection failed: {e}")
    
    # Show schema if available
    if st.session_state.schema_info:
        st.sidebar.subheader("📋 Database Schema")
        tables = st.session_state.db_manager.get_tables()
        
        selected_table = st.sidebar.selectbox("Select table to view:", tables)
        
        if selected_table:
            table_info = st.session_state.db_manager.get_table_info(selected_table)
            st.sidebar.write(f"**Rows:** {table_info['row_count']}")
            st.sidebar.write(f"**Columns:** {table_info['column_count']}")
            
            with st.sidebar.expander("View columns"):
                for col in table_info['columns']:
                    st.write(f"• {col['name']} ({col['type']})")


def sidebar_llm_config():
    """Sidebar section for LLM configuration."""
    st.sidebar.header("🤖 LLM Configuration")
    
    # Show current configuration from .env
    st.sidebar.info(f"**Provider:** {config.llm_provider}\n\n**Model:** {config.get_llm_model_name()}")
    
    # Check if API key is loaded
    api_key = config.get_llm_api_key()
    if api_key:
        st.sidebar.success("✅ API Key loaded from .env")
        # Store in session state for LLM client
        st.session_state.api_key = api_key
        st.session_state.llm_provider = config.llm_provider
    else:
        st.sidebar.error("❌ No API key found in .env file")
        st.sidebar.info("Add your API key to the .env file")
    
    # Optional: Allow override in UI (advanced users)
    with st.sidebar.expander("⚙️ Override Configuration"):
        provider = st.selectbox(
            "LLM Provider:",
            ["gemini", "openai", "anthropic"],
            index=["gemini", "openai", "anthropic"].index(config.llm_provider)
        )
        
        st.session_state.llm_provider = provider
        
        # Model selection
        model_options = {
            "gemini": ["gemini-2.0-flash-exp", "gemini-1.5-flash", "gemini-1.5-pro"],
            "openai": ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
            "anthropic": ["claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"]
        }
        
        selected_model = st.selectbox(
            "Model:",
            model_options.get(provider, []),
            help="Select the model to use"
        )
        
        if selected_model:
            if provider == "gemini":
                config.gemini_model = selected_model
            elif provider == "openai":
                config.openai_model = selected_model
            elif provider == "anthropic":
                config.anthropic_model = selected_model


def sidebar_ml_config():
    """Sidebar section for ML configuration."""
    with st.sidebar.expander("⚙️ ML Configuration"):
        config.test_size = st.slider(
            "Test set size:",
            0.1, 0.5, config.test_size,
            help="Proportion of data for testing"
        )
        
        config.missing_strategy = st.selectbox(
            "Missing value strategy:",
            ["drop", "impute"],
            index=["drop", "impute"].index(config.missing_strategy)
        )
        
        config.scaling_method = st.selectbox(
            "Feature scaling:",
            ["standard", "minmax", "robust", "none"],
            index=["standard", "minmax", "robust", "none"].index(config.scaling_method)
        )


def main_query_interface():
    """Main interface for natural language queries."""
    st.title("📊 Speak2Data: Natural Language to Analytics & ML")
    st.markdown("Ask questions about your data or request machine learning models in plain English.")
    
    # Check if data source and LLM are configured
    if not st.session_state.db_manager:
        st.warning("⚠️ Please configure a data source in the sidebar.")
        return
    
    llm_client = initialize_llm_client()
    if not llm_client:
        st.warning("⚠️ Please configure LLM API key in the sidebar.")
        return
    
    # Query input
    query = st.text_area(
        "Ask a question or request an analysis:",
        height=100,
        placeholder="Examples:\n- Show me monthly sales trends by region\n- Predict customer churn based on transaction history\n- Cluster patients into risk groups based on vitals",
        help="Enter your question in natural language"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        run_button = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
    
    if run_button and query:
        run_analysis_pipeline(query, llm_client)


def run_analysis_pipeline(query: str, llm_client):
    """Execute the complete analysis pipeline."""
    
    # Create progress container
    progress_container = st.container()
    
    with progress_container:
        st.subheader("🔄 Analysis Progress")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Task Understanding
            status_text.text("1/7 Understanding task...")
            progress_bar.progress(1/7)
            
            task_info = infer_task(query, st.session_state.schema_info, llm_client)
            st.success(f"✅ Task identified: {task_info['task_type']}")
            
            # Step 2: SQL Generation
            status_text.text("2/7 Generating SQL query...")
            progress_bar.progress(2/7)
            
            sql, error = generate_sql_with_retry(
                task_info,
                st.session_state.schema_info,
                st.session_state.db_manager,
                llm_client
            )
            
            if error:
                st.warning(f"SQL generation warning: {error}")
            
            st.success("✅ SQL query generated")
            
            # Step 3: Data Extraction
            status_text.text("3/7 Extracting data...")
            progress_bar.progress(3/7)
            
            df = st.session_state.db_manager.run_query(sql)
            st.success(f"✅ Retrieved {len(df)} rows, {len(df.columns)} columns")
            
            # Refine task understanding with actual columns
            task_info = refine_task_understanding(task_info, list(df.columns), df)
            
            # Auto-suggest target column based on column names and query for supervised tasks
            supervised_tasks = ['classification', 'regression', 'time_series_forecast']
            if task_info['task_type'] in supervised_tasks and not task_info.get('target_column'):
                # Try to infer from query and column names
                query_lower = query.lower()
                potential_targets = []
                
                # Look for keywords in query
                predict_keywords = ['predict', 'forecast', 'estimate', 'classify', 'determine']
                if any(keyword in query_lower for keyword in predict_keywords):
                    # Check if any column name appears after these keywords
                    for col in df.columns:
                        if col.lower() in query_lower:
                            potential_targets.append(col)
                
                # If still no target, suggest the last column or numeric columns
                if not potential_targets:
                    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                    if numeric_cols:
                        potential_targets = [numeric_cols[-1]]
                    else:
                        potential_targets = [df.columns[-1]]
                
                if potential_targets:
                    task_info['target_column'] = potential_targets[0]
            
            # Step 4: Data Preprocessing
            status_text.text("4/7 Preprocessing data...")
            progress_bar.progress(4/7)
            
            # Get target column if needed for supervised tasks
            target_column = task_info.get('target_column')
            
            supervised_tasks = ['classification', 'regression', 'time_series_forecast']
            if task_info['task_type'] in supervised_tasks and not target_column:
                # Ask user to select target
                st.warning(f"⚠️ {task_info['task_type'].replace('_', ' ').title()} requires a target column")
                target_column = st.selectbox(
                    "Select target column for prediction:",
                    options=df.columns.tolist(),
                    help="Choose the column you want to predict"
                )
                
                if target_column:
                    task_info['target_column'] = target_column
                else:
                    st.error("Please select a target column to continue")
                    return
            
            data_bundle = preprocess_data(
                df,
                task_info['task_type'],
                target_column=target_column,
                test_size=config.test_size,
                random_state=config.random_state
            )
            
            st.success("✅ Data preprocessed")
            
            # Step 5: Model Training
            status_text.text("5/7 Training models...")
            progress_bar.progress(5/7)
            
            pipeline_result = run_pipeline(task_info['task_type'], data_bundle)
            
            if pipeline_result.best_model_name != "None":
                st.success(f"✅ Best model: {pipeline_result.best_model_name}")
            else:
                st.info("ℹ️ No model training required for this task")
            
            # Step 6: Visualization
            status_text.text("6/7 Creating visualizations...")
            progress_bar.progress(6/7)
            
            figures = create_visualizations(task_info['task_type'], data_bundle, pipeline_result)
            st.success(f"✅ Created {len(figures)} visualizations")
            
            # Step 7: Logging
            status_text.text("7/7 Logging experiment...")
            progress_bar.progress(1.0)
            
            dataset_info = {
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": list(df.columns)
            }
            
            metrics = pipeline_result.metrics_comparison.to_dict() if not pipeline_result.metrics_comparison.empty else None
            
            experiment_logger.log_experiment(
                query=query,
                task_type=task_info['task_type'],
                task_understanding=task_info,
                sql_query=sql,
                dataset_info=dataset_info,
                model_name=pipeline_result.best_model_name,
                metrics=metrics,
                config_dict=config.to_dict(),
                success=True
            )
            
            st.success("✅ Analysis complete!")
            
            # Store results in session state
            st.session_state.query_result = {
                'query': query,
                'task_info': task_info,
                'sql': sql,
                'df': df,
                'data_bundle': data_bundle,
                'pipeline_result': pipeline_result,
                'figures': figures
            }
            
            # Clear progress
            progress_container.empty()
            
            # Display results
            display_results()
        
        except Exception as e:
            st.error(f"❌ Error during analysis: {e}")
            st.code(traceback.format_exc())
            
            # Log failed experiment
            experiment_logger.log_experiment(
                query=query,
                error=str(e),
                success=False
            )


def display_results():
    """Display analysis results in tabs."""
    if not st.session_state.query_result:
        return
    
    result = st.session_state.query_result
    
    # Create tabs
    tabs = st.tabs([
        "📊 Overview",
        "📄 Data Preview",
        "🤖 Model & Metrics",
        "📈 Visualizations",
        "💡 LLM Explanation",
        "📋 Experiment Log"
    ])
    
    # Tab 1: Overview
    with tabs[0]:
        st.subheader("Analysis Overview")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Task Type", result['task_info']['task_type'].replace('_', ' ').title())
        
        with col2:
            st.metric("Data Rows", len(result['df']))
        
        with col3:
            if result['pipeline_result'].best_model_name != "None":
                st.metric("Best Model", result['pipeline_result'].best_model_name)
        
        st.subheader("🎯 Task Understanding")
        
        # Display task info in a more visual way
        task_info = result['task_info']
        
        st.markdown(f"**Task Type:** `{task_info['task_type'].replace('_', ' ').title()}`")
        st.markdown(f"**Explanation:** {task_info.get('explanation', 'N/A')}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if task_info.get('relevant_tables'):
                st.markdown("**📊 Relevant Tables:**")
                for table in task_info['relevant_tables']:
                    st.markdown(f"- `{table}`")
            
            if task_info.get('target_column'):
                st.markdown(f"**🎯 Target Column:** `{task_info['target_column']}`")
            
            if task_info.get('time_column'):
                st.markdown(f"**📅 Time Column:** `{task_info['time_column']}`")
        
        with col2:
            if task_info.get('feature_columns'):
                st.markdown("**🔧 Feature Columns:**")
                for col in task_info['feature_columns'][:10]:
                    st.markdown(f"- `{col}`")
                if len(task_info['feature_columns']) > 10:
                    st.markdown(f"*... and {len(task_info['feature_columns']) - 10} more*")
            
            if task_info.get('grouping'):
                st.markdown("**📑 Grouping:**")
                for group in task_info['grouping']:
                    st.markdown(f"- `{group}`")
        
        if task_info.get('filters'):
            st.markdown(f"**🔍 Filters:** `{task_info['filters']}`")
        
        if task_info.get('aggregations'):
            st.markdown("**📈 Aggregations:**")
            agg_df = pd.DataFrame([
                {"Column": k, "Function": v} 
                for k, v in task_info['aggregations'].items()
            ])
            st.dataframe(agg_df, hide_index=True, use_container_width=True)
        
        st.markdown("---")
        st.subheader("💾 Generated SQL Query")
        st.code(result['sql'], language="sql")
        
        st.markdown("---")
        st.subheader("📋 Query Result")
        
        # Show result metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Rows Returned", len(result['df']))
        with col2:
            st.metric("Columns", len(result['df'].columns))
        
        # Display the query result
        st.dataframe(result['df'].head(50), use_container_width=True)
        
        if len(result['df']) > 50:
            st.info(f"Showing first 50 rows of {len(result['df'])} total rows. See 'Data Preview' tab for more.")
    
    # Tab 2: Data Preview
    with tabs[1]:
        st.subheader("📄 Data Preview")
        
        # Show data shape
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", len(result['df']))
        with col2:
            st.metric("Total Columns", len(result['df'].columns))
        with col3:
            memory_mb = result['df'].memory_usage(deep=True).sum() / 1024 / 1024
            st.metric("Memory Usage", f"{memory_mb:.2f} MB")
        
        st.markdown("---")
        st.dataframe(result['df'].head(100), use_container_width=True)
        
        st.markdown("---")
        st.subheader("📊 Column Statistics")
        
        stats = get_column_statistics(result['df'])
        
        # Create a more visual statistics display
        stats_data = []
        for col_name, col_stats in stats.items():
            row = {
                "Column": col_name,
                "Type": col_stats['inferred_type'].title(),
                "Missing": f"{col_stats['missing_percent']:.1f}%",
                "Unique": col_stats['unique_count']
            }
            
            if col_stats['inferred_type'] == 'numeric':
                row['Min'] = f"{col_stats.get('min', 'N/A'):.2f}" if col_stats.get('min') is not None else 'N/A'
                row['Max'] = f"{col_stats.get('max', 'N/A'):.2f}" if col_stats.get('max') is not None else 'N/A'
                row['Mean'] = f"{col_stats.get('mean', 'N/A'):.2f}" if col_stats.get('mean') is not None else 'N/A'
            elif col_stats['inferred_type'] == 'categorical':
                top_values = col_stats.get('top_values', {})
                if top_values:
                    top_val = list(top_values.keys())[0]
                    row['Top Value'] = f"{top_val} ({top_values[top_val]})"
            
            stats_data.append(row)
        
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, hide_index=True, use_container_width=True)
    
    # Tab 3: Model & Metrics
    with tabs[2]:
        st.subheader("🤖 Model Performance")
        
        pipeline_result = result['pipeline_result']
        
        if pipeline_result.best_model_name != "None":
            # Highlight best model
            st.success(f"🏆 **Best Model:** {pipeline_result.best_model_name}")
            
            if not pipeline_result.metrics_comparison.empty:
                st.markdown("---")
                st.subheader("📊 Metrics Comparison")
                
                # Style the metrics dataframe
                metrics_df = pipeline_result.metrics_comparison
                
                # Highlight the best model row
                def highlight_best(row):
                    if row['model'] == pipeline_result.best_model_name:
                        return ['background-color: #d4edda'] * len(row)
                    return [''] * len(row)
                
                styled_df = metrics_df.style.apply(highlight_best, axis=1).format({
                    col: "{:.4f}" for col in metrics_df.columns if col != 'model'
                })
                
                st.dataframe(styled_df, use_container_width=True, hide_index=True)
                
                # Show metric cards for best model
                st.markdown("---")
                st.subheader("🎯 Best Model Metrics")
                
                best_metrics = metrics_df[metrics_df['model'] == pipeline_result.best_model_name].iloc[0].to_dict()
                metric_cols = [col for col in best_metrics.keys() if col != 'model']
                
                cols = st.columns(min(len(metric_cols), 4))
                for idx, metric in enumerate(metric_cols):
                    with cols[idx % len(cols)]:
                        st.metric(
                            label=metric.replace('_', ' ').upper(),
                            value=f"{best_metrics[metric]:.4f}"
                        )
            
            if pipeline_result.feature_importances:
                st.markdown("---")
                st.subheader("🔍 Top Feature Importances")
                
                importance_df = pd.DataFrame([
                    {"Feature": k, "Importance": v}
                    for k, v in list(pipeline_result.feature_importances.items())[:10]
                ])
                
                # Create a horizontal bar chart for better visualization
                import plotly.graph_objects as go
                fig = go.Figure(go.Bar(
                    x=importance_df['Importance'],
                    y=importance_df['Feature'],
                    orientation='h',
                    marker=dict(color='steelblue')
                ))
                fig.update_layout(
                    title="Feature Importance",
                    xaxis_title="Importance Score",
                    yaxis_title="",
                    height=400,
                    yaxis=dict(autorange="reversed")
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("ℹ️ No model training required for this task type.")
    
    # Tab 4: Visualizations
    with tabs[3]:
        st.subheader("Visualizations")
        
        figures = result['figures']
        
        if figures:
            for name, fig in figures.items():
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No visualizations generated.")
    
    # Tab 5: LLM Explanation
    with tabs[4]:
        st.subheader("AI-Generated Explanation")
        
        # Generate explanation
        try:
            llm_client = initialize_llm_client()
            if llm_client:
                explanation = generate_explanation(result, llm_client)
                st.markdown(explanation)
            else:
                st.warning("LLM client not available for explanation.")
        except Exception as e:
            st.error(f"Failed to generate explanation: {e}")
    
    # Tab 6: Experiment Log
    with tabs[5]:
        st.subheader("📋 Recent Experiments")
        
        recent = experiment_logger.get_recent_experiments(20)
        
        if not recent.empty:
            # Format the dataframe
            recent['timestamp'] = pd.to_datetime(recent['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
            recent['task_type'] = recent['task_type'].str.replace('_', ' ').str.title()
            recent['success'] = recent['success'].map({1: '✅', 0: '❌'})
            
            # Rename columns for better display
            recent = recent.rename(columns={
                'id': 'ID',
                'timestamp': 'Time',
                'query': 'Query',
                'task_type': 'Task Type',
                'model_name': 'Model',
                'success': 'Status'
            })
            
            st.dataframe(recent, use_container_width=True, hide_index=True)
        else:
            st.info("No experiments logged yet.")
        
        st.markdown("---")
        st.subheader("📊 Experiment Statistics")
        stats = experiment_logger.get_experiment_statistics()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Experiments", stats['total_experiments'])
        
        with col2:
            st.metric("Success Rate", f"{stats['success_rate']:.1%}")
        
        with col3:
            st.metric("Task Types", len(stats['by_task_type']))
        
        if stats['by_task_type']:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### 📑 Experiments by Task Type")
                task_df = pd.DataFrame([
                    {"Task Type": (k.replace('_', ' ').title() if k else "Unknown"), "Count": v}
                    for k, v in stats['by_task_type'].items()
                ])
                
                import plotly.express as px
                fig = px.bar(task_df, x='Count', y='Task Type', orientation='h',
                            color='Count', color_continuous_scale='Blues')
                fig.update_layout(showlegend=False, height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if stats.get('top_models'):
                    st.markdown("##### 🏆 Top Models Used")
                    model_df = pd.DataFrame([
                        {"Model": k, "Count": v}
                        for k, v in stats['top_models'].items()
                    ])
                    
                    fig = px.pie(model_df, values='Count', names='Model',
                                color_discrete_sequence=px.colors.sequential.RdBu)
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)


def generate_explanation(result: Dict[str, Any], llm_client) -> str:
    """Generate natural language explanation of results."""
    from pathlib import Path
    
    # Load prompt template
    prompt_path = config.prompts_dir / "explanation_prompt.txt"
    if prompt_path.exists():
        prompt_template = prompt_path.read_text(encoding='utf-8')
    else:
        return "Explanation prompt not found."
    
    # Fill in template
    task_info_str = str(result['task_info'])
    dataset_info_str = f"Rows: {len(result['df'])}, Columns: {len(result['df'].columns)}"
    
    metrics_str = ""
    if not result['pipeline_result'].metrics_comparison.empty:
        metrics_str = result['pipeline_result'].metrics_comparison.to_string()
    
    feature_imp_str = ""
    if result['pipeline_result'].feature_importances:
        top_features = list(result['pipeline_result'].feature_importances.items())[:5]
        feature_imp_str = ", ".join([f"{k}: {v:.3f}" for k, v in top_features])
    
    viz_str = ", ".join(result['figures'].keys())
    
    # Add prediction results for prediction tasks
    predictions_str = ""
    task_type = result['task_info']['task_type']
    
    if task_type in ['classification', 'regression', 'time_series_forecast'] and result['pipeline_result'].predictions is not None:
        data_bundle = result['data_bundle']
        predictions = result['pipeline_result'].predictions
        
        # Create predictions summary
        if task_type == 'classification':
            y_test = data_bundle.y_test
            pred_df = pd.DataFrame({
                'Actual': y_test.values,
                'Predicted': predictions
            })
            predictions_str = f"\n\nPrediction Results (first 10):\n{pred_df.head(10).to_string()}\n"
            
            # Add class distribution
            unique, counts = np.unique(predictions, return_counts=True)
            pred_distribution = dict(zip(unique, counts))
            predictions_str += f"\nPredicted Class Distribution: {pred_distribution}"
            
        elif task_type == 'regression':
            y_test = data_bundle.y_test
            pred_df = pd.DataFrame({
                'Actual': y_test.values,
                'Predicted': predictions,
                'Error': y_test.values - predictions
            })
            predictions_str = f"\n\nPrediction Results (first 10):\n{pred_df.head(10).to_string()}\n"
            predictions_str += f"\nPrediction Range: [{predictions.min():.2f}, {predictions.max():.2f}]"
    
    prompt = prompt_template.replace("{task_info}", task_info_str)
    prompt = prompt.replace("{dataset_info}", dataset_info_str)
    prompt = prompt.replace("{metrics}", metrics_str)
    prompt = prompt.replace("{feature_importances}", feature_imp_str)
    prompt = prompt.replace("{visualizations}", viz_str)
    
    # Add predictions section to prompt
    if predictions_str:
        prompt += predictions_str
    
    # Generate explanation
    explanation = llm_client.complete(prompt, temperature=0.7)
    
    return explanation


def main():
    """Main application entry point."""
    
    # Sidebar
    sidebar_data_source()
    st.sidebar.markdown("---")
    sidebar_llm_config()
    st.sidebar.markdown("---")
    sidebar_ml_config()
    
    # Main content
    main_query_interface()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "**Speak2Data** is a research-grade system for natural language to automated analytics and ML. "
        "Built with Streamlit, SQLAlchemy, scikit-learn, and LLMs."
    )


if __name__ == "__main__":
    main()
