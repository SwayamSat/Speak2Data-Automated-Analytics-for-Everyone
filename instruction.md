You are ai engineer helping me build a full research-grade project from scratch.

## 0. Project Overview

Build a complete end-to-end system called **“Speak2Data: Natural Language to Automated Analytics and ML”**.

High-level pipeline (this is critical):

**Natural Language query → Task understanding → SQL dataset extraction → Data preprocessing → Model training → Evaluation → Visualization → Interactive dashboard**

The system must be:

- **Database-agnostic by schema**: any *relational* database from any sector (finance, healthcare, retail, etc.) as long as it has tables/columns. No hardcoded domain assumptions.
- **Research-ready**: structure the code and experiments so I can write a research paper.
- **Modular & clean**: separate concerns clearly (UI, DB layer, LLM layer, ML pipeline, experiment logging, visualization).

We are using **Python** with:

- **Streamlit** for the web UI / dashboard
- **SQLAlchemy** for database handling
- **pandas** for data frames
- **scikit-learn** for ML
- **Plotly** (or altair) for visualizations in Streamlit
- **An LLM (Gemini / OpenAI / other)** for:
  - Task understanding
  - SQL generation
  - Natural language explanations of results

Design the code so the LLM provider is abstracted (e.g., a `LLMClient` class that can be swapped).

---

## 1. Project Structure

Create a clear, modular structure like this:

- `app.py`  
  Main Streamlit app. Orchestrates the pipeline:  
  NL query → Task understanding → SQL → Data → Preprocessing → ML → Evaluation → Visualization → Dashboard.

- `config.py`  
  Handles configuration (LLM API keys, DB connection strings, model defaults, paths).

- `db_manager.py`  
  Database and dataset handling:
  - Connect using a `DATABASE_URL` or uploaded SQLite/CSV/Excel file.
  - Discover tables and columns.
  - Run SQL queries safely and return pandas DataFrames.

- `llm_task_understanding.py`  
  LLM-based task understanding:
  - Takes natural language query + schema.
  - Returns a structured description:
    - task_type: `"descriptive_analytics" | "aggregation" | "classification" | "regression" | "time_series_forecast" | "clustering" | "comparison" | "correlation_analysis"`
    - inferred_target_column (if applicable)
    - relevant feature columns
    - suggested filters / time range / grouping
    - an English explanation of what will be done.

- `llm_sql_generator.py`  
  - Given task understanding + schema, generate SQL query (or queries).
  - Ensure only **read-only** SQL (no INSERT/UPDATE/DELETE/DROP).
  - Optionally use LLM again to fix SQL if DB error occurs.

- `data_preprocessing.py`  
  - Clean and preprocess the DataFrame:
    - Handle missing values (configurable strategies).
    - Detect data types (numeric, categorical, datetime).
    - Encode categorical features (One-Hot or Ordinal).
    - Normalize/scale numeric features where appropriate.
    - Optionally train/test split for supervised tasks.
  - Return:
    - `X_train, X_test, y_train, y_test` for supervised tasks
    - `X` processed for unsupervised tasks (clustering, PCA, etc.)
    - metadata: original column names, encoders, scalers, type info.

- `ml_pipeline.py`  
  This is the heart of the ML pipeline (see section 3 below).

- `visualization.py`
  - Functions to generate appropriate visualizations based on:
    - task_type
    - data types (numeric vs categorical)
    - prediction vs actuals
  - Use Plotly for interactive charts compatible with Streamlit:
    - Histograms, bar charts, line charts, scatter plots, box plots.
    - ROC curves, confusion matrices, feature importance plots.
  - Automatically suggest and build:
    - descriptive plots for descriptive tasks,
    - evaluation plots for ML tasks.

- `experiment_logging.py`
  - Log every experiment into a local lightweight store (SQLite or JSON):
    - timestamp
    - NL query
    - task understanding JSON
    - generated SQL
    - dataset information (rows, columns, domain if known)
    - chosen model(s)
    - metrics (accuracy, F1, RMSE, etc.)
  - This is important to make the system **research-ready**: I need to later analyze performance across many queries and databases.

- `utils.py`
  - Shared utilities (schema printing, type inference, safe casting, etc.).

- `prompts/`
  - `task_understanding_prompt.txt`
  - `sql_generation_prompt.txt`
  - `explanation_prompt.txt`
  Store well-structured prompt templates here so they’re reusable and can be modified for experiments.

- `requirements.txt`
  Include everything needed for this project.

---

## 2. Streamlit App Flow (`app.py`)

**2.1. Sidebar – Data Source**

- Allow user to choose:
  - Option 1: Connect to an existing DB using a `DATABASE_URL`.
  - Option 2: Upload a file (`.db`, `.sqlite`, `.csv`, `.xlsx`, `.xls`, `.parquet`).
- Use `db_manager.DatabaseManager`:
  - If `.db`/`.sqlite`: connect directly.
  - If CSV/Excel/Parquet: load into a temporary SQLite DB and treat it as a table.
- Show:
  - List of tables.
  - For a chosen table, show list of columns with inferred types.
- This must be **schema-agnostic**: no domain knowledge hardcoded.

**2.2. Main – Natural Language Query**

- Text input: “Ask a question about your data or request a model”
  Examples:
  - “Show me monthly sales trends by region.”
  - “Predict customer churn based on their transaction history.”
  - “Cluster patients into risk groups based on their vitals and lab tests.”
- Button: “Run Analysis”

**2.3. On submit: core pipeline (call the modules in order)**

1. **Schema extraction**  
   - Use `DatabaseManager` to fetch schema:
     - List of tables and columns.
     - Data types if possible.
   - Feed this schema to the LLM as context.

2. **Task understanding (LLM)**  
   - Call `llm_task_understanding.infer_task(query, schema)` to get a structured JSON-like result:
     - task_type
     - candidate table(s)
     - candidate target_column
     - candidate feature_columns
     - optional filters (WHERE conditions)
     - grouping / aggregation keys
     - natural language explanation of what’s planned.

3. **SQL generation (LLM)**  
   - Pass task understanding + schema into `llm_sql_generator.generate_sql(...)`.
   - The output is a **SELECT-only** SQL query (or minimal sequence of SELECTs).
   - If the query fails (SQLAlchemy raises an error), catch it and:
     - send the error + SQL back to the LLM to request a correction.
   - Finally, get a working SQL query string.

4. **SQL dataset extraction**  
   - Use `DatabaseManager.run_query(sql)` to get a pandas DataFrame.
   - Handle the case where query returns:
     - zero rows, or
     - too many rows (sample down safely, but record that in logs).

5. **Data preprocessing**  
   - Based on `task_type`, call functions from `data_preprocessing`:
     - For descriptive / aggregation tasks:
       - Simple type inference and minor cleaning.
     - For supervised ML tasks (classification, regression, time-series forecasting):
       - Identify target column; if missing, ask user to confirm from a dropdown.
       - Split into train/test (default 80/20).
       - Encode categorical features.
       - Handle missing values (drop or impute).
     - For clustering:
       - Focus on numeric/categorical-friendly features; scaled numeric data.

6. **Model training + Evaluation (ML pipeline)**  
   - Use `ml_pipeline.run_pipeline(task_type, preprocessed_data, config)`:
     - Train appropriate models (see section 3 for details).
     - Evaluate using proper metrics.
     - Return:
       - Trained models (or best model)
       - Evaluation metrics
       - Predictions (for test set)
       - Any additional artefacts (feature importances, cluster labels, etc.).

7. **Visualization + Dashboard**  
   - Use `visualization` module to:
     - Build descriptive plots (e.g., bar, line, scatter, etc.)
     - Build evaluation plots:
       - classification: confusion matrix, ROC curve (if binary), precision-recall.
       - regression: predicted vs actual scatter, residual plots.
       - clustering: 2D projections with cluster labels (e.g., PCA).
       - time-series: actual vs forecast line charts.
   - Render these in Streamlit:
     - Tabs: “Overview”, “Data Preview”, “Model & Metrics”, “Visualizations”, “LLM Explanation”, “Experiment Log”.
   - Also ask the LLM to generate:
     - A concise, human-readable explanation of the results, e.g.:
       - “The model achieves 0.87 F1-score. The most important features are …”
       - “Cluster 0 appears to represent high-risk patients with …”

8. **Experiment logging**  
   - After each run, log:
     - Query
     - Task understanding result
     - SQL
     - Basic dataset stats (rows, column names)
     - Metrics
     - Model names
     - Timestamp
   - Show recent experiments in a table in the “Experiment Log” tab.

---

## 3. ML Pipeline Design (`ml_pipeline.py`)

Design `ml_pipeline.py` as a **research-friendly AutoML-like pipeline**.

### 3.1. Supported task types and models

- `task_type = "descriptive_analytics" | "aggregation"`
  - No ML training, only:
    - groupby/aggregation
    - descriptive statistics
    - distribution plots.
- `task_type = "classification"`
  - Models to compare:
    - Logistic Regression
    - RandomForestClassifier
    - XGBoost or GradientBoostingClassifier (if available)
  - Metrics:
    - Accuracy
    - Precision, Recall, F1-score (macro/weighted)
    - ROC-AUC (if binary)
  - Choose the “best” model based on F1 (or user-configurable).
- `task_type = "regression"`
  - Models:
    - LinearRegression
    - RandomForestRegressor
    - GradientBoostingRegressor
  - Metrics:
    - MAE, MSE, RMSE, R²
- `task_type = "clustering"`
  - Models:
    - KMeans
    - (optional) DBSCAN
  - Metrics:
    - Silhouette score (if labels are not given)
  - Perform simple hyperparameter sweep over K (e.g., K=2..6) and report best.
- `task_type = "time_series_forecast"`
  - Keep it simple:
    - Maybe univariate models using rolling window + RandomForestRegressor.
    - Or use statsmodels if available.
  - Metrics: MAE, RMSE on hold-out period.

### 3.2. Pipeline structure

Implement a main entry:

```python
def run_pipeline(task_type: str, data_bundle: DataBundle, config: PipelineConfig) -> PipelineResult:
    ...


Where data_bundle contains:

For supervised tasks:

X_train, X_test, y_train, y_test

For unsupervised:

X and maybe original_df, id_column if needed

metadata about column types and encoders.

PipelineResult should include:

best_model_name

all_models_metrics (dict of model → metrics)

predictions (aligned with X_test or original data)

feature_importances where available

plots_data (anything needed for visualization functions)

3.3. Research angles in ML pipeline

This is important. Build the pipeline so that we can later publish a paper:

Multiple model comparison:

For each task, train multiple models, not just one.

Store metrics for all models.

Provide a structured JSON or dict describing all results.

Explainability (basic):

For tree-based models, compute:

Feature importances and expose them to the visualization layer.

For linear models, show coefficients.

Cross-database generality:

Keep code generic so it works for any relational DB.

Avoid domain-specific feature engineering.

Logging for experiments:

Store metrics + context per run in experiment_logging.py.

4. LLM Components and Prompts

Implement an LLMClient class with methods:

complete(prompt: str, **kwargs) -> str

structured_complete(prompt: str, schema: dict, **kwargs) -> dict

Then build on top:

infer_task(query, schema) in llm_task_understanding.py

Use a prompt template like:

“You are a data analyst assistant. Given a user question and a database schema, classify the task and fill a JSON structure …”

Make sure the function returns a Python dict with clearly defined keys.

generate_sql(task_info, schema) in llm_sql_generator.py

Prompt: “You are an expert SQL developer. Only produce a SELECT query compatible with SQLAlchemy + SQLite/Postgres. Don’t modify schema.”

Optionally have a function to repair SQL when an error occurs.

explain_results(task_info, metrics, plots_summary, dataset_info)

Ask the LLM to produce a short, student-like, non-flashy explanation suitable for a project report.

Keep all prompt templates in prompts/ as plain text.

5. Visualization Layer (visualization.py)

Implement functions like:

plot_descriptive_overview(df, metadata)

plot_classification_metrics(y_true, y_pred, probabilities=None)

Confusion matrix (as heatmap)

ROC curve if possible

plot_regression_metrics(y_true, y_pred)

Pred vs actual

Residuals

plot_clustering_results(X, labels, metadata)

2D projection via PCA or t-SNE

plot_time_series(actual_series, forecast_series, timestamps)

Each function returns Plotly figures or Streamlit-renderable objects.

In app.py, create Streamlit tabs:

“Overview”

“Data Preview”

“Model & Metrics”

“Visualizations”

“LLM Explanation”

“Experiment Log”

6. Research Features (Make It Unique)

To make this project strong for a research paper, build in these unique angles:

Cross-domain, schema-agnostic NL → ML system
The system should work without retraining on:

finance DBs

medical DBs

retail DBs

generic tabular DBs
Only schema is given to the LLM.

Task inference correctness evaluation hooks
Provide code-level hooks to:

Log the inferred task_type and let the user override it (e.g., dropdown).

Later, we can evaluate how often the LLM’s task inference matches user’s intended task.

Automated experiment logging
Every run is an experiment with:

Domain (if user sets a tag)

Query type

LLM decisions

Model choice and metrics
This allows later offline evaluation and ablation studies, for example:

Effect of different prompts

Effect of different models

Comparison across domains

Prompt configurability for research
Store prompts in text files and add a simple UI in Streamlit to:

View current prompts

Switch between “baseline” and “research” prompt variants.
This makes it easier to run controlled experiments.

Explainability view
Add a tab that:

Shows feature importances

Asks LLM to explain why these features might matter in domain-agnostic terms (e.g., “spending frequency may relate to churn”).

Structure the code so these research features are cleanly separated and can be extended.

7. Implementation Style

Use type hints.

Use clear, descriptive function and class names.

Write docstrings for public functions.

Avoid hardcoding domain-specific logic; everything should be based on schema and task type.

Prefer pure functions for preprocessing and ML training where possible.

8. Deliverables

By following this specification, create:

All the Python modules listed above with working code.

A working Streamlit application that:

Accepts NL query.

Understands the task.

Generates SQL.

Extracts data.

Preprocesses it.

Trains relevant ML models.

Evaluates them.

Visualizes results.

Logs experiments.

Code that is easy to extend for research and paper-writing.

Please now generate the actual code files and functions according to this plan.