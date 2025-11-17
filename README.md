# Speak2Data: Natural Language to Automated Analytics and ML

A research-grade, end-to-end system that transforms natural language queries into automated data analytics and machine learning pipelines.

## 🎯 Overview

Speak2Data is a **schema-agnostic**, **database-independent** system that:

1. Understands natural language queries about data
2. Automatically generates SQL queries
3. Extracts and preprocesses data
4. Trains multiple ML models (classification, regression, clustering)
5. Evaluates and compares models
6. Creates interactive visualizations
7. Logs all experiments for research analysis

**Key Features:**
- 🧠 LLM-powered task understanding and SQL generation
- 🔄 Support for multiple LLM providers (Gemini, OpenAI, Anthropic)
- 📊 Works with any relational database (SQLite, PostgreSQL, MySQL, etc.)
- 📁 Direct file upload support (CSV, Excel, SQLite, Parquet)
- 🤖 Automated ML pipeline with model comparison
- 📈 Rich interactive visualizations with Plotly
- 📝 Complete experiment logging for research
- 🎨 Clean, modular, extensible architecture

## 🏗️ Architecture

```
Natural Language Query
    ↓
Task Understanding (LLM) → task_type, target, features
    ↓
SQL Generation (LLM) → SELECT query
    ↓
Data Extraction → pandas DataFrame
    ↓
Data Preprocessing → encoding, scaling, train/test split
    ↓
ML Pipeline → multiple models trained & evaluated
    ↓
Visualization → Plotly charts
    ↓
Interactive Dashboard (Streamlit)
```

## 📦 Installation

### 1. Clone or download this project

```bash
cd "nl 2 sql"
```

### 2. Create virtual environment (recommended)

```bash
python -m venv nlpenv
nlpenv\Scripts\activate  # On Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up API keys

Create a `.env` file or set environment variables:

```bash
# For Gemini
GEMINI_API_KEY=your_gemini_key

# For OpenAI
OPENAI_API_KEY=your_openai_key

# For Anthropic
ANTHROPIC_API_KEY=your_anthropic_key

# Choose default provider
LLM_PROVIDER=gemini  # or openai, anthropic
```

## 🚀 Usage

### Run the Streamlit app

```bash
streamlit run app.py
```

### Basic workflow

1. **Configure Data Source** (sidebar):
   - Upload a file (CSV, Excel, SQLite, Parquet), OR
   - Connect to a database URL

2. **Configure LLM** (sidebar):
   - Select provider (Gemini, OpenAI, or Anthropic)
   - Enter API key
   - Choose model

3. **Ask a Question**:
   - "Show me monthly sales trends by region"
   - "Predict customer churn based on transaction history"
   - "Cluster patients into risk groups based on vitals"

4. **View Results** in tabs:
   - Overview: Task understanding & SQL
   - Data Preview: Sample data & statistics
   - Model & Metrics: Performance comparison
   - Visualizations: Interactive charts
   - LLM Explanation: AI-generated insights
   - Experiment Log: All past experiments

## 📂 Project Structure

```
nl 2 sql/
├── app.py                      # Main Streamlit application
├── config.py                   # Configuration management
├── utils.py                    # Shared utilities
├── db_manager.py              # Database connections & queries
├── llm_client.py              # LLM abstraction layer
├── llm_task_understanding.py  # Task inference from NL
├── llm_sql_generator.py       # SQL generation with LLM
├── data_preprocessing.py      # Data cleaning & encoding
├── ml_pipeline.py             # Multi-model ML training
├── visualization.py           # Plotly visualizations
├── experiment_logging.py      # Experiment tracking
├── prompts/                   # LLM prompt templates
│   ├── task_understanding_prompt.txt
│   ├── sql_generation_prompt.txt
│   └── explanation_prompt.txt
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🎓 Supported Task Types

1. **Descriptive Analytics**: Show data, statistics, distributions
2. **Aggregation**: GROUP BY and aggregate functions
3. **Classification**: Predict categorical targets (LogisticRegression, RandomForest, GradientBoosting)
4. **Regression**: Predict numeric targets (LinearRegression, RandomForest, GradientBoosting)
5. **Clustering**: Group similar records (KMeans, DBSCAN)
6. **Time Series Forecast**: Predict future values
7. **Comparison**: Compare groups or segments
8. **Correlation Analysis**: Analyze relationships

## 🔬 Research Features

### Experiment Logging

Every analysis is logged to `experiments.db` with:
- Natural language query
- Task understanding (JSON)
- Generated SQL
- Dataset information
- Model names and metrics
- Configuration used
- Timestamp

### Model Comparison

For ML tasks, multiple models are trained and compared:
- Classification: 3 models (Logistic, RandomForest, GradientBoosting)
- Regression: 3 models (Linear, RandomForest, GradientBoosting)
- Clustering: KMeans with multiple K values + DBSCAN

### Metrics Tracking

- **Classification**: Accuracy, Precision, Recall, F1, ROC-AUC
- **Regression**: MAE, MSE, RMSE, R²
- **Clustering**: Silhouette score, inertia

### Schema-Agnostic Design

No domain knowledge is hardcoded. The system works with:
- Finance databases
- Healthcare records
- Retail data
- E-commerce transactions
- Any relational database with tables and columns

## 🔧 Configuration

Edit `config.py` or use environment variables to customize:

```python
# LLM settings
LLM_PROVIDER=gemini
GEMINI_MODEL=gemini-1.5-flash

# ML settings
TEST_SIZE=0.2
RANDOM_STATE=42
CV_FOLDS=5

# Preprocessing
MISSING_STRATEGY=drop  # or impute
CATEGORICAL_ENCODING=onehot  # or label
SCALING_METHOD=standard  # or minmax, robust, none
```

## 📊 Example Queries

### Descriptive Analytics
- "Show me the distribution of customer ages"
- "What are the summary statistics for sales by region?"

### Aggregation
- "Calculate average order value by month"
- "Count the number of transactions per customer"

### Classification
- "Predict whether a customer will churn based on their purchase history"
- "Classify patients as high or low risk based on lab results"

### Regression
- "Predict house prices based on square footage and location"
- "Estimate next month's revenue based on historical data"

### Clustering
- "Group customers into segments based on their behavior"
- "Cluster products by sales patterns"

## 🛠️ Extending the System

### Add a new LLM provider

1. Create a new client class in `llm_client.py`:
```python
class NewProviderClient(LLMClient):
    def complete(self, prompt: str, **kwargs) -> str:
        # Implementation
```

2. Update `create_llm_client()` factory function

### Add a new ML model

1. In `ml_pipeline.py`, add to the models dict:
```python
models = {
    "YourModel": YourModelClass(params),
    ...
}
```

### Customize prompts

Edit the prompt templates in `prompts/` directory to improve:
- Task understanding accuracy
- SQL generation quality
- Result explanations

## 📝 License

This is a research and educational project.

## 🤝 Contributing

This is designed as a complete research project. Feel free to:
- Extend with new task types
- Add more ML models
- Improve prompts
- Add new data sources
- Enhance visualizations

## 📧 Support

For issues or questions about the implementation, refer to the code documentation and comments throughout the modules.

## 🎯 Research Applications

Use this system to study:
1. **Cross-domain generalization**: How well does NL→SQL work across different domains?
2. **Task inference accuracy**: How often does the LLM correctly identify the task type?
3. **Model selection strategies**: Which models perform best for different data characteristics?
4. **Prompt engineering**: How do different prompts affect SQL quality and task understanding?
5. **Automated ML**: Can LLMs enable fully automated data science pipelines?

## 🔍 Next Steps

1. Run the app with sample data
2. Try different query types
3. Explore the experiment log
4. Customize prompts for your domain
5. Add domain-specific models or preprocessing
6. Analyze experiment data for research insights

---

**Built with:** Streamlit • SQLAlchemy • scikit-learn • Plotly • Google Gemini / OpenAI / Anthropic
