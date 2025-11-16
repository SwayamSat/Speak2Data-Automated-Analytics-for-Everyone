# 📊 Speak2Data: Automated Analytics for Everyone

Speak2Data is a **domain-free, schema-agnostic** data analysis platform that allows non-technical users to analyze ANY database by asking questions in natural language. The system automatically adapts to your database structure, interprets your questions, generates SQL queries, executes them, and can even build and run machine learning pipelines—all in one seamless flow.

## ✨ Features

### Universal Database Support 🗄️
- **Works with ANY sector**: Finance, Healthcare, Retail, HR, Manufacturing, IoT, Education, etc.
- **Multiple file formats**: SQLite (.db), CSV (.csv), Excel (.xlsx, .xls), Parquet (.parquet)
- **Automatic schema detection**: Discovers tables and columns dynamically
- **Zero configuration**: Upload any database and start asking questions immediately
- **No hardcoded assumptions**: Fully adapts to your data structure

### Intelligent Analysis 🤖
- **Natural Language Processing**: Ask questions in plain English using Google Gemini Pro
- **Automatic SQL Generation**: Converts your questions into optimized SQL queries for YOUR schema
- **Smart Schema Awareness**: AI understands your specific tables and columns
- **Context-aware suggestions**: Query suggestions tailored to your database structure

### Powerful Visualizations 📊
- **Auto-generated charts**: Bar charts, line charts, scatter plots, heatmaps
- **Interactive visualizations**: Powered by Plotly for dynamic data exploration
- **Smart visualization selection**: Automatically chooses appropriate chart types

### Machine Learning 🔮
- **Domain-agnostic ML**: Works with any tabular data
- **Automated pipelines**: Classification, regression, clustering
- **No feature engineering required**: Automatically handles any column types
- **Performance metrics**: R², RMSE, accuracy, and feature importance

### User Experience ✨
- **Real-time Data Analysis**: Instant results with comprehensive explanations
- **User-friendly Interface**: Clean, modern Streamlit-based web application
- **Sample Database**: Pre-loaded business data for testing and learning

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Google Gemini Pro API key

### Installation

1. **Clone or download the project**
   ```bash
   git clone <repository-url>
   cd NLPToSQL
   ```

2. **Create and activate virtual environment (Recommended)**
   ```bash
   # Create virtual environment
   python -m venv nlpenv
   
   # Activate virtual environment
   # On Windows:
   nlpenv\Scripts\activate
   # On macOS/Linux:
   source nlpenv/bin/activate
   
   # Or use the provided scripts:
   # Windows Command Prompt:
   activate_env.bat
   # Windows PowerShell:
   .\activate_env.ps1
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   - Copy `env_template.txt` to `.env`
   - Add your Google Gemini Pro API key:
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   DATABASE_URL=sqlite:///business_data.db
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

6. **Open your browser**
   - Navigate to `http://localhost:8501`
   - Start asking questions!

## 📋 Project Structure

```
NLPToSQL/
├── app.py                 # Main Streamlit application
├── nlp_module.py         # Natural language processing with Gemini Pro
├── sql_generator.py      # SQL query generation
├── ml_pipeline.py        # Machine learning pipeline
├── db_module.py          # Database management and sample data
├── utils.py              # Utility functions and helpers
├── requirements.txt      # Python dependencies
├── env_template.txt      # Environment variables template
├── README.md            # This file
└── database/            # Database files (auto-generated)
    └── business_data.db
```

## 💡 Usage Examples

### Example 1: Retail/E-commerce Database
**Tables**: customers, orders, products, sales
- "Show me the total sales by product category"
- "What are the top 10 customers by order value?"
- "Which products are most profitable?"
- "Predict customer churn based on order history"

### Example 2: Healthcare Database
**Tables**: patients, visits, medications, doctors
- "Show me patient visit trends by month"
- "What are the most prescribed medications?"
- "Find patients with multiple visits this year"
- "Analyze patient demographics by condition"

### Example 3: Financial Database
**Tables**: accounts, transactions, customers, loans
- "What's the total transaction volume by account type?"
- "Show me high-value transactions over $10,000"
- "Analyze loan default rates by customer segment"
- "Predict loan approval based on customer history"

### Example 4: HR Database
**Tables**: employees, departments, payroll, attendance
- "Show me average salary by department"
- "What's the attendance rate by employee?"
- "Find employees with highest performance ratings"
- "Analyze turnover rates across departments"

### Example 5: IoT Sensor Database
**Tables**: sensors, readings, locations, alerts
- "Show me sensor readings over time"
- "What are the average temperature readings by location?"
- "Find sensors with abnormal readings"
- "Predict sensor failures based on historical data"

### Generic Queries (Work with Any Schema)
- "Show me all tables and what data is available"
- "What are the most common values in [column name]?"
- "Give me a summary of the data"
- "What patterns can you find?"
- "Run machine learning analysis on this data"

## 🔧 Technical Details

### Architecture

1. **Frontend**: Streamlit web application
2. **NLP Engine**: Google Gemini Pro API for natural language understanding
3. **Database**: SQLite with realistic business data
4. **ML Pipeline**: Scikit-learn for automated machine learning
5. **Visualization**: Plotly for interactive charts

### Key Components

- **NLPProcessor**: Parses natural language queries and extracts intent
- **SQLGenerator**: Converts parsed queries into SQL statements
- **MLPipeline**: Automated machine learning workflow
- **DatabaseManager**: Handles database operations and sample data
- **VisualizationGenerator**: Creates interactive charts and graphs

### Schema-Agnostic Architecture

The application automatically adapts to ANY database schema:

**Default Sample Schema** (for testing):
- **customers**: Customer information and segments
- **products**: Product catalog with categories and pricing
- **orders**: Order details and status
- **order_items**: Individual items within orders
- **sales**: Sales transactions with regional data

**Your Custom Schema**:
- Upload any database file
- System automatically discovers all tables and columns
- AI generates queries using YOUR exact schema
- No configuration or setup required

## 🛠️ Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
# Google Gemini API Configuration
GEMINI_API_KEY=your_gemini_api_key_here

# Database Configuration
DATABASE_URL=sqlite:///business_data.db
```

### API Key Setup

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Add it to your `.env` file

## 🌍 Universal Database Support

### Supported File Formats

Speak2Data works with multiple database and data file formats:

#### 1. SQLite Databases (.db, .sqlite, .sqlite3)
- Direct connection to existing SQLite databases
- Works with any schema structure
- Perfect for local development and testing

#### 2. CSV Files (.csv)
- Automatically imported into SQLite
- Table name derived from filename
- Preserves column names and data types

#### 3. Excel Files (.xlsx, .xls)
- Imports first sheet or specified sheet
- Handles multiple data types
- Converts to SQLite for querying

#### 4. Parquet Files (.parquet)
- Efficient columnar storage format
- Fast import and query performance
- Ideal for large datasets

### How It Works

1. **Upload any database file** through the web interface
2. **Automatic schema detection** discovers all tables and columns
3. **AI analyzes structure** and generates contextual query suggestions
4. **Ask questions naturally** - the system understands YOUR data
5. **Get instant insights** with SQL execution and visualizations

### Sample Data

The application includes a pre-loaded business database for testing:

- **1,000 customers** with demographic information
- **200 products** across multiple categories
- **5,000 orders** with various statuses
- **10,000 sales records** with regional data

But you can replace this with ANY database from ANY domain!

## 🎯 Use Cases

### Universal Data Analysis
- **Any Industry**: Retail, Healthcare, Finance, HR, Manufacturing, Education, Government
- **Any Data Source**: CRM, ERP, IoT sensors, transaction logs, survey data
- **Any Question**: From simple counts to complex ML predictions

### Business Analysts
- Quick data exploration without SQL knowledge
- Automated report generation across any domain
- Trend analysis and forecasting for any metrics

### Domain Experts
- Medical researchers analyzing patient data
- Financial analysts exploring transaction patterns
- HR professionals reviewing employee metrics
- Operations managers monitoring sensor data

### Data Scientists
- Rapid prototyping of ML models on new datasets
- Automated feature engineering for any schema
- Model performance evaluation across domains

## 🔍 Troubleshooting

### Common Issues

1. **API Key Error**
   - Ensure your Gemini Pro API key is correctly set in `.env`
   - Check that the API key has proper permissions

2. **Database Connection Error**
   - Verify that the database file is created properly
   - Check file permissions in the project directory

3. **Import Errors**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check Python version compatibility

4. **Memory Issues**
   - For large datasets, consider adding LIMIT clauses to queries
   - Monitor system memory usage

### Getting Help

- Check the error messages in the Streamlit interface
- Review the console output for detailed error information
- Ensure all environment variables are properly configured

## 🚀 Advanced Usage

### Working with Custom Databases

1. **Prepare your database file**
   - SQLite: Use existing .db file
   - CSV: Export from Excel, database, or any data source
   - Excel: Use .xlsx or .xls format
   - Parquet: For large datasets

2. **Upload through the web interface**
   - Click "Upload Database or Data File"
   - Select your file
   - System automatically detects schema

3. **Start asking questions**
   - AI generates suggestions based on YOUR schema
   - Use natural language with your own table/column names
   - System adapts all queries to your structure

### Best Practices

#### For Optimal Performance
- Use specific column names in queries
- Add appropriate filters to limit data size
- For large CSV/Excel files, consider Parquet format

#### For Better AI Understanding
- Use descriptive table and column names
- Keep naming conventions consistent
- Avoid special characters in names

#### For Machine Learning
- Ensure sufficient data (>100 rows recommended)
- Clean missing values when possible
- Use numeric columns for regression predictions

## 📈 Future Enhancements

- Live database connections (PostgreSQL, MySQL, SQL Server)
- More advanced ML algorithms and ensemble methods
- Real-time data streaming capabilities
- Custom dashboard creation and templates
- Export functionality for reports and visualizations
- Multi-table joins with automatic relationship detection
- Data quality assessment and recommendations

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Google Gemini Pro for natural language processing
- Streamlit for the web interface
- Plotly for interactive visualizations
- Scikit-learn for machine learning capabilities
- SQLAlchemy for database operations

---

**Speak2Data** - Making data analysis accessible to everyone through natural language! 🚀
