# 📊 Speak2Data: Automated Analytics for Everyone

Speak2Data is a powerful web-based platform that allows non-technical users to analyze business data by asking questions in natural language. The system interprets your questions, generates SQL queries, executes them on a business database, and can even build and run machine learning pipelines—all in one seamless flow.

## ✨ Features

- **Natural Language Processing**: Ask questions in plain English using Google Gemini Pro
- **Automatic SQL Generation**: Converts your questions into optimized SQL queries
- **Interactive Visualizations**: Auto-generated charts and graphs using Plotly
- **Machine Learning Integration**: Automated ML pipelines for predictions and analysis
- **Real-time Data Analysis**: Instant results with comprehensive explanations
- **User-friendly Interface**: Clean, modern Streamlit-based web application
- **Sample Business Data**: Pre-loaded with realistic business datasets

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

### Data Retrieval Queries
- "Show me the total sales by product category"
- "What are the top 10 customers by order value?"
- "Which products are most profitable?"
- "Show me sales trends over the last 6 months"

### Machine Learning Queries
- "Predict customer churn based on order history"
- "Cluster customers based on their purchase behavior"
- "Forecast sales for next quarter"
- "Classify products by profitability"

### Analytics Queries
- "What's the average order value by customer segment?"
- "Show me the correlation between product price and sales"
- "Which regions have the highest growth?"

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

### Database Schema

The application includes sample business data with the following tables:

- **customers**: Customer information and segments
- **products**: Product catalog with categories and pricing
- **orders**: Order details and status
- **order_items**: Individual items within orders
- **sales**: Sales transactions with regional data

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

## 📊 Sample Data

The application automatically generates realistic business data including:

- **1,000 customers** with demographic information
- **200 products** across multiple categories
- **5,000 orders** with various statuses
- **10,000 sales records** with regional data

## 🎯 Use Cases

### Business Analysts
- Quick data exploration without SQL knowledge
- Automated report generation
- Trend analysis and forecasting

### Managers
- High-level business insights
- Performance dashboards
- Strategic decision support

### Data Scientists
- Rapid prototyping of ML models
- Automated feature engineering
- Model performance evaluation

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

### Custom Queries

You can extend the system by:

1. **Adding new tables** to the database schema
2. **Customizing ML models** in the pipeline
3. **Creating new visualizations** in the utils module
4. **Extending NLP prompts** for better query understanding

### Performance Optimization

- Use specific column names in queries
- Add appropriate filters to limit data size
- Consider indexing for frequently queried columns

## 📈 Future Enhancements

- Support for additional database types (PostgreSQL, MySQL)
- More advanced ML algorithms and ensemble methods
- Real-time data streaming capabilities
- Custom dashboard creation
- Export functionality for reports and visualizations

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
