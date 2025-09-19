# NLP to SQL Converter 🤖

**Convert natural language queries into SQL and retrieve data with automatic visualizations using Google's Gemini AI**

## 📋 Project Status - Phase 1 (60% Complete)

This is the **2nd Monthly Report** version focusing on core NLP-to-SQL functionality:

✅ **Completed Features:**
- Natural language query processing using Google Gemini AI
- Automatic SQL generation from user queries
- Database query execution and data retrieval  
- Automatic visualization generation (bar charts, histograms, pie charts, scatter plots)
- AI-powered insights and query explanations
- Interactive query interface with examples
- Database management with sample data

🔄 **Future Features (Phase 2):**
- Machine Learning pipeline integration
- Predictive analytics and forecasting
- Advanced visualization options
- Query optimization suggestions

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Setup API Key
1. Get your Google AI API key from: https://makersuite.google.com/app/apikey
2. Open `.env` file
3. Replace `your_google_ai_api_key_here` with your actual API key

### 3. Run the Application
```bash
streamlit run app.py
```

### 4. Initialize Database
- Click "Initialize Sample Database" in the sidebar
- This creates sample tables: customers, products, orders, order_items, sales_reps

## 🔧 Technology Stack

- **Frontend:** Streamlit
- **AI Model:** Google Gemini Pro
- **Database:** SQLite
- **Visualizations:** Plotly
- **Environment:** Python 3.8+

## 📊 Example Queries

Try these natural language queries:
- "Show me all customers from New York"
- "What are the total sales by month?"
- "Find customers with the highest purchase amounts"
- "Show product performance by category"
- "Which products have the best profit margins?"

## 📁 Project Structure

```
├── app.py              # Main Streamlit application
├── nlp_module.py       # NLP processing with Gemini AI
├── db_module.py        # Database management
├── requirements.txt    # Python dependencies
├── .env               # Environment variables (API key)
├── database/
│   └── sample.db      # SQLite database (created automatically)
└── README.md          # This file
```

## 🎯 Development Notes

This version demonstrates:
1. **NLP Processing:** Converting natural language to structured queries
2. **SQL Generation:** Automatic SQL creation from user intent
3. **Data Retrieval:** Executing queries and retrieving results
4. **Visualization:** Automatic chart generation based on data types
5. **AI Insights:** Intelligent analysis of query results

Perfect for showcasing NLP-to-SQL capabilities in presentations and demos!

## 🔐 Security

- API keys are stored in `.env` file (not committed to version control)
- SQL injection protection with query validation
- Read-only database operations for safety

## 📞 Support

For questions about this implementation, refer to the code comments and documentation within each module.