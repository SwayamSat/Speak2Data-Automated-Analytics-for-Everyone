# 📊 Speak2Data Project Overview

## 🎯 Project Summary

Speak2Data is a comprehensive web-based platform that enables non-technical users to analyze business data through natural language queries. The system leverages Google Gemini Pro for natural language understanding, automatically generates SQL queries, and can build machine learning pipelines—all in one seamless workflow.

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit UI  │    │   NLP Module    │    │  SQL Generator  │
│                 │◄──►│  (Gemini Pro)   │◄──►│                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Visualization  │    │  ML Pipeline    │    │  Database       │
│   Generator     │    │ (Scikit-learn)  │    │  Manager        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📁 File Structure

```
NLPToSQL/
├── app.py                 # Main Streamlit application
├── nlp_module.py         # Natural language processing
├── sql_generator.py      # SQL query generation
├── ml_pipeline.py        # Machine learning pipeline
├── db_module.py          # Database management
├── utils.py              # Utility functions
├── setup.py              # Setup script
├── requirements.txt      # Dependencies
├── env_template.txt      # Environment template
├── README.md            # Documentation
├── PROJECT_OVERVIEW.md  # This file
└── database/            # Auto-generated database files
```

## 🔧 Key Components

### 1. Natural Language Processing (`nlp_module.py`)
- **Purpose**: Parse user queries using Google Gemini Pro
- **Features**: Intent recognition, entity extraction, query classification
- **Output**: Structured query information for SQL generation

### 2. SQL Generation (`sql_generator.py`)
- **Purpose**: Convert parsed queries into SQL statements
- **Features**: Query optimization, syntax validation, JOIN handling
- **Output**: Executable SQL queries

### 3. Machine Learning Pipeline (`ml_pipeline.py`)
- **Purpose**: Automated ML workflow for predictions and analysis
- **Features**: Data preprocessing, model training, evaluation
- **Algorithms**: Random Forest, Logistic Regression, K-Means clustering

### 4. Database Management (`db_module.py`)
- **Purpose**: Handle database operations and sample data
- **Features**: SQLite integration, sample data generation, query execution
- **Data**: 1,000 customers, 200 products, 5,000 orders, 10,000 sales records

### 5. Visualization (`utils.py`)
- **Purpose**: Generate interactive charts and data visualizations
- **Features**: Auto-chart generation, Plotly integration, dashboard creation
- **Charts**: Bar, line, scatter, histogram, pie, heatmap

### 6. Main Application (`app.py`)
- **Purpose**: Streamlit web interface
- **Features**: Query input, results display, ML analysis, interactive filters
- **UI**: Modern, responsive design with sidebar navigation

## 🚀 Key Features

### Natural Language Queries
- "Show me sales by category"
- "Predict customer churn"
- "What are the top products?"
- "Cluster customers by behavior"

### Automated SQL Generation
- Converts natural language to SQL
- Handles complex queries with JOINs
- Optimizes query performance
- Validates syntax

### Machine Learning Integration
- Automatic model selection
- Feature engineering
- Performance evaluation
- Prediction explanations

### Interactive Visualizations
- Auto-generated charts
- Multiple chart types
- Interactive filters
- Dashboard creation

## 📊 Sample Data Schema

### Customers Table
- `customer_id`, `name`, `email`, `phone`
- `city`, `state`, `registration_date`
- `customer_segment`

### Products Table
- `product_id`, `name`, `category`, `subcategory`
- `price`, `cost`, `supplier`, `launch_date`

### Orders Table
- `order_id`, `customer_id`, `order_date`
- `total_amount`, `status`, `shipping_city`, `shipping_state`

### Sales Table
- `sale_id`, `product_id`, `customer_id`, `sale_date`
- `quantity`, `unit_price`, `total_amount`
- `region`, `sales_rep`

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **NLP**: Google Gemini Pro API
- **Database**: SQLite with SQLAlchemy
- **ML**: Scikit-learn
- **Visualization**: Plotly
- **Language**: Python 3.8+

## 📈 Use Cases

### Business Analysts
- Quick data exploration
- Automated reporting
- Trend analysis

### Managers
- High-level insights
- Performance dashboards
- Strategic decisions

### Data Scientists
- Rapid prototyping
- Automated feature engineering
- Model evaluation

## 🔒 Security Features

- API key management via environment variables
- Input validation and sanitization
- Error handling and user-friendly messages
- No sensitive data exposure

## 🚀 Getting Started

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Up Environment**
   ```bash
   cp env_template.txt .env
   # Edit .env and add your GEMINI_API_KEY
   ```

3. **Run Application**
   ```bash
   streamlit run app.py
   ```

4. **Access Interface**
   - Open browser to `http://localhost:8501`
   - Start asking questions!

## 🧪 Testing

The application includes comprehensive testing:
- Database module functionality
- ML pipeline operations
- Utility functions
- SQL generation
- Import validation

Run tests with:
```bash
python setup.py
```

## 📚 Documentation

- **README.md**: Complete setup and usage guide
- **Code Comments**: Inline documentation throughout
- **Type Hints**: Full type annotations for better code understanding
- **Error Messages**: User-friendly error handling

## 🔮 Future Enhancements

- Support for additional databases (PostgreSQL, MySQL)
- More advanced ML algorithms
- Real-time data streaming
- Custom dashboard creation
- Export functionality
- Multi-language support

## 🤝 Contributing

The project is designed for extensibility:
- Modular architecture
- Clear separation of concerns
- Comprehensive error handling
- Easy to add new features

## 📄 License

Open source under MIT License - feel free to use and modify!

---

**Speak2Data** - Making data analysis accessible to everyone through natural language! 🚀
