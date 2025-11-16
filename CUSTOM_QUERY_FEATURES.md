# Custom Query Features with Gemini API

## Overview
Enhanced custom query generation using Google Gemini API with robust response handling, smart JSON extraction, and comprehensive error recovery.

---

## 🚀 Key Features

### 1. **AI-Powered Query Guidance**
Get intelligent guidance before executing your query:
- **Understanding**: AI interprets your question
- **Query Type**: Identifies whether it's retrieval, analysis, prediction, etc.
- **Suggested Approach**: Step-by-step methodology
- **Key Insights**: Important considerations
- **SQL Hints**: Query structure recommendations
- **Visualization Suggestions**: Appropriate chart types
- **Follow-up Questions**: Related queries to explore

**How to Use:**
1. Type your question in the text area
2. Click **"💡 Get AI Guidance"** button
3. Review the comprehensive guidance
4. Click **"🔍 Analyze"** to execute

---

### 2. **Robust API Response Handling**

#### JSON Extraction
Multiple strategies to extract valid JSON from API responses:
- Remove markdown code blocks (```json, ```)
- Regex pattern matching for JSON objects
- Direct JSON parsing
- Smart detection of JSON boundaries
- Graceful fallback if parsing fails

#### Error Recovery
- **Quota Limits**: Automatic fallback to local processing
- **Rate Limiting**: Exponential backoff retry logic
- **Model Fallback**: Switches between available models
- **Malformed Responses**: Falls back to rule-based parsing

---

### 3. **Enhanced SQL Generation**

#### Response Cleaning
```python
# Automatically removes:
- Markdown code blocks (```sql, ```)
- Explanatory text before/after query
- Common prefixes like "Here's the query:"
- Trailing text after semicolon
```

#### SQL Validation
```python
# Validates:
✓ Starts with valid SQL keyword (SELECT, WITH, etc.)
✓ Has FROM clause (for SELECT queries)
✓ Balanced parentheses
✓ No dangerous patterns (DROP, TRUNCATE, etc.)
✓ Non-empty query
```

#### Smart Fallback
If AI generation fails, uses schema-aware fallback:
- Analyzes available tables and columns
- Identifies numeric, categorical, date columns
- Generates appropriate aggregations
- Adds sensible ORDER BY and LIMIT clauses

---

### 4. **AI-Powered Results Explanation**

#### Rich Explanations
- **Business-Friendly Language**: Non-technical insights
- **Key Metrics**: Highlights important numbers
- **Patterns**: Identifies trends and outliers
- **Actionable Insights**: Business implications

#### Fallback Explanations
If API quota exceeded, generates comprehensive analysis:
```markdown
📊 Data Analysis Results:
• Records Found: 1,234 rows
• Data Columns: 15 fields
• Key Metrics: Total value of 456,789 across 8 numeric columns
• Top Performer: total_amount leads with 123,456
• Averages: price: 45.67, quantity: 23
```

---

### 5. **Intelligent Follow-up Questions**

#### Context-Aware Suggestions
Based on your query and results:
- Drill-down questions for specific segments
- Comparison questions across categories
- Trend analysis suggestions
- Relationship exploration between columns
- Prediction and forecasting options

#### Interactive Exploration
- Click any question to automatically ask it
- Questions adapt to your data structure
- Limited to 5 most relevant suggestions

---

## 🛠️ Technical Implementation

### New Methods in `nlp_module.py`

#### 1. `_extract_json_from_response(response_text)`
Robust JSON extraction with multiple fallback strategies.

**Parameters:**
- `response_text`: Raw API response

**Returns:**
- Parsed JSON dictionary

**Strategies:**
1. Remove markdown formatting
2. Regex pattern matching
3. Direct JSON parsing
4. Boundary detection
5. Error with helpful message

---

#### 2. `_clean_sql_response(sql_text)`
Cleans SQL responses by removing non-query text.

**Parameters:**
- `sql_text`: Raw SQL from API

**Returns:**
- Clean SQL query string

**Cleaning Steps:**
1. Remove markdown blocks
2. Find SQL keyword start
3. Remove trailing explanations
4. Strip common prefixes
5. Trim whitespace

---

#### 3. `_is_valid_sql(sql_query)`
Basic SQL validation before execution.

**Parameters:**
- `sql_query`: SQL to validate

**Returns:**
- `True` if valid, `False` otherwise

**Checks:**
- Non-empty query
- Valid SQL keyword start
- FROM clause for SELECT
- Balanced parentheses
- No dangerous operations

---

#### 4. `generate_custom_query(user_intent, context)`
Generate comprehensive query guidance.

**Parameters:**
- `user_intent`: User's question
- `context`: Optional dict with schema, previous results

**Returns:**
```json
{
  "understanding": "Brief interpretation",
  "query_type": "data_retrieval|analysis|prediction",
  "suggested_approach": "Step-by-step guide",
  "key_insights": ["insight1", "insight2"],
  "sql_hint": "Query structure hint",
  "visualization_suggestions": ["chart1", "chart2"],
  "follow_up_questions": ["q1", "q2", "q3"]
}
```

**Use Cases:**
- Pre-query planning
- Understanding complex requests
- Learning SQL patterns
- Exploring data relationships

---

## 📊 Usage Examples

### Example 1: Basic Query with Guidance
```python
# User asks: "What are the top selling products?"

# 1. Click "Get AI Guidance"
guidance = {
    "understanding": "User wants to see products ranked by sales volume",
    "query_type": "data_retrieval",
    "suggested_approach": "1. Identify sales table\n2. Group by product\n3. Sum quantities\n4. Order descending\n5. Limit to top 10",
    "key_insights": [
        "Consider time period for analysis",
        "Revenue vs quantity - which metric matters?",
        "May need to join product details table"
    ],
    "sql_hint": "SELECT product_name, SUM(quantity) FROM sales GROUP BY product_name ORDER BY SUM(quantity) DESC LIMIT 10",
    "visualization_suggestions": ["bar chart", "horizontal bar chart"],
    "follow_up_questions": [
        "What's the revenue from top products?",
        "How do sales compare month over month?",
        "Which categories have highest sales?"
    ]
}

# 2. Click "Analyze" to execute
# Results display with AI explanation and follow-up suggestions
```

---

### Example 2: Complex Analysis
```python
# User asks: "Compare sales performance across regions for Q4"

# AI generates:
{
    "understanding": "Regional sales comparison for specific time period",
    "query_type": "comparison",
    "suggested_approach": "1. Filter by Q4 dates\n2. Group by region\n3. Calculate metrics (sum, avg, count)\n4. Compare values\n5. Visualize with charts",
    "key_insights": [
        "Define Q4 date range clearly",
        "Consider multiple metrics: revenue, units, orders",
        "Look for seasonal patterns",
        "Identify outlier regions"
    ],
    "sql_hint": "Use WHERE for date filtering, GROUP BY region, multiple aggregations",
    "visualization_suggestions": ["grouped bar chart", "map", "pie chart"],
    "follow_up_questions": [
        "Which products drive regional differences?",
        "How does Q4 compare to other quarters?",
        "What's the growth rate by region?"
    ]
}
```

---

### Example 3: Predictive Query
```python
# User asks: "Predict customer churn based on purchase patterns"

# AI generates:
{
    "understanding": "ML prediction task - identify at-risk customers",
    "query_type": "prediction",
    "suggested_approach": "1. Extract customer features\n2. Calculate engagement metrics\n3. Build classification model\n4. Score customers\n5. Identify high-risk segment",
    "key_insights": [
        "Define 'churn' clearly (no purchase in X days)",
        "Key features: recency, frequency, monetary value",
        "Need historical data with known outcomes",
        "Consider seasonal patterns"
    ],
    "sql_hint": "Calculate RFM metrics, days since last purchase, purchase frequency",
    "visualization_suggestions": ["risk distribution", "feature importance", "ROC curve"],
    "follow_up_questions": [
        "What features predict churn best?",
        "How many customers are at high risk?",
        "What actions reduce churn rate?"
    ]
}
```

---

## 🔧 Configuration

### API Key Setup
```bash
# .env file
GEMINI_API_KEY=your_api_key_here
```

### Error Suppression
```python
# Automatic gRPC warning suppression
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GRPC_TRACE'] = ''
os.environ['GLOG_minloglevel'] = '2'
```

### Model Selection
```python
# Preferred models (in order)
preferred_models = ['gemini-pro', 'gemini-1.5-pro']

# Automatic fallback to available models
# Avoids experimental models by default
```

---

## ⚡ Performance Optimizations

### 1. **Retry Logic**
- Exponential backoff: 1s, 2s, 4s
- Automatic model switching
- Graceful degradation

### 2. **Response Caching**
- Session state management
- Avoids duplicate API calls
- Faster follow-up queries

### 3. **Fallback Processing**
- Schema-aware rule-based parsing
- No API calls required
- Instant response generation

---

## 🎯 Best Practices

### For Users
1. **Be Specific**: "Show me Q4 sales by region" vs "Show me sales"
2. **Use Guidance**: Click "Get AI Guidance" for complex queries
3. **Explore Follow-ups**: Click suggested questions to drill deeper
4. **Review SQL**: Check generated SQL in debug expander
5. **Iterate**: Refine questions based on results

### For Developers
1. **Always use `_extract_json_from_response()`** for API responses
2. **Validate SQL with `_is_valid_sql()`** before execution
3. **Provide fallbacks** for all AI-powered features
4. **Handle quota errors gracefully** with user-friendly messages
5. **Test with edge cases**: empty data, single row, missing columns

---

## 🐛 Troubleshooting

### Issue: "API Quota Exceeded"
**Solution:**
- Wait 60 seconds and retry
- Check usage at: https://ai.dev/usage?tab=rate-limit
- Upgrade plan at: https://ai.google.dev/gemini-api/docs/rate-limits
- App continues working with fallback processing

### Issue: "Could not extract JSON"
**Solution:**
- Automatic fallback to rule-based parsing
- Check API response in logs
- May indicate model output format change

### Issue: "Invalid SQL generated"
**Solution:**
- Review schema in debug info
- Try rephrasing question
- Use "Get AI Guidance" first
- Fallback query executed automatically

### Issue: "Model not found"
**Solution:**
- Automatic model switching enabled
- Check available models: `genai.list_models()`
- Verify API key permissions

---

## 📈 Future Enhancements

### Planned Features
1. **Query History Intelligence**: Learn from past queries
2. **Multi-turn Conversations**: Context-aware follow-ups
3. **Custom Prompt Templates**: Domain-specific optimizations
4. **A/B Testing**: Compare different query approaches
5. **Query Optimization**: Performance suggestions
6. **Natural Language Results**: Voice-ready explanations
7. **Multi-language Support**: Questions in any language

---

## 🔐 Security Considerations

### SQL Injection Prevention
- AI-generated queries validated
- No user input directly in SQL
- Dangerous patterns blocked
- Parameterized queries where possible

### API Key Security
- Stored in environment variables
- Never committed to version control
- Rate limiting respected
- Error messages sanitized

---

## 📚 API Reference

### Main Entry Points

#### App-Level
```python
# Get AI Guidance Button (app.py)
st.button("💡 Get AI Guidance")
→ Calls: nlp_processor.generate_custom_query()

# Analyze Button (app.py)
st.button("🔍 Analyze")  
→ Calls: process_query() → sql_generator.generate_query()
```

#### NLP Module
```python
# Generate comprehensive guidance
guidance = nlp_processor.generate_custom_query(
    user_intent="Your question here",
    context={'schema': db_schema, 'previous_query': last_query}
)

# Parse query to structured format
parsed = nlp_processor.parse_query(user_query)

# Generate SQL from parsed query
sql = nlp_processor.generate_sql_query(parsed)

# Explain results in natural language
explanation = nlp_processor.explain_results(query, results_df, "sql")

# Suggest follow-up questions
questions = nlp_processor.suggest_follow_up_questions(query, results_df)
```

---

## 🎨 UI Enhancements

### Visual Elements
- **🔍 Analyze**: Primary action button (blue)
- **💡 Get AI Guidance**: Secondary action button
- **🤖 AI-Powered Analysis**: Results section with styled card
- **💭 Explore Further**: Follow-up questions section
- **❓ Question pills**: Interactive suggestion buttons

### User Feedback
- Loading spinners during API calls
- Progress messages for long operations
- Friendly error messages
- Quota limit warnings
- Success indicators

---

## 📊 Metrics & Analytics

### Trackable Metrics
1. **API Usage**: Calls per session, quota consumption
2. **Query Success Rate**: Valid queries / total attempts
3. **Fallback Rate**: Fallback usage / total queries
4. **User Engagement**: Follow-up questions clicked
5. **Response Time**: API latency, total processing time

### Monitoring
```python
# Add to app.py for tracking
if 'api_calls' not in st.session_state:
    st.session_state.api_calls = 0
if 'fallback_count' not in st.session_state:
    st.session_state.fallback_count = 0

# Increment on each use
st.session_state.api_calls += 1  # After API call
st.session_state.fallback_count += 1  # On fallback
```

---

## ✅ Testing Checklist

### Unit Tests
- [ ] JSON extraction with various formats
- [ ] SQL validation with edge cases
- [ ] Error handling for quota limits
- [ ] Fallback quality for common queries
- [ ] Response cleaning accuracy

### Integration Tests
- [ ] End-to-end query flow
- [ ] Follow-up question generation
- [ ] Multi-turn conversations
- [ ] Schema updates reflected
- [ ] Error recovery paths

### User Acceptance Tests
- [ ] Guidance accuracy for domain queries
- [ ] SQL correctness on test databases
- [ ] Explanation quality and clarity
- [ ] Follow-up relevance
- [ ] UI responsiveness

---

## 🎓 Learning Resources

### For Users
1. **Example Queries**: Check `USAGE_GUIDE.md`
2. **Video Tutorial**: [Coming Soon]
3. **FAQ**: See troubleshooting section above

### For Developers
1. **Gemini API Docs**: https://ai.google.dev/docs
2. **Prompt Engineering**: https://ai.google.dev/docs/prompt_best_practices
3. **SQLAlchemy**: https://docs.sqlalchemy.org/
4. **Streamlit**: https://docs.streamlit.io/

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:
1. Additional response parsing strategies
2. Domain-specific prompt templates
3. Enhanced fallback algorithms
4. Performance optimizations
5. UI/UX enhancements

---

## 📄 License
MIT License - See LICENSE file for details

---

## 🙏 Acknowledgments
- Google Gemini API for powerful NLP capabilities
- Streamlit for amazing web framework
- SQLAlchemy for database abstraction
- Open source community for inspiration

---

**Version**: 2.0  
**Last Updated**: 2025-11-16  
**Status**: Production Ready ✅
