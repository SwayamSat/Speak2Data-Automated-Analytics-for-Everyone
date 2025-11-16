# Testing Custom Query Features - Quick Guide

## 🚀 Quick Start Test

### Step 1: Start the Application
```powershell
cd "D:\Projects\NL TO SQL"
.\activate_env.ps1
streamlit run app.py
```

### Step 2: Upload a Database
1. Click "Upload Database"
2. Choose any `.db`, `.csv`, `.xlsx`, or `.parquet` file
3. Wait for schema detection confirmation

---

## 🧪 Test Scenarios

### Test 1: Basic Custom Query with AI Guidance ⭐

**Goal**: Verify AI guidance feature works

**Steps**:
1. Type: "What are the top 5 items by value?"
2. Click **"💡 Get AI Guidance"** button
3. Review the guidance panel

**Expected Results**:
✅ Panel expands showing:
- Understanding of your query
- Query type (e.g., "data_retrieval")
- Suggested approach (step-by-step)
- Key insights (3-5 bullets)
- SQL hint (sample query structure)
- Visualization suggestions
- Follow-up questions (3-5)

**If API Quota Exceeded**:
✅ Shows fallback guidance with generic suggestions
✅ App continues to work normally

---

### Test 2: Execute Query and Get AI Explanation ⭐⭐

**Goal**: Verify end-to-end query execution with AI insights

**Steps**:
1. Type: "Show me total sales by category"
2. Click **"🔍 Analyze"** button
3. Wait for results

**Expected Results**:
✅ Query Analysis section shows generated SQL
✅ Data Table displays results
✅ **🤖 AI-Powered Analysis** section shows:
   - Natural language explanation
   - Key metrics highlighted
   - Business insights
   - Styled blue card with formatted text

**Fallback Behavior** (if quota exceeded):
✅ Shows "⚠️ API quota limit reached. Using basic analysis."
✅ Displays comprehensive fallback analysis with:
   - 📊 Data Analysis Results
   - Record count
   - Column info
   - Key metrics summary
   - 💡 Insight paragraph

---

### Test 3: Follow-up Questions ⭐⭐

**Goal**: Test interactive exploration

**Steps**:
1. Execute any query (see Test 2)
2. Scroll to **"💭 Explore Further"** section
3. Click any suggested question button

**Expected Results**:
✅ 3-5 follow-up questions displayed as buttons
✅ Each button has ❓ emoji and full question text
✅ Clicking a button:
   - Loads question into text area
   - Page refreshes
   - Ready to analyze new question

**Fallback Behavior**:
✅ Generic but relevant questions based on data structure
✅ Questions reference actual column names from results

---

### Test 4: Complex Query Handling ⭐⭐⭐

**Goal**: Test robust response parsing

**Steps**:
1. Type: "Compare average values across different categories and show trends over time"
2. Click **"💡 Get AI Guidance"**
3. Click **"🔍 Analyze"**

**Expected Results**:
✅ Guidance provides detailed approach for complex query
✅ SQL generated handles multiple tables/joins if needed
✅ Results include appropriate visualizations (grouped bar chart, line chart)
✅ Explanation mentions comparisons and trends
✅ Follow-up questions probe deeper into specific segments

---

### Test 5: Error Recovery ⭐⭐⭐

**Goal**: Test graceful degradation when API fails

**Test 5a: Empty Query**
```
Steps: Leave text area empty, click "Analyze"
Expected: ⚠️ "Please enter a question to analyze."
```

**Test 5b: Invalid Schema Reference**
```
Steps: Type "Show me data from nonexistent_table"
Expected: 
- SQL generated uses closest matching table
- Or fallback SQL uses first available table
- Query executes without crash
```

**Test 5c: API Quota Limit**
```
Steps: Make many API calls rapidly (15-20 queries)
Expected:
- "⚠️ API quota limit reached" message
- Fallback processing kicks in automatically
- All features continue working
- No crashes or blank screens
```

---

## 📊 Feature-by-Feature Verification

### JSON Extraction (`_extract_json_from_response`)

**Test**: Check debug expander after query execution

**Verify**:
- [ ] Parsed Query shows valid JSON structure
- [ ] No markdown artifacts (```json, ```)
- [ ] All required fields present
- [ ] No parsing errors in logs

---

### SQL Cleaning (`_clean_sql_response`)

**Test**: Look at "Generated SQL" display

**Verify**:
- [ ] No markdown formatting (```sql)
- [ ] No explanatory text like "Here's the query:"
- [ ] Starts directly with SELECT/WITH
- [ ] Ends at semicolon (or before explanatory text)
- [ ] Clean, executable SQL

---

### SQL Validation (`_is_valid_sql`)

**Test**: Try various query types

**Valid Queries to Test**:
```sql
SELECT * FROM table
SELECT col1, col2 FROM table WHERE condition
WITH cte AS (...) SELECT * FROM cte
```

**Invalid Queries (should be caught)**:
```sql
This is not SQL
FROM table SELECT *  (wrong order)
SELECT * WHERE condition  (missing FROM)
SELECT ((( unbalanced parentheses
```

**Verify**:
- [ ] Valid queries execute
- [ ] Invalid queries trigger fallback
- [ ] No SQL injection patterns allowed

---

### Custom Query Generation (`generate_custom_query`)

**Test**: Click "💡 Get AI Guidance" multiple times

**Verify**:
- [ ] Response includes all 7 fields
- [ ] Understanding matches your intent
- [ ] Suggested approach is logical
- [ ] SQL hint is schema-aware
- [ ] Follow-up questions are relevant
- [ ] Fallback works if API fails

---

## 🎯 Performance Tests

### Response Time

**Measure**:
1. Start timer when clicking button
2. Stop when results appear

**Targets**:
- [ ] AI Guidance: < 5 seconds
- [ ] Query Analysis: < 3 seconds  
- [ ] Query Execution: < 2 seconds
- [ ] Explanation: < 5 seconds
- [ ] Follow-ups: < 3 seconds

**Note**: Times may vary based on:
- API response speed
- Database size
- Query complexity

---

### Memory Usage

**Monitor**: Check "Memory Usage" metric in Results Summary

**Verify**:
- [ ] Small datasets (< 1000 rows): < 100 KB
- [ ] Medium datasets (1000-10000 rows): < 1 MB
- [ ] Large datasets (> 10000 rows): Shown with KB/MB units
- [ ] No memory leaks on multiple queries

---

## 🐛 Known Issues & Workarounds

### Issue 1: API Rate Limit Reached
**Symptom**: "429" error or quota messages  
**Workaround**: Wait 60 seconds, or continue using fallback features  
**Prevention**: Reduce query frequency during testing

### Issue 2: Complex SQL Not Parsed
**Symptom**: Generated SQL has syntax errors  
**Workaround**: Click "Get AI Guidance" first, then simplify question  
**Prevention**: Use clearer, more specific questions

### Issue 3: Slow Initial Load
**Symptom**: First query takes longer  
**Explanation**: Model initialization overhead  
**Normal**: Subsequent queries much faster

---

## ✅ Final Checklist

### Before Deployment
- [ ] All 5 test scenarios pass
- [ ] Error recovery works gracefully
- [ ] API quota fallbacks functional
- [ ] UI displays properly on different screen sizes
- [ ] No console errors in browser dev tools
- [ ] Session state persists correctly
- [ ] Database upload works with all formats

### User Experience
- [ ] Buttons clearly labeled with emojis
- [ ] Loading spinners show during processing
- [ ] Error messages are user-friendly
- [ ] Results are easy to understand
- [ ] Follow-up questions are clickable
- [ ] Guidance panel is informative

### Performance
- [ ] Queries execute in reasonable time
- [ ] Page doesn't freeze during API calls
- [ ] Memory usage stays reasonable
- [ ] No infinite loops or hangs

---

## 📝 Test Results Template

```
Test Date: ___________
Tester: ___________
Environment: Local / Cloud

Test 1 - AI Guidance:        [ ] PASS  [ ] FAIL  [ ] SKIP
Test 2 - Query Execution:    [ ] PASS  [ ] FAIL  [ ] SKIP
Test 3 - Follow-up Questions:[ ] PASS  [ ] FAIL  [ ] SKIP
Test 4 - Complex Queries:    [ ] PASS  [ ] FAIL  [ ] SKIP
Test 5 - Error Recovery:     [ ] PASS  [ ] FAIL  [ ] SKIP

Notes:
_________________________________________________________________
_________________________________________________________________
_________________________________________________________________

Issues Found:
_________________________________________________________________
_________________________________________________________________
_________________________________________________________________

Overall Status: [ ] READY FOR PRODUCTION  [ ] NEEDS FIXES
```

---

## 🎓 Tips for Effective Testing

### 1. Test with Real Data
Use actual business databases, not just sample data. Real data exposes edge cases.

### 2. Test Edge Cases
- Empty tables
- Single row tables  
- Tables with 100+ columns
- Very long text columns
- NULL values
- Special characters in data

### 3. Test Different Query Types
- Simple lookups: "Show me customer X"
- Aggregations: "Total sales by category"
- Comparisons: "Compare Q1 vs Q2"
- Trends: "Sales over last 6 months"
- Top N: "Top 10 products"
- Complex joins: "Customers with no orders"

### 4. Test User Workflows
- New user first time
- Power user rapid queries
- Exploration with follow-ups
- Copy/paste results
- Export data

### 5. Monitor Logs
Watch terminal output for:
- API warnings
- Fallback triggers
- Error messages
- Performance issues

---

## 🚨 Emergency Rollback

If critical issues found:

```powershell
# Restore previous version (if using git)
git log --oneline  # Find commit before changes
git checkout <commit_hash> nlp_module.py
git checkout <commit_hash> app.py

# Restart app
streamlit run app.py
```

---

## 📞 Support

If you encounter issues:

1. Check logs in terminal
2. Review `CUSTOM_QUERY_FEATURES.md` for troubleshooting
3. Test with fallback mode (disconnect API key temporarily)
4. Verify database schema is valid
5. Try with sample data first

---

**Happy Testing!** 🎉

---

**Version**: 2.0  
**Last Updated**: 2025-11-16
