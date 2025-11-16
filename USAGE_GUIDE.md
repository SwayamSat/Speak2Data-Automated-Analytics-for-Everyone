# 🌍 Universal Database Usage Guide

## Quick Start with Different Database Types

### 1. SQLite Database (.db, .sqlite, .sqlite3)

#### Using Existing Database
```python
from db_module import DatabaseManager

# Connect to existing SQLite database
db_manager = DatabaseManager(db_path="path/to/your/database.db")

# Or use the web interface:
# 1. Open Speak2Data app
# 2. Upload your .db file
# 3. Start asking questions!
```

#### Example Databases
- **Medical**: patient_records.db
- **Financial**: transactions.db  
- **HR**: employee_data.db
- **IoT**: sensor_data.db

---

### 2. CSV Files (.csv)

#### From File
```python
from db_module import DatabaseManager

# Import CSV into SQLite automatically
db_manager = DatabaseManager.create_from_csv("data.csv")

# Or specify table name
db_manager = DatabaseManager.create_from_csv("data.csv", table_name="my_data")
```

#### Via Web Interface
1. Open Speak2Data
2. Click "Upload Database or Data File"
3. Select your CSV file
4. System automatically:
   - Imports data into SQLite
   - Detects column types
   - Creates table with filename
5. Ask questions immediately!

#### CSV Best Practices
- ✅ First row should contain column names
- ✅ Use consistent data types per column
- ✅ Avoid special characters in column names
- ✅ Use underscores instead of spaces: `customer_id` not `Customer ID`

---

### 3. Excel Files (.xlsx, .xls)

#### From File
```python
from db_module import DatabaseManager

# Import Excel (first sheet)
db_manager = DatabaseManager.create_from_excel("data.xlsx")

# Import specific sheet
db_manager = DatabaseManager.create_from_excel("data.xlsx", sheet_name="Sales Data")

# Custom table name
db_manager = DatabaseManager.create_from_excel("data.xlsx", table_name="sales")
```

#### Via Web Interface
1. Upload Excel file
2. System imports first sheet automatically
3. Table name derived from filename
4. Ready to query!

#### Excel Best Practices
- ✅ Use first row for column headers
- ✅ Keep data in a single continuous table (no merged cells)
- ✅ Remove formatting/colors (optional but cleaner)
- ✅ Export complex sheets to CSV first for better compatibility

---

### 4. Parquet Files (.parquet)

#### From File
```python
from db_module import DatabaseManager

# Import Parquet file
db_manager = DatabaseManager.create_from_parquet("data.parquet")

# Custom table name
db_manager = DatabaseManager.create_from_parquet("data.parquet", table_name="my_table")
```

#### Via Web Interface
1. Upload .parquet file
2. Automatic import to SQLite
3. Preserves data types efficiently
4. Start querying!

#### When to Use Parquet
- ✅ Large datasets (>100MB)
- ✅ Need columnar storage efficiency
- ✅ Working with data science tools
- ✅ Want fast query performance

---

### 5. Pandas DataFrame (Programmatic)

#### From Code
```python
import pandas as pd
from db_module import DatabaseManager

# Create or load DataFrame
df = pd.read_csv("data.csv")
# or
df = pd.read_excel("data.xlsx")
# or create from scratch

# Import into database
db_manager = DatabaseManager.create_from_dataframe(df, table_name="my_data")

# Now use with NLP queries
from nlp_module import NLPProcessor
from sql_generator import SQLGenerator

nlp = NLPProcessor(schema_info={"tables": db_manager.get_table_schema()})
sql_gen = SQLGenerator(nlp)

result = sql_gen.generate_query("Show me top 10 records")
```

---

## Domain-Specific Examples

### Healthcare Database

#### Sample Schema
```sql
-- patients.csv
patient_id, name, date_of_birth, gender, diagnosis
1, John Doe, 1980-05-15, M, Diabetes
2, Jane Smith, 1975-08-22, F, Hypertension

-- visits.csv
visit_id, patient_id, visit_date, doctor_name, notes
1, 1, 2024-01-15, Dr. Johnson, Routine checkup
2, 1, 2024-02-10, Dr. Johnson, Follow-up
```

#### Usage
```bash
# Upload patients.csv and visits.csv
# System creates two tables: patients, visits
```

#### Natural Language Queries
- "Show me all patients with diabetes"
- "What's the average number of visits per patient?"
- "Which doctors have the most patients?"
- "Show me patient visit trends over time"
- "Predict readmission risk based on visit history"

---

### Financial/Banking Database

#### Sample Schema
```sql
-- accounts.csv
account_id, customer_name, account_type, balance, opening_date
1001, Alice Johnson, Savings, 15000.00, 2020-01-15
1002, Bob Smith, Checking, 3500.00, 2019-05-20

-- transactions.csv
transaction_id, account_id, transaction_date, amount, transaction_type
1, 1001, 2024-01-10, -500.00, Withdrawal
2, 1001, 2024-01-15, 1000.00, Deposit
```

#### Natural Language Queries
- "Show me accounts with balance over $10,000"
- "What's the total transaction volume by account type?"
- "Find accounts with suspicious transaction patterns"
- "Show me monthly transaction trends"
- "Predict account closure based on transaction history"

---

### HR/Employee Database

#### Sample Schema
```sql
-- employees.csv
employee_id, name, department, position, salary, hire_date
1, John Doe, Engineering, Senior Dev, 95000, 2020-03-15
2, Jane Smith, Marketing, Manager, 85000, 2019-07-01

-- attendance.csv
attendance_id, employee_id, date, status, hours_worked
1, 1, 2024-01-15, Present, 8.0
2, 1, 2024-01-16, Present, 8.5
```

#### Natural Language Queries
- "Show me average salary by department"
- "What's the attendance rate for each employee?"
- "Find employees hired in the last year"
- "Show me department headcount over time"
- "Predict employee turnover based on attendance"

---

### IoT Sensor Database

#### Sample Schema
```sql
-- sensors.csv
sensor_id, location, sensor_type, installation_date
S001, Warehouse A, Temperature, 2023-01-15
S002, Warehouse A, Humidity, 2023-01-15

-- readings.csv
reading_id, sensor_id, timestamp, value
1, S001, 2024-01-15 10:00:00, 22.5
2, S001, 2024-01-15 10:15:00, 23.1
```

#### Natural Language Queries
- "Show me temperature readings over the last 24 hours"
- "What's the average humidity by location?"
- "Find sensors with readings outside normal range"
- "Show me sensor data trends by hour"
- "Predict sensor failures based on reading patterns"

---

### Retail/E-commerce Database

#### Sample Schema
```sql
-- customers.csv
customer_id, name, email, city, registration_date
1, John Doe, john@example.com, New York, 2023-01-15

-- orders.csv
order_id, customer_id, order_date, total_amount, status
1001, 1, 2024-01-10, 150.00, Completed

-- products.csv
product_id, name, category, price
101, Laptop, Electronics, 999.99
```

#### Natural Language Queries
- "Show me top customers by order value"
- "What are the best-selling products?"
- "Show me sales trends by category"
- "What's the average order value by city?"
- "Predict customer churn based on order history"

---

## Common Patterns

### Pattern 1: Single Table Analysis
```
Upload: sales_data.csv
Query: "Show me total sales by month"
Query: "What's the average sale amount?"
Query: "Find the top 10 sales"
```

### Pattern 2: Multi-Table Analysis
```
Upload: 
  - customers.csv
  - orders.csv
  - products.csv
  
Query: "Show me customer orders with product names"
Query: "What's the total revenue per customer?"
Query: "Which products are most popular?"
```

### Pattern 3: Time Series Analysis
```
Upload: sensor_readings.csv (with timestamp column)
Query: "Show me readings over time"
Query: "What are the hourly trends?"
Query: "Predict next week's values"
```

### Pattern 4: Classification/Prediction
```
Upload: employee_data.csv
Query: "Predict employee turnover"
Query: "Classify employees by performance"
Query: "What factors predict high performance?"
```

---

## Tips & Best Practices

### Data Preparation
1. **Clean column names**: Use lowercase, underscores, no spaces
   - ✅ `customer_id`, `order_date`, `total_amount`
   - ❌ `Customer ID`, `Order Date`, `Total Amount`

2. **Consistent data types**: Keep each column's data type consistent
   - ✅ All dates in same format: `2024-01-15`
   - ❌ Mixed formats: `01/15/2024`, `15-Jan-2024`

3. **Handle missing values**: Remove or fill missing values
   - Remove rows with critical missing data
   - Fill numeric nulls with 0 or median
   - Fill text nulls with "Unknown" or "N/A"

4. **Use meaningful names**: Descriptive table and column names
   - ✅ `monthly_sales`, `customer_transactions`
   - ❌ `data1`, `table2`, `col_x`

### File Size Recommendations
- **CSV/Excel**: Up to 500MB (then consider Parquet)
- **Parquet**: Any size, optimized for large datasets
- **SQLite**: Up to 140TB (practically unlimited for most uses)

### Query Performance
- Add filters to limit result size: "Show me sales in 2024"
- Use specific columns: "Show me name and sales" not "Show all data"
- For large tables, use TOP/LIMIT: "Show top 1000 records"

---

## Troubleshooting

### Issue: "Table doesn't exist"
**Solution**: Check file uploaded correctly, verify table name in sidebar schema view

### Issue: "Column doesn't exist"  
**Solution**: Check actual column names in sidebar, AI suggests available columns

### Issue: "No results found"
**Solution**: Verify data exists, try broader query, check filters

### Issue: "Memory error with large file"
**Solution**: Use Parquet format, filter data before upload, or split into multiple files

---

## Next Steps

1. **Start simple**: Upload a single CSV with clean data
2. **Test basic queries**: "Show me all data", "Count total records"
3. **Try aggregations**: "Show me totals by category"
4. **Explore ML**: "Predict X based on Y"
5. **Upload more tables**: Add related data files
6. **Ask complex questions**: Multi-table joins, time series, predictions

**Remember**: The system adapts to YOUR data. No configuration needed! 🚀
