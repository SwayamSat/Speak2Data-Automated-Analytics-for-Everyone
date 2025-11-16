# 🔄 Migration Guide: Domain-Specific to Domain-Free

## Overview

This guide helps you migrate from domain-specific implementations to the new universal, domain-free Speak2Data platform.

## No Breaking Changes! ✅

**Good News**: The new domain-free architecture is **100% backward compatible**. Your existing code will continue to work without any changes.

### What Still Works

```python
# All existing code continues to work
from db_module import DatabaseManager
from nlp_module import NLPProcessor
from sql_generator import SQLGenerator

# Create default database (still works)
db = DatabaseManager()

# Use custom database (still works)
db = DatabaseManager(db_path="business_data.db")

# All existing methods work
schema = db.get_table_schema()
results = db.execute_query("SELECT * FROM customers LIMIT 10")
```

## New Capabilities

### 1. Import CSV Files

**Before** (You had to manually convert):
```python
import pandas as pd
import sqlite3

# Manual process
df = pd.read_csv("data.csv")
conn = sqlite3.connect("temp.db")
df.to_sql("data", conn, if_exists="replace")
conn.close()

# Then use database
db = DatabaseManager(db_path="temp.db")
```

**Now** (One line):
```python
from db_module import DatabaseManager

# Automatic import
db = DatabaseManager.create_from_csv("data.csv")
# Ready to use!
```

### 2. Import Excel Files

**Before** (Manual steps):
```python
import pandas as pd
import sqlite3

df = pd.read_excel("data.xlsx")
conn = sqlite3.connect("temp.db")
df.to_sql("data", conn, if_exists="replace")
conn.close()

db = DatabaseManager(db_path="temp.db")
```

**Now**:
```python
db = DatabaseManager.create_from_excel("data.xlsx")
```

### 3. Import Parquet Files

**New Feature**:
```python
db = DatabaseManager.create_from_parquet("data.parquet")
```

### 4. Import DataFrame Directly

**Before** (Workaround):
```python
import pandas as pd
import sqlite3

df = pd.DataFrame(...)  # Your data
conn = sqlite3.connect("temp.db")
df.to_sql("data", conn, if_exists="replace")
conn.close()

db = DatabaseManager(db_path="temp.db")
```

**Now**:
```python
db = DatabaseManager.create_from_dataframe(df, table_name="my_data")
```

## Web Interface Changes

### File Upload

**Before**:
- Only SQLite databases (.db, .sqlite, .sqlite3)

**Now**:
- SQLite databases (.db, .sqlite, .sqlite3)
- CSV files (.csv)
- Excel files (.xlsx, .xls)
- Parquet files (.parquet)

### UI Text Changes

**Before**:
```
"Upload Custom Database"
type=['db', 'sqlite', 'sqlite3']
```

**Now**:
```
"Upload Database or Data File"
type=['db', 'sqlite', 'sqlite3', 'csv', 'xlsx', 'xls', 'parquet']
```

## Code Migration Examples

### Example 1: Migrating Custom Data Loading

**Old Way** (Custom script):
```python
# load_data.py
import pandas as pd
import sqlite3

def load_custom_data(csv_path):
    df = pd.read_csv(csv_path)
    conn = sqlite3.connect("custom.db")
    df.to_sql("data", conn, if_exists="replace", index=False)
    conn.close()
    print("Data loaded!")

# Usage
load_custom_data("sales.csv")

# Then in app
from db_module import DatabaseManager
db = DatabaseManager(db_path="custom.db")
```

**New Way** (Built-in):
```python
from db_module import DatabaseManager

# One line!
db = DatabaseManager.create_from_csv("sales.csv")
```

### Example 2: Migrating Multi-File Loading

**Old Way**:
```python
import pandas as pd
import sqlite3

# Load multiple files
df1 = pd.read_csv("customers.csv")
df2 = pd.read_csv("orders.csv")
df3 = pd.read_csv("products.csv")

# Create database
conn = sqlite3.connect("business.db")
df1.to_sql("customers", conn, if_exists="replace", index=False)
df2.to_sql("orders", conn, if_exists="replace", index=False)
df3.to_sql("products", conn, if_exists="replace", index=False)
conn.close()

# Use database
db = DatabaseManager(db_path="business.db")
```

**New Way** (Option 1 - Sequential):
```python
from db_module import DatabaseManager
import pandas as pd
from sqlalchemy import create_engine
import tempfile
import os

# Create temp database path
temp_db = os.path.join(tempfile.gettempdir(), "business.db")
engine = create_engine(f"sqlite:///{temp_db}")

# Load all files
pd.read_csv("customers.csv").to_sql("customers", engine, if_exists="replace", index=False)
pd.read_csv("orders.csv").to_sql("orders", engine, if_exists="replace", index=False)
pd.read_csv("products.csv").to_sql("products", engine, if_exists="replace", index=False)

# Use database
db = DatabaseManager(db_path=temp_db)
```

**New Way** (Option 2 - Via UI):
```
1. Upload customers.csv → creates customers table
2. System asks: "Upload more files?"
3. Upload orders.csv → adds orders table to same DB
4. Upload products.csv → adds products table to same DB

Note: Currently supports single file upload. 
Multi-file support via UI is on roadmap.
```

### Example 3: Migrating Domain-Specific Code

**Old Way** (Hardcoded business logic):
```python
# Old domain-specific code
def get_top_customers():
    query = "SELECT name, SUM(total) FROM customers JOIN orders ON customers.id = orders.customer_id GROUP BY name ORDER BY SUM(total) DESC LIMIT 10"
    return db.execute_query(query)

def get_product_sales():
    query = "SELECT category, SUM(price * quantity) FROM products JOIN sales ON products.id = sales.product_id GROUP BY category"
    return db.execute_query(query)
```

**New Way** (Domain-free):
```python
from nlp_module import NLPProcessor
from sql_generator import SQLGenerator

# Initialize with ANY schema
schema = db.get_table_schema()
nlp = NLPProcessor(schema_info={"tables": schema})
sql_gen = SQLGenerator(nlp)

# Natural language queries work with ANY domain
result = sql_gen.generate_query("Show me top 10 items by total value")
sql = result["sql_query"]
data = db.execute_query(sql)

# Works for:
# - "Show me top customers" (retail)
# - "Show me top patients" (healthcare)
# - "Show me top accounts" (banking)
# - "Show me top sensors" (IoT)
```

## Schema Changes

### Before (Hardcoded)

**Old schema references**:
```python
# DON'T DO THIS ANYMORE
def get_sales_by_category():
    # Assumes 'sales' and 'category' exist
    return db.execute_query("SELECT category, SUM(amount) FROM sales GROUP BY category")
```

### After (Dynamic)

**New schema-aware code**:
```python
# DO THIS INSTEAD
schema = db.get_table_schema()

# Check what tables exist
if 'sales' in schema:
    # Use sales table
    pass
elif 'transactions' in schema:
    # Use transactions table
    pass

# Or better - use NLP to generate query
result = sql_gen.generate_query("Show me totals by category")
data = db.execute_query(result["sql_query"])
```

## Database Connection Changes

### No Changes Needed!

All existing connection methods still work:

```python
# SQLite file (works as before)
db = DatabaseManager(db_path="data.db")

# Connection string (works as before)
db = DatabaseManager(connection_string="postgresql://user:pass@host/dbname")

# Default database (works as before)
db = DatabaseManager()  # Creates sample business_data.db
```

## ML Pipeline Changes

### No Changes Needed!

ML pipeline already works with any DataFrame:

```python
from ml_pipeline_simple import SimpleMLPipeline

# Works with ANY data
ml = SimpleMLPipeline()
analysis = ml.analyze_data(df, target_column="any_column_name")
features, target = ml.prepare_data(df, target_column="any_column_name")
results = ml.train_model(features, target, problem_type="regression")
```

## NLP/SQL Generator Changes

### No Changes Needed!

Already domain-agnostic:

```python
# Works with ANY schema
nlp = NLPProcessor(schema_info={"tables": schema})
sql_gen = SQLGenerator(nlp)

# Generates queries for YOUR database
result = sql_gen.generate_query("Show me top 10 records")
```

## Best Practices for Migration

### 1. Remove Hardcoded Table/Column Names

**Before**:
```python
def get_customer_orders():
    return db.execute_query("SELECT * FROM customers JOIN orders ON customers.id = orders.customer_id")
```

**After**:
```python
def get_related_data(entity1, entity2, relationship_hint):
    query_text = f"Show me {entity1} with their {entity2}"
    result = sql_gen.generate_query(query_text)
    return db.execute_query(result["sql_query"])

# Works for:
# - get_related_data("customers", "orders", "customer_id")
# - get_related_data("patients", "visits", "patient_id")
# - get_related_data("accounts", "transactions", "account_id")
```

### 2. Use Schema Introspection

**Before**:
```python
# Assumed schema
tables = ["customers", "orders", "products"]
```

**After**:
```python
# Discover schema
schema = db.get_table_schema()
tables = list(schema.keys())
print(f"Found {len(tables)} tables: {', '.join(tables)}")
```

### 3. Generate Queries Dynamically

**Before**:
```python
# Hardcoded queries
QUERIES = {
    "top_customers": "SELECT name, SUM(total) FROM customers...",
    "best_products": "SELECT name, SUM(sales) FROM products...",
}
```

**After**:
```python
# Generate queries dynamically
def run_query(natural_language):
    result = sql_gen.generate_query(natural_language)
    if result["is_valid"]:
        return db.execute_query(result["sql_query"])
    else:
        return f"Error: {result.get('error', 'Invalid query')}"

# Works with any question
data = run_query("Show me top items by value")
```

## Checklist for Migration

- [ ] Review code for hardcoded table names → Replace with schema introspection
- [ ] Review code for hardcoded column names → Use dynamic queries
- [ ] Test with different databases (healthcare, finance, etc.)
- [ ] Update documentation to reflect domain-free nature
- [ ] Add file upload capability if building custom UI
- [ ] Test CSV/Excel/Parquet import functionality
- [ ] Verify ML pipeline works with different schemas
- [ ] Update error handling for unknown tables/columns

## Common Pitfalls

### ❌ Pitfall 1: Assuming Tables Exist
```python
# Bad - assumes 'customers' exists
df = db.execute_query("SELECT * FROM customers")
```

```python
# Good - check first
schema = db.get_table_schema()
if 'customers' in schema:
    df = db.execute_query("SELECT * FROM customers")
else:
    print(f"Available tables: {', '.join(schema.keys())}")
```

### ❌ Pitfall 2: Hardcoded Column Names
```python
# Bad - assumes columns exist
result = sql_gen.generate_query("Show me total sales by category")
# What if there's no 'sales' or 'category' column?
```

```python
# Good - let AI handle it
result = sql_gen.generate_query("Show me totals grouped by type")
# AI finds appropriate columns from actual schema
```

### ❌ Pitfall 3: Not Handling Import Errors
```python
# Bad - no error handling
db = DatabaseManager.create_from_csv("data.csv")
```

```python
# Good - handle errors
try:
    db = DatabaseManager.create_from_csv("data.csv")
    print("✓ Database loaded successfully")
except Exception as e:
    print(f"✗ Error loading database: {e}")
    # Fallback or error recovery
```

## Testing Your Migration

### Test Suite

```python
# test_migration.py
from db_module import DatabaseManager
from nlp_module import NLPProcessor
from sql_generator import SQLGenerator

def test_csv_import():
    """Test CSV import functionality"""
    db = DatabaseManager.create_from_csv("test_data.csv")
    schema = db.get_table_schema()
    assert len(schema) > 0, "No tables found"
    print("✓ CSV import works")

def test_schema_discovery():
    """Test schema discovery"""
    db = DatabaseManager.create_from_csv("test_data.csv")
    schema = db.get_table_schema()
    assert isinstance(schema, dict), "Schema should be a dict"
    assert len(schema) > 0, "Schema should have tables"
    print(f"✓ Schema discovery works: {list(schema.keys())}")

def test_query_generation():
    """Test dynamic query generation"""
    db = DatabaseManager.create_from_csv("test_data.csv")
    schema = db.get_table_schema()
    nlp = NLPProcessor(schema_info={"tables": schema})
    sql_gen = SQLGenerator(nlp)
    
    result = sql_gen.generate_query("Show me all data")
    assert result["is_valid"], "Query should be valid"
    print("✓ Query generation works")

def test_query_execution():
    """Test query execution"""
    db = DatabaseManager.create_from_csv("test_data.csv")
    df = db.execute_query("SELECT * FROM test_data LIMIT 10")
    assert len(df) > 0, "Should have results"
    print("✓ Query execution works")

if __name__ == "__main__":
    test_csv_import()
    test_schema_discovery()
    test_query_generation()
    test_query_execution()
    print("\n✅ All migration tests passed!")
```

## Support

### Getting Help

If you encounter issues during migration:

1. **Check the error message** - Often tells you what's missing
2. **Review the schema** - Use `db.get_table_schema()` to see what exists
3. **Test with sample data** - Use default database first
4. **Check documentation** - README.md and USAGE_GUIDE.md
5. **Review examples** - See DOMAIN_FREE_ARCHITECTURE.md

### Reporting Issues

When reporting issues, include:
- Database schema (from `get_table_schema()`)
- Error message
- Sample data (if possible)
- Expected vs actual behavior

## Summary

**Key Points**:
1. ✅ No breaking changes - existing code works as-is
2. 🆕 New file import methods available
3. 🌍 Now works with ANY database domain
4. 🔍 Schema introspection is automatic
5. 🤖 AI adapts to YOUR data structure

**Migration is optional** - existing functionality preserved while new capabilities added.

**Start using new features** by simply uploading CSV/Excel/Parquet files through the UI or using the new static factory methods in code.

🚀 **Your application is now truly universal!**
