import sqlite3
import pandas as pd
from typing import List, Optional
import os

class DatabaseManager:
    def __init__(self, db_path: str = "database/sample.db"):
        self.db_path = db_path
        self.ensure_database_directory()
    
    def ensure_database_directory(self):
        """Ensure database directory exists"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
    
    def initialize_database(self):
        """Initialize database with sample data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create customers table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS customers (
            customer_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT,
            city TEXT,
            state TEXT,
            country TEXT DEFAULT 'USA',
            registration_date DATE,
            age INTEGER,
            income DECIMAL(10,2)
        )
        ''')
        
        # Create products table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS products (
            product_id INTEGER PRIMARY KEY,
            product_name TEXT NOT NULL,
            category TEXT,
            subcategory TEXT,
            price DECIMAL(10,2),
            cost DECIMAL(10,2),
            brand TEXT,
            launch_date DATE
        )
        ''')
        
        # Create orders table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS orders (
            order_id INTEGER PRIMARY KEY,
            customer_id INTEGER,
            order_date DATE,
            total_amount DECIMAL(10,2),
            status TEXT DEFAULT 'completed',
            shipping_cost DECIMAL(10,2) DEFAULT 0,
            discount_amount DECIMAL(10,2) DEFAULT 0,
            FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
        )
        ''')
        
        # Create order_items table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS order_items (
            item_id INTEGER PRIMARY KEY,
            order_id INTEGER,
            product_id INTEGER,
            quantity INTEGER,
            unit_price DECIMAL(10,2),
            discount_percent DECIMAL(5,2) DEFAULT 0,
            FOREIGN KEY (order_id) REFERENCES orders (order_id),
            FOREIGN KEY (product_id) REFERENCES products (product_id)
        )
        ''')
        
        # Create sales_reps table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS sales_reps (
            rep_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            region TEXT,
            hire_date DATE,
            commission_rate DECIMAL(5,4)
        )
        ''')
        
        # Insert sample data
        self._insert_sample_data(cursor)
        
        conn.commit()
        conn.close()
        print("Database initialized successfully with sample data")
    
    def _insert_sample_data(self, cursor):
        """Insert comprehensive sample data into tables"""
        # Sample customers with more diverse data
        customers_data = [
            (1, 'John Doe', 'john@email.com', 'New York', 'NY', 'USA', '2023-01-15', 35, 75000.00),
            (2, 'Jane Smith', 'jane@email.com', 'Los Angeles', 'CA', 'USA', '2023-02-20', 28, 65000.00),
            (3, 'Bob Johnson', 'bob@email.com', 'Chicago', 'IL', 'USA', '2023-03-10', 42, 85000.00),
            (4, 'Alice Brown', 'alice@email.com', 'New York', 'NY', 'USA', '2023-04-05', 31, 70000.00),
            (5, 'Charlie Wilson', 'charlie@email.com', 'Miami', 'FL', 'USA', '2023-05-12', 39, 90000.00),
            (6, 'Diana Martinez', 'diana@email.com', 'Phoenix', 'AZ', 'USA', '2023-06-18', 33, 68000.00),
            (7, 'Edward Davis', 'edward@email.com', 'Seattle', 'WA', 'USA', '2023-07-22', 45, 95000.00),
            (8, 'Fiona Lee', 'fiona@email.com', 'Boston', 'MA', 'USA', '2023-08-14', 29, 72000.00)
        ]
        
        cursor.executemany('''
        INSERT OR REPLACE INTO customers 
        (customer_id, name, email, city, state, country, registration_date, age, income) 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', customers_data)
        
        # Sample products with more variety
        products_data = [
            (1, 'Laptop Pro', 'Electronics', 'Computers', 1299.99, 800.00, 'TechBrand', '2023-01-01'),
            (2, 'Wireless Mouse', 'Electronics', 'Accessories', 29.99, 15.00, 'TechBrand', '2023-01-01'),
            (3, 'Office Chair', 'Furniture', 'Seating', 199.99, 120.00, 'ComfortPlus', '2023-02-01'),
            (4, 'Desk Lamp', 'Furniture', 'Lighting', 49.99, 25.00, 'BrightLight', '2023-02-01'),
            (5, 'Smartphone', 'Electronics', 'Mobile', 699.99, 400.00, 'MobileTech', '2023-03-01'),
            (6, 'Tablet', 'Electronics', 'Mobile', 399.99, 250.00, 'MobileTech', '2023-03-15'),
            (7, 'Keyboard', 'Electronics', 'Accessories', 79.99, 40.00, 'TechBrand', '2023-04-01'),
            (8, 'Monitor', 'Electronics', 'Displays', 299.99, 180.00, 'DisplayCorp', '2023-04-15')
        ]
        
        cursor.executemany('''
        INSERT OR REPLACE INTO products 
        (product_id, product_name, category, subcategory, price, cost, brand, launch_date) 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', products_data)
        
        # Sample orders with time series data
        orders_data = [
            (1, 1, '2023-06-01', 1329.98, 'completed', 10.00, 0.00),
            (2, 2, '2023-06-02', 729.98, 'completed', 15.00, 50.00),
            (3, 3, '2023-06-03', 249.98, 'completed', 12.00, 0.00),
            (4, 1, '2023-06-04', 49.99, 'completed', 5.00, 0.00),
            (5, 4, '2023-06-05', 1999.96, 'shipped', 20.00, 100.00),
            (6, 5, '2023-07-01', 479.98, 'completed', 8.00, 20.00),
            (7, 6, '2023-07-15', 1579.97, 'completed', 18.00, 80.00),
            (8, 7, '2023-08-01', 379.98, 'completed', 10.00, 0.00),
            (9, 8, '2023-08-15', 699.99, 'processing', 12.00, 30.00),
            (10, 2, '2023-09-01', 899.98, 'completed', 15.00, 0.00)
        ]
        
        cursor.executemany('''
        INSERT OR REPLACE INTO orders 
        (order_id, customer_id, order_date, total_amount, status, shipping_cost, discount_amount) 
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', orders_data)
        
        # Sample order items
        order_items_data = [
            (1, 1, 1, 1, 1299.99, 0.0),
            (2, 1, 2, 1, 29.99, 0.0),
            (3, 2, 5, 1, 699.99, 5.0),
            (4, 2, 2, 1, 29.99, 0.0),
            (5, 3, 3, 1, 199.99, 0.0),
            (6, 3, 4, 1, 49.99, 0.0),
            (7, 4, 4, 1, 49.99, 0.0),
            (8, 5, 1, 1, 1299.99, 3.0),
            (9, 5, 5, 1, 699.99, 2.0),
            (10, 6, 6, 1, 399.99, 0.0),
            (11, 6, 7, 1, 79.99, 0.0),
            (12, 7, 1, 1, 1299.99, 5.0),
            (13, 7, 8, 1, 299.99, 0.0),
            (14, 8, 6, 1, 399.99, 0.0),
            (15, 9, 5, 1, 699.99, 4.0),
            (16, 10, 1, 1, 1299.99, 0.0)
        ]
        
        cursor.executemany('''
        INSERT OR REPLACE INTO order_items 
        (item_id, order_id, product_id, quantity, unit_price, discount_percent) 
        VALUES (?, ?, ?, ?, ?, ?)
        ''', order_items_data)
        
        # Sample sales reps
        sales_reps_data = [
            (1, 'Mike Johnson', 'East Coast', '2022-01-15', 0.05),
            (2, 'Sarah Davis', 'West Coast', '2022-03-20', 0.045),
            (3, 'Tom Wilson', 'Central', '2021-11-10', 0.055)
        ]
        
        cursor.executemany('''
        INSERT OR REPLACE INTO sales_reps 
        (rep_id, name, region, hire_date, commission_rate) 
        VALUES (?, ?, ?, ?, ?)
        ''', sales_reps_data)
    
    def execute_query(self, sql_query: str) -> Optional[pd.DataFrame]:
        """Execute SQL query and return results as DataFrame"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Add safety check for potentially harmful queries
            query_upper = sql_query.upper().strip()
            if any(dangerous in query_upper for dangerous in ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER']):
                print(f"Potentially dangerous query blocked: {sql_query}")
                return None
            
            df = pd.read_sql_query(sql_query, conn)
            conn.close()
            return df
        except Exception as e:
            print(f"Error executing query: {e}")
            print(f"Query: {sql_query}")
            return None
    
    def get_table_names(self) -> List[str]:
        """Get list of all table names in the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
            tables = [row[0] for row in cursor.fetchall()]
            conn.close()
            return tables
        except Exception as e:
            print(f"Error getting table names: {e}")
            return []
    
    def get_schema(self) -> str:
        """Get detailed database schema as formatted string"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            schema_info = []
            tables = self.get_table_names()
            
            for table in tables:
                cursor.execute(f"PRAGMA table_info({table})")
                columns = cursor.fetchall()
                
                schema_info.append(f"\nTable: {table}")
                schema_info.append("Columns:")
                for col in columns:
                    # col = (cid, name, type, notnull, dflt_value, pk)
                    col_info = f"  - {col[1]} ({col[2]})"
                    if col[5]:  # primary key
                        col_info += " [PRIMARY KEY]"
                    if col[3]:  # not null
                        col_info += " [NOT NULL]"
                    schema_info.append(col_info)
                
                # Get sample data
                cursor.execute(f"SELECT * FROM {table} LIMIT 3")
                sample_rows = cursor.fetchall()
                if sample_rows:
                    schema_info.append("Sample data:")
                    for row in sample_rows:
                        schema_info.append(f"  {row}")
            
            conn.close()
            return "\n".join(schema_info)
        
        except Exception as e:
            return f"Error getting schema: {e}"
    
    def get_table_sample(self, table_name: str, limit: int = 5) -> Optional[pd.DataFrame]:
        """Get sample data from a specific table"""
        query = f"SELECT * FROM {table_name} LIMIT {limit}"
        return self.execute_query(query)
    
    def get_table_info(self, table_name: str) -> dict:
        """Get detailed information about a specific table"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get column info
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            
            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            row_count = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                'table_name': table_name,
                'columns': [{'name': col[1], 'type': col[2], 'primary_key': bool(col[5])} for col in columns],
                'row_count': row_count
            }
        except Exception as e:
            print(f"Error getting table info: {e}")
            return {}
