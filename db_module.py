"""
Database module for Speak2Data platform.
Handles SQLite database setup, sample data generation, and query execution.
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
import os
from typing import List, Dict, Any, Optional


class DatabaseManager:
    """Manages database operations for the Speak2Data platform."""
    
    def __init__(self, db_path: str = "business_data.db", custom_db_path: Optional[str] = None):
        """Initialize database manager.
        
        Args:
            db_path: Path to default SQLite database file
            custom_db_path: Path to custom uploaded database file
        """
        if custom_db_path:
            self.db_path = custom_db_path
            self.is_custom_db = True
        else:
            self.db_path = db_path
            self.is_custom_db = False
            
        self.engine = create_engine(f"sqlite:///{self.db_path}")
        
        # Only create tables and populate data for default database
        if not self.is_custom_db:
            self._create_tables()
            self._populate_sample_data()
    
    def _create_tables(self):
        """Create database tables for business data."""
        with self.engine.connect() as conn:
            # Customers table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS customers (
                    customer_id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    email TEXT UNIQUE,
                    phone TEXT,
                    city TEXT,
                    state TEXT,
                    registration_date DATE,
                    customer_segment TEXT
                )
            """))
            
            # Products table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS products (
                    product_id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    category TEXT,
                    subcategory TEXT,
                    price DECIMAL(10,2),
                    cost DECIMAL(10,2),
                    supplier TEXT,
                    launch_date DATE
                )
            """))
            
            # Orders table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS orders (
                    order_id INTEGER PRIMARY KEY,
                    customer_id INTEGER,
                    order_date DATE,
                    total_amount DECIMAL(10,2),
                    status TEXT,
                    shipping_city TEXT,
                    shipping_state TEXT,
                    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
                )
            """))
            
            # Order items table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS order_items (
                    order_item_id INTEGER PRIMARY KEY,
                    order_id INTEGER,
                    product_id INTEGER,
                    quantity INTEGER,
                    unit_price DECIMAL(10,2),
                    total_price DECIMAL(10,2),
                    FOREIGN KEY (order_id) REFERENCES orders(order_id),
                    FOREIGN KEY (product_id) REFERENCES products(product_id)
                )
            """))
            
            # Sales table (aggregated view)
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS sales (
                    sale_id INTEGER PRIMARY KEY,
                    product_id INTEGER,
                    customer_id INTEGER,
                    sale_date DATE,
                    quantity INTEGER,
                    unit_price DECIMAL(10,2),
                    total_amount DECIMAL(10,2),
                    region TEXT,
                    sales_rep TEXT,
                    FOREIGN KEY (product_id) REFERENCES products(product_id),
                    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
                )
            """))
            
            conn.commit()
    
    def _populate_sample_data(self):
        """Generate and populate sample business data."""
        with self.engine.connect() as conn:
            # Check if data already exists
            result = conn.execute(text("SELECT COUNT(*) FROM customers")).fetchone()
            if result[0] > 0:
                return  # Data already exists
            
            # Generate sample data
            self._generate_customers(conn)
            self._generate_products(conn)
            self._generate_orders(conn)
            self._generate_sales(conn)
            
            conn.commit()
    
    def _generate_customers(self, conn):
        """Generate sample customer data."""
        np.random.seed(42)
        n_customers = 1000
        
        cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 
                 'San Antonio', 'San Diego', 'Dallas', 'San Jose', 'Austin', 'Jacksonville']
        states = ['NY', 'CA', 'IL', 'TX', 'AZ', 'PA', 'TX', 'CA', 'TX', 'CA', 'TX', 'FL']
        segments = ['Premium', 'Standard', 'Basic', 'Enterprise']
        
        customers_data = []
        for i in range(n_customers):
            city_idx = np.random.randint(0, len(cities))
            segment_idx = np.random.randint(0, len(segments))
            
            customer = {
                'customer_id': i + 1,
                'name': f"Customer {i + 1}",
                'email': f"customer{i + 1}@email.com",
                'phone': f"555-{np.random.randint(100, 999)}-{np.random.randint(1000, 9999)}",
                'city': cities[city_idx],
                'state': states[city_idx],
                'registration_date': (datetime.now() - timedelta(days=np.random.randint(1, 1095))).strftime('%Y-%m-%d'),
                'customer_segment': segments[segment_idx]
            }
            customers_data.append(customer)
        
        df = pd.DataFrame(customers_data)
        df.to_sql('customers', conn, if_exists='append', index=False)
    
    def _generate_products(self, conn):
        """Generate sample product data."""
        np.random.seed(42)
        n_products = 200
        
        categories = ['Electronics', 'Clothing', 'Home & Garden', 'Sports', 'Books', 'Toys']
        subcategories = {
            'Electronics': ['Smartphones', 'Laptops', 'Tablets', 'Accessories'],
            'Clothing': ['Men\'s', 'Women\'s', 'Children\'s', 'Accessories'],
            'Home & Garden': ['Furniture', 'Decor', 'Tools', 'Appliances'],
            'Sports': ['Fitness', 'Outdoor', 'Team Sports', 'Water Sports'],
            'Books': ['Fiction', 'Non-Fiction', 'Educational', 'Children\'s'],
            'Toys': ['Action Figures', 'Board Games', 'Educational', 'Outdoor']
        }
        
        suppliers = ['Supplier A', 'Supplier B', 'Supplier C', 'Supplier D', 'Supplier E']
        
        products_data = []
        for i in range(n_products):
            category = np.random.choice(categories)
            subcategory = np.random.choice(subcategories[category])
            base_price = np.random.uniform(10, 500)
            cost = base_price * np.random.uniform(0.4, 0.7)
            
            product = {
                'product_id': i + 1,
                'name': f"{subcategory} Product {i + 1}",
                'category': category,
                'subcategory': subcategory,
                'price': round(base_price, 2),
                'cost': round(cost, 2),
                'supplier': np.random.choice(suppliers),
                'launch_date': (datetime.now() - timedelta(days=np.random.randint(1, 730))).strftime('%Y-%m-%d')
            }
            products_data.append(product)
        
        df = pd.DataFrame(products_data)
        df.to_sql('products', conn, if_exists='append', index=False)
    
    def _generate_orders(self, conn):
        """Generate sample order data."""
        np.random.seed(42)
        n_orders = 5000
        
        statuses = ['Completed', 'Pending', 'Shipped', 'Cancelled', 'Returned']
        
        orders_data = []
        for i in range(n_orders):
            customer_id = np.random.randint(1, 1001)
            order_date = datetime.now() - timedelta(days=np.random.randint(1, 365))
            total_amount = np.random.uniform(20, 1000)
            status = np.random.choice(statuses, p=[0.6, 0.1, 0.2, 0.05, 0.05])
            
            order = {
                'order_id': i + 1,
                'customer_id': customer_id,
                'order_date': order_date.strftime('%Y-%m-%d'),
                'total_amount': round(total_amount, 2),
                'status': status,
                'shipping_city': f"City {np.random.randint(1, 50)}",
                'shipping_state': f"ST{np.random.randint(1, 10)}"
            }
            orders_data.append(order)
        
        df = pd.DataFrame(orders_data)
        df.to_sql('orders', conn, if_exists='append', index=False)
    
    def _generate_sales(self, conn):
        """Generate sample sales data."""
        np.random.seed(42)
        n_sales = 10000
        
        regions = ['North', 'South', 'East', 'West', 'Central']
        sales_reps = ['John Smith', 'Jane Doe', 'Mike Johnson', 'Sarah Wilson', 'David Brown']
        
        sales_data = []
        for i in range(n_sales):
            product_id = np.random.randint(1, 201)
            customer_id = np.random.randint(1, 1001)
            sale_date = datetime.now() - timedelta(days=np.random.randint(1, 365))
            quantity = np.random.randint(1, 10)
            unit_price = np.random.uniform(5, 200)
            total_amount = quantity * unit_price
            
            sale = {
                'sale_id': i + 1,
                'product_id': product_id,
                'customer_id': customer_id,
                'sale_date': sale_date.strftime('%Y-%m-%d'),
                'quantity': quantity,
                'unit_price': round(unit_price, 2),
                'total_amount': round(total_amount, 2),
                'region': np.random.choice(regions),
                'sales_rep': np.random.choice(sales_reps)
            }
            sales_data.append(sale)
        
        df = pd.DataFrame(sales_data)
        df.to_sql('sales', conn, if_exists='append', index=False)
    
    def execute_query(self, query: str) -> pd.DataFrame:
        """Execute SQL query and return results as DataFrame.
        
        Args:
            query: SQL query string
            
        Returns:
            DataFrame with query results
        """
        try:
            # Validate input
            if not query or not query.strip():
                raise Exception("Empty query provided")
            
            # Clean up the query - remove semicolons and extra whitespace
            clean_query = query.strip().rstrip(';')
            
            # Basic validation - check if query looks valid
            if not clean_query.upper().startswith(('SELECT', 'WITH', 'PRAGMA')):
                raise Exception(f"Invalid query type. Only SELECT queries are supported. Got: {clean_query[:20]}...")
            
            # Validate query against schema before execution (non-blocking)
            validation_error = self._validate_query_against_schema(clean_query)
            if validation_error:
                # Log validation error but allow execution attempt
                print(f"Query validation warning: {validation_error}")
            
            with self.engine.connect() as conn:
                try:
                    result = conn.execute(text(clean_query))
                    columns = result.keys()
                    data = result.fetchall()
                    
                    # Check if we got any results
                    if not columns:
                        raise Exception("Query returned no columns")
                    
                    return pd.DataFrame(data, columns=columns)
                    
                except Exception as db_err:
                    # Re-raise with better error message
                    error_msg = str(db_err)
                    
                    # Provide more helpful error messages
                    if "no such column" in error_msg.lower():
                        schema = self.get_table_schema()
                        # Find the problematic column by analyzing the query
                        problematic_table = None
                        for table, cols in schema.items():
                            if table.lower() in clean_query.lower():
                                problematic_table = table
                                break
                        
                        if problematic_table:
                            available_cols = schema[problematic_table]
                            raise Exception(
                                f"Column doesn't exist in table '{problematic_table}'. "
                                f"Available columns: {', '.join(available_cols)}. "
                                f"Please check your question."
                            )
                        else:
                            raise Exception(
                                f"Column doesn't exist. Available tables: {', '.join(list(schema.keys()))}. "
                                f"Please check your question."
                            )
                    elif "no such table" in error_msg.lower():
                        schema = self.get_table_schema()
                        available_tables = list(schema.keys())
                        raise Exception(
                            f"Table doesn't exist. Available tables: {', '.join(available_tables)}. "
                            f"Please check your question."
                        )
                    else:
                        raise Exception(f"Database Error: {error_msg}")
                        
        except Exception as e:
            # Re-raise with clearer message
            raise e
    
    def _validate_query_against_schema(self, query: str) -> Optional[str]:
        """Validate SQL query against database schema.
        
        Args:
            query: SQL query string
            
        Returns:
            Error message if validation fails, None otherwise
        """
        try:
            import re
            schema = self.get_table_schema()
            available_tables = list(schema.keys())
            
            # Extract table names from query
            query_upper = query.upper()
            
            # Find tables in FROM clause
            from_match = re.search(r'FROM\s+(\w+)', query_upper)
            tables_in_query = []
            if from_match:
                table_name = from_match.group(1).lower()
                tables_in_query.append(table_name)
            
            # Find tables in JOIN clauses
            join_matches = re.findall(r'JOIN\s+(\w+)', query_upper)
            tables_in_query.extend([t.lower() for t in join_matches])
            
            # Check if all tables exist
            for table in tables_in_query:
                if table not in [t.lower() for t in available_tables]:
                    return f"The requested table '{table}' doesn't exist in the database. Available tables: {', '.join(available_tables)}"
            
            # Extract column names (basic check)
            # This is a simplified validation - we check if columns from the schema are used
            # We can't easily validate all columns without parsing the full SQL
            return None
            
        except Exception:
            # If validation fails, don't block the query - let the database handle it
            return None
    
    def get_table_schema(self) -> Dict[str, List[str]]:
        """Get schema information for all tables.
        
        Returns:
            Dictionary mapping table names to their column lists
        """
        schema = {}
        with self.engine.connect() as conn:
            if self.is_custom_db:
                # For custom databases, discover all tables dynamically
                result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
                tables = [row[0] for row in result.fetchall()]
            else:
                # For default database, use known tables
                tables = ['customers', 'products', 'orders', 'order_items', 'sales']
            
            for table in tables:
                try:
                    result = conn.execute(text(f"PRAGMA table_info({table})"))
                    columns = [row[1] for row in result.fetchall()]
                    if columns:  # Only add tables that exist and have columns
                        schema[table] = columns
                except Exception:
                    # Skip tables that can't be accessed
                    continue
        return schema
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get database information.
        
        Returns:
            Dictionary with database information
        """
        info = {
            'is_custom': self.is_custom_db,
            'path': self.db_path,
            'tables': list(self.get_table_schema().keys())
        }
        return info
    
    def get_sample_queries(self) -> List[str]:
        """Get sample queries for demonstration.
        
        Returns:
            List of sample SQL queries
        """
        return [
            "SELECT COUNT(*) as total_customers FROM customers",
            "SELECT category, COUNT(*) as product_count FROM products GROUP BY category",
            "SELECT status, COUNT(*) as order_count FROM orders GROUP BY status",
            "SELECT c.customer_segment, SUM(o.total_amount) as total_sales FROM customers c JOIN orders o ON c.customer_id = o.customer_id GROUP BY c.customer_segment",
            "SELECT p.category, SUM(s.total_amount) as total_sales FROM products p JOIN sales s ON p.product_id = s.product_id GROUP BY p.category ORDER BY total_sales DESC",
            "SELECT DATE(sale_date) as date, SUM(total_amount) as daily_sales FROM sales GROUP BY DATE(sale_date) ORDER BY date DESC LIMIT 30"
        ]
