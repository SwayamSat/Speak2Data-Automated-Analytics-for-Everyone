"""
Database module for Speak2Data platform.
Handles multiple database types (SQLite, PostgreSQL, MySQL, etc.) with universal schema discovery and query execution.
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text, inspect, MetaData
from sqlalchemy.exc import SQLAlchemyError
import os
import tempfile
from typing import List, Dict, Any, Optional, Union
import re
from urllib.parse import urlparse
from pathlib import Path


class DatabaseManager:
    """Manages database operations for multiple database types."""
    
    SUPPORTED_DB_TYPES = {
        'sqlite': ['sqlite', 'sqlite3', 'db'],
        'postgresql': ['postgresql', 'postgres'],
        'mysql': ['mysql', 'mariadb'],
        'mssql': ['mssql', 'sqlserver'],
        'oracle': ['oracle'],
    }
    
    def __init__(self, 
                 db_path: Optional[str] = None,
                 connection_string: Optional[str] = None,
                 db_type: Optional[str] = None,
                 custom_db_path: Optional[str] = None):
        """Initialize database manager.
        
        Args:
            db_path: Path to SQLite database file (for default or file-based databases)
            connection_string: SQLAlchemy connection string (e.g., 'postgresql://user:pass@host:port/dbname')
            db_type: Database type ('sqlite', 'postgresql', 'mysql', etc.) - auto-detected if not provided
            custom_db_path: Path to custom uploaded database file (deprecated, use db_path)
        """
        self.engine = None
        self.db_type = None
        self.connection_string = None
        self.is_custom_db = False
        self.db_path = None
        
        # Handle deprecated custom_db_path parameter
        if custom_db_path:
            db_path = custom_db_path
            self.is_custom_db = True
        
        # Determine database connection method
        if connection_string:
            # Connection string provided (PostgreSQL, MySQL, etc.)
            self.connection_string = connection_string
            self.db_type = db_type or self._detect_db_type_from_connection_string(connection_string)
            self._create_engine_from_connection_string()
        elif db_path:
            # File path provided (SQLite)
            self.db_path = db_path
            self.db_type = 'sqlite'
            self.is_custom_db = True
            self._create_engine_from_file()
        else:
            # Default: create sample SQLite database
            self.db_path = "business_data.db"
            self.db_type = 'sqlite'
            self.is_custom_db = False
            self._create_engine_from_file()
            self._create_tables()
            self._populate_sample_data()
        
        # Validate connection
        self._validate_connection()
    
    def _detect_db_type_from_connection_string(self, connection_string: str) -> str:
        """Detect database type from connection string."""
        connection_string_lower = connection_string.lower()
        
        if connection_string_lower.startswith('postgresql://') or connection_string_lower.startswith('postgres://'):
            return 'postgresql'
        elif connection_string_lower.startswith('mysql://') or connection_string_lower.startswith('mysql+pymysql://'):
            return 'mysql'
        elif connection_string_lower.startswith('mssql://') or connection_string_lower.startswith('mssql+pyodbc://'):
            return 'mssql'
        elif connection_string_lower.startswith('oracle://') or connection_string_lower.startswith('oracle+cx_oracle://'):
            return 'oracle'
        elif connection_string_lower.startswith('sqlite:///'):
            return 'sqlite'
        else:
            # Try to parse as URL
            try:
                parsed = urlparse(connection_string)
                scheme = parsed.scheme.split('+')[0]  # Remove driver prefix
                return scheme
            except Exception:
                raise ValueError(f"Unable to detect database type from connection string: {connection_string}")
    
    def _create_engine_from_connection_string(self):
        """Create SQLAlchemy engine from connection string."""
        try:
            self.engine = create_engine(
                self.connection_string,
                pool_pre_ping=True,  # Verify connections before using
                echo=False  # Set to True for SQL query logging
            )
        except Exception as e:
            raise ValueError(f"Failed to create database connection: {str(e)}")
    
    def _create_engine_from_file(self):
        """Create SQLAlchemy engine from file path (SQLite)."""
        try:
            # Normalize path for SQLite
            if self.db_path:
                # Handle absolute and relative paths
                if not os.path.isabs(self.db_path):
                    # Relative path - create in current directory
                    self.db_path = os.path.abspath(self.db_path)
                
                # Ensure directory exists
                db_dir = os.path.dirname(self.db_path)
                if db_dir and not os.path.exists(db_dir):
                    os.makedirs(db_dir, exist_ok=True)
                
                # Create SQLite connection string
                sqlite_url = f"sqlite:///{self.db_path}"
                self.engine = create_engine(
                    sqlite_url,
                    pool_pre_ping=True,
                    connect_args={"check_same_thread": False}  # Allow multiple threads for SQLite
                )
            else:
                raise ValueError("Database path is required for file-based databases")
        except Exception as e:
            raise ValueError(f"Failed to create database connection to file '{self.db_path}': {str(e)}")
    
    def _validate_connection(self):
        """Validate database connection."""
        try:
            with self.engine.connect() as conn:
                # Test connection with a simple query
                if self.db_type == 'sqlite':
                    conn.execute(text("SELECT 1"))
                else:
                    # For other databases, use database-specific test query
                    conn.execute(text("SELECT 1"))
        except SQLAlchemyError as e:
            raise ConnectionError(f"Failed to connect to database: {str(e)}")
        except Exception as e:
            raise ConnectionError(f"Database connection validation failed: {str(e)}")
    
    def _create_tables(self):
        """Create database tables for business data (only for default SQLite database)."""
        if self.db_type != 'sqlite' or self.is_custom_db:
            return  # Only create tables for default SQLite database
        
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
        """Generate and populate sample business data (only for default database)."""
        if self.db_type != 'sqlite' or self.is_custom_db:
            return  # Only populate for default SQLite database
        
        with self.engine.connect() as conn:
            # Check if data already exists
            try:
                result = conn.execute(text("SELECT COUNT(*) FROM customers")).fetchone()
                if result[0] > 0:
                    return  # Data already exists
            except Exception:
                pass  # Table doesn't exist yet, continue to create data
            
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
    
    def get_table_schema(self) -> Dict[str, List[str]]:
        """Get schema information for all tables using SQLAlchemy inspector (database-agnostic).
        
        Returns:
            Dictionary mapping table names to their column lists
        """
        schema = {}
        
        try:
            inspector = inspect(self.engine)
            tables = inspector.get_table_names()
            
            # Also get views if supported
            try:
                views = inspector.get_view_names()
                tables.extend(views)
            except Exception:
                pass  # Views might not be supported by all databases
            
            for table_name in tables:
                try:
                    # Get columns for this table
                    columns = inspector.get_columns(table_name)
                    column_names = [col['name'] for col in columns]
                    
                    if column_names:  # Only add tables that have columns
                        schema[table_name] = column_names
                except Exception as e:
                    # Skip tables that can't be accessed
                    print(f"Warning: Could not access table '{table_name}': {str(e)}")
                    continue
        
        except Exception as e:
            # Fallback to SQL-based schema discovery for SQLite
            if self.db_type == 'sqlite':
                try:
                    with self.engine.connect() as conn:
                        result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
                        tables = [row[0] for row in result.fetchall()]
                        
                        for table in tables:
                            try:
                                result = conn.execute(text(f"PRAGMA table_info({table})"))
                                columns = [row[1] for row in result.fetchall()]
                                if columns:
                                    schema[table] = columns
                            except Exception:
                                continue
                except Exception:
                    raise ValueError(f"Failed to discover database schema: {str(e)}")
            else:
                raise ValueError(f"Failed to discover database schema: {str(e)}")
        
        return schema
    
    def get_detailed_schema(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed schema information including column types and constraints.
        
        Returns:
            Dictionary with detailed schema information
        """
        detailed_schema = {}
        
        try:
            inspector = inspect(self.engine)
            tables = inspector.get_table_names()
            
            for table_name in tables:
                try:
                    columns = inspector.get_columns(table_name)
                    primary_keys = inspector.get_primary_keys(table_name)
                    
                    # Get foreign keys
                    foreign_keys = []
                    try:
                        fks = inspector.get_foreign_keys(table_name)
                        foreign_keys = [fk for fk in fks]
                    except Exception:
                        pass
                    
                    detailed_schema[table_name] = {
                        'columns': {col['name']: {
                            'type': str(col['type']),
                            'nullable': col.get('nullable', True),
                            'default': col.get('default'),
                        } for col in columns},
                        'primary_keys': primary_keys,
                        'foreign_keys': foreign_keys
                    }
                except Exception:
                    continue
        except Exception as e:
            raise ValueError(f"Failed to get detailed schema: {str(e)}")
        
        return detailed_schema
    
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
                raise ValueError("Empty query provided")
            
            # Clean up the query - remove semicolons and extra whitespace
            clean_query = query.strip().rstrip(';')
            
            # Basic validation - check if query looks valid
            query_upper = clean_query.upper().strip()
            if not query_upper.startswith(('SELECT', 'WITH', 'PRAGMA', 'SHOW', 'DESCRIBE', 'DESC', 'EXPLAIN')):
                raise ValueError(f"Invalid query type. Only SELECT and read-only queries are supported. Got: {clean_query[:50]}...")
            
            # Validate query against schema before execution (non-blocking)
            validation_error = self._validate_query_against_schema(clean_query)
            if validation_error:
                # Log validation error but allow execution attempt
                print(f"Query validation warning: {validation_error}")
            
            with self.engine.connect() as conn:
                try:
                    result = conn.execute(text(clean_query))
                    
                    # Check if result has columns (SELECT query)
                    if result.returns_rows:
                        columns = result.keys()
                        data = result.fetchall()
                        
                        # Check if we got any results
                        if not columns:
                            raise ValueError("Query returned no columns")
                        
                        return pd.DataFrame(data, columns=columns)
                    else:
                        # Non-SELECT query (e.g., PRAGMA, SHOW)
                        # Return empty DataFrame with success message
                        return pd.DataFrame({'status': ['Query executed successfully']})
                    
                except SQLAlchemyError as db_err:
                    # Re-raise with better error message
                    error_msg = str(db_err)
                    
                    # Provide more helpful error messages
                    if "no such column" in error_msg.lower() or "column" in error_msg.lower() and "does not exist" in error_msg.lower():
                        schema = self.get_table_schema()
                        # Find the problematic column by analyzing the query
                        problematic_table = None
                        for table, cols in schema.items():
                            if table.lower() in clean_query.lower():
                                problematic_table = table
                                break
                        
                        if problematic_table:
                            available_cols = schema[problematic_table]
                            raise ValueError(
                                f"Column doesn't exist in table '{problematic_table}'. "
                                f"Available columns: {', '.join(available_cols)}. "
                                f"Please check your question."
                            )
                        else:
                            raise ValueError(
                                f"Column doesn't exist. Available tables: {', '.join(list(schema.keys()))}. "
                                f"Please check your question."
                            )
                    elif "no such table" in error_msg.lower() or "table" in error_msg.lower() and "does not exist" in error_msg.lower():
                        schema = self.get_table_schema()
                        available_tables = list(schema.keys())
                        raise ValueError(
                            f"Table doesn't exist. Available tables: {', '.join(available_tables)}. "
                            f"Please check your question."
                        )
                    else:
                        raise ValueError(f"Database Error: {error_msg}")
                        
        except ValueError as e:
            # Re-raise ValueError as-is
            raise e
        except Exception as e:
            # Re-raise with clearer message
            raise ValueError(f"Query execution failed: {str(e)}")
    
    def _validate_query_against_schema(self, query: str) -> Optional[str]:
        """Validate SQL query against database schema.
        
        Args:
            query: SQL query string
            
        Returns:
            Error message if validation fails, None otherwise
        """
        try:
            schema = self.get_table_schema()
            available_tables = list(schema.keys())
            
            if not available_tables:
                return "No tables found in database"
            
            # Extract table names from query (basic pattern matching)
            query_upper = query.upper()
            
            # Find tables in FROM clause
            from_pattern = r'FROM\s+(\w+)'
            from_matches = re.findall(from_pattern, query_upper)
            
            # Find tables in JOIN clauses
            join_pattern = r'JOIN\s+(\w+)'
            join_matches = re.findall(join_pattern, query_upper)
            
            tables_in_query = [t.lower() for t in from_matches + join_matches]
            
            # Check if all tables exist
            for table in tables_in_query:
                if table not in [t.lower() for t in available_tables]:
                    return f"The requested table '{table}' doesn't exist in the database. Available tables: {', '.join(available_tables)}"
            
            return None
            
        except Exception:
            # If validation fails, don't block the query - let the database handle it
            return None
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get database information.
        
        Returns:
            Dictionary with database information
        """
        schema = self.get_table_schema()
        
        info = {
            'db_type': self.db_type,
            'is_custom': self.is_custom_db,
            'tables': list(schema.keys()),
            'table_count': len(schema),
            'total_columns': sum(len(cols) for cols in schema.values())
        }
        
        if self.db_path:
            info['path'] = self.db_path
        elif self.connection_string:
            # Mask password in connection string
            try:
                parsed = urlparse(self.connection_string)
                if parsed.password:
                    masked = self.connection_string.replace(parsed.password, '***')
                    info['connection_string'] = masked
                else:
                    info['connection_string'] = self.connection_string
            except Exception:
                info['connection_string'] = '***'
        
        return info
    
    def get_sample_queries(self) -> List[str]:
        """Get sample queries for demonstration based on available tables.
        
        Returns:
            List of sample SQL queries
        """
        schema = self.get_table_schema()
        
        if not schema:
            return ["SELECT 1"]  # Default query if no tables
        
        # Get first table
        first_table = list(schema.keys())[0]
        first_table_cols = schema[first_table]
        
        sample_queries = []
        
        # Basic count query
        sample_queries.append(f"SELECT COUNT(*) as total_count FROM {first_table}")
        
        # If we have multiple tables, create join query
        if len(schema) > 1:
            tables = list(schema.keys())
            table1, table2 = tables[0], tables[1]
            cols1 = schema[table1]
            cols2 = schema[table2]
            
            # Find common column names (potential join keys)
            common_cols = set(cols1) & set(cols2)
            if common_cols:
                join_col = list(common_cols)[0]
                sample_queries.append(
                    f"SELECT {table1}.*, {table2}.* FROM {table1} JOIN {table2} ON {table1}.{join_col} = {table2}.{join_col} LIMIT 10"
                )
        
        # Group by query if we have categorical columns
        categorical_keywords = ['category', 'type', 'status', 'segment', 'name', 'city', 'state']
        for table, cols in schema.items():
            for col in cols:
                if any(keyword in col.lower() for keyword in categorical_keywords):
                    sample_queries.append(f"SELECT {col}, COUNT(*) as count FROM {table} GROUP BY {col} LIMIT 10")
                    break
        
        # Limit to 5 sample queries
        return sample_queries[:5]
    
    def test_connection(self) -> bool:
        """Test database connection.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            self._validate_connection()
            return True
        except Exception:
            return False
    
    @staticmethod
    def create_from_file(file_path: str) -> 'DatabaseManager':
        """Create DatabaseManager from file path.
        
        Args:
            file_path: Path to database file
            
        Returns:
            DatabaseManager instance
        """
        return DatabaseManager(db_path=file_path)
    
    @staticmethod
    def create_from_connection_string(connection_string: str, db_type: Optional[str] = None) -> 'DatabaseManager':
        """Create DatabaseManager from connection string.
        
        Args:
            connection_string: SQLAlchemy connection string
            db_type: Database type (auto-detected if not provided)
            
        Returns:
            DatabaseManager instance
        """
        return DatabaseManager(connection_string=connection_string, db_type=db_type)
    
    @staticmethod
    def create_from_csv(csv_path: str, table_name: Optional[str] = None) -> 'DatabaseManager':
        """Create DatabaseManager from CSV file by importing into SQLite.
        
        Args:
            csv_path: Path to CSV file
            table_name: Optional table name (defaults to filename without extension)
            
        Returns:
            DatabaseManager instance with imported data
        """
        # Read CSV file
        df = pd.read_csv(csv_path)
        
        # Determine table name
        if table_name is None:
            table_name = Path(csv_path).stem.replace(' ', '_').replace('-', '_').lower()
        
        # Create temporary SQLite database
        temp_dir = tempfile.gettempdir()
        temp_db_path = os.path.join(temp_dir, f"imported_{table_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}.db")
        
        # Create SQLite database and import data
        engine = create_engine(f"sqlite:///{temp_db_path}")
        df.to_sql(table_name, engine, if_exists='replace', index=False)
        
        # Create and return DatabaseManager
        return DatabaseManager(db_path=temp_db_path)
    
    @staticmethod
    def create_from_excel(excel_path: str, sheet_name: Optional[str] = None, 
                         table_name: Optional[str] = None) -> 'DatabaseManager':
        """Create DatabaseManager from Excel file by importing into SQLite.
        
        Args:
            excel_path: Path to Excel file
            sheet_name: Optional sheet name (defaults to first sheet)
            table_name: Optional table name (defaults to filename without extension)
            
        Returns:
            DatabaseManager instance with imported data
        """
        # Read Excel file
        if sheet_name:
            df = pd.read_excel(excel_path, sheet_name=sheet_name)
        else:
            df = pd.read_excel(excel_path)
        
        # Determine table name
        if table_name is None:
            table_name = Path(excel_path).stem.replace(' ', '_').replace('-', '_').lower()
        
        # Create temporary SQLite database
        temp_dir = tempfile.gettempdir()
        temp_db_path = os.path.join(temp_dir, f"imported_{table_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}.db")
        
        # Create SQLite database and import data
        engine = create_engine(f"sqlite:///{temp_db_path}")
        df.to_sql(table_name, engine, if_exists='replace', index=False)
        
        # Create and return DatabaseManager
        return DatabaseManager(db_path=temp_db_path)
    
    @staticmethod
    def create_from_parquet(parquet_path: str, table_name: Optional[str] = None) -> 'DatabaseManager':
        """Create DatabaseManager from Parquet file by importing into SQLite.
        
        Args:
            parquet_path: Path to Parquet file
            table_name: Optional table name (defaults to filename without extension)
            
        Returns:
            DatabaseManager instance with imported data
        """
        # Read Parquet file
        df = pd.read_parquet(parquet_path)
        
        # Determine table name
        if table_name is None:
            table_name = Path(parquet_path).stem.replace(' ', '_').replace('-', '_').lower()
        
        # Create temporary SQLite database
        temp_dir = tempfile.gettempdir()
        temp_db_path = os.path.join(temp_dir, f"imported_{table_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}.db")
        
        # Create SQLite database and import data
        engine = create_engine(f"sqlite:///{temp_db_path}")
        df.to_sql(table_name, engine, if_exists='replace', index=False)
        
        # Create and return DatabaseManager
        return DatabaseManager(db_path=temp_db_path)
    
    @staticmethod
    def create_from_dataframe(df: pd.DataFrame, table_name: str = "data") -> 'DatabaseManager':
        """Create DatabaseManager from pandas DataFrame by importing into SQLite.
        
        Args:
            df: pandas DataFrame
            table_name: Table name for the data
            
        Returns:
            DatabaseManager instance with imported data
        """
        # Clean table name
        table_name = table_name.replace(' ', '_').replace('-', '_').lower()
        
        # Create temporary SQLite database
        temp_dir = tempfile.gettempdir()
        temp_db_path = os.path.join(temp_dir, f"imported_{table_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}.db")
        
        # Create SQLite database and import data
        engine = create_engine(f"sqlite:///{temp_db_path}")
        df.to_sql(table_name, engine, if_exists='replace', index=False)
        
        # Create and return DatabaseManager
        return DatabaseManager(db_path=temp_db_path)
