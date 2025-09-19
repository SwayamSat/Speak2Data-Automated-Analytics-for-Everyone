-- Sample Database Schema for NL to SQL Project

-- Customers table
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT UNIQUE,
    city TEXT,
    state TEXT,
    country TEXT DEFAULT 'USA',
    registration_date DATE,
    age INTEGER,
    income DECIMAL(10,2)
);

-- Products table
CREATE TABLE products (
    product_id INTEGER PRIMARY KEY,
    product_name TEXT NOT NULL,
    category TEXT,
    subcategory TEXT,
    price DECIMAL(10,2),
    cost DECIMAL(10,2),
    brand TEXT,
    launch_date DATE
);

-- Orders table
CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY,
    customer_id INTEGER,
    order_date DATE,
    total_amount DECIMAL(10,2),
    status TEXT DEFAULT 'completed',
    shipping_cost DECIMAL(10,2) DEFAULT 0,
    discount_amount DECIMAL(10,2) DEFAULT 0,
    FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
);

-- Order Items table
CREATE TABLE order_items (
    item_id INTEGER PRIMARY KEY,
    order_id INTEGER,
    product_id INTEGER,
    quantity INTEGER,
    unit_price DECIMAL(10,2),
    discount_percent DECIMAL(5,2) DEFAULT 0,
    FOREIGN KEY (order_id) REFERENCES orders (order_id),
    FOREIGN KEY (product_id) REFERENCES products (product_id)
);

-- Sales Representatives table
CREATE TABLE sales_reps (
    rep_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    region TEXT,
    hire_date DATE,
    commission_rate DECIMAL(5,4)
);

-- Customer Sales Rep Assignments
CREATE TABLE customer_assignments (
    assignment_id INTEGER PRIMARY KEY,
    customer_id INTEGER,
    rep_id INTEGER,
    assignment_date DATE,
    FOREIGN KEY (customer_id) REFERENCES customers (customer_id),
    FOREIGN KEY (rep_id) REFERENCES sales_reps (rep_id)
);

-- Sample data insertion queries
INSERT INTO customers (customer_id, name, email, city, state, registration_date, age, income) VALUES
(1, 'John Doe', 'john@email.com', 'New York', 'NY', '2023-01-15', 35, 75000.00),
(2, 'Jane Smith', 'jane@email.com', 'Los Angeles', 'CA', '2023-02-20', 28, 65000.00),
(3, 'Bob Johnson', 'bob@email.com', 'Chicago', 'IL', '2023-03-10', 42, 85000.00),
(4, 'Alice Brown', 'alice@email.com', 'New York', 'NY', '2023-04-05', 31, 70000.00),
(5, 'Charlie Wilson', 'charlie@email.com', 'Miami', 'FL', '2023-05-12', 39, 90000.00);

INSERT INTO products (product_id, product_name, category, subcategory, price, cost, brand, launch_date) VALUES
(1, 'Laptop Pro', 'Electronics', 'Computers', 1299.99, 800.00, 'TechBrand', '2023-01-01'),
(2, 'Wireless Mouse', 'Electronics', 'Accessories', 29.99, 15.00, 'TechBrand', '2023-01-01'),
(3, 'Office Chair', 'Furniture', 'Seating', 199.99, 120.00, 'ComfortPlus', '2023-02-01'),
(4, 'Desk Lamp', 'Furniture', 'Lighting', 49.99, 25.00, 'BrightLight', '2023-02-01'),
(5, 'Smartphone', 'Electronics', 'Mobile', 699.99, 400.00, 'MobileTech', '2023-03-01');

INSERT INTO orders (order_id, customer_id, order_date, total_amount, status) VALUES
(1, 1, '2023-06-01', 1329.98, 'completed'),
(2, 2, '2023-06-02', 729.98, 'completed'),
(3, 3, '2023-06-03', 249.98, 'completed'),
(4, 1, '2023-06-04', 49.99, 'completed'),
(5, 4, '2023-06-05', 1999.96, 'shipped');

INSERT INTO order_items (item_id, order_id, product_id, quantity, unit_price) VALUES
(1, 1, 1, 1, 1299.99),
(2, 1, 2, 1, 29.99),
(3, 2, 5, 1, 699.99),
(4, 2, 2, 1, 29.99),
(5, 3, 3, 1, 199.99),
(6, 3, 4, 1, 49.99),
(7, 4, 4, 1, 49.99),
(8, 5, 1, 1, 1299.99),
(9, 5, 5, 1, 699.99);
