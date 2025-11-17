"""
Generate sample datasets for testing the Speak2Data system.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path


def create_sales_data(n_rows=1000):
    """Create a sample sales dataset."""
    np.random.seed(42)
    
    regions = ['North', 'South', 'East', 'West']
    products = ['Product A', 'Product B', 'Product C', 'Product D', 'Product E']
    
    start_date = datetime(2023, 1, 1)
    
    data = {
        'order_id': range(1, n_rows + 1),
        'date': [start_date + timedelta(days=np.random.randint(0, 365)) for _ in range(n_rows)],
        'region': np.random.choice(regions, n_rows),
        'product': np.random.choice(products, n_rows),
        'quantity': np.random.randint(1, 100, n_rows),
        'unit_price': np.random.uniform(10, 500, n_rows).round(2),
        'customer_id': np.random.randint(1, 200, n_rows),
    }
    
    df = pd.DataFrame(data)
    df['total_amount'] = df['quantity'] * df['unit_price']
    
    return df


def create_customer_churn_data(n_rows=500):
    """Create a sample customer churn dataset for classification."""
    np.random.seed(42)
    
    data = {
        'customer_id': range(1, n_rows + 1),
        'age': np.random.randint(18, 70, n_rows),
        'tenure_months': np.random.randint(1, 72, n_rows),
        'monthly_charges': np.random.uniform(20, 150, n_rows).round(2),
        'total_charges': np.random.uniform(100, 8000, n_rows).round(2),
        'num_products': np.random.randint(1, 5, n_rows),
        'num_support_calls': np.random.randint(0, 10, n_rows),
        'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_rows),
        'payment_method': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_rows),
    }
    
    df = pd.DataFrame(data)
    
    # Create churn target based on features (for realistic patterns)
    churn_prob = (
        0.1 +
        0.3 * (df['contract_type'] == 'Month-to-month') +
        0.2 * (df['num_support_calls'] > 5) +
        0.15 * (df['tenure_months'] < 12) -
        0.1 * (df['num_products'] > 2)
    )
    
    df['churned'] = (np.random.random(n_rows) < churn_prob).astype(int)
    
    return df


def create_house_prices_data(n_rows=300):
    """Create a sample house prices dataset for regression."""
    np.random.seed(42)
    
    neighborhoods = ['Downtown', 'Suburb A', 'Suburb B', 'Rural']
    
    data = {
        'house_id': range(1, n_rows + 1),
        'square_feet': np.random.randint(800, 4000, n_rows),
        'bedrooms': np.random.randint(1, 6, n_rows),
        'bathrooms': np.random.randint(1, 4, n_rows),
        'year_built': np.random.randint(1950, 2024, n_rows),
        'neighborhood': np.random.choice(neighborhoods, n_rows),
        'has_garage': np.random.choice([0, 1], n_rows),
        'lot_size': np.random.randint(2000, 20000, n_rows),
    }
    
    df = pd.DataFrame(data)
    
    # Create price based on features (for realistic patterns)
    base_price = 50000
    df['price'] = (
        base_price +
        df['square_feet'] * np.random.uniform(100, 150, n_rows) +
        df['bedrooms'] * 20000 +
        df['bathrooms'] * 15000 +
        (2024 - df['year_built']) * -500 +
        df['has_garage'] * 25000 +
        (df['neighborhood'] == 'Downtown') * 100000
    )
    
    # Add some noise
    df['price'] = df['price'] + np.random.normal(0, 30000, n_rows)
    df['price'] = df['price'].round(2)
    
    return df


def main():
    """Generate and save sample datasets."""
    output_dir = Path("sample_data")
    output_dir.mkdir(exist_ok=True)
    
    print("Generating sample datasets...")
    
    # Sales data
    sales_df = create_sales_data(1000)
    sales_path = output_dir / "sales_data.csv"
    sales_df.to_csv(sales_path, index=False)
    print(f"✅ Created {sales_path} ({len(sales_df)} rows)")
    
    # Customer churn data
    churn_df = create_customer_churn_data(500)
    churn_path = output_dir / "customer_churn.csv"
    churn_df.to_csv(churn_path, index=False)
    print(f"✅ Created {churn_path} ({len(churn_df)} rows)")
    
    # House prices data
    houses_df = create_house_prices_data(300)
    houses_path = output_dir / "house_prices.csv"
    houses_df.to_csv(houses_path, index=False)
    print(f"✅ Created {houses_path} ({len(houses_df)} rows)")
    
    print("\nSample datasets created successfully!")
    print("\nYou can now:")
    print("1. Run: streamlit run app.py")
    print("2. Upload one of these files from the sidebar")
    print("3. Try queries like:")
    print("   - 'Show me sales trends by region'")
    print("   - 'Predict customer churn'")
    print("   - 'What factors affect house prices?'")


if __name__ == "__main__":
    main()
