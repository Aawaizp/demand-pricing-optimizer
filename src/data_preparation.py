# src/data_preparation.py
# This script generates realistic time series sales data for FMCG products

import pandas as pd  # Work with tables/data
import numpy as np  # Math and random numbers
from datetime import datetime, timedelta  # Work with dates
import random  # Generate random values

# Set random seed so we get same results every time (reproducible)
np.random.seed(42)
random.seed(42)

def load_product_data():
    """
    Load the BigBasket product data from CSV file
    Returns: DataFrame with product information
    """
    print("📂 Loading BigBasket product data...")
    
    # Read the CSV file
    df = pd.read_csv('data/raw/BigBasket Products.csv')
    
    print(f"✅ Loaded {len(df)} products")
    return df

def select_top_products(df, n_products=20):
    """
    Select top N products from different categories
    We don't want 27,000 products - too much! Just pick 20 popular ones
    
    Args:
        df: DataFrame with all products
        n_products: How many products to select (default 20)
    Returns: DataFrame with selected products
    """
    print(f"\n🎯 Selecting top {n_products} products from different categories...")
    
    # Filter out products with missing prices
    df_clean = df.dropna(subset=['sale_price', 'market_price'])
    
    # Filter products with good ratings (above 3.5)
    df_clean = df_clean[df_clean['rating'] > 3.5]
    
    # Get products from major FMCG categories
    fmcg_categories = [
        'Beauty & Hygiene',
        'Beverages', 
        'Cleaning & Household',
        'Foodgrains, Oil & Masala',
        'Snacks & Branded Foods',
        'Kitchen, Garden & Pets'
    ]
    
    # Filter for FMCG categories only
    df_fmcg = df_clean[df_clean['category'].isin(fmcg_categories)]
    
    # Sample products from each category
    products_per_category = n_products // len(fmcg_categories)
    selected_products = []
    
    for category in fmcg_categories:
        # Get products from this category
        cat_products = df_fmcg[df_fmcg['category'] == category]
        
        # Sample random products from this category
        if len(cat_products) > 0:
            sample_size = min(products_per_category, len(cat_products))
            sampled = cat_products.sample(n=sample_size, random_state=42)
            selected_products.append(sampled)
    
    # Combine all selected products
    final_products = pd.concat(selected_products, ignore_index=True)
    
    # If we don't have enough, sample more
    if len(final_products) < n_products:
        remaining = n_products - len(final_products)
        extra = df_fmcg.sample(n=remaining, random_state=42)
        final_products = pd.concat([final_products, extra], ignore_index=True)
    
    # Take only the first n_products
    final_products = final_products.head(n_products)
    
    print(f"✅ Selected {len(final_products)} products")
    print(f"Categories: {final_products['category'].unique()}")
    
    return final_products

def generate_sales_timeseries(products_df, start_date='2022-01-01', end_date='2024-10-31'):
    """
    Generate realistic daily sales data for each product
    
    Args:
        products_df: DataFrame with selected products
        start_date: Start date for sales data
        end_date: End date for sales data
    Returns: DataFrame with date, product_id, quantity_sold, revenue
    """
    print("\n📈 Generating time series sales data...")
    
    # Convert dates to datetime
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # Create date range (every day from start to end)
    date_range = pd.date_range(start=start, end=end, freq='D')
    
    print(f"   Date range: {start_date} to {end_date} ({len(date_range)} days)")
    
    all_sales = []  # Store all sales records here
    
    # Loop through each product
    for idx, product in products_df.iterrows():
        product_id = idx
        product_name = product['product']
        sale_price = product['sale_price']
        category = product['category']
        
        print(f"   Generating sales for: {product_name[:50]}...")
        
        # Base demand (how many units sold per day on average)
        # Different categories have different demand levels
        if 'Beverages' in category or 'Snacks' in category:
            base_demand = random.randint(50, 150)  # High demand
        elif 'Beauty' in category or 'Cleaning' in category:
            base_demand = random.randint(30, 80)  # Medium demand
        else:
            base_demand = random.randint(20, 60)  # Lower demand
        
        # Generate daily sales for this product
        for date in date_range:
            # Add seasonality (sales go up during festivals/holidays)
            month = date.month
            day_of_week = date.dayofweek
            
            # Weekend boost (Saturday=5, Sunday=6)
            weekend_boost = 1.3 if day_of_week >= 5 else 1.0
            
            # Festival season boost (Oct-Dec in India - Diwali, Christmas)
            if month in [10, 11, 12]:
                season_boost = 1.4
            # Summer boost (Apr-Jun)
            elif month in [4, 5, 6]:
                season_boost = 1.2
            else:
                season_boost = 1.0
            
            # Add trend (sales grow over time by 0.5% per month)
            months_since_start = (date.year - start.year) * 12 + (date.month - start.month)
            trend = 1 + (0.005 * months_since_start)
            
            # Calculate final quantity sold for this day
            quantity = base_demand * weekend_boost * season_boost * trend
            
            # Add random noise (realistic variation)
            quantity = quantity * np.random.uniform(0.85, 1.15)

            
            # Round to integer
            quantity = max(1, int(quantity))  # At least 1 unit sold
            
            # Calculate revenue
            revenue = quantity * sale_price
            
            # Store this sale record
            all_sales.append({
                'date': date,
                'product_id': product_id,
                'product_name': product_name,
                'category': category,
                'quantity_sold': quantity,
                'sale_price': sale_price,
                'revenue': revenue
            })
    
    # Convert list to DataFrame
    sales_df = pd.DataFrame(all_sales)
    
    print(f"✅ Generated {len(sales_df)} sales records")
    print(f"   Total revenue: ₹{sales_df['revenue'].sum():,.2f}")
    
    return sales_df

def add_pricing_features(sales_df, products_df):
    """
    Add features needed for price optimization model
    
    Args:
        sales_df: DataFrame with sales data
        products_df: DataFrame with product info
    Returns: DataFrame with additional features
    """
    print("\n🔧 Adding pricing optimization features...")
    
    # Merge to get market_price
    sales_df = sales_df.merge(
        products_df[['product', 'market_price', 'rating']],
        left_on='product_name',
        right_on='product',
        how='left'
    )
    
    # Calculate discount percentage
    sales_df['discount_pct'] = ((sales_df['market_price'] - sales_df['sale_price']) / 
                                 sales_df['market_price'] * 100)
    
    # Add competitor price (simulate competitor having slightly different prices)
    sales_df['competitor_price'] = sales_df['sale_price'] * np.random.uniform(0.95, 1.1, len(sales_df))
    
    # Add day of week (0=Monday, 6=Sunday)
    sales_df['day_of_week'] = sales_df['date'].dt.dayofweek
    
    # Add month
    sales_df['month'] = sales_df['date'].dt.month
    
    # Add is_weekend flag
    sales_df['is_weekend'] = (sales_df['day_of_week'] >= 5).astype(int)
    
    # Add is_festival_season flag (Oct-Dec)
    sales_df['is_festival_season'] = sales_df['month'].isin([10, 11, 12]).astype(int)
    
    # Drop duplicate 'product' column
    sales_df = sales_df.drop(columns=['product'])
    
    print("✅ Added features: discount_pct, competitor_price, day_of_week, month, is_weekend, is_festival_season")
    
    return sales_df

def save_processed_data(sales_df, products_df):
    """
    Save cleaned data to processed folder
    """
    print("\n💾 Saving processed data...")
    
    # Save sales data
    sales_df.to_csv('data/processed/sales_data.csv', index=False)
    print(f"   ✅ Saved sales_data.csv ({len(sales_df)} rows)")
    
    # Save selected products info
    products_df.to_csv('data/processed/products_info.csv', index=False)
    print(f"   ✅ Saved products_info.csv ({len(products_df)} rows)")
    
    # Print summary statistics
    print("\n📊 Summary Statistics:")
    print(f"   Date range: {sales_df['date'].min()} to {sales_df['date'].max()}")
    print(f"   Total products: {sales_df['product_id'].nunique()}")
    print(f"   Total sales records: {len(sales_df)}")
    print(f"   Average daily sales per product: {sales_df.groupby('product_id')['quantity_sold'].mean().mean():.2f} units")
    print(f"   Total revenue: ₹{sales_df['revenue'].sum():,.2f}")

def main():
    """
    Main function to run entire data preparation pipeline
    """
    print("=" * 60)
    print("🚀 FMCG DEMAND FORECASTING - DATA PREPARATION")
    print("=" * 60)
    
    # Step 1: Load product data
    products_df = load_product_data()
    
    # Step 2: Select top products
    selected_products = select_top_products(products_df, n_products=20)
    
    # Step 3: Generate time series sales data
    sales_df = generate_sales_timeseries(selected_products)
    
    # Step 4: Add pricing features
    sales_df = add_pricing_features(sales_df, selected_products)
    
    # Step 5: Save processed data
    save_processed_data(sales_df, selected_products)
    
    print("\n" + "=" * 60)
    print("✅ DATA PREPARATION COMPLETE!")
    print("=" * 60)
    print("\n📁 Check 'data/processed/' folder for output files")

# This runs when you execute the script
if __name__ == "__main__":
    main()