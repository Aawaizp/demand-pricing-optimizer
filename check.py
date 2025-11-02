# quick_check.py - Quick look at our dataset

import pandas as pd  # Import pandas to read CSV files

# Read the CSV file into a pandas DataFrame (like a table)
df = pd.read_csv('data/raw/BigBasket Products.csv')

# Print first 5 rows
print("First 5 rows of data:")
print(df.head())

# Print column names
print("\nColumn names:")
print(df.columns.tolist())

# Print dataset shape (rows, columns)
print(f"\nDataset has {df.shape[0]} rows and {df.shape[1]} columns")

# Print data types
print("\nData types:")
print(df.dtypes)