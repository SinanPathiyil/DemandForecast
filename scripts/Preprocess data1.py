import pandas as pd
import numpy as np

# Read your CSV file
file_path = r'C:\Users\Strix\Desktop\Boehm Tech\demand forecasting\data\product demo list.csv'
df = pd.read_csv(file_path)

print("Column names in your dataset:")
print(df.columns.tolist())
print(f"\nFirst few rows:")
print(df.head(10))

# Forward fill product names first (before filtering)
df['Particulars'] = df['Particulars'].ffill()

# Remove rows where Date is NaN (these are header rows)
df = df[df['Date'].notna()]

# Remove "Total" rows
df = df[~df['Particulars'].str.contains('Total', na=False, case=False)]

# Rename columns for clarity
df.columns = ['Product_Name', 'Date', 'Qty']

# Remove rows where Date or Qty are NaN
df = df.dropna(subset=['Date', 'Qty'])

# Convert Date to datetime (DD-MM-YYYY format)
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')
df = df.dropna(subset=['Date'])

# Convert Qty to numeric
df['Qty'] = pd.to_numeric(df['Qty'], errors='coerce')
df = df.dropna(subset=['Qty'])

# Extract Year and Week number (ISO calendar)
df['Year'] = df['Date'].dt.isocalendar().year
df['Week'] = df['Date'].dt.isocalendar().week

# Create Year-Week identifier (e.g., "2021-W49")
df['Year_Week'] = df['Year'].astype(str) + '-W' + df['Week'].astype(str).str.zfill(2)

# Group by Product Name and Year-Week, sum the quantities
df_weekly = df.groupby(['Product_Name', 'Year_Week', 'Year', 'Week'])['Qty'].sum().reset_index()

# Rename for final output
df_weekly.columns = ['Product_Name', 'Week', 'Year', 'Week_Number', 'Total_Quantity']

# Sort by Product Name, Year, and Week
df_weekly = df_weekly.sort_values(['Product_Name', 'Year', 'Week_Number']).reset_index(drop=True)

# Display first few rows
print("\nWeekly Aggregated Dataset:")
print(df_weekly.head(30))
print(f"\nTotal rows: {len(df_weekly)}")

# Save to Excel
df_weekly.to_excel('./data/demand_prediction_weekly.xlsx', index=False)
print("\nDataset saved as 'demand_prediction_weekly.xlsx'")

# Optional: Check unique products and weeks
print(f"\nUnique Products: {df_weekly['Product_Name'].nunique()}")
print(f"Total Weeks: {df_weekly['Week'].nunique()}")

# Check data distribution
print("\nData distribution by Product:")
print(df_weekly.groupby('Product_Name')['Total_Quantity'].agg(['count', 'sum', 'mean']))