import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Read your CSV file
file_path = r'C:\Users\Strix\Desktop\Boehm Tech\demand forecasting\data\product demo list.csv'
df = pd.read_csv(file_path)

# Forward fill product names first (before filtering)
df['Particulars'] = df['Particulars'].ffill()

# Remove rows where Date is NaN (these are header rows)
df = df[df['Date'].notna()]

# Remove "Total" rows
df = df[~df['Particulars'].str.contains('Total', na=False, case=False)]

# Rename columns for clarity
df.columns = ['Product_Name', 'Date', 'Qty']

# Remove rows where Qty is NaN
df = df.dropna(subset=['Qty'])

# Convert Date to datetime (DD-MM-YYYY format)
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')
df = df.dropna(subset=['Date'])

# Convert Qty to numeric
df['Qty'] = pd.to_numeric(df['Qty'], errors='coerce')
df = df.dropna(subset=['Qty'])

# Extract Year and Week number (ISO calendar)
df['Year'] = df['Date'].dt.isocalendar().year
df['Week'] = df['Date'].dt.isocalendar().week

# Create Year-Week identifier
df['Year_Week'] = df['Year'].astype(str) + '-W' + df['Week'].astype(str).str.zfill(2)

# Group by Product Name and Year-Week
df_weekly = df.groupby(['Product_Name', 'Year_Week', 'Year', 'Week'])['Qty'].sum().reset_index()
df_weekly.columns = ['Product_Name', 'Week', 'Year', 'Week_Number', 'Total_Quantity']

print("Original data summary:")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
print(f"Products: {df_weekly['Product_Name'].unique()}")

# ===== GENERATE SYNTHETIC DATA FOR 2015-2020 =====
np.random.seed(42)

products = df_weekly['Product_Name'].unique()
synthetic_data = []

# Calculate average quantities and seasonality patterns for each product from 2021-2025 data
product_stats = {}
for product in products:
    product_data = df_weekly[df_weekly['Product_Name'] == product]
    product_stats[product] = {
        'mean_qty': product_data['Total_Quantity'].mean(),
        'std_qty': product_data['Total_Quantity'].std(),
        'min_qty': product_data['Total_Quantity'].min(),
        'max_qty': product_data['Total_Quantity'].max()
    }

# Generate synthetic data for 2015-2020 (104 weeks per year)
for product in products:
    stats = product_stats[product]
    mean = stats['mean_qty']
    std = max(stats['std_qty'], mean * 0.3)  # Ensure reasonable std
    
    for year in range(2015, 2021):
        for week in range(1, 53):  # 52 weeks per year
            # Add seasonality: higher demand in certain quarters
            seasonal_factor = 1.0 + 0.2 * np.sin(2 * np.pi * week / 52)
            
            # Generate quantity with trend and noise
            trend_factor = 1.0 + (year - 2015) * 0.05  # Slight upward trend
            qty = mean * seasonal_factor * trend_factor + np.random.normal(0, std)
            qty = max(1, int(qty))  # Ensure positive integer
            
            year_week = f"{year}-W{week:02d}"
            synthetic_data.append({
                'Product_Name': product,
                'Week': year_week,
                'Year': year,
                'Week_Number': week,
                'Total_Quantity': qty
            })

# Convert synthetic data to DataFrame
df_synthetic = pd.DataFrame(synthetic_data)

# Combine synthetic data (2015-2020) with actual data (2021-2025)
df_combined = pd.concat([df_synthetic, df_weekly], ignore_index=True)

# Sort by Product Name, Year, and Week
df_combined = df_combined.sort_values(['Product_Name', 'Year', 'Week_Number']).reset_index(drop=True)

# Display summary
print("\n" + "="*60)
print("COMBINED DATASET (2015-2025):")
print("="*60)
print(df_combined.head(30))
print(f"\nTotal rows: {len(df_combined)}")
print(f"Date range: 2015 to 2025")
print(f"Unique Products: {df_combined['Product_Name'].nunique()}")

# Data distribution
print("\nData distribution by Product:")
print(df_combined.groupby('Product_Name')['Total_Quantity'].agg(['count', 'sum', 'mean', 'min', 'max']))

# Save to Excel
df_combined.to_excel('./data/demand_prediction_weekly_2015_2025.xlsx', index=False)
print("\n✓ Dataset saved as 'demand_prediction_weekly_2015_2025.xlsx'")
print(f"✓ Synthetic data: 2015-2020 (generated based on {len(df_weekly)} actual records)")
print(f"✓ Actual data: 2021-2025")