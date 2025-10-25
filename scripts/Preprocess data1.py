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

# Remove 2021 data (incomplete year with only 4 weeks)
df = df[df['Year'] >= 2022]

# Create Year-Week identifier (e.g., "2022-W01")
df['Year_Week'] = df['Year'].astype(str) + '-W' + df['Week'].astype(str).str.zfill(2)

# Group by Product Name and Year-Week, sum the quantities
df_weekly = df.groupby(['Product_Name', 'Year_Week', 'Year', 'Week'])['Qty'].sum().reset_index()

# Rename for final output
df_weekly.columns = ['Product_Name', 'Week', 'Year', 'Week_Number', 'Total_Quantity']

# Get global min and max year across all data (2022 onwards only)
global_min_year = int(df_weekly['Year'].min())
global_max_year = int(df_weekly['Year'].max())

print(f"\nData range: {global_min_year} to {global_max_year}")

# Create complete date range with all weeks for each product
products = sorted(df_weekly['Product_Name'].unique())
all_data = []

for product in products:
    product_data = df_weekly[df_weekly['Product_Name'] == product]
    
    # Generate all year-week combinations from global min to global max year
    for year in range(global_min_year, global_max_year + 1):
        for week_num in range(1, 54):  # Generate all 53 weeks
            year_week = f"{year}-W{week_num:02d}"
            
            # Check if this week exists in data for this product
            week_data = product_data[(product_data['Year'] == year) & (product_data['Week_Number'] == week_num)]
            
            if len(week_data) > 0:
                qty = week_data['Total_Quantity'].values[0]
            else:
                qty = 0  # Fill missing weeks with zero
            
            all_data.append({
                'Product_Name': product,
                'Week': year_week,
                'Year': year,
                'Week_Number': week_num,
                'Total_Quantity': int(qty)
            })

# Create final dataframe with all weeks
df_weekly_filled = pd.DataFrame(all_data)
df_weekly_filled = df_weekly_filled.sort_values(['Product_Name', 'Year', 'Week_Number']).reset_index(drop=True)

# Display first few rows
print("\nWeekly Aggregated Dataset (2022 onwards, zero-filled missing weeks):")
print(df_weekly_filled.head(30))
print(f"\nTotal rows: {len(df_weekly_filled)}")

# Save to Excel
df_weekly_filled.to_excel('./data/demand_prediction_weekly.xlsx', index=False)
print("\nDataset saved as 'demand_prediction_weekly.xlsx'")

# Optional: Check unique products and weeks
print(f"\nUnique Products: {df_weekly_filled['Product_Name'].nunique()}")
print(f"Total Weeks: {df_weekly_filled['Week'].nunique()}")

# Check data distribution
print("\nData distribution by Product:")
print(df_weekly_filled.groupby('Product_Name')['Total_Quantity'].agg(['count', 'sum', 'mean']))