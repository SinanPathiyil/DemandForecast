import pandas as pd

# Read your aggregated weekly data
file_path = r'C:\Users\Strix\Desktop\Boehm Tech\demand forecasting\data\demand_prediction_weekly.xlsx'
df = pd.read_excel(file_path)

print("Original dataset:")
print(f"Total rows: {len(df)}")
print(f"Unique products: {df['Product_Name'].nunique()}")
print("\nAll products:")
print(df['Product_Name'].unique())

# List of products to keep
products_to_keep = [
    'CLINMISKIN GEL',
    'DESWIN  TAB',
    'K GLIM-M 1MG',
    'MEFORNIX P',
    'MONTEMAC FX TAB'
]

# Filter data to keep only these products
df_filtered = df[df['Product_Name'].isin(products_to_keep)].copy()

print(f"\n\nFiltered dataset:")
print(f"Total rows: {len(df_filtered)}")
print(f"Unique products: {df_filtered['Product_Name'].nunique()}")
print("\nProducts kept:")
print(df_filtered['Product_Name'].unique())

# Sort by Product Name, Year, and Week Number
df_filtered = df_filtered.sort_values(['Product_Name', 'Year', 'Week_Number']).reset_index(drop=True)

# Display first few rows
print("\nFirst few rows of filtered data:")
print(df_filtered.head(20))

# Save filtered data to Excel
output_path = r'C:\Users\Strix\Desktop\Boehm Tech\demand forecasting\data\XGB_prediction_data.xlsx'
df_filtered.to_excel(output_path, index=False)
print(f"\n\nFiltered dataset saved to: {output_path}")
print(f"Total rows in filtered data: {len(df_filtered)}")