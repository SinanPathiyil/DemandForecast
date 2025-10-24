import pandas as pd

# Read your dataset from the 'Data' sheet
df = pd.read_excel(r'C:\Users\Strix\Desktop\Boehm Tech\demand forecasting\data\Pharmaceutical_sales_ data.xlsx', sheet_name='Data')

# Print column names to verify
print("Column names in your dataset:")
print(df.columns.tolist())

# You already have Year and Month columns, so use them directly
# Create a Year-Month identifier for reference
df['Year_Month'] = df['Year'].astype(str) + '-' + df['Month']

# Select only the columns you need
df_filtered = df[['Product Name', 'Quantity', 'Year', 'Month', 'Year_Month']]

# Group by Product Name, Year, and Month
df_aggregated = df_filtered.groupby(['Product Name', 'Year', 'Month', 'Year_Month'])['Quantity'].sum().reset_index()

# Rename columns for clarity
df_aggregated.columns = ['Product_Name', 'Year', 'Month', 'Year_Month', 'Total_Quantity']

# Sort by Product Name, Year, and Month for better readability
df_aggregated = df_aggregated.sort_values(['Product_Name', 'Year', 'Month']).reset_index(drop=True)

# Display first few rows
print("Aggregated Dataset (Year-Wise Month Grouping):")
print(df_aggregated.head(15))
print(f"\nTotal rows: {len(df_aggregated)}")

# Save to Excel file
df_aggregated.to_excel('./data/demand_prediction_dataset.xlsx', index=False)
print("\nDataset saved as 'demand_prediction_dataset.xlsx'")

# Optional: Check unique products, years, and months
print(f"\nUnique Products: {df_aggregated['Product_Name'].nunique()}")
print(f"Unique Years: {df_aggregated['Year'].nunique()}")
print(f"Unique Months: {df_aggregated['Month'].nunique()}")
print(f"Total Year-Month Combinations: {df_aggregated['Year_Month'].nunique()}")

# Check data distribution
print("\nData distribution by Product:")
print(df_aggregated.groupby('Product_Name').size())