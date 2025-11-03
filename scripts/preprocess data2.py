import pandas as pd
import os

# Read your aggregated weekly data
file_path = r'C:\Users\Strix\Desktop\Boehm Tech\demand forecasting\data\demand_prediction_weekly.xlsx'
df = pd.read_excel(file_path)

print("Column names in your dataset:")
print(df.columns.tolist())
print(f"\nFirst few rows:")
print(df.head(10))

# Create output directory if it doesn't exist
output_dir = './data/product_files'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"\nCreated directory: {output_dir}")

# Get unique products
products = sorted(df['Product_Name'].unique())
print(f"\nTotal unique products: {len(products)}")
print("Products:")
for i, product in enumerate(products, 1):
    print(f"  {i}. {product}")

# Create separate Excel file for each product
for product in products:
    # Filter data for this product
    product_data = df[df['Product_Name'] == product].copy()
    
    # Sort by Year and Week_Number
    product_data = product_data.sort_values(['Year', 'Week_Number']).reset_index(drop=True)
    
    # Create filename (replace special characters with underscore)
    safe_name = product.replace('*', '').replace('/', '_').replace(':', '_').replace('?', '_').strip()
    filename = f"{safe_name}.xlsx"
    filepath = os.path.join(output_dir, filename)
    
    # Save to Excel
    product_data.to_excel(filepath, index=False)
    
    print(f"\nSaved: {filename} ({len(product_data)} rows)")

print(f"\n\nAll files saved to: {output_dir}")