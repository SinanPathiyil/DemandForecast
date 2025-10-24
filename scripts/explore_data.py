import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Step 1: Load dataset
# -----------------------------
df = pd.read_excel('./data/sample_pharmacy_data_enhanced.xlsx')

# Convert Month to datetime
df['Month'] = pd.to_datetime(df['Month'])

print("✅ Dataset loaded successfully!")
print("\nPreview of data:")
print(df.head())

# -----------------------------
# Step 2: Summary statistics
# -----------------------------
print("\n📊 Basic Statistics:")
print(df.describe())

print("\nUnique medicines:", df['Medicine'].nunique())
print("Date range:", df['Month'].min().strftime('%b %Y'), "→", df['Month'].max().strftime('%b %Y'))

# -----------------------------
# Step 3: Demand trend visualization (single medicine)
# -----------------------------
medicine_name = 'Paracetamol'
subset = df[df['Medicine'] == medicine_name]

plt.figure(figsize=(10, 5))
plt.plot(subset['Month'], subset['Units_Dispensed'], marker='o', label='Units Dispensed')
plt.title(f'Demand Trend for {medicine_name}')
plt.xlabel('Month')
plt.ylabel('Units Dispensed')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------
# Step 4: External factor influence (Outbreak vs Demand)
# -----------------------------
plt.figure(figsize=(10, 5))
sns.scatterplot(data=subset, x='Outbreak_Index', y='Units_Dispensed', hue='Month', palette='coolwarm')
plt.title(f'Effect of Outbreak Index on {medicine_name} Demand')
plt.xlabel('Outbreak Index (0–1)')
plt.ylabel('Units Dispensed')
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------
# Step 5: Stockouts and expiry correlation
# -----------------------------
plt.figure(figsize=(10, 5))
sns.scatterplot(data=subset, x='Expiry_Loss_%', y='Units_Dispensed', size='Stockout_Incident', hue='Stockout_Incident', palette='viridis', sizes=(50, 200))
plt.title(f'{medicine_name}: Expiry Loss vs Demand (Stockouts Highlighted)')
plt.xlabel('Expiry Loss (%)')
plt.ylabel('Units Dispensed')
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------
# Step 6: Correlation heatmap
# -----------------------------
corr_features = ['Units_Dispensed', 'Purchase_Lead_Time_Days', 'Delivery_Delay_Days', 'Stockout_Incident', 'Expiry_Loss_%', 'Outbreak_Index']
corr = df[corr_features].corr()

plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix — Demand vs Other Factors')
plt.tight_layout()
plt.show()

# -----------------------------
# Step 7: Multi-medicine comparison
# -----------------------------
plt.figure(figsize=(12,6))
sns.lineplot(data=df, x='Month', y='Units_Dispensed', hue='Medicine')
plt.title('Monthly Demand Trends for All Medicines')
plt.xlabel('Month')
plt.ylabel('Units Dispensed')
plt.grid(True)
plt.tight_layout()
plt.show()
