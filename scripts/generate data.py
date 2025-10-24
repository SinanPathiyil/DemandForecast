import pandas as pd
import numpy as np
import os

# -----------------------------
# Step 1: Define basic parameters
# -----------------------------
medicines = [
    'Paracetamol', 'Amoxicillin', 'Amlodipine', 'Metformin', 'Insulin',
    'Cough Syrup', 'Antacid', 'Losartan', 'Cefixime', 'Vitamin D'
]

months = pd.date_range(start='2023-01-01', periods=24, freq='M')
np.random.seed(42)

data = []

# -----------------------------
# Step 2: Simulate synthetic data
# -----------------------------
for med in medicines:
    base_demand = np.random.randint(500, 1500)
    seasonal_factor = np.sin(np.linspace(0, 3 * np.pi, 24)) * 100
    random_noise = np.random.normal(0, 50, 24)
    
    # Monthly consumption pattern
    units_dispensed = base_demand + seasonal_factor + random_noise
    units_dispensed = np.clip(units_dispensed, 200, None)  # avoid negative
    
    # Simulate purchase and delivery timelines
    purchase_lead_time_days = np.random.randint(5, 15, 24)  # time between order and arrival
    delivery_delay_days = np.random.choice([0, 1, 2, 3, 5], 24, p=[0.6, 0.2, 0.1, 0.05, 0.05])
    
    # Stockout incidents (0 or 1)
    stockout = np.random.choice([0, 1], 24, p=[0.9, 0.1])
    
    # Expiry losses (as percentage of total stock)
    expiry_loss_percent = np.random.uniform(0, 3, 24)  # 0–3%
    
    # External factors (e.g., seasonal outbreaks)
    outbreak_index = np.sin(np.linspace(0, 2 * np.pi, 24)) * 0.5 + np.random.uniform(-0.1, 0.1, 24)
    outbreak_index = np.clip(outbreak_index, 0, 1)
    
    for i in range(24):
        data.append({
            'Medicine': med,
            'Month': months[i],
            'Units_Dispensed': int(round(units_dispensed[i])),
            'Purchase_Lead_Time_Days': purchase_lead_time_days[i],
            'Delivery_Delay_Days': delivery_delay_days[i],
            'Stockout_Incident': stockout[i],
            'Expiry_Loss_%': round(expiry_loss_percent[i], 2),
            'Outbreak_Index': round(outbreak_index[i], 2)
        })

# -----------------------------
# Step 3: Create DataFrame
# -----------------------------
df = pd.DataFrame(data)

# -----------------------------
# Step 4: Save to Excel
# -----------------------------

import os

output_dir = './data'
os.makedirs(output_dir, exist_ok=True)
df.to_excel(os.path.join(output_dir, 'sample_pharmacy_data_enhanced.xlsx'), index=False)

print("✅ Enhanced dataset generated: data/sample_pharmacy_data_enhanced.xlsx")
