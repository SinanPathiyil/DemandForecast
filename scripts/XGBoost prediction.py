# ==============================================
# 📦 Single Week Prediction for Selected Products
# ==============================================

import pandas as pd
import numpy as np
import os
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from tkinter import Tk, filedialog

# ------------------- 1️⃣ Load Data -------------------
root = Tk()
root.withdraw()
file_path = filedialog.askopenfilename(
    title="Select Prediction Data File",
    filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
)

if not file_path:
    print("❌ No file selected. Exiting...")
    exit()

df = pd.read_excel(file_path)
print(f"✅ Loaded data from: {os.path.basename(file_path)}\n")

# Define medicines to predict
selected_medicines = ['CLINMISKIN GEL', 'DESWIN  TAB', 'K GLIM-M 1MG', 'MEFORNIX P', 'MONTEMAC FX TAB']

print(f"🔍 Checking for selected medicines in data...\n")

# ------------------- 2️⃣ Function to process each medicine -------------------
def forecast_medicine_next_week(medicine_name, df):
    """Generate next week forecast for a single medicine"""
    
    # Check if medicine exists in dataframe
    if medicine_name not in df['Product_Name'].unique():
        print(f"⚠️  '{medicine_name}' not found in dataset. Skipping...")
        return None
    
    df_med = df[df['Product_Name'] == medicine_name].copy()
    df_med = df_med.sort_values('Week').reset_index(drop=True)
    
    # Convert Week strings to datetime
    df_med['Week'] = pd.to_datetime(df_med['Week'] + '-1', format='%Y-W%W-%w')
    
    # Feature Engineering
    df_med['Month'] = np.ceil(df_med['Week_Number'] / 4.33).astype(int).clip(upper=12)
    df_med['Quarter'] = ((df_med['Month'] - 1) // 3 + 1).astype(int)
    df_med['Is_Year_Start'] = (df_med['Week_Number'] <= 4).astype(int)
    df_med['Is_Year_End'] = (df_med['Week_Number'] >= 48).astype(int)
    df_med['Sin_Week'] = np.sin(2 * np.pi * df_med['Week_Number'] / 52)
    df_med['Cos_Week'] = np.cos(2 * np.pi * df_med['Week_Number'] / 52)
    
    # Lag features
    for lag in range(1, 13):
        df_med[f'lag_{lag}'] = df_med['Total_Quantity'].shift(lag)
    
    # Rolling statistics
    df_med['rolling_mean_3'] = df_med['Total_Quantity'].shift(1).rolling(window=3).mean()
    df_med['rolling_mean_5'] = df_med['Total_Quantity'].shift(1).rolling(window=5).mean()
    df_med['rolling_mean_6'] = df_med['Total_Quantity'].shift(1).rolling(window=6).mean()
    df_med['rolling_std_6'] = df_med['Total_Quantity'].shift(1).rolling(window=6).std()
    df_med['rolling_mean_8'] = df_med['Total_Quantity'].shift(1).rolling(window=8).mean()
    df_med['rolling_std_4'] = df_med['Total_Quantity'].shift(1).rolling(window=4).std()
    
    df_med = df_med.dropna().reset_index(drop=True)
    
    # Load model based on medicine name from file
    model = XGBRegressor()
    model_dir = "./saved models"
    model_path = os.path.join(model_dir, f"xgboost_{medicine_name}.json")
    
    try:
        model.load_model(model_path)
        print(f"✅ Model loaded: xgboost_{medicine_name}.json")
    except Exception as e:
        print(f"❌ Model not found for '{medicine_name}'")
        print(f"   Expected: {model_path}")
        return None
    
    # Recreate scaler
    scaler = StandardScaler()
    X_train = df_med.drop(columns=['Total_Quantity', 'Week', 'Product_Name'])
    scaler.fit(X_train)
    
    # Define feature columns
    feature_cols = [c for c in df_med.columns 
                    if c not in ['Total_Quantity', 'Week', 'Product_Name']]
    
    # Prepare for forecasting
    last_12_quantities = list(df_med['Total_Quantity'].astype(float).values[-12:][::-1])
    last_row = df_med.iloc[-1].copy()
    start_year = int(last_row['Year'])
    start_week = int(last_row['Week_Number'])
    last_date = df_med['Week'].iloc[-1]
    
    # Single Week Prediction (Next Week Only)
    wk = start_week + 1
    yr = start_year
    if wk > 52:
        wk -= 52
        yr += 1
    
    # Temporal features
    month = int(np.ceil(wk / 4.33))
    month = min(month, 12)
    quarter = ((month - 1) // 3) + 1
    is_year_start = int(wk <= 4)
    is_year_end = int(wk >= 48)
    sin_week = np.sin(2 * np.pi * wk / 52)
    cos_week = np.cos(2 * np.pi * wk / 52)
    
    # Lag features
    lags = {f'lag_{j}': (last_12_quantities[j-1] if j-1 < len(last_12_quantities) else 0.0) 
            for j in range(1, 13)}
    
    # Safe functions
    def safe_mean(arr, k):
        if len(arr) < k:
            return float(np.mean(arr)) if len(arr) > 0 else 0.0
        return float(np.mean(arr[:k]))
    
    def safe_std(arr, k):
        if len(arr) < 2 or k < 2:
            return 0.0
        return float(np.std(arr[:k], ddof=1))
    
    # Rolling statistics
    rm3 = safe_mean(last_12_quantities, 3)
    rm5 = safe_mean(last_12_quantities, 5)
    rm6 = safe_mean(last_12_quantities, 6)
    rs6 = safe_std(last_12_quantities, 6)
    rm8 = safe_mean(last_12_quantities, 8)
    rs4 = safe_std(last_12_quantities, 4)
    
    # Build feature row
    row = {
        'Year': yr,
        'Week_Number': wk,
        'Month': month,
        'Quarter': quarter,
        'Is_Year_Start': is_year_start,
        'Is_Year_End': is_year_end,
        'Sin_Week': sin_week,
        'Cos_Week': cos_week,
        **lags,
        'rolling_mean_3': rm3,
        'rolling_mean_5': rm5,
        'rolling_mean_6': rm6,
        'rolling_std_6': rs6,
        'rolling_mean_8': rm8,
        'rolling_std_4': rs4
    }
    
    row_df = pd.DataFrame([row])
    row_df = row_df[feature_cols]
    
    # Predict
    X_input = scaler.transform(row_df)
    pred = model.predict(X_input)[0]
    
    next_week_label = f"{yr}-W{wk:02d}"
    next_date = last_date + pd.Timedelta(weeks=1)
    
    return {
        'Product': medicine_name,
        'Last_Actual_Week': f"{int(last_row['Year'])}-W{int(last_row['Week_Number']):02d}",
        'Last_Actual_Quantity': int(last_row['Total_Quantity']),
        'Next_Predicted_Week': next_week_label,
        'Next_Predicted_Quantity': int(round(pred))
    }

# ------------------- 3️⃣ Process Selected Products -------------------
all_predictions = []

for medicine_name in selected_medicines:
    print(f"\n🔄 Processing: {medicine_name}")
    prediction = forecast_medicine_next_week(medicine_name, df)
    
    if prediction is not None:
        all_predictions.append(prediction)
        print(f"   Prediction: {prediction['Next_Predicted_Quantity']} units for {prediction['Next_Predicted_Week']}")
    print()

# ------------------- 4️⃣ Create Output -------------------
if all_predictions:
    result_df = pd.DataFrame(all_predictions)
    
    # Save to single Excel file
    output_file = "./outputs/1_week_prediction_XGB.xlsx"
    result_df.to_excel(output_file, index=False)
    
    print("\n" + "="*70)
    print("✅ NEXT WEEK PREDICTIONS - ALL PRODUCTS")
    print("="*70)
    print(result_df.to_string(index=False))
    print("="*70)
    print(f"\n💾 Saved to: {output_file}")
else:
    print("❌ No predictions generated!")