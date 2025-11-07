# ==============================================
# 🤝 Unified 1-Week Forecast (XGB + LSTM)
# Picks model based on selected medicine lists
# Output format matches your XGB script
# ==============================================

import pandas as pd
import numpy as np
import os
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.models import load_model
from tkinter import Tk, filedialog

# ------------------- 0️⃣ Configure model lists -------------------
xgb_medicines = ['CLINMISKIN GEL', 'DESWIN  TAB', 'MONTEMAC FX TAB', 'K GLIM-M 1MG']
lstm_medicines = ['AJAY SENSITIVE PLUS  --40', 'AMOCARE CV 625', 'MEFORNIX-P TAB', 'MEFTAL-P TAB']

# You can override this if you want to run only a subset:
selected_medicines = xgb_medicines + lstm_medicines

# ------------------- 1️⃣ Load Data -------------------
root = Tk()
root.withdraw()
file_path = filedialog.askopenfilename(
    title="Select Prediction Data File",
    filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
)

if not file_path:
    print("❌ No file selected. Exiting...")
    raise SystemExit

df = pd.read_excel(file_path)
print(f"✅ Loaded data from: {os.path.basename(file_path)}\n")

# Basic required columns check
required_cols = {'Product_Name', 'Week', 'Year', 'Week_Number', 'Total_Quantity'}
missing = [c for c in required_cols if c not in df.columns]
if missing:
    print(f"❌ Missing required columns in data: {missing}")
    raise SystemExit

# ------------------- 2️⃣ Helpers -------------------
def next_week_label_from_row(row):
    """Compute (year, week) for next week given a row with Year, Week_Number."""
    yr = int(row['Year'])
    wk = int(row['Week_Number']) + 1
    if wk > 52:
        wk -= 52
        yr += 1
    return yr, wk, f"{yr}-W{wk:02d}"

def convert_week_to_datetime(series):
    """Convert Week strings like '2024-W31' to datetime (Mon of that ISO week)."""
    # Keep consistent with your XGB script usage:
    # Use %Y-W%W-%w (Monday based) by appending '-1' (Mon)
    return pd.to_datetime(series + '-1', format='%Y-W%W-%w', errors='coerce')

# ------------------- 3️⃣ XGB: 1-week forecast -------------------
def forecast_medicine_next_week_xgb(medicine_name, df):
    """Generate next week forecast for a single medicine using XGBoost."""
    if medicine_name not in df['Product_Name'].unique():
        print(f"⚠️  '{medicine_name}' not found in dataset. Skipping...")
        return None
    
    df_med = df[df['Product_Name'] == medicine_name].copy()
    df_med = df_med.sort_values('Week').reset_index(drop=True)

    # Convert Week strings to datetime (consistent with your script)
    df_med['Week'] = convert_week_to_datetime(df_med['Week'])

    # Feature Engineering
    df_med['Month'] = np.ceil(df_med['Week_Number'] / 4.33).astype(int).clip(upper=12)
    df_med['Quarter'] = ((df_med['Month'] - 1) // 3 + 1).astype(int)
    df_med['Is_Year_Start'] = (df_med['Week_Number'] <= 4).astype(int)
    df_med['Is_Year_End'] = (df_med['Week_Number'] >= 48).astype(int)
    df_med['Sin_Week'] = np.sin(2 * np.pi * df_med['Week_Number'] / 52)
    df_med['Cos_Week'] = np.cos(2 * np.pi * df_med['Week_Number'] / 52)

    # Lags
    for lag in range(1, 13):
        df_med[f'lag_{lag}'] = df_med['Total_Quantity'].shift(lag)

    # Rolling stats
    df_med['rolling_mean_3'] = df_med['Total_Quantity'].shift(1).rolling(window=3).mean()
    df_med['rolling_mean_5'] = df_med['Total_Quantity'].shift(1).rolling(window=5).mean()
    df_med['rolling_mean_6'] = df_med['Total_Quantity'].shift(1).rolling(window=6).mean()
    df_med['rolling_std_6']  = df_med['Total_Quantity'].shift(1).rolling(window=6).std()
    df_med['rolling_mean_8'] = df_med['Total_Quantity'].shift(1).rolling(window=8).mean()
    df_med['rolling_std_4']  = df_med['Total_Quantity'].shift(1).rolling(window=4).std()

    df_med = df_med.dropna().reset_index(drop=True)
    if df_med.empty:
        print(f"⚠️ Not enough history for '{medicine_name}'. Skipping...")
        return None

    # Load model
    model = XGBRegressor()
    model_dir = "./saved models"
    model_path = os.path.join(model_dir, f"xgboost_{medicine_name}.json")

    try:
        model.load_model(model_path)
        print(f"✅ Model loaded: xgboost_{medicine_name}.json")
    except Exception:
        print(f"❌ Model not found for '{medicine_name}'")
        print(f"   Expected: {model_path}")
        return None

    # Recreate scaler
    feature_cols = [c for c in df_med.columns if c not in ['Total_Quantity', 'Week', 'Product_Name']]
    X_train = df_med[feature_cols]
    scaler = StandardScaler().fit(X_train)

    # Last actual info
    last_row = df_med.iloc[-1].copy()
    last_date = df_med['Week'].iloc[-1]
    last_actual_qty = int(last_row['Total_Quantity'])
    yr, wk, next_label = next_week_label_from_row(last_row)

    # Build next-week feature row
    month = int(np.ceil(wk / 4.33))
    month = min(month, 12)
    quarter = ((month - 1) // 3) + 1
    is_year_start = int(wk <= 4)
    is_year_end = int(wk >= 48)
    sin_week = np.sin(2 * np.pi * wk / 52)
    cos_week = np.cos(2 * np.pi * wk / 52)

    # Prepare last 12 actuals (most recent first)
    last_12 = list(df_med['Total_Quantity'].astype(float).values[-12:][::-1])

    def safe_mean(arr, k):
        if len(arr) < 1: return 0.0
        return float(np.mean(arr[:min(k, len(arr))]))

    def safe_std(arr, k):
        k = min(k, len(arr))
        if k < 2: return 0.0
        return float(np.std(arr[:k], ddof=1))

    lags = {f'lag_{j}': (last_12[j-1] if j-1 < len(last_12) else 0.0) for j in range(1, 13)}
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
        'rolling_mean_3': safe_mean(last_12, 3),
        'rolling_mean_5': safe_mean(last_12, 5),
        'rolling_mean_6': safe_mean(last_12, 6),
        'rolling_std_6':  safe_std(last_12, 6),
        'rolling_mean_8': safe_mean(last_12, 8),
        'rolling_std_4':  safe_std(last_12, 4)
    }

    row_df = pd.DataFrame([row])[feature_cols]
    X_input = scaler.transform(row_df)
    pred = model.predict(X_input)[0]

    return {
        'Product': medicine_name,
        'Last_Actual_Week': f"{int(last_row['Year'])}-W{int(last_row['Week_Number']):02d}",
        'Last_Actual_Quantity': last_actual_qty,
        'Next_Predicted_Week': next_label,
        'Next_Predicted_Quantity': int(round(float(pred)))
    }

# ------------------- 4️⃣ LSTM: 1-week forecast -------------------
def forecast_medicine_next_week_lstm(medicine_name, df):
    """Generate next week forecast for a single medicine using LSTM (time_steps=4)."""
    if medicine_name not in df['Product_Name'].unique():
        print(f"⚠️  '{medicine_name}' not found in dataset. Skipping...")
        return None

    df_med = df[df['Product_Name'] == medicine_name].copy()
    # Aggregate if multiple rows per week exist
    df_med = (df_med
              .groupby(['Year', 'Week_Number', 'Week'], as_index=False)['Total_Quantity']
              .sum())
    df_med = df_med.sort_values(['Year', 'Week_Number']).reset_index(drop=True)

    # Convert Week to datetime (for consistency / sanity, not used in features)
    df_med['Week_dt'] = convert_week_to_datetime(df_med['Week'])

    if len(df_med) < 5:
        print(f"⚠️ Not enough history (<5) for '{medicine_name}'. Skipping...")
        return None

    # Prepare scaler on Total_Quantity
    scaler = MinMaxScaler()
    qty = df_med[['Total_Quantity']].values.astype(float)
    scaled = scaler.fit_transform(qty)

    # Need last 4 points as input (time_steps=4)
    time_steps = 4
    last_seq = scaled[-time_steps:].reshape(1, time_steps, 1)

    # Load LSTM model
    model_dir = "./saved models"
    model_path = os.path.join(model_dir, f"lstm_{medicine_name}.keras")
    try:
        model = load_model(model_path)
        print(f"✅ Model loaded: lstm_{medicine_name}.keras")
    except Exception:
        print(f"❌ LSTM model not found for '{medicine_name}'")
        print(f"   Expected: {model_path}")
        return None

    # Predict 1 week ahead
    next_scaled = model.predict(last_seq, verbose=0)[0, 0]
    next_qty = scaler.inverse_transform(np.array([[next_scaled]])).flatten()[0]

    # Last actual info
    last_row = df_med.iloc[-1]
    yr, wk, next_label = next_week_label_from_row(last_row)

    return {
        'Product': medicine_name,
        'Last_Actual_Week': f"{int(last_row['Year'])}-W{int(last_row['Week_Number']):02d}",
        'Last_Actual_Quantity': int(round(float(last_row['Total_Quantity']))),
        'Next_Predicted_Week': next_label,
        'Next_Predicted_Quantity': int(round(float(next_qty)))
    }

# ------------------- 5️⃣ Run for selected medicines -------------------
all_predictions = []

print("🔍 Checking selected medicines and choosing models...\n")
for medicine_name in selected_medicines:
    print(f"\n🔄 Processing: {medicine_name}")

    if medicine_name in xgb_medicines:
        pred = forecast_medicine_next_week_xgb(medicine_name, df)
    elif medicine_name in lstm_medicines:
        pred = forecast_medicine_next_week_lstm(medicine_name, df)
    else:
        print(f"⚠️ '{medicine_name}' not in XGB or LSTM lists. Skipping...")
        pred = None

    if pred is not None:
        all_predictions.append(pred)
        print(f"   Prediction: {pred['Next_Predicted_Quantity']} units for {pred['Next_Predicted_Week']}")
    print()

# ------------------- 6️⃣ Create Output -------------------
if all_predictions:
    result_df = pd.DataFrame(all_predictions)
    os.makedirs("./outputs", exist_ok=True)
    output_file = "./outputs/1_week_prediction_ALL.xlsx"
    result_df.to_excel(output_file, index=False)

    print("\n" + "="*70)
    print("✅ NEXT WEEK PREDICTIONS - ALL PRODUCTS")
    print("="*70)
    print(result_df.to_string(index=False))
    print("="*70)
    print(f"\n💾 Saved to: {output_file}")
else:
    print("❌ No predictions generated!")
