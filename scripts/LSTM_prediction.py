import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# ===============================
# 1️⃣ Load trained model and data
# ===============================
model = load_model('../saved models/best_lstm.keras')

# Load the same dataset used for training
file_path = "../data/demand_prediction_weekly.xlsx"
sales_data = pd.read_excel(file_path)

# Select the medicine
selected_medicine = input("Enter the medicine name: ")

# Filter for that medicine
product_df = sales_data[sales_data['Product_Name'] == selected_medicine].copy()
product_df = product_df.groupby('Week')['Total_Quantity'].sum().reset_index()

# Sort by week
product_df = product_df.sort_values(by='Week')

# ===============================
# 2️⃣ Prepare the last known data
# ===============================
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(product_df[['Total_Quantity']])

time_steps = 2  # must match your LSTM training setting

# Take last `time_steps` values as input
last_sequence = scaled_data[-time_steps:].reshape(1, time_steps, 1)

# ===============================
# 3️⃣ Predict next 12 weeks (≈3 months)
# ===============================
future_predictions = []

for i in range(12):  # 12 weeks = 3 months
    next_pred = model.predict(last_sequence)              # shape: (1, 1)
    future_predictions.append(next_pred[0, 0])
    
    # Reshape next_pred correctly to (1, 1, 1)
    next_pred_reshaped = np.reshape(next_pred, (1, 1, 1))
    
    # Slide window forward
    last_sequence = np.concatenate((last_sequence[:, 1:, :], next_pred_reshaped), axis=1)


# Inverse transform predictions
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# ===============================
# 4️⃣ Prepare future week labels
# ===============================
# Convert 'YYYY-Www' to actual date (Monday of that ISO week)
last_week_str = product_df['Week'].iloc[-1]
year, week = map(int, last_week_str.split('-W'))
last_week = pd.to_datetime(f'{year}-W{week}-1', format='%G-W%V-%u')

future_weeks = pd.date_range(start=last_week + pd.Timedelta(weeks=1), periods=12, freq='W-MON')

# ===============================
# 6️⃣ Display forecast results
# ===============================
future_weeks_iso = [f"{d.isocalendar().year}-W{d.isocalendar().week:02d}" for d in future_weeks]

forecast_df = pd.DataFrame({
    'Week': future_weeks,
    'Week Number': future_weeks_iso,
    'Predicted_Quantity': future_predictions.flatten().astype(int)
})

print("\n📅 Next 3-Month Forecast:")
print(forecast_df)
