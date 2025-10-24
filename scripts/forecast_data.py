import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_percentage_error
import os

# Load your dataset
df = pd.read_excel('./data/sample_pharmacy_data_enhanced.xlsx')

# Convert month column
df['Month'] = pd.to_datetime(df['Month'])

# Select a single medicine for forecasting
medicine = 'Paracetamol'
data = df[df['Medicine'] == medicine].sort_values('Month')

# Feature engineering (combine external factors)
data['Adjusted_Demand'] = (
    data['Units_Dispensed']
    + 0.3 * data['Outbreak_Index'] * data['Units_Dispensed']
    # + 0.2 * data['Campaign_Indicator'] * data['Units_Dispensed']
    - 0.1 * data['Delivery_Delay_Days'] * 5
    - 0.05 * data['Expiry_Loss_%'] * 10
)

data = data.set_index('Month')

# Model training (SARIMA can handle seasonality)
model = SARIMAX(
    data['Adjusted_Demand'],
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12),
    enforce_stationarity=False,
    enforce_invertibility=False
)

model_fit = model.fit(disp=False)

# Forecast next 3 months
forecast_steps = 3
forecast = model_fit.forecast(steps=forecast_steps)
forecast_dates = pd.date_range(data.index[-1] + pd.offsets.MonthEnd(), periods=forecast_steps, freq='M')

# Combine forecast into DataFrame
forecast_df = pd.DataFrame({
    'Month': forecast_dates,
    'Predicted_Units': forecast.values
})

# Plot
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['Units_Dispensed'], label='Actual', marker='o')
plt.plot(forecast_df['Month'], forecast_df['Predicted_Units'], label='Forecast', marker='x', linestyle='--')
plt.title(f'{medicine} Demand Forecast (Next 3 Months)')
plt.xlabel('Month')
plt.ylabel('Units Dispensed')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print(forecast_df)




#evaluate data
train = data['Adjusted_Demand'][:-3]
test = data['Adjusted_Demand'][-3:]

# Fit SARIMA (handles seasonality)
model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12),
                enforce_stationarity=False, enforce_invertibility=False)
model_fit = model.fit(disp=False)

# Predict the last 3 months
pred = model_fit.forecast(steps=3)

# Mean Absolute Percentage Error
mape = mean_absolute_percentage_error(test, pred) * 100
print(f"MAPE for {medicine}: {mape:.2f}%")





############forecast dashboard###################
forecast_steps = 3
results = []


# Create output folder if it doesn't exist
os.makedirs('./outputs', exist_ok=True)

# Step 3: Loop over all medicines
for med in df['Medicine'].unique():
    subset = df[df['Medicine'] == med].sort_values('Month').copy()
    
    # Adjust demand using external factors (basic adjustment)
    subset['Adjusted_Demand'] = subset['Units_Dispensed'] \
        - 5*subset['Stockout_Incident'] \
        - 0.05*subset['Expiry_Loss_%']*subset['Units_Dispensed'] \
        + 0.2*subset['Outbreak_Index']*subset['Units_Dispensed']
    
    subset = subset.set_index('Month')
    
    # SARIMA model
    try:
        model = SARIMAX(
            subset['Adjusted_Demand'],
            order=(1,1,1),
            seasonal_order=(1,1,1,12),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        model_fit = model.fit(disp=False)
        forecast = model_fit.forecast(steps=forecast_steps)
        
        # Save forecast for each month
        for i in range(forecast_steps):
            results.append({
                'Medicine': med,
                'Month': forecast.index[i].strftime('%b %Y'),
                'Forecast_Units': round(forecast.values[i])
            })
    except Exception as e:
        print(f"⚠️ Forecast failed for {med}: {e}")

# -----------------------------
# Step 4: Save to Excel
# -----------------------------
forecast_table = pd.DataFrame(results)
forecast_table.to_excel('./outputs/forecast_summary_newdata.xlsx', index=False)
print("✅ Forecast summary saved to outputs/forecast_summary_newdata.xlsx")

