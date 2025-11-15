import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import random, numpy as np
random.seed(42)
np.random.seed(42)

import pandas as pd


df = pd.read_excel('../data/demand_prediction_weekly1.xlsx')  # <-- change path if needed

medicine_name = input("Enter the medicine name from the list above: ")
#medicine_name = 'MEFORNIX-P TAB'
df_med = df[df['Product_Name'] == medicine_name].copy()

# Sort by week
#df_med = df_med.sort_values('Week')
df = df.sort_values('Week').reset_index(drop=True)



# ---- Derive Approximate Month ----
# Since there are ~4.3 weeks per month, convert week number to month roughly
df_med['Month'] = np.ceil(df_med['Week_Number'] / 4.33).astype(int)
df_med['Month'] = df_med['Month'].clip(upper=12)


# ---- Derive Quarter ----
df_med['Quarter'] = ((df_med['Month'] - 1) // 3 + 1).astype(int)

# ---- Flags for Year Start/End ----
df_med['Is_Year_Start'] = (df_med['Week_Number'] <= 4).astype(int)
df_med['Is_Year_End'] = (df_med['Week_Number'] >= 48).astype(int)

df_med['Sin_Week'] = np.sin(2 * np.pi * df_med['Week_Number'] / 52)
df_med['Cos_Week'] = np.cos(2 * np.pi * df_med['Week_Number'] / 52)


for lag in range(1, 13):  # 12 weeks (3 months)
    df_med[f'lag_{lag}'] = df_med['Total_Quantity'].shift(lag)

df_med['rolling_mean_3'] = df_med['Total_Quantity'].shift(1).rolling(window=3).mean()
df_med['rolling_mean_5'] = df_med['Total_Quantity'].shift(1).rolling(window=5).mean()



df_med['rolling_mean_6'] = df_med['Total_Quantity'].shift(1).rolling(6).mean()
df_med['rolling_std_6'] = df_med['Total_Quantity'].shift(1).rolling(6).std()


df_med['rolling_mean_8'] = df_med['Total_Quantity'].shift(1).rolling(window=8).mean()
df_med['rolling_std_4'] = df_med['Total_Quantity'].shift(1).rolling(window=4).std()

df_med = df_med.dropna()


from pickle import TRUE

from sympy import false


X_med = df_med.drop(columns=['Total_Quantity', 'Week', 'Product_Name'])
y_med = df_med['Total_Quantity']

# Drop non-numeric columns if any
#X_med = X_med.apply(pd.to_numeric, errors='ignore')
#non_numeric_cols = X_med.select_dtypes(include=['object']).columns
#if len(non_numeric_cols) > 0:
#    X_med = X_med.drop(columns=non_numeric_cols)

X_train, X_test, y_train, y_test = train_test_split(
    X_med, y_med, test_size=0.2, shuffle=False
)

#X_train, X_valid, y_train, y_valid = train_test_split(
#    X_train, y_train, test_size=0.1, shuffle=False)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
a = X_train
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from xgboost import XGBRegressor

"""

| Parameter        | Meaning           | Effect                  | Typical Range |
| ---------------- | ----------------- | ----------------------- | ------------- |
| n_estimators     | number of trees   | More trees = better fit | 300–1000      |
| learning_rate    | Step size         | Lower = more stable     | 0.01–0.1      |
| max_depth        | Tree depth        | Higher = more complex   | 3–10          |
| subsample        | Row sampling      | Prevents overfitting    | 0.5–1.0       |
| colsample_bytree | Column sampling   | Prevents overfitting    | 0.6–1.0       |
| min_child_weight | Min data per leaf | Higher = simpler        | 1–10          |
| gamma            | Split threshold   | Higher = conservative   | 0–1           |
| reg_lambda       | L2 regularization | Higher = less overfit   | 0.1–10        |

"""


model = XGBRegressor(
    objective= "reg:squarederror",  # reg:squarederror,  reg:absoluteerror  use if there are sudden hikes
    n_estimators=500,
    learning_rate=0.01,
    max_depth=4,
    subsample=1,
    colsample_bytree=1,
    min_child_weight=1,
    gamma=0.1,
    reg_lambda=1.0,
)



model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ✅ Compute metrics directly without shifting
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.2f}")

final_xgb = model


