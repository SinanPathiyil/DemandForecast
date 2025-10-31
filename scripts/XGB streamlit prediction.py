import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
import os

# ======================= Page Configuration =======================
st.set_page_config(page_title="Medicine Demand Forecast", layout="wide")
st.title("📊 8-Week Medicine Demand Forecasting")
st.markdown("---")

# ======================= Medicine Selection =======================
medicines = [
    "MONTEMAC FX TAB",
    "MEFORNIX P",
    "DESWIN  TAB",
    "CLINMISKIN GEL",
    "K GLIM-M 1MG"
]

medicine_name = st.selectbox(
    "🔍 Select Medicine Name:",
    options=medicines,
    index=0
)

st.markdown("---")

# ======================= Load & Process Data =======================
@st.cache_data
def load_and_process_data(medicine_name):
    """Load data and create predictions"""
    file_path = "../data/demand_prediction_weekly.xlsx"
    df = pd.read_excel(file_path)
    
    if medicine_name not in df['Product_Name'].unique():
        st.error(f"❌ Medicine '{medicine_name}' not found in dataset!")
        return None, None, None
    
    df_med = df[df['Product_Name'] == medicine_name].copy()
    df_med = df_med.sort_values('Week').reset_index(drop=True)
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
    
    # Load Model
    model = XGBRegressor()
    model_dir = "../saved models"
    model_path = os.path.join(model_dir, f"xgboost_{medicine_name}.json")
    
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found: {model_path}")
        return None, None, None
    
    model.load_model(model_path)
    
    # Recreate Scaler
    scaler = StandardScaler()
    X_train = df_med.drop(columns=['Total_Quantity', 'Week', 'Product_Name'])
    scaler.fit(X_train)
    
    feature_cols = [c for c in df_med.columns 
                    if c not in ['Total_Quantity', 'Week', 'Product_Name']]
    
    # Prepare Initial Lag History
    last_12_quantities = list(df_med['Total_Quantity'].astype(float).values[-12:][::-1])
    last_row = df_med.iloc[-1].copy()
    start_year = int(last_row['Year'])
    start_week = int(last_row['Week_Number'])
    last_date = df_med['Week'].iloc[-1]
    
    # Iterative Forecasting
    n_weeks = 8
    predictions = []
    pred_weeks = []
    pred_dates = []
    
    prev_qty = last_12_quantities.copy()
    
    for i in range(n_weeks):
        wk = start_week + i + 1
        yr = start_year
        while wk > 52:
            wk -= 52
            yr += 1
        
        month = int(np.ceil(wk / 4.33))
        month = min(month, 12)
        quarter = ((month - 1) // 3) + 1
        is_year_start = int(wk <= 4)
        is_year_end = int(wk >= 48)
        sin_week = np.sin(2 * np.pi * wk / 52)
        cos_week = np.cos(2 * np.pi * wk / 52)
        
        lags = {f'lag_{j}': (prev_qty[j-1] if j-1 < len(prev_qty) else 0.0) 
                for j in range(1, 13)}
        
        def safe_mean(arr, k):
            if len(arr) < k:
                return float(np.mean(arr)) if len(arr) > 0 else 0.0
            return float(np.mean(arr[:k]))
        
        def safe_std(arr, k):
            if len(arr) < 2 or k < 2:
                return 0.0
            return float(np.std(arr[:k], ddof=1))
        
        rm3 = safe_mean(prev_qty, 3)
        rm5 = safe_mean(prev_qty, 5)
        rm6 = safe_mean(prev_qty, 6)
        rs6 = safe_std(prev_qty, 6)
        rm8 = safe_mean(prev_qty, 8)
        rs4 = safe_std(prev_qty, 4)
        
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
        
        X_input = scaler.transform(row_df)
        pred = model.predict(X_input)[0]
        
        predictions.append(float(pred))
        pred_weeks.append(f"{yr}-W{wk:02d}")
        pred_dates.append(last_date + pd.Timedelta(weeks=i+1))
        
        prev_qty.insert(0, float(pred))
        prev_qty = prev_qty[:12]
    
    future_df = pd.DataFrame({
        'Week': pred_weeks,
        'Predicted_Quantity': [int(round(p)) for p in predictions]
    })
    
    return df_med, future_df, (predictions, pred_dates, pred_weeks)

# ======================= Execute Prediction =======================
if medicine_name:
    with st.spinner("🔄 Processing forecast..."):
        df_med, future_df, plot_data = load_and_process_data(medicine_name)
    
    if df_med is not None and future_df is not None:
        predictions, pred_dates, pred_weeks = plot_data
        
        # ======================= Layout: Left (DataFrame) + Right (2 Plots) =======================
        col_left, col_right = st.columns([1, 2])
        
        # LEFT COLUMN: Display DataFrame
        with col_left:
            st.subheader("📋 Forecast Results")
            st.dataframe(future_df, use_container_width=True, height=600)
            
            # Download Button
            csv = future_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇ Download Forecast CSV",
                data=csv,
                file_name=f"forecast_{medicine_name.replace(' ', '_')}.csv",
                mime="text/csv"
            )
        
        # RIGHT COLUMN: Two Plots Stacked Vertically
        with col_right:
            # Plot 1: Last 12 Weeks + Next 8 Weeks (Detailed)
            st.subheader("📈 Demand Trend (Last 12 + Next 8 Weeks)")
            fig1, ax1 = plt.subplots(figsize=(10, 5))
            
            last_12_dates = df_med['Week'].iloc[-12:].values
            last_12_actuals = df_med['Total_Quantity'].iloc[-12:].values
            
            ax1.plot(last_12_dates, last_12_actuals, marker='o', 
                    label='Actual Demand (Last 12 Weeks)', 
                    linewidth=2, markersize=6, color='blue')
            
            ax1.plot(pred_dates, predictions, marker='s', linestyle='--', 
                    label='Forecasted Demand (Next 8 Weeks)',
                    linewidth=2, markersize=6, color='red')
            
            last_12_weeks = [f"{int(yr)}-W{int(wk):02d}" for yr, wk in 
                            zip(df_med['Year'].iloc[-12:].values, 
                                df_med['Week_Number'].iloc[-12:].values)]
            
            all_dates = list(last_12_dates) + list(pred_dates)
            all_labels = last_12_weeks + pred_weeks
            
            ax1.set_xticks(all_dates)
            ax1.set_xticklabels(all_labels, rotation=45, ha='right', fontsize=8)
            ax1.set_xlabel("Week", fontsize=10)
            ax1.set_ylabel("Quantity", fontsize=10)
            ax1.legend(fontsize=9)
            ax1.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig1)
            
            st.markdown("---")
            
            # Plot 2: Full Historical + Forecast
            st.subheader("📊 Complete Forecast Overview")
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            
            future_dates = pd.date_range(df_med['Week'].iloc[-1], 
                                        periods=len(predictions) + 1, freq='W')[1:]
            
            ax2.plot(df_med['Week'], df_med['Total_Quantity'], 
                    label='Actual Demand', marker='o', linewidth=1.5, color='green')
            ax2.plot(future_dates, predictions, 
                    label='Forecasted Demand', marker='o', linestyle='--', 
                    linewidth=2, color='orange')
            
            ax2.set_xlabel("Week", fontsize=10)
            ax2.set_ylabel("Quantity", fontsize=10)
            ax2.legend(fontsize=9)
            ax2.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig2)
        
        st.success(f"✅ Forecast complete for '{medicine_name}'!")