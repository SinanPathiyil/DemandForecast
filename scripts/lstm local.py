import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from scikeras.wrappers import KerasRegressor


# 1. Data Cleaning and Normalization for the new dataset
def preprocess_data(df, product_name):
    """
    Preprocess data for a single product:
    - Filters product-specific data
    - Orders months correctly
    - Normalizes Total_Quantity
    """
    
    # Define month order so data can be sorted chronologically
    month_order = ["January", "February", "March", "April", "May", "June",
                   "July", "August", "September", "October", "November", "December"]

    # Filter data for a particular product
    product_df = df[df['Product_Name'] == product_name].copy()
    
    # Sort by month order
    product_df['Month'] = pd.Categorical(product_df['Month'], categories=month_order, ordered=True)
    product_df = product_df.sort_values('Month')
    
    # Extract Total_Quantity as numpy array
    data = product_df['Total_Quantity'].values.reshape(-1, 1)
    
    # Handle missing values (if any)
    data = np.nan_to_num(data)
    
    # Normalize between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    return scaled_data, scaler, product_df


# 2. Create supervised learning dataset
def create_dataset(data, time_steps=3):
    """
    Creates sequences of data for LSTM input:
    - X: past time steps
    - y: next step prediction target
    """
    X, y = [], []
    for i in range(time_steps, len(data)):
        X.append(data[i-time_steps:i, 0])  # previous 'time_steps' month quantities
        y.append(data[i, 0])               # next month's quantity
    X = np.array(X)
    y = np.array(y)
    return X, y




def create_multilayer_lstm_model(units=64, dropout_rate=0.0, activation='relu', 
                                optimizer='adam', time_steps=10):
    """
    Build the multilayer LSTM model as described in the paper
    """
    model = Sequential()

    # Define the input layer explicitly
    model.add(Input(shape=(time_steps, 1)))
    
    # First LSTM layer with return_sequences=True for multilayer
    model.add(LSTM(units=units, 
                   return_sequences=True, 
                   input_shape=(time_steps, 1),
                   activation=activation))
    model.add(Dropout(dropout_rate))
    
    # Second LSTM layer (final layer with return_sequences=False)
    model.add(LSTM(units=units, 
                   activation=activation))
    model.add(Dropout(dropout_rate))
    
    # Dense output layer
    model.add(Dense(units=1))
    
    # Compile model with MSE loss as mentioned in paper
    model.compile(optimizer=optimizer, loss='mse')
    
    return model




def build_optimized_lstm():
    """
    Implement grid search as described in the methodology
    """
    # Grid search parameters from Table 2
    param_grid = {
        'units': [64],  # Paper used 64 units
        'dropout_rate': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        'activation': ['relu', 'tanh', 'sigmoid', 'hard_sigmoid'],
        'batch_size': [8, 16, 32, 48],
        'epochs': [50, 100, 150, 200, 250],
        'optimizer': ['Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    }
    
    # Create KerasRegressor wrapper
    model = KerasRegressor(
        build_fn=create_multilayer_lstm_model,
        verbose=0
    )
    
    # Perform grid search
    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    
    return grid



def train_pharmaceutical_lstm(sales_data):
    """
    Complete training pipeline following the paper's methodology
    """
    # Step 1: Preprocess data
    scaled_data, scaler = preprocess_data(sales_data, "Abatatriptan")
    
    # Step 2: Create supervised learning dataset
    X, y = create_dataset(scaled_data, time_steps=10)
    
    # Step 3: Split data as described in paper
    # Training: Jan 2012 - July 2017
    # Testing: Aug 2017 - March 2019  
    # Validation: April 2019 - Dec 2020
    
    train_size = int(0.7 * len(X))
    test_size = int(0.85 * len(X))
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:test_size], y[train_size:test_size]
    X_val, y_val = X[test_size:], y[test_size:]
    
    # Reshape for LSTM input
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
    
    # Step 4: Build optimized model using best parameters from paper
    model = create_multilayer_lstm_model(
        units=64,
        dropout_rate=0.0,  # Best parameter from their grid search
        activation='relu',  # You may need to test this
        optimizer='adam',
        time_steps=10
    )
    
    # Step 5: Train the model
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=2,
        validation_data=(X_val, y_val),
        verbose=1,
        shuffle=False
    )
    
    return model, scaler, history






def evaluate_model(model, X_test, y_test, scaler):
    """
    Evaluate using RMSE and SMAPE as in the paper
    """
    # Make predictions
    predictions = model.predict(X_test)
    
    # Inverse transform to original scale
    predictions = scaler.inverse_transform(predictions)
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Calculate RMSE (equation 13)
    rmse = np.sqrt(np.mean((y_test_actual - predictions) ** 2))
    
    # Calculate SMAPE (equation 12)
    smape = np.mean(2 * np.abs(y_test_actual - predictions) / 
                   (np.abs(y_test_actual) + np.abs(predictions)))
    
    return rmse, smape, predictions
