import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input
from keras.optimizers import Adam
from scikeras.wrappers import KerasRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input

from keras.activations import relu, tanh, selu
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping



# 1. Data Cleaning and Normalization for the new dataset
def preprocess_data(df, product_name):
    """
    Preprocess data for a single product:
    - Filters product-specific data
    - Sorts by year and week number
    - Normalizes TotalQuantity
    """
    # Filter data for selected product
    product_df = df[df['Product_Name'] == product_name].copy()

    # Sort by Year THEN WeekNumber
    product_df = product_df.sort_values(['Year', 'Week_Number']).reset_index(drop=True)

    # Extract TotalQuantity as numpy array
    data = product_df['Total_Quantity'].values.reshape(-1, 1)

    # Handle missing values
    data = np.nan_to_num(data)

    # Normalize between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    #scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    return scaled_data, scaler, product_df, data



# 2. Create supervised learning dataset USING SCALED DATA
def create_dataset(scaled_data, time_steps=12):
    """
    Creates sequences of data for LSTM input:
    - X: past time steps
    - y: next step prediction target
    """

    X, y = [], []
    for i in range(len(scaled_data) - time_steps):
        X.append(scaled_data[i:(i + time_steps), 0])  # take past demand
        y.append(scaled_data[i + time_steps, 0])      # predict next demand
    X = np.array(X)
    y = np.array(y)
    return X, y



def create_multilayer_lstm_model(units=64, dropout_rate=0.0, activation=relu, 
                                optimizer=Adam(learning_rate=0.001), time_steps=2):
    """
    Build the multilayer LSTM model as described in the paper
    """
    model = Sequential()

    # Define the input layer explicitly
    model.add(Input(shape=(time_steps, 1)))
    
    # First LSTM layer with return_sequences=True for multilayer
    model.add(LSTM(units=units, 
                   return_sequences=True, 
                   #input_shape=(time_steps, 1),
                   activation=activation))
    model.add(Dropout(rate=dropout_rate))
    
    # Second LSTM layer (final layer with return_sequences=False)
    model.add(LSTM(units=64, 
                   return_sequences=False,
                   activation=activation))
    model.add(Dropout(rate=dropout_rate))
    

    #model.add(Dense(units=64, activation='relu'))

    # Dense output layer
    model.add(Dense(units=1))
    
    # Compile model with MSE loss as mentioned in paper
    model.compile(optimizer=optimizer, loss='mse', metrics=["accuracy"])
    
    return model

from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def evaluate_model(model, X_test, y_test, scaler, time_steps):
    """
    Evaluate LSTM model performance after aligning shifted predictions.
    """
    # Predict
    predictions = model.predict(X_test)

    # Inverse transform
    predictions = scaler.inverse_transform(predictions)
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Align predictions to actual values (same shift as plotting)
    shifted_predictions = predictions[time_steps:]              # drop first `time_steps`
    trimmed_actual = y_test_actual[:-time_steps]   # match lengths

    # Compute metrics
    rmse = np.sqrt(mean_squared_error(trimmed_actual, shifted_predictions))
    mae = mean_absolute_error(trimmed_actual, shifted_predictions)

    # SMAPE
    smape = np.mean(
        2 * np.abs(trimmed_actual - shifted_predictions) /
        (np.abs(trimmed_actual) + np.abs(shifted_predictions) + 1e-8)  # avoid div by 0
    )

    # Approx accuracy (as in your logic)
    accuracy = max(0, (1 - (mae / np.mean(trimmed_actual))) * 100)

    return rmse, smape, shifted_predictions, trimmed_actual, accuracy, mae


def train_pharmaceutical_lstm(sales_data, product_to_be_predicted):
    """
    Complete training pipeline following the paper's methodology
    """
    # Step 1: Preprocess data
    scaled_data, scaler, product_df, data = preprocess_data(sales_data, product_to_be_predicted)

    
    ts = 4  # time steps

    # Step 2: Create supervised learning dataset
    X, y = create_dataset(scaled_data, time_steps=ts)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
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
        units=128,
        dropout_rate=0.0,  # Best parameter from their grid search
        activation = relu,  # You may need to test this
        optimizer= Adam(learning_rate=1e-4),
        time_steps=ts
    )
    
    early_stop = EarlyStopping(
        monitor='val_loss', 
        patience=3, 
        restore_best_weights=True)

    # Step 5: Train the model
    history = model.fit(
        X_train, y_train,
        epochs=200,
        callbacks=[early_stop],
        batch_size=8,
        validation_data=(X_val, y_val),
        verbose=2,
        shuffle=False
    )


    #rmse, smape, predictions, y_test_actual, accuracy, mae = evaluate_model(model, X_test, y_test, scaler)
    rmse, smape, shifted_predictions, trimmed_actual, accuracy, mae = evaluate_model(model, X_test, y_test, scaler, ts)


    return model, scaler, history, rmse, smape, shifted_predictions, trimmed_actual, accuracy, mae, X, y

import pandas as pd
import matplotlib.pyplot as plt

# 1️⃣ Load dataset
file_path = "../data/demand_prediction_weekly1.xlsx"
sales_data = pd.read_excel(file_path)

# 2️⃣ Display available medicines
unique_medicines = sales_data['Product_Name'].unique()
print("Available Medicines:\n", unique_medicines)

# 3️⃣ Select medicine dynamically
selected_medicine = input("Enter the medicine name from the list above: ")

# 4️⃣ Train LSTM model (includes preprocessing + evaluation internally)
#model, scaler, history, rmse, smape, predictions, y_test_actual, accuracy, mae, X, y = train_pharmaceutical_lstm(sales_data, selected_medicine)

model, scaler, history, rmse, smape, shifted_predictions, trimmed_actual, accuracy, mae, X, y = train_pharmaceutical_lstm(sales_data, selected_medicine)

# 5️⃣ Print model performance
print(f"\n📊 Model Evaluation for {selected_medicine}")
print("RMSE",rmse)
print("SMAPE:" ,smape)
print(" Mean Absolute Error (MAE):",mae)
print(f"✅ Accuracy (approx): {accuracy:.2f}%")