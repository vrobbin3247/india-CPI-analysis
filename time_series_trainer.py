import numpy as np
import tensorflow as tf
import random
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import streamlit as st
import pandas as pd

# Set seeds for reproducibility
SEED = 190
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

class TimeSeriesPredictor:
    def __init__(self, window_size=12):
        self.window_size = window_size
        self.scaler = StandardScaler()

    def prepare_sequences(self, data):
        """Create sequences for training"""
        X, y = [], []
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))

        for i in range(len(scaled_data) - self.window_size):
            X.append(scaled_data[i:(i + self.window_size)].flatten())
            y.append(scaled_data[i + self.window_size])

        return np.array(X), np.array(y)

    def build_model(self, input_shape):
        model = Sequential([
            Dense(16, activation='relu', input_shape=(input_shape,)),
            Dropout(0.15),
            Dense(8, activation='relu'),
            Dropout(0.15),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model

    def train_and_evaluate(self, data):
        # Prepare sequences
        X, y = self.prepare_sequences(data)

        # Set the test set to the last 12 months
        test_size = 12
        train_size = len(X) - test_size

        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Build and train model
        model = self.build_model(self.window_size)

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True
        )

        history = model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=200,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=1
        )

        # Make predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        # Inverse transform predictions
        train_pred = self.scaler.inverse_transform(train_pred)
        test_pred = self.scaler.inverse_transform(test_pred)

        # Get original scale data for metrics
        original_train = data[self.window_size:self.window_size + len(train_pred)]
        original_test = data[-test_size:]  # Last 12 months

        # Calculate metrics
        train_scores = self.calculate_metrics(original_train, train_pred)
        test_scores = self.calculate_metrics(original_test, test_pred)

        return model, train_pred, test_pred, train_scores, test_scores, history

    def calculate_metrics(self, actual, predicted):
        mse = mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual, predicted)
        r2 = r2_score(actual, predicted)

        return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}


    def plot_results(self, data, train_pred, test_pred, history):
        # Prepare data for plotting
        train_index = range(self.window_size, self.window_size + len(train_pred))
        test_index = range(self.window_size + len(train_pred),
                           self.window_size + len(train_pred) + len(test_pred))

        # Convert data to Pandas DataFrame
        df_actual = pd.DataFrame({'Actual': data})
        df_train = pd.DataFrame({'Train Predictions': train_pred.flatten()}, index=train_index)
        df_test = pd.DataFrame({'Test Predictions': test_pred.flatten()}, index=test_index)

        # Combine all into a single DataFrame
        df_combined = df_actual.join(df_train, how='outer').join(df_test, how='outer')

        # Display prediction plot
        st.write("### Time Series Prediction")
        st.line_chart(df_combined)

        # Prepare training history DataFrame
        df_history = pd.DataFrame({
            'Training Loss': history.history['loss'],
            'Validation Loss': history.history['val_loss']
        })

        # Display loss plot
        st.write("### Model Loss During Training")
        st.line_chart(df_history)

def train_model(data):
    predictor = TimeSeriesPredictor(window_size=12)
    model, train_pred, test_pred, train_scores, test_scores, history = predictor.train_and_evaluate(data)

    print("\nTraining Metrics:")
    for metric, value in train_scores.items():
        print(f"{metric}: {value:.4f}")

    print("\nTest Metrics:")
    for metric, value in test_scores.items():
        print(f"{metric}: {value:.4f}")

    predictor.plot_results(data, train_pred, test_pred, history)

    return model, predictor