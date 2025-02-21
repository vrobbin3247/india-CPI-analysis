import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import joblib

import numpy as np
import tensorflow as tf
import random
import os

# Set seeds for reproducibility
# SEED = 66
SEED = 135
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
            Dropout(0.2),
            Dense(8, activation='relu'),
            Dropout(0.2),
            Dense(1)
        ])

        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model

    def train_and_evaluate(self, data):
        # Prepare sequences
        X, y = self.prepare_sequences(data)

        # Split data
        train_size = int(0.8 * len(X))
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
        original_test = data[self.window_size + len(train_pred):self.window_size + len(train_pred) + len(test_pred)]

        # Calculate metrics
        train_scores = self.calculate_metrics(original_train, train_pred)
        test_scores = self.calculate_metrics(original_test, test_pred)

        return model, train_pred, test_pred, train_scores, test_scores, history

    def calculate_metrics(self, actual, predicted):
        mse = mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual, predicted)
        r2 = r2_score(actual, predicted)

        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }

    def plot_results(self, data, train_pred, test_pred, history):
        plt.figure(figsize=(15, 10))

        # Plot 1: Actual vs Predicted
        plt.subplot(2, 1, 1)
        plt.plot(data, label='Actual', linewidth=2)

        # Plot predictions
        train_index = range(self.window_size, self.window_size + len(train_pred))
        test_index = range(self.window_size + len(train_pred),
                           self.window_size + len(train_pred) + len(test_pred))

        plt.plot(train_index, train_pred, label='Train Predictions',
                 linestyle='dashed', linewidth=2)
        plt.plot(test_index, test_pred, label='Test Predictions',
                 linestyle='dashed', linewidth=2)

        plt.title('Time Series Prediction')
        plt.legend()

        # Plot 2: Training History
        plt.subplot(2, 1, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss During Training')
        plt.legend()

        plt.tight_layout()
        plt.show()


def main():
    # Your data
    data = np.array(
        [105.1, 105.8, 106, 106.4, 107.2, 108.9, 110.7, 112.1, 114.2, 115.5, 117.4, 115.5, 114.2, 114, 114.6,
         115.4, 116, 117, 119.5, 120.7, 120.9, 121, 121.1, 120.3, 120.3, 120.6, 121.1, 121.5, 122.4, 124.1,
         124.7, 126.1, 127, 127.7, 128.3, 127.9, 128.1, 127.9, 128, 129, 130.3, 131.9, 133, 133.5, 133.4,
         133.8, 133.6, 132.8, 132.4, 132.6, 132.8, 132.9, 133.3, 133.9, 136.2, 137.8, 137.6, 138.3, 140,
         139.8, 139.3, 138.5, 138.7, 139.1, 139.8, 140.5, 141.8, 142.5, 142.1, 142.2, 142.4, 141.9, 141, 141,
         141.2, 141.7, 142.4, 143.6, 144.9, 145.7, 146.7, 148.3, 149.9, 152.3, 151.9, 150.4, 149.8, 151.9,
         151.2, 152.7, 154.7, 155.4, 157.5, 159.8, 160.7, 158.5, 156.8, 156.7, 156.7, 157.6, 161.1, 162.1,
         163.2, 163.6, 164, 166.3, 167.6, 167, 166.4, 166.7, 168.7, 170.8, 172.5, 173.6, 174.3, 175.3, 176.4,
         177.9, 177.8, 177.1, 177.8, 177.9, 178, 178.8, 179.8, 181.9, 187.6, 187.6, 185.8, 187, 188.2, 187.6,
         187.3, 187.4, 187.8, 188.5, 189.4, 192.2, 195.3, 195.4, 196.7, 199.5, 199.4, 198.4, 196])

    # Initialize and run predictor
    predictor = TimeSeriesPredictor(window_size=12)
    model, train_pred, test_pred, train_scores, test_scores, history = predictor.train_and_evaluate(data)

    joblib.dump(predictor.scaler, "../scaler.pkl")
    print("Scaler saved successfully!")
    # Print metrics
    print("\nTraining Metrics:")
    for metric, value in train_scores.items():
        print(f"{metric}: {value:.4f}")

    print("\nTest Metrics:")
    for metric, value in test_scores.items():
        print(f"{metric}: {value:.4f}")

    # Plot results
    predictor.plot_results(data, train_pred, test_pred, history)





    return model, predictor


if __name__ == "__main__":
    model, predictor = main()
    model.save("lstm_rural_model_12.keras")
    print("Model saved successfully in Keras format!")