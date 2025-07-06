from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import numpy as np

def build_lstm(time_step):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def plot_lstm_predictions(df, scaler, time_step, train_predict, test_predict, y_train, y_test):
    train_predict = scaler.inverse_transform(train_predict.reshape(-1, 1))
    test_predict = scaler.inverse_transform(test_predict.reshape(-1, 1))
    y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1))
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

    plt.figure(figsize=(12, 6))
    plt.plot(df['Close'].values, label='Original')
    plt.plot(np.arange(time_step+1, len(train_predict)+time_step+1), train_predict.flatten(), label='Train Predict')
    plt.plot(np.arange(len(train_predict)+2*time_step+2, len(df)-1), test_predict.flatten(), label='Test Predict')
    plt.title("Stock Price Prediction with LSTM")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.show()
