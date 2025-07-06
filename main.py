from preprocess import download_stock_data, prepare_lstm_data
from arima_model import run_arima
from lstm_model import build_lstm, plot_lstm_predictions
from sklearn.model_selection import train_test_split

# Parameters
ticker = "AAPL"
time_step = 60

# Download and plot data
df = download_stock_data(ticker)

# Run ARIMA
run_arima(df)

# Prepare data for LSTM
X, y, scaler = prepare_lstm_data(df, time_step)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build and train LSTM
model = build_lstm(time_step)
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64, verbose=1)

# Predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Plot
plot_lstm_predictions(df, scaler, time_step, train_predict, test_predict, y_train, y_test)
