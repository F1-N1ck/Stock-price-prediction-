from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import numpy as np

def run_arima(df, steps=30):
    series = df['Close']
    model = ARIMA(series, order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)

    plt.figure(figsize=(12, 6))
    plt.plot(series, label='Historical')
    plt.plot(np.arange(len(series), len(series)+steps), forecast, label='ARIMA Forecast')
    plt.title("Stock Price Forecast with ARIMA")
    plt.legend()
    plt.show()

    return forecast
