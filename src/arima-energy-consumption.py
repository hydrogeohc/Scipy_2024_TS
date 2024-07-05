import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pmdarima import auto_arima
import warnings
warnings.filterwarnings('ignore')

# Generate sample energy consumption data
def generate_energy_data(n_samples=1000):
    date_rng = pd.date_range(start='2020-01-01', end='2022-12-31', freq='D')
    df = pd.DataFrame(date_rng, columns=['timestamp'])
    
    # Create trend
    trend = np.linspace(0, 20, n_samples)
    
    # Create seasonality
    season = 10 * np.sin(2 * np.pi * np.arange(n_samples) / 365.25)
    
    # Create random noise
    noise = np.random.normal(0, 5, n_samples)
    
    df['energy_consumption'] = 100 + trend + season + noise
    return df

# Generate data
df = generate_energy_data()
df.set_index('timestamp', inplace=True)

# Split data into train and test sets
train_data = df[:-30]  # Use all but last 30 days for training
test_data = df[-30:]   # Use last 30 days for testing

# Automatically find the best ARIMA parameters
model_auto = auto_arima(train_data['energy_consumption'], start_p=0, start_q=0, max_p=5, max_q=5, m=7,
                        start_P=0, seasonal=True, d=1, D=1, trace=True, error_action='ignore',
                        suppress_warnings=True, stepwise=True)

print(f"Best ARIMA parameters: {model_auto.order}")

# Fit the ARIMA model
model = ARIMA(train_data['energy_consumption'], order=model_auto.order)
results = model.fit()

# Make predictions
forecast = results.forecast(steps=30)

# Calculate error metrics
mae = mean_absolute_error(test_data['energy_consumption'], forecast)
rmse = np.sqrt(mean_squared_error(test_data['energy_consumption'], forecast))
print(f"Mean Absolute Error: {mae}")
print(f"Root Mean Squared Error: {rmse}")

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(test_data.index, test_data['energy_consumption'], label='Actual')
plt.plot(test_data.index, forecast, label='Forecast')
plt.title('Energy Consumption Forecast')
plt.xlabel('Date')
plt.ylabel('Energy Consumption')
plt.legend()
plt.show()

# Generate a forecast for the next 30 days
future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=30, freq='D')
future_forecast = results.forecast(steps=30)

# Plot historical data and future forecast
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['energy_consumption'], label='Historical')
plt.plot(future_dates, future_forecast, label='Forecast')
plt.title('Energy Consumption Forecast for Next 30 Days')
plt.xlabel('Date')
plt.ylabel('Energy Consumption')
plt.legend()
plt.show()

# Model diagnostics
results.plot_diagnostics(figsize=(15, 12))
plt.show()

# Print model summary
print(results.summary())

# Perform decomposition of the time series
from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(df['energy_consumption'], model='additive', period=365)

plt.figure(figsize=(12, 10))
plt.subplot(411)
plt.plot(decomposition.observed)
plt.title('Observed')
plt.subplot(412)
plt.plot(decomposition.trend)
plt.title('Trend')
plt.subplot(413)
plt.plot(decomposition.seasonal)
plt.title('Seasonal')
plt.subplot(414)
plt.plot(decomposition.resid)
plt.title('Residual')
plt.tight_layout()
plt.show()
