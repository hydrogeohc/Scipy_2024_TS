import pandas as pd
import numpy as np
from nixtlats import TimeGPT
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Generate sample energy consumption data
def generate_energy_data(n_samples=1000):
    date_rng = pd.date_range(start='2020-01-01', end='2022-12-31', freq='D')
    df = pd.DataFrame(date_rng, columns=['timestamp'])
    df['energy_consumption'] = np.random.normal(loc=100, scale=20, size=len(df))
    df['temperature'] = np.random.normal(loc=20, scale=5, size=len(df))
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    return df

# Generate data
df = generate_energy_data()
df.set_index('timestamp', inplace=True)

# Prepare data for TimeGPT
train_data = df[:-30]  # Use all but last 30 days for training
test_data = df[-30:]   # Use last 30 days for testing

# Initialize TimeGPT model
model = TimeGPT()

# Fit the model and make predictions
forecast = model.forecast(
    df=train_data,
    h=30,  # Forecast horizon
    time_col='timestamp',
    target_col='energy_consumption',
    static_features=['day_of_week', 'month'],
    time_varying_known_features=['temperature'],
    freq='D'
)

# Extract predictions
predictions = forecast['energy_consumption']

# Evaluate the model
mae = mean_absolute_error(test_data['energy_consumption'], predictions)
rmse = np.sqrt(mean_squared_error(test_data['energy_consumption'], predictions))
print(f"Mean Absolute Error: {mae}")
print(f"Root Mean Squared Error: {rmse}")

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(test_data.index, test_data['energy_consumption'], label='Actual')
plt.plot(predictions.index, predictions, label='Predicted')
plt.title('Energy Consumption Forecast')
plt.xlabel('Date')
plt.ylabel('Energy Consumption')
plt.legend()
plt.show()

# Generate a forecast for the next 30 days
future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=30, freq='D')
future_df = pd.DataFrame(index=future_dates)
future_df['temperature'] = np.random.normal(loc=20, scale=5, size=len(future_df))
future_df['day_of_week'] = future_df.index.dayofweek
future_df['month'] = future_df.index.month

future_forecast = model.forecast(
    df=df,
    h=30,
    time_col='timestamp',
    target_col='energy_consumption',
    static_features=['day_of_week', 'month'],
    time_varying_known_features=['temperature'],
    freq='D',
    future_covariates=future_df
)

future_predictions = future_forecast['energy_consumption']

# Plot historical data and future forecast
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['energy_consumption'], label='Historical')
plt.plot(future_predictions.index, future_predictions, label='Forecast')
plt.title('Energy Consumption Forecast for Next 30 Days')
plt.xlabel('Date')
plt.ylabel('Energy Consumption')
plt.legend()
plt.show()

# Feature importance
importance = model.get_feature_importance()
print("\nFeature Importance:")
print(importance)

# Anomaly detection
anomalies = model.detect_anomalies(
    df=df,
    time_col='timestamp',
    target_col='energy_consumption',
    static_features=['day_of_week', 'month'],
    time_varying_known_features=['temperature'],
    freq='D'
)

print("\nDetected Anomalies:")
print(anomalies)
