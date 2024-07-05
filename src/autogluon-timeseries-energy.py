import pandas as pd
import numpy as np
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
import matplotlib.pyplot as plt

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

# Prepare data for AutoGluon-TimeSeries
ts_df = TimeSeriesDataFrame.from_data_frame(
    df,
    id_column=None,  # Single time series, so no id column
    timestamp_column='timestamp',
    target_column='energy_consumption',
    static_feature_columns=[],
    known_covariates_columns=['temperature', 'day_of_week', 'month']
)

# Split data into train and test sets
train_data = ts_df.slice(None, -30)  # Use all but last 30 days for training
test_data = ts_df.slice(-30, None)   # Use last 30 days for testing

# Initialize and train the TimeSeriesPredictor
predictor = TimeSeriesPredictor(
    prediction_length=30,  # Forecast horizon
    path="autogluon_ts_model",
    target="energy_consumption",
    eval_metric="MASE",
    ignore_time_index=False
)

predictor.fit(
    train_data=train_data,
    time_limit=600,  # Time limit in seconds
    presets="medium_quality"
)

# Make predictions
predictions = predictor.predict(test_data)

# Evaluate the model
evaluation = predictor.evaluate(test_data)
print(f"Evaluation results: {evaluation}")

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
future_dates = pd.date_range(start=ts_df.index[-1] + pd.Timedelta(days=1), periods=30, freq='D')
future_df = pd.DataFrame({
    'timestamp': future_dates,
    'temperature': np.random.normal(loc=20, scale=5, size=30),
    'day_of_week': future_dates.dayofweek,
    'month': future_dates.month
})

future_ts_df = TimeSeriesDataFrame.from_data_frame(
    future_df,
    id_column=None,
    timestamp_column='timestamp',
    static_feature_columns=[],
    known_covariates_columns=['temperature', 'day_of_week', 'month']
)

future_predictions = predictor.predict(future_ts_df)

# Plot historical data and future forecast
plt.figure(figsize=(12, 6))
plt.plot(ts_df.index, ts_df['energy_consumption'], label='Historical')
plt.plot(future_predictions.index, future_predictions, label='Forecast')
plt.title('Energy Consumption Forecast for Next 30 Days')
plt.xlabel('Date')
plt.ylabel('Energy Consumption')
plt.legend()
plt.show()

# Feature importance
feature_importance = predictor.feature_importance(test_data)
print("Feature Importance:")
print(feature_importance)

# Model performance report
performance = predictor.leaderboard(test_data)
print("\nModel Performance:")
print(performance)
