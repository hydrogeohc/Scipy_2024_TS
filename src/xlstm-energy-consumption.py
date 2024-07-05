import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam

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

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)

# Prepare data for LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length), :])
        y.append(data[i + seq_length, 0])
    return np.array(X), np.array(y)

seq_length = 30  # Use 30 days of history to predict the next day
X, y = create_sequences(scaled_data, seq_length)

# Split data into train and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build the xLSTM model
class xLSTM(LSTM):
    def __init__(self, units, return_sequences=False, **kwargs):
        super(xLSTM, self).__init__(units, return_sequences=return_sequences, **kwargs)
        self.units = units

    def call(self, inputs):
        output, hidden_state, cell_state = super(xLSTM, self).call(inputs)
        return [output, hidden_state, cell_state]

model = Sequential([
    xLSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    xLSTM(50),
    Dense(1)
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Make predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Inverse transform predictions
train_predictions = scaler.inverse_transform(np.concatenate((train_predictions, np.zeros((len(train_predictions), 3))), axis=1))[:, 0]
test_predictions = scaler.inverse_transform(np.concatenate((test_predictions, np.zeros((len(test_predictions), 3))), axis=1))[:, 0]

# Calculate error metrics
train_rmse = np.sqrt(mean_squared_error(df['energy_consumption'][:train_size], train_predictions))
test_rmse = np.sqrt(mean_squared_error(df['energy_consumption'][train_size:], test_predictions))
train_mae = mean_absolute_error(df['energy_consumption'][:train_size], train_predictions)
test_mae = mean_absolute_error(df['energy_consumption'][train_size:], test_predictions)

print(f"Train RMSE: {train_rmse}")
print(f"Test RMSE: {test_rmse}")
print(f"Train MAE: {train_mae}")
print(f"Test MAE: {test_mae}")

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(df.index[seq_length:train_size], train_predictions, label='Train Predictions')
plt.plot(df.index[train_size+seq_length:], test_predictions, label='Test Predictions')
plt.plot(df.index, df['energy_consumption'], label='Actual')
plt.title('Energy Consumption Forecast')
plt.xlabel('Date')
plt.ylabel('Energy Consumption')
plt.legend()
plt.show()

# Feature importance analysis
def get_feature_importance(model, X):
    feature_importance = []
    for i in range(X.shape[2]):
        X_modified = X.copy()
        X_modified[:, :, i] = 0
        predictions = model.predict(X_modified)
        mse = mean_squared_error(y_test, predictions)
        feature_importance.append(mse)
    return feature_importance

feature_importance = get_feature_importance(model, X_test)
features = ['energy_consumption', 'temperature', 'day_of_week', 'month']

plt.figure(figsize=(10, 6))
plt.bar(features, feature_importance)
plt.title('Feature Importance')
plt.xlabel('Features')
plt.ylabel('Mean Squared Error (when feature is removed)')
plt.show()

# Generate forecast for next 30 days
last_sequence = scaled_data[-seq_length:]
forecast = []

for _ in range(30):
    next_pred = model.predict(last_sequence.reshape(1, seq_length, 4))
    forecast.append(next_pred[0, 0])
    last_sequence = np.roll(last_sequence, -1, axis=0)
    last_sequence[-1] = np.concatenate((next_pred, [0, 0, 0]))

forecast = scaler.inverse_transform(np.concatenate((forecast, np.zeros((len(forecast), 3))), axis=1))[:, 0]

future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=30, freq='D')

plt.figure(figsize=(12, 6))
plt.plot(df.index, df['energy_consumption'], label='Historical')
plt.plot(future_dates, forecast, label='Forecast')
plt.title('Energy Consumption Forecast for Next 30 Days')
plt.xlabel('Date')
plt.ylabel('Energy Consumption')
plt.legend()
plt.show()
