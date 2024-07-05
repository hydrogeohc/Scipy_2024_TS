import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from llmtime import LLMTime
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Generate sample energy consumption data
def generate_energy_data(n_samples=1000):
    date_rng = pd.date_range(start='2020-01-01', end='2022-12-31', freq='D')
    df = pd.DataFrame(date_rng, columns=['date'])
    df['energy_consumption'] = np.random.normal(loc=100, scale=20, size=len(df))
    df['temperature'] = np.random.normal(loc=20, scale=5, size=len(df))
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    return df

# Generate and prepare data
df = generate_energy_data()
df.set_index('date', inplace=True)

# Normalize the data
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

# Prepare data for LLMTime
X = df_scaled[['temperature', 'day_of_week', 'month']]
y = df_scaled['energy_consumption']

# Split data into train and test sets
train_size = int(len(df) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Initialize the language model
model_name = "distilgpt2"  # You can use a larger model if you have the computational resources
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Initialize LLMTime
llmtime = LLMTime(
    model=model,
    tokenizer=tokenizer,
    max_length=512,
    num_beams=5,
    no_repeat_ngram_size=2,
    early_stopping=True
)

# Fit the model
llmtime.fit(X_train, y_train)

# Make predictions
y_pred = llmtime.predict(X_test)

# Inverse transform the predictions and actual values
y_pred_original = scaler.inverse_transform(np.column_stack((y_pred, X_test)))[:, 0]
y_test_original = scaler.inverse_transform(np.column_stack((y_test.values.reshape(-1, 1), X_test)))[:, 0]

# Calculate Mean Absolute Error
mae = np.mean(np.abs(y_pred_original - y_test_original))
print(f"Mean Absolute Error: {mae}")

# Plot results
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test_original, label='Actual')
plt.plot(y_test.index, y_pred_original, label='Predicted')
plt.title('Energy Consumption Forecast')
plt.xlabel('Date')
plt.ylabel('Energy Consumption')
plt.legend()
plt.show()

# Generate a forecast for the next 30 days
last_date = df.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq='D')
future_df = pd.DataFrame(index=future_dates)
future_df['temperature'] = np.random.normal(loc=20, scale=5, size=len(future_df))
future_df['day_of_week'] = future_df.index.dayofweek
future_df['month'] = future_df.index.month

future_df_scaled = pd.DataFrame(scaler.transform(future_df), columns=future_df.columns, index=future_df.index)

future_pred = llmtime.predict(future_df_scaled)
future_pred_original = scaler.inverse_transform(np.column_stack((future_pred, future_df_scaled)))[:, 0]

plt.figure(figsize=(12, 6))
plt.plot(df.index, df['energy_consumption'], label='Historical')
plt.plot(future_dates, future_pred_original, label='Forecast')
plt.title('Energy Consumption Forecast for Next 30 Days')
plt.xlabel('Date')
plt.ylabel('Energy Consumption')
plt.legend()
plt.show()
