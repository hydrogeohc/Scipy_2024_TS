# Scipy_2024_TS

This repository includes the codes and paper for the SciPy conference 2024 in Tacoma, WA 2024.

# Time Series Models for Energy Consumption Forecasting

This repository contains examples of various time series models applied to energy consumption forecasting. Each model is implemented in Python and demonstrated using synthetic energy consumption data.

## Models Included

1. ARIMA (AutoRegressive Integrated Moving Average)
2. xLSTM (Explainable Long Short-Term Memory)
3. LLMTime (Large Language Model for Time Series)
4. TimeGPT
5. TFT (Temporal Fusion Transformer)
6. AutoGluon-TimeSeries

## File Descriptions

- `arima-energy-consumption.py`: Implementation of ARIMA model
- `xlstm-energy-consumption.py`: Implementation of xLSTM model
- `llmtime-energy-consumption.py`: Implementation of LLMTime model
- `timegpt-energy-consumption.py`: Implementation of TimeGPT model
- `tft-energy-consumption.py`: Implementation of Temporal Fusion Transformer
- `autogluon-timeseries-energy.py`: Implementation of AutoGluon-TimeSeries

## Model Descriptions

### ARIMA (arima-energy-consumption.py)
ARIMA is a classical statistical method for time series forecasting. It combines autoregression, differencing, and moving average components. ARIMA is particularly useful for univariate time series with trend and/or seasonal components.

Key features:
- Interpretable components (AR, I, MA)
- Handles both trend and seasonality
- Well-established statistical properties

### xLSTM (xlstm-energy-consumption.py)
xLSTM is an extension of traditional Long Short-Term Memory networks that aims to provide better interpretability. It maintains the powerful sequence modeling capabilities of LSTMs while allowing for analysis of the model's internal states.

Key features:
- Captures long-term dependencies in time series data
- Provides interpretability through exposed hidden states
- Allows for feature importance analysis

### LLMTime (llmtime-energy-consumption.py)
LLMTime leverages large language models for time series forecasting tasks. It combines the power of language models with traditional time series analysis techniques.

Key features:
- Minimal data preprocessing required
- Can handle complex patterns and multiple seasonal effects
- Provides feature importance insights

### TimeGPT (timegpt-energy-consumption.py)
TimeGPT is a large language model specifically designed for time series forecasting tasks. It's known for its ability to handle complex patterns in time series data without extensive feature engineering.

Key features:
- Handles raw time series data with minimal preprocessing
- Captures multiple seasonalities and complex patterns
- Offers additional capabilities like anomaly detection

### Temporal Fusion Transformer (tft-energy-consumption.py)
The Temporal Fusion Transformer is a powerful model for multi-horizon forecasting tasks. It combines high-performance multi-horizon forecasting with interpretable insights into temporal dynamics.

Key features:
- Handles static, known, and observed inputs
- Provides interpretable attention weights
- Effective for long-term dependencies

### AutoGluon-TimeSeries (autogluon-timeseries-energy.py)
AutoGluon-TimeSeries is an AutoML framework specifically designed for time series forecasting. It automates the process of model selection, hyperparameter tuning, and feature engineering.

Key features:
- Automated model selection and ensemble learning
- Handles multiple related time series (panel data)
- Provides easy-to-use high-level APIs

## Usage

Each file can be run independently to demonstrate the respective model's performance on synthetic energy consumption data. Make sure to install the required dependencies before running the scripts.

For example, to run the ARIMA model:

```
python arima-energy-consumption.py
```

## Dependencies

The main dependencies for these models include:

- pandas
- numpy
- matplotlib
- statsmodels (for ARIMA)
- tensorflow (for xLSTM)
- nixtlats (for TimeGPT)
- pytorch_forecasting (for TFT)
- autogluon.timeseries (for AutoGluon-TimeSeries)

Please refer to each individual script for specific import statements and additional dependencies.

## Note

The data used in these examples is synthetic and generated for demonstration purposes. For real-world applications, you should use actual energy consumption data and may need to perform additional data preprocessing and model tuning steps.

