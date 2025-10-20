import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
import time
warnings.filterwarnings('ignore')

# Generate sample energy consumption data
def generate_energy_data(n_samples=1000, seed=42):
    np.random.seed(seed)
    date_rng = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
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
print("Generating energy consumption data...")
df = generate_energy_data()
df.set_index('timestamp', inplace=True)

# Split data into train and test sets
train_data = df[:-30]  # Use all but last 30 days for training
test_data = df[-30:]   # Use last 30 days for testing

print(f"Training data: {len(train_data)} samples")
print(f"Test data: {len(test_data)} samples")

# Import and initialize TimesFM model
try:
    import timesfm

    print("\nInitializing TimesFM model...")
    start_time = time.time()

    # Load the pretrained TimesFM model (200M parameters, version 2.5)
    model = timesfm.TimesFm(
        context_len=512,  # Context length for input
        horizon_len=30,   # Forecast horizon (30 days ahead)
        input_patch_len=32,
        output_patch_len=128,
        num_layers=20,
        model_dims=1280,
        backend='gpu',  # Use 'cpu' if GPU not available
    )

    # Load checkpoint from Hugging Face
    model.load_from_checkpoint(repo_id="google/timesfm-1.0-200m-pytorch")

    init_time = time.time() - start_time
    print(f"Model initialized in {init_time:.2f} seconds")

    # Prepare data for TimesFM (expects list of arrays)
    train_values = train_data['energy_consumption'].values

    # Make forecast
    print("\nGenerating forecast...")
    forecast_start = time.time()

    # TimesFM expects input as list of time series
    forecast_result = model.forecast(
        inputs=[train_values],
        freq=[0],  # Frequency (0 for daily data)
    )

    # Extract the forecast for test period (last 30 points)
    forecast = forecast_result.forecast[0][:30]

    forecast_time = time.time() - forecast_start
    print(f"Forecast generated in {forecast_time:.2f} seconds")

    # Calculate error metrics
    mae = mean_absolute_error(test_data['energy_consumption'], forecast)
    rmse = np.sqrt(mean_squared_error(test_data['energy_consumption'], forecast))

    print(f"\n{'='*50}")
    print(f"TimesFM Performance Metrics")
    print(f"{'='*50}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Model Initialization Time: {init_time:.2f} seconds")
    print(f"Forecast Generation Time: {forecast_time:.2f} seconds")
    print(f"Total Runtime: {init_time + forecast_time:.2f} seconds")
    print(f"{'='*50}")

    # Plot results - Test Set Prediction
    plt.figure(figsize=(14, 6))
    plt.plot(test_data.index, test_data['energy_consumption'],
             label='Actual', linewidth=2, marker='o', markersize=4)
    plt.plot(test_data.index, forecast,
             label='TimesFM Forecast', linewidth=2, marker='s', markersize=4, linestyle='--')
    plt.title('TimesFM: Energy Consumption Forecast (Test Set)', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Energy Consumption', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/Users/hydrogeo/Downloads/Scipy_2024_TS/results/timesfm_test_forecast.png', dpi=300)
    plt.show()

    # Generate forecast for next 30 days beyond the dataset
    print("\nGenerating future forecast for next 30 days...")
    future_forecast_result = model.forecast(
        inputs=[df['energy_consumption'].values],
        freq=[0],
    )
    future_forecast = future_forecast_result.forecast[0][:30]

    # Plot historical data and future forecast
    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=30, freq='D')

    plt.figure(figsize=(14, 6))
    plt.plot(df.index[-180:], df['energy_consumption'][-180:],
             label='Historical (Last 180 days)', linewidth=2, alpha=0.7)
    plt.plot(future_dates, future_forecast,
             label='TimesFM Forecast (Next 30 days)', linewidth=2,
             marker='s', markersize=4, linestyle='--', color='red')
    plt.axvline(x=df.index[-1], color='gray', linestyle=':', linewidth=2, label='Forecast Start')
    plt.title('TimesFM: Energy Consumption Forecast for Next 30 Days', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Energy Consumption', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/Users/hydrogeo/Downloads/Scipy_2024_TS/results/timesfm_future_forecast.png', dpi=300)
    plt.show()

    # Plot prediction error distribution
    errors = test_data['energy_consumption'].values - forecast

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(errors, bins=20, edgecolor='black', alpha=0.7)
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
    plt.title('TimesFM: Prediction Error Distribution', fontsize=12, fontweight='bold')
    plt.xlabel('Error (Actual - Predicted)', fontsize=10)
    plt.ylabel('Frequency', fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.scatter(test_data['energy_consumption'], forecast, alpha=0.6, s=50)
    plt.plot([test_data['energy_consumption'].min(), test_data['energy_consumption'].max()],
             [test_data['energy_consumption'].min(), test_data['energy_consumption'].max()],
             'r--', linewidth=2, label='Perfect Prediction')
    plt.title('TimesFM: Actual vs Predicted', fontsize=12, fontweight='bold')
    plt.xlabel('Actual Energy Consumption', fontsize=10)
    plt.ylabel('Predicted Energy Consumption', fontsize=10)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/Users/hydrogeo/Downloads/Scipy_2024_TS/results/timesfm_error_analysis.png', dpi=300)
    plt.show()

    # Save results to CSV
    results_df = pd.DataFrame({
        'Date': test_data.index,
        'Actual': test_data['energy_consumption'].values,
        'Forecast': forecast,
        'Error': errors,
        'Absolute_Error': np.abs(errors)
    })
    results_df.to_csv('/Users/hydrogeo/Downloads/Scipy_2024_TS/results/timesfm_results.csv', index=False)
    print("\nResults saved to: results/timesfm_results.csv")

    print("\n" + "="*50)
    print("TimesFM Model Summary")
    print("="*50)
    print("Model: Google TimesFM 1.0 (200M parameters)")
    print("Architecture: Transformer-based foundation model")
    print(f"Context Length: 512 time steps")
    print(f"Forecast Horizon: 30 days")
    print("Training: Pretrained on diverse time series datasets")
    print("="*50)

except ImportError:
    print("\n" + "="*50)
    print("ERROR: TimesFM not installed!")
    print("="*50)
    print("\nTo install TimesFM, run:")
    print("1. pip install timesfm")
    print("   OR")
    print("2. Clone the repository and install:")
    print("   git clone https://github.com/google-research/timesfm.git")
    print("   cd timesfm")
    print("   pip install -e .[torch]")
    print("\nNote: TimesFM requires PyTorch or JAX/Flax backend")
    print("="*50)

except Exception as e:
    print("\n" + "="*50)
    print(f"ERROR: {str(e)}")
    print("="*50)
    print("\nPlease ensure:")
    print("1. TimesFM is properly installed")
    print("2. PyTorch or JAX backend is available")
    print("3. You have internet connection to download the model checkpoint")
    print("="*50)
