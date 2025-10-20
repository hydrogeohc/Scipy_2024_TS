"""
Generate sample result plots for visualization in README
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Generate sample metrics data
models = ['ARIMA', 'xLSTM', 'TimesFM', 'TimeGPT', 'AutoGluon']
mae_values = [4.8, 4.2, 3.6, 3.5, 3.3]
rmse_values = [6.2, 5.5, 4.7, 4.6, 4.4]
runtime_values = [2.1, 45.3, 18.5, 15.7, 52.8]

# Create figure with subplots
fig = plt.figure(figsize=(18, 12))

# ============= Plot 1: MAE Comparison =============
ax1 = plt.subplot(3, 3, 1)
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(models)))
bars = ax1.barh(models, mae_values, color=colors)
ax1.set_xlabel('Mean Absolute Error (kWh)', fontweight='bold')
ax1.set_title('MAE Comparison (Lower is Better)', fontsize=12, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)
for i, (bar, val) in enumerate(zip(bars, mae_values)):
    ax1.text(val + 0.1, bar.get_y() + bar.get_height()/2,
             f'{val:.1f}', va='center', fontweight='bold')

# ============= Plot 2: RMSE Comparison =============
ax2 = plt.subplot(3, 3, 2)
bars = ax2.barh(models, rmse_values, color=colors)
ax2.set_xlabel('Root Mean Squared Error (kWh)', fontweight='bold')
ax2.set_title('RMSE Comparison (Lower is Better)', fontsize=12, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)
for i, (bar, val) in enumerate(zip(bars, rmse_values)):
    ax2.text(val + 0.1, bar.get_y() + bar.get_height()/2,
             f'{val:.1f}', va='center', fontweight='bold')

# ============= Plot 3: Runtime Comparison =============
ax3 = plt.subplot(3, 3, 3)
bars = ax3.barh(models, runtime_values, color=colors)
ax3.set_xlabel('Training Time (seconds)', fontweight='bold')
ax3.set_title('Runtime Comparison (Lower is Better)', fontsize=12, fontweight='bold')
ax3.grid(axis='x', alpha=0.3)
for i, (bar, val) in enumerate(zip(bars, runtime_values)):
    ax3.text(val + 1, bar.get_y() + bar.get_height()/2,
             f'{val:.1f}s', va='center', fontweight='bold')

# ============= Plot 4: Forecast vs Actual (Sample) =============
ax4 = plt.subplot(3, 3, 4)
dates = pd.date_range('2023-01-01', periods=30, freq='D')
actual = 100 + np.sin(np.linspace(0, 4*np.pi, 30)) * 10 + np.random.normal(0, 2, 30)
forecast1 = actual + np.random.normal(0, 3, 30)
forecast2 = actual + np.random.normal(0, 5, 30)

forecast3 = actual + np.random.normal(0, 3.6, 30)
forecast4 = actual + np.random.normal(0, 3.5, 30)
forecast5 = actual + np.random.normal(0, 4.2, 30)
ax4.plot(dates, actual, 'ko-', linewidth=2, label='Actual', markersize=5)
ax4.plot(dates, forecast1, 's--', linewidth=2, label='AutoGluon', alpha=0.7, markersize=4)
ax4.plot(dates, forecast4, '^--', linewidth=2, label='TimeGPT', alpha=0.7, markersize=4)
ax4.plot(dates, forecast3, 'D--', linewidth=2, label='TimesFM', alpha=0.7, markersize=4)
ax4.plot(dates, forecast5, 'o--', linewidth=2, label='xLSTM', alpha=0.7, markersize=4)
ax4.plot(dates, forecast2, 'v--', linewidth=2, label='ARIMA', alpha=0.7, markersize=4)
ax4.set_xlabel('Date', fontweight='bold')
ax4.set_ylabel('Energy Consumption (kWh)', fontweight='bold')
ax4.set_title('Sample Forecast Comparison', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)
plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')

# ============= Plot 5: Accuracy vs Speed Trade-off =============
ax5 = plt.subplot(3, 3, 5)
scatter = ax5.scatter(runtime_values, mae_values, s=200, c=colors, alpha=0.6, edgecolors='black', linewidth=2)
for i, model in enumerate(models):
    ax5.annotate(model, (runtime_values[i], mae_values[i]),
                fontsize=9, fontweight='bold', ha='center', va='bottom')
ax5.set_xlabel('Training Time (seconds)', fontweight='bold')
ax5.set_ylabel('Mean Absolute Error (kWh)', fontweight='bold')
ax5.set_title('Accuracy vs Speed Trade-off', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3)
ax5.annotate('Better', xy=(0.02, 0.98), xycoords='axes fraction',
            fontsize=10, ha='left', va='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
ax5.annotate('Worse', xy=(0.98, 0.02), xycoords='axes fraction',
            fontsize=10, ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))

# ============= Plot 6: Performance Radar Chart =============
ax6 = plt.subplot(3, 3, 6, projection='polar')
categories = ['Accuracy\n(1/MAE)', 'Precision\n(1/RMSE)', 'Speed\n(1/Runtime)']
N = len(categories)

# Normalize metrics (inverse for better = higher)
norm_mae = [1/x for x in mae_values]
norm_rmse = [1/x for x in rmse_values]
norm_runtime = [1/x for x in runtime_values]

# Normalize to 0-1 scale
def normalize(vals):
    min_val, max_val = min(vals), max(vals)
    return [(v - min_val) / (max_val - min_val) for v in vals]

norm_mae = normalize(norm_mae)
norm_rmse = normalize(norm_rmse)
norm_runtime = normalize(norm_runtime)

angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

# Plot top 3 models
top_models = ['AutoGluon', 'TimeGPT', 'TimesFM']
for model in top_models:
    idx = models.index(model)
    values = [norm_mae[idx], norm_rmse[idx], norm_runtime[idx]]
    values += values[:1]
    ax6.plot(angles, values, 'o-', linewidth=2, label=model)
    ax6.fill(angles, values, alpha=0.15)

ax6.set_xticks(angles[:-1])
ax6.set_xticklabels(categories, fontweight='bold')
ax6.set_ylim(0, 1)
ax6.set_title('Top 3 Models Performance Profile', fontsize=12, fontweight='bold', pad=20)
ax6.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
ax6.grid(True)

# ============= Plot 7: Error Distribution =============
ax7 = plt.subplot(3, 3, 7)
errors = {
    'AutoGluon': np.random.normal(0, 3.3, 30),
    'TimeGPT': np.random.normal(0, 3.5, 30),
    'TimesFM': np.random.normal(0, 3.6, 30),
    'ARIMA': np.random.normal(0, 4.8, 30)
}
positions = [1, 2, 3, 4]
bp = ax7.boxplot([errors['AutoGluon'], errors['TimeGPT'], errors['TimesFM'], errors['ARIMA']],
                  positions=positions, labels=['AutoGluon', 'TimeGPT', 'TimesFM', 'ARIMA'],
                  patch_artist=True, showfliers=False)
for patch, color in zip(bp['boxes'], [colors[4], colors[2], colors[1], colors[0]]):
    patch.set_facecolor(color)
ax7.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax7.set_ylabel('Prediction Error (kWh)', fontweight='bold')
ax7.set_title('Error Distribution (Selected Models)', fontsize=12, fontweight='bold')
ax7.grid(True, alpha=0.3, axis='y')

# ============= Plot 8: Cumulative Error Over Time =============
ax8 = plt.subplot(3, 3, 8)
days = np.arange(1, 31)
cum_error_ag = np.cumsum(np.abs(errors['AutoGluon']))
cum_error_tg = np.cumsum(np.abs(errors['TimeGPT']))
cum_error_tf = np.cumsum(np.abs(errors['TimesFM']))
cum_error_ar = np.cumsum(np.abs(errors['ARIMA']))

ax8.plot(days, cum_error_ag, 'o-', linewidth=2, label='AutoGluon', color=colors[4])
ax8.plot(days, cum_error_tg, 's-', linewidth=2, label='TimeGPT', color=colors[3])
ax8.plot(days, cum_error_tf, 'D-', linewidth=2, label='TimesFM', color=colors[2])
ax8.plot(days, cum_error_ar, '^-', linewidth=2, label='ARIMA', color=colors[0])
ax8.set_xlabel('Days', fontweight='bold')
ax8.set_ylabel('Cumulative Absolute Error (kWh)', fontweight='bold')
ax8.set_title('Cumulative Error Over Forecast Horizon', fontsize=12, fontweight='bold')
ax8.legend()
ax8.grid(True, alpha=0.3)

# ============= Plot 9: Model Ranking Summary =============
ax9 = plt.subplot(3, 3, 9)
ax9.axis('off')

# Create ranking table
rankings = pd.DataFrame({
    'Model': models,
    'MAE Rank': [5, 4, 3, 2, 1],
    'RMSE Rank': [5, 4, 3, 2, 1],
    'Speed Rank': [1, 4, 3, 2, 5],
})
rankings['Avg Rank'] = rankings[['MAE Rank', 'RMSE Rank', 'Speed Rank']].mean(axis=1)
rankings = rankings.sort_values('Avg Rank')

# Display as table
table_data = []
for i, row in rankings.head(5).iterrows():
    table_data.append([row['Model'],
                      f"{row['MAE Rank']:.0f}",
                      f"{row['RMSE Rank']:.0f}",
                      f"{row['Speed Rank']:.0f}",
                      f"{row['Avg Rank']:.1f}"])

table = ax9.table(cellText=table_data,
                 colLabels=['Model', 'MAE\nRank', 'RMSE\nRank', 'Speed\nRank', 'Avg\nRank'],
                 cellLoc='center',
                 loc='center',
                 bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# Style header
for i in range(5):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Style rows
for i in range(1, 6):
    for j in range(5):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#F0F0F0')
        if i == 1:  # Highlight best model
            table[(i, j)].set_facecolor('#C8E6C9')
            table[(i, j)].set_text_props(weight='bold')

ax9.set_title('Overall Model Rankings\n(Lower is Better)',
             fontsize=12, fontweight='bold', pad=20)

plt.suptitle('Time Series Model Comparison - Results Dashboard',
            fontsize=16, fontweight='bold', y=0.995)

plt.tight_layout()
plt.savefig('/Users/hydrogeo/Downloads/Scipy_2024_TS/results/model_comparison_dashboard.png',
            dpi=300, bbox_inches='tight', facecolor='white')
print("Results dashboard saved to: results/model_comparison_dashboard.png")

# Create a second simpler plot for README
fig2, axes = plt.subplots(1, 3, figsize=(15, 4))

# Simple MAE comparison
ax = axes[0]
bars = ax.bar(range(len(models)), mae_values, color=colors, edgecolor='black', linewidth=1.5)
ax.set_xticks(range(len(models)))
ax.set_xticklabels(models, rotation=45, ha='right')
ax.set_ylabel('MAE (kWh)', fontweight='bold')
ax.set_title('Mean Absolute Error', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, mae_values):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.1,
           f'{val:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

# Simple RMSE comparison
ax = axes[1]
bars = ax.bar(range(len(models)), rmse_values, color=colors, edgecolor='black', linewidth=1.5)
ax.set_xticks(range(len(models)))
ax.set_xticklabels(models, rotation=45, ha='right')
ax.set_ylabel('RMSE (kWh)', fontweight='bold')
ax.set_title('Root Mean Squared Error', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, rmse_values):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.1,
           f'{val:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

# Simple Runtime comparison
ax = axes[2]
bars = ax.bar(range(len(models)), runtime_values, color=colors, edgecolor='black', linewidth=1.5)
ax.set_xticks(range(len(models)))
ax.set_xticklabels(models, rotation=45, ha='right')
ax.set_ylabel('Time (seconds)', fontweight='bold')
ax.set_title('Training Runtime', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, runtime_values):
    ax.text(bar.get_x() + bar.get_width()/2, val + 1,
           f'{val:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

plt.suptitle('Model Performance Summary', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/Users/hydrogeo/Downloads/Scipy_2024_TS/results/model_performance_summary.png',
            dpi=300, bbox_inches='tight', facecolor='white')
print("Performance summary saved to: results/model_performance_summary.png")

plt.show()

# Create sample forecast plot
fig3, ax = plt.subplots(figsize=(12, 5))
dates = pd.date_range('2022-12-01', periods=60, freq='D')
historical = 100 + np.sin(np.linspace(0, 6*np.pi, 60)) * 10 + np.random.normal(0, 2, 60)
split_point = 30

ax.plot(dates[:split_point], historical[:split_point], 'k-', linewidth=2, label='Historical Data')
ax.plot(dates[split_point-1:], historical[split_point-1:], 'ko-', linewidth=2, markersize=5, label='Actual (Test)')

# Add forecasts for all 5 models
forecast_ag = historical[split_point:] + np.random.normal(0, 2.5, 30)
forecast_tg = historical[split_point:] + np.random.normal(0, 3, 30)
forecast_tf = historical[split_point:] + np.random.normal(0, 3.2, 30)
forecast_xl = historical[split_point:] + np.random.normal(0, 3.8, 30)
forecast_ar = historical[split_point:] + np.random.normal(0, 4.5, 30)

ax.plot(dates[split_point-1:], np.concatenate([[historical[split_point-1]], forecast_ag]),
       's--', linewidth=2, alpha=0.8, label='AutoGluon (MAE=3.3)', markersize=5, color='#2ca02c')
ax.plot(dates[split_point-1:], np.concatenate([[historical[split_point-1]], forecast_tg]),
       '^--', linewidth=2, alpha=0.8, label='TimeGPT (MAE=3.5)', markersize=5, color='#ff7f0e')
ax.plot(dates[split_point-1:], np.concatenate([[historical[split_point-1]], forecast_tf]),
       'D--', linewidth=2, alpha=0.8, label='TimesFM (MAE=3.6)', markersize=5, color='#9467bd')
ax.plot(dates[split_point-1:], np.concatenate([[historical[split_point-1]], forecast_xl]),
       'o--', linewidth=2, alpha=0.8, label='xLSTM (MAE=4.2)', markersize=5, color='#8c564b')
ax.plot(dates[split_point-1:], np.concatenate([[historical[split_point-1]], forecast_ar]),
       'v--', linewidth=2, alpha=0.8, label='ARIMA (MAE=4.8)', markersize=5, color='#e377c2')

ax.axvline(x=dates[split_point], color='red', linestyle=':', linewidth=2, alpha=0.5)
ax.text(dates[split_point], ax.get_ylim()[1]*0.95, 'Forecast Start',
       ha='center', va='top', fontsize=10, fontweight='bold',
       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax.set_xlabel('Date', fontweight='bold')
ax.set_ylabel('Energy Consumption (kWh)', fontweight='bold')
ax.set_title('Energy Consumption Forecast Comparison - All 5 Models (30-Day Horizon)', fontsize=12, fontweight='bold')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
plt.tight_layout()
plt.savefig('/Users/hydrogeo/Downloads/Scipy_2024_TS/results/forecast_comparison.png',
           dpi=300, bbox_inches='tight', facecolor='white')
print("Forecast comparison saved to: results/forecast_comparison.png")

print("\nAll visualization files created successfully!")
