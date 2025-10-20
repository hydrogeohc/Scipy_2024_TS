"""
Generate an architecture diagram for the time series model comparison framework
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Set up the figure
fig, ax = plt.subplots(1, 1, figsize=(16, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Define colors
color_data = '#E8F4F8'
color_prep = '#B8E6F0'
color_eval = '#C8E6C9'
color_viz = '#F8BBD0'

# Title
ax.text(5, 9.5, 'Time Series Forecasting Framework Architecture',
        fontsize=20, fontweight='bold', ha='center', va='center')

# 1. Data Layer
data_box = FancyBboxPatch((0.5, 7.5), 2, 1.2,
                          boxstyle="round,pad=0.1",
                          edgecolor='#0277BD', facecolor=color_data, linewidth=2)
ax.add_patch(data_box)
ax.text(1.5, 8.3, 'Data Generation', fontsize=12, fontweight='bold', ha='center', va='center')
ax.text(1.5, 8.0, 'Synthetic Energy', fontsize=9, ha='center', va='center')
ax.text(1.5, 7.75, 'Consumption Data', fontsize=9, ha='center', va='center')

# 2. Preprocessing Layer
prep_box = FancyBboxPatch((3.5, 7.5), 2, 1.2,
                          boxstyle="round,pad=0.1",
                          edgecolor='#01579B', facecolor=color_prep, linewidth=2)
ax.add_patch(prep_box)
ax.text(4.5, 8.3, 'Preprocessing', fontsize=12, fontweight='bold', ha='center', va='center')
ax.text(4.5, 8.0, '• Train/Test Split', fontsize=8, ha='center', va='center')
ax.text(4.5, 7.75, '• Normalization', fontsize=8, ha='center', va='center')

# Arrow from Data to Preprocessing
arrow1 = FancyArrowPatch((2.5, 8.1), (3.5, 8.1),
                        arrowstyle='->', mutation_scale=20, linewidth=2, color='#0277BD')
ax.add_patch(arrow1)

# 3. Model Layer (Five parallel models)
model_names = ['ARIMA\n(Statistical)', 'xLSTM\n(Deep Learning)', 'TimesFM\n(Foundation)', 'TimeGPT\n(API-based)', 'AutoGluon\n(AutoML)']
model_y_positions = [6.5, 5.2, 3.9, 2.6, 1.3]
color_models = ['#FFE5B4', '#FFD699', '#FFCC80', '#FFB366', '#FFA64D']

for i, (name, y_pos) in enumerate(zip(model_names, model_y_positions)):
    model_box = FancyBboxPatch((3.5, y_pos - 0.5), 2, 1.0,
                              boxstyle="round,pad=0.1",
                              edgecolor='#E65100', facecolor=color_models[i], linewidth=2)
    ax.add_patch(model_box)
    ax.text(4.5, y_pos + 0.25, name.split('\n')[0], fontsize=10, fontweight='bold', ha='center', va='center')
    ax.text(4.5, y_pos - 0.05, name.split('\n')[1], fontsize=7, ha='center', va='center', style='italic')

    # Arrows from preprocessing to models
    arrow = FancyArrowPatch((4.5, 7.5), (4.5, y_pos + 0.5),
                           arrowstyle='->', mutation_scale=15, linewidth=1.5,
                           color='#01579B', linestyle='--', alpha=0.6)
    ax.add_patch(arrow)

# Add model-specific details
details = [
    ['Auto params', 'Fast (2s)'],
    ['2x50 LSTM', '20 epochs'],
    ['Zero-shot', '200M params'],
    ['API-based', 'Pre-trained'],
    ['Ensemble', 'Auto-tuned']
]

for i, (detail, y_pos) in enumerate(zip(details, model_y_positions)):
    for j, line in enumerate(detail):
        ax.text(4.5, y_pos - 0.3 - j*0.12, line, fontsize=6.5, ha='center', va='center')

# 4. Evaluation Layer
eval_box = FancyBboxPatch((6.5, 4.0), 2, 2.5,
                          boxstyle="round,pad=0.1",
                          edgecolor='#2E7D32', facecolor=color_eval, linewidth=2)
ax.add_patch(eval_box)
ax.text(7.5, 6.2, 'Evaluation Metrics', fontsize=12, fontweight='bold', ha='center', va='center')
ax.text(7.5, 5.8, '────────────', fontsize=10, ha='center', va='center')
ax.text(7.5, 5.5, 'MAE', fontsize=10, ha='center', va='center', fontweight='bold')
ax.text(7.5, 5.25, '(Mean Absolute Error)', fontsize=7, ha='center', va='center')
ax.text(7.5, 4.9, 'RMSE', fontsize=10, ha='center', va='center', fontweight='bold')
ax.text(7.5, 4.65, '(Root Mean Squared Error)', fontsize=7, ha='center', va='center')
ax.text(7.5, 4.3, 'Runtime', fontsize=10, ha='center', va='center', fontweight='bold')
ax.text(7.5, 4.05, '(Training Time)', fontsize=7, ha='center', va='center')

# Arrows from models to evaluation
for y_pos in model_y_positions:
    arrow = FancyArrowPatch((5.5, y_pos), (6.5, 5.25),
                           arrowstyle='->', mutation_scale=15, linewidth=1.5,
                           color='#E65100', alpha=0.6)
    ax.add_patch(arrow)

# 5. Visualization Layer
viz_box = FancyBboxPatch((6.5, 1.0), 2, 2.0,
                         boxstyle="round,pad=0.1",
                         edgecolor='#C2185B', facecolor=color_viz, linewidth=2)
ax.add_patch(viz_box)
ax.text(7.5, 2.7, 'Visualization', fontsize=12, fontweight='bold', ha='center', va='center')
ax.text(7.5, 2.35, '• Forecast plots', fontsize=9, ha='center', va='center')
ax.text(7.5, 2.1, '• Metric comparison', fontsize=9, ha='center', va='center')
ax.text(7.5, 1.85, '• Performance tables', fontsize=9, ha='center', va='center')
ax.text(7.5, 1.6, '• Runtime analysis', fontsize=9, ha='center', va='center')
ax.text(7.5, 1.35, '• Error distribution', fontsize=9, ha='center', va='center')

# Arrow from evaluation to visualization
arrow_viz = FancyArrowPatch((7.5, 4.0), (7.5, 3.0),
                           arrowstyle='->', mutation_scale=20, linewidth=2, color='#2E7D32')
ax.add_patch(arrow_viz)

# Add legend box
legend_box = FancyBboxPatch((0.5, 0.5), 2, 2.5,
                           boxstyle="round,pad=0.1",
                           edgecolor='#424242', facecolor='#FAFAFA', linewidth=2)
ax.add_patch(legend_box)
ax.text(1.5, 2.8, 'Design Choices', fontsize=11, fontweight='bold', ha='center', va='center')
ax.text(1.5, 2.5, '────────────', fontsize=8, ha='center', va='center')

legend_items = [
    ('• Test Size: 30 days', 2.2),
    ('• Seq Length: 30 days', 1.95),
    ('• LSTM Layers: 2x50', 1.7),
    ('• LSTM Epochs: 20', 1.45),
    ('• Batch Size: 32', 1.2),
    ('• Seed: 42', 0.95),
]

for text, y in legend_items:
    ax.text(1.5, y, text, fontsize=8, ha='center', va='center')

# Add workflow notes
workflow_box = FancyBboxPatch((0.5, 3.5), 2, 3.5,
                             boxstyle="round,pad=0.1",
                             edgecolor='#424242', facecolor='#FFF9C4', linewidth=2)
ax.add_patch(workflow_box)
ax.text(1.5, 6.7, 'Workflow', fontsize=11, fontweight='bold', ha='center', va='center')
ax.text(1.5, 6.4, '────────────', fontsize=8, ha='center', va='center')

workflow_steps = [
    '1. Generate synthetic data',
    '   with trend + seasonality',
    '',
    '2. Split into train (970)',
    '   and test (30) samples',
    '',
    '3. Train 5 models in',
    '   parallel with timing',
    '',
    '4. Evaluate on test set',
    '   using MAE & RMSE',
    '',
    '5. Compare and visualize',
    '   results side-by-side',
]

y_start = 6.0
for i, step in enumerate(workflow_steps):
    ax.text(1.5, y_start - i*0.2, step, fontsize=7.5, ha='center', va='center')

# Add title box for methodology
method_box = FancyBboxPatch((0.3, 8.7), 9.4, 0.6,
                           boxstyle="round,pad=0.05",
                           edgecolor='#1565C0', facecolor='#E3F2FD', linewidth=2, alpha=0.3)
ax.add_patch(method_box)

# Add footer with key insights
footer_text = "Key Insight: Framework compares statistical, deep learning, foundation models (zero-shot), and AutoML approaches"
ax.text(5, 0.3, footer_text, fontsize=8.5, ha='center', va='center',
        style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig('/Users/hydrogeo/Downloads/Scipy_2024_TS/results/architecture_diagram.png',
            dpi=300, bbox_inches='tight', facecolor='white')
print("Architecture diagram saved to: results/architecture_diagram.png")
plt.show()
