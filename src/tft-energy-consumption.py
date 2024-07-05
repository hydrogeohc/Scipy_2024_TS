import pandas as pd
import numpy as np
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import MAE
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor

# Generate sample data
def generate_energy_data(n_samples=1000):
    date_rng = pd.date_range(start='2020-01-01', end='2022-12-31', freq='H')
    df = pd.DataFrame(date_rng, columns=['timestamp'])
    df['energy_consumption'] = np.random.normal(loc=100, scale=20, size=len(df))
    df['temperature'] = np.random.normal(loc=20, scale=5, size=len(df))
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['hour'] = df['timestamp'].dt.hour
    df['month'] = df['timestamp'].dt.month
    return df

# Prepare data
df = generate_energy_data()
df['time_idx'] = df.index

# Create dataset
max_encoder_length = 24 * 7  # Use past week for encoding
max_prediction_length = 24 * 2  # Predict next two days

training_cutoff = df['time_idx'].max() - max_prediction_length

training = TimeSeriesDataSet(
    df[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="energy_consumption",
    group_ids=["month"],  # Assuming each month is a separate group
    static_categoricals=[],
    static_reals=[],
    time_varying_known_reals=["hour", "day_of_week", "month", "temperature"],
    time_varying_unknown_reals=["energy_consumption"],
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
)

# Create validation set
validation = TimeSeriesDataSet.from_dataset(training, df, min_prediction_idx=training_cutoff + 1)

# Create data loaders
batch_size = 128
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

# Configure the model
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=1e-3,
    hidden_size=32,
    attention_head_size=1,
    dropout=0.1,
    hidden_continuous_size=16,
    loss=MAE(),
    log_interval=10,
    reduce_on_plateau_patience=4
)

# Configure trainer
trainer = pl.Trainer(
    max_epochs=100,
    gpus=0,
    gradient_clip_val=0.1,
    limit_train_batches=50,
    callbacks=[EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min"),
               LearningRateMonitor(logging_interval="epoch")]
)

# Fit the model
trainer.fit(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)

# Make predictions
best_model_path = trainer.checkpoint_callback.best_model_path
best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

predictions = best_tft.predict(val_dataloader, return_index=True, return_decoder_lengths=True)
print(predictions)
