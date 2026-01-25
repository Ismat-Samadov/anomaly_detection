"""
Fix LOF model by retraining with novelty=True for production inference
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
import joblib
import json
from pathlib import Path

print("="*80)
print("FIXING LOF MODEL - RETRAINING WITH NOVELTY=TRUE")
print("="*80)

# Load existing artifacts to get the data shape and features
print("\nLoading existing artifacts...")
artifacts_dir = Path('artifacts')
feature_config = json.load(open(artifacts_dir / 'feature_config.json'))
old_pipeline = joblib.load(artifacts_dir / 'production_pipeline.joblib')

print(f"✓ Loaded existing config")
print(f"  Features: {feature_config['n_features']}")
print(f"  Feature columns: {feature_config['feature_columns']}")

# Load the data files
print("\nLoading data from data directory...")
data_dir = Path('data')

COLUMN_RENAME_MAP = {
    "TARİX": "timestamp",
    "XÜSUSİ ÇƏKİ\n(kq/m3)": "density_kg_m3",
    "TƏZYİQLƏR\nFƏRQİ (kPa)": "pressure_diff_kpa",
    "TƏZYİQ (kPa)": "pressure_kpa",
    "TEMPERATUR\n(C)": "temperature_c",
    "SAATLIQ\nSƏRF(min m3)": "hourly_flow_m3",
    "SƏRF (min m3)": "total_flow_m3",
}

# Load all location files
dfs = []
for location in ['Mardakan', 'Sumqayit', 'Turkan']:
    df = pd.read_csv(data_dir / f'{location}.csv')
    df = df.rename(columns=COLUMN_RENAME_MAP)
    df['location'] = location
    dfs.append(df)
    print(f"  ✓ Loaded {location}: {len(df):,} rows")

df = pd.concat(dfs, ignore_index=True)
print(f"\n✓ Combined dataset: {len(df):,} rows")

# Clean nulls
df = df.dropna()
print(f"✓ After cleaning: {len(df):,} rows")

# Parse timestamp
df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True, format='mixed')

# Feature engineering
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['month'] = df['timestamp'].dt.month
df['year'] = df['timestamp'].dt.year

# One-hot encode location with 'loc' prefix
df = pd.get_dummies(df, columns=['location'], prefix='loc', drop_first=False)

# Select features
X = df[feature_config['feature_columns']].values
print(f"\n✓ Feature matrix: {X.shape}")

# Create new LOF model with novelty=True
print("\n" + "="*80)
print("TRAINING NEW LOF MODEL WITH NOVELTY=TRUE")
print("="*80)

contamination_rate = 0.01
print(f"\nContamination rate: {contamination_rate}")

lof = LocalOutlierFactor(
    n_neighbors=20,
    contamination=contamination_rate,
    novelty=True,  # CRITICAL: Enable for production
    n_jobs=-1
)

# Create new pipeline
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nFitting LOF model...")
lof.fit(X_scaled)
print("✓ Model trained successfully")

# Create production pipeline
production_pipeline = Pipeline([
    ('scaler', scaler),
    ('model', lof)
])

print("\n" + "="*80)
print("TESTING NEW MODEL")
print("="*80)

# Test predictions
print("\nTesting predict() method...")
test_predictions = production_pipeline.predict(X_scaled[:100])
print(f"✓ Predictions work! Shape: {test_predictions.shape}")
print(f"  Anomalies in test sample: {(test_predictions == -1).sum()}/100")

# Test on full dataset
print("\nPredicting on full dataset...")
all_predictions = production_pipeline.predict(X_scaled)
anomaly_count = (all_predictions == -1).sum()
anomaly_rate = anomaly_count / len(all_predictions) * 100
print(f"✓ Full dataset predictions complete")
print(f"  Total: {len(all_predictions):,}")
print(f"  Normal: {(all_predictions == 1).sum():,} ({100-anomaly_rate:.2f}%)")
print(f"  Anomalies: {anomaly_count:,} ({anomaly_rate:.2f}%)")

# Save new pipeline
print("\n" + "="*80)
print("SAVING NEW PRODUCTION PIPELINE")
print("="*80)

# Backup old pipeline
import shutil
from datetime import datetime
backup_path = artifacts_dir / f"production_pipeline_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
shutil.copy(artifacts_dir / 'production_pipeline.joblib', backup_path)
print(f"✓ Backed up old pipeline to: {backup_path}")

# Save new pipeline
joblib.dump(production_pipeline, artifacts_dir / 'production_pipeline.joblib')
print(f"✓ Saved new pipeline to: {artifacts_dir / 'production_pipeline.joblib'}")

# Also save standalone LOF model
joblib.dump(lof, artifacts_dir / 'best_model_local_outlier_factor.joblib')
print(f"✓ Saved LOF model to: {artifacts_dir / 'best_model_local_outlier_factor.joblib'}")

# Update feature config to reflect current model
feature_config['best_model_name'] = 'Local Outlier Factor'
feature_config['contamination_rate'] = contamination_rate
with open(artifacts_dir / 'feature_config.json', 'w') as f:
    json.dump(feature_config, f, indent=2)
print(f"✓ Updated feature config")

# Verify the saved model
print("\n" + "="*80)
print("VERIFICATION")
print("="*80)

loaded_pipeline = joblib.load(artifacts_dir / 'production_pipeline.joblib')
loaded_lof = loaded_pipeline.named_steps['model']
print(f"\n✓ Reloaded pipeline")
print(f"  Model type: {type(loaded_lof).__name__}")
print(f"  Novelty setting: {loaded_lof.novelty}")
print(f"  Has predict(): {hasattr(loaded_lof, 'predict')}")

# Test loaded pipeline
test_pred = loaded_pipeline.predict(X_scaled[:10])
print(f"\n✓ Loaded pipeline works!")
print(f"  Test predictions: {test_pred}")

print("\n" + "="*80)
print("SUCCESS! MODEL READY FOR PRODUCTION")
print("="*80)
print("\nThe production pipeline now uses LOF with novelty=True")
print("Restart the FastAPI server to use the new model.")
