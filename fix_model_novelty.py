"""
Quick fix script to retrain LOF model with novelty=True for production inference
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
import joblib
import json
from pathlib import Path

print("Loading training data...")
df = pd.read_csv('data/gas_pipeline_data.csv', parse_dates=['timestamp'])

print(f"Dataset shape: {df.shape}")
print(f"Anomaly rate: {df['crash'].mean() * 100:.2f}%")

# Feature engineering
print("\nEngineering features...")
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['month'] = df['timestamp'].dt.month
df['year'] = df['timestamp'].dt.year

# One-hot encode location (using 'loc' prefix to match production code)
location_dummies = pd.get_dummies(df['location'], prefix='loc')
df = pd.concat([df, location_dummies], axis=1)

# Select features
feature_columns = [
    'density_kg_m3',
    'pressure_diff_kpa',
    'pressure_kpa',
    'temperature_c',
    'hourly_flow_m3',
    'total_flow_m3',
    'hour',
    'day_of_week',
    'month',
    'year',
    'loc_Mardakan',
    'loc_Sumqayit',
    'loc_Turkan'
]

X = df[feature_columns]
y = df['crash'].apply(lambda x: -1 if x == 1 else 1)  # LOF uses -1 for anomalies

print(f"\nFeature matrix shape: {X.shape}")
print(f"Features: {feature_columns}")

# Create pipeline with novelty=True
print("\nTraining LocalOutlierFactor with novelty=True...")
contamination_rate = df['crash'].mean()
print(f"Contamination rate: {contamination_rate:.4f}")

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LocalOutlierFactor(
        n_neighbors=20,
        contamination=contamination_rate,
        novelty=True,  # CRITICAL: Enable novelty detection for production
        n_jobs=-1
    ))
])

# Fit the pipeline (with novelty=True, we can only fit, not fit_predict)
pipeline.fit(X)

print("✓ Model trained successfully with novelty=True")

# Save artifacts
artifacts_dir = Path('artifacts')
artifacts_dir.mkdir(exist_ok=True)

print("\nSaving artifacts...")
joblib.dump(pipeline, artifacts_dir / 'production_pipeline.joblib')
print(f"✓ Saved: {artifacts_dir / 'production_pipeline.joblib'}")

# Save feature config
feature_config = {
    'feature_columns': feature_columns,
    'n_features': len(feature_columns),
    'contamination_rate': float(contamination_rate)
}
with open(artifacts_dir / 'feature_config.json', 'w') as f:
    json.dump(feature_config, f, indent=2)
print(f"✓ Saved: {artifacts_dir / 'feature_config.json'}")

# Test the model
print("\n" + "="*60)
print("TESTING PRODUCTION INFERENCE")
print("="*60)

# Load the saved model
loaded_pipeline = joblib.load(artifacts_dir / 'production_pipeline.joblib')
print("\n✓ Model loaded successfully")

# Test on a few samples
test_samples = X.head(10)
predictions = loaded_pipeline.predict(test_samples)
print(f"\n✓ Predictions work! Sample predictions: {predictions}")
print(f"  Anomalies detected in test: {(predictions == -1).sum()}/10")

print("\n" + "="*60)
print("SUCCESS! Model is ready for production use")
print("="*60)
print("\nThe dashboard should now work with real ML predictions.")
print("Restart the FastAPI server to load the new model.")
