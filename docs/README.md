# Natural Gas Pipeline Anomaly Detection

**End-to-End Machine Learning Pipeline for Operational Anomaly Detection**

This project implements a production-ready anomaly detection system for natural gas pipeline operations monitoring across three stations in Azerbaijan: Mardakan, Sumqayit, and Turkan.

## 📋 Project Overview

- **Objective**: Detect operational anomalies in gas pipeline sensor data
- **Approach**: Unsupervised anomaly detection with multiple algorithm comparison
- **Data**: ~174,000 hourly sensor readings (2018-2024)
- **Models**: 5 algorithms benchmarked and compared

## 📁 Directory Structure

```
crash_detection/
├── data/                  # Raw CSV files (Mardakan, Sumqayit, Turkan)
├── notebooks/             # Jupyter notebooks
│   └── anomaly_detection_pipeline.ipynb
├── charts/                # Generated visualizations (PNG)
├── outputs/               # Metrics, tables, results (CSV/JSON)
├── artifacts/             # Trained models, scalers, configs (joblib)
└── README.md             # This file
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib jupyter
```

### Running the Notebook

1. Open Jupyter Notebook:
```bash
cd /Users/ismatsamadov/crash_detection
jupyter notebook notebooks/anomaly_detection_pipeline.ipynb
```

2. Run all cells sequentially (Cell → Run All)

3. Results will be automatically saved to:
   - `/charts` - Visualizations
   - `/outputs` - CSV/JSON results
   - `/artifacts` - Trained models

## 📊 Features

### Data Processing
- ✅ Automated data loading and validation
- ✅ Comprehensive data quality assessment
- ✅ Null value handling and cleaning
- ✅ Temporal feature engineering (hour, day_of_week, month, year)
- ✅ One-hot encoding for categorical features
- ✅ StandardScaler normalization (fitted on train only)

### Models Implemented

1. **Isolation Forest** - Tree-based ensemble
2. **Local Outlier Factor (LOF)** - Density-based local outlier detection
3. **One-Class SVM** - Support vector boundary detection
4. **DBSCAN** - Density-based clustering
5. **Elliptic Envelope** - Robust covariance estimation

### Model Evaluation
- ✅ Train/Validation/Test split (70%/10%/20%)
- ✅ Comprehensive benchmarking metrics
- ✅ Rate consistency analysis
- ✅ Training time comparison
- ✅ Anomaly score distribution analysis

### Outputs Generated

#### Charts (`/charts/`)
1. `01_correlation_matrix.png` - Feature correlation heatmap
2. `02_feature_distributions.png` - Sensor feature distributions
3. `03_model_comparison.png` - Model benchmarking (4 subplots)
4. `04_anomaly_analysis.png` - Results analysis (4 subplots)

#### Data Outputs (`/outputs/`)
1. `00_SUMMARY_REPORT.txt` - Comprehensive execution summary
2. `01_data_quality_report_raw.csv` - Initial data quality metrics
3. `02_descriptive_statistics.csv` - Feature statistics
4. `03_model_comparison.csv` - Model performance comparison
5. `04_anomaly_results.csv` - Full predictions (CSV)
6. `04_anomaly_results.json` - Full predictions (JSON)
7. `05_anomalies_by_location.csv` - Location-based analysis
8. `06_anomalies_by_hour.csv` - Temporal analysis

#### Model Artifacts (`/artifacts/`)
1. `scaler.joblib` - Fitted StandardScaler
2. `best_model_*.joblib` - Best performing model
3. `feature_config.json` - Feature configuration
4. `production_pipeline.joblib` - End-to-end pipeline

## 🔧 Configuration

All configuration is centralized in the notebook's Configuration section:

```python
RANDOM_SEED = 42                    # Reproducibility
CONTAMINATION_RATE = 0.01          # Expected anomaly rate (1%)
TEST_SIZE = 0.2                    # Test set proportion (20%)
VAL_SIZE = 0.1                     # Validation set proportion (10%)
```

## 📈 Key Results

- **Total Records**: ~174,000 hourly observations
- **Time Coverage**: January 2018 - August 2024 (6.7 years)
- **Features**: 6 sensor + 4 temporal + 3 location = 13 total
- **Expected Anomaly Rate**: 1%
- **Best Model**: Automatically selected based on train-val consistency

## 🔄 Production Inference

```python
import joblib
import json

# Load production pipeline
pipeline = joblib.load("artifacts/production_pipeline.joblib")
config = json.load(open("artifacts/feature_config.json"))

# Prepare new data (must match feature columns)
# new_data shape: (n_samples, 13 features)

# Predict
predictions = pipeline.predict(new_data)
# Returns: 1 (normal) or -1 (anomaly)
```

## 📝 Data Schema

### Input CSV Columns (Azerbaijani)
- `TARİX` → timestamp
- `XÜSUSİ ÇƏKİ\n(kq/m3)` → density_kg_m3
- `TƏZYİQLƏR\nFƏRQİ (kPa)` → pressure_diff_kpa
- `TƏZYİQ (kPa)` → pressure_kpa
- `TEMPERATUR\n(C)` → temperature_c
- `SAATLIQ\nSƏRF(min m3)` → hourly_flow_m3
- `SƏRF (min m3)` → total_flow_m3

### Engineered Features
- `hour` (0-23)
- `day_of_week` (0-6, Monday=0)
- `month` (1-12)
- `year`
- `loc_Mardakan`, `loc_Sumqayit`, `loc_Turkan` (one-hot)

## 🎯 Model Selection Criteria

The best model is automatically selected based on:
1. **Rate Consistency**: Lowest absolute difference between train and validation anomaly rates
2. **Expected Contamination**: Closeness to 1% target rate
3. **Computational Efficiency**: Training time

## 📌 Best Practices Implemented

✅ **Reproducibility**: Fixed random seeds throughout
✅ **No Data Leakage**: Scaler fitted on train only
✅ **Modular Code**: Reusable functions
✅ **Clear Documentation**: Markdown explanations
✅ **Comprehensive Outputs**: Charts, metrics, models
✅ **Production-Ready**: Pipeline for inference
✅ **Schema Validation**: Automated column checks

## 🔬 Methodology

### 1. Data Validation
- Load 3 location CSV files
- Validate schema and detect missing values
- Remove incomplete rows (~2% data loss)

### 2. Feature Engineering
- Extract temporal patterns (hour, day, month, year)
- Encode categorical location variable
- Normalize all features (StandardScaler)

### 3. Model Training
- Split: 70% train, 10% validation, 20% test
- Train 5 different anomaly detection algorithms
- Evaluate on validation set

### 4. Model Selection
- Compare models on multiple criteria
- Select best based on consistency
- Final evaluation on test set

### 5. Production Deployment
- Create full dataset predictions
- Save production pipeline
- Generate comprehensive reports

## 📊 Evaluation Metrics

- **Anomaly Rate**: Percentage of data flagged as anomalous
- **Rate Consistency**: Train-val rate difference (lower = better)
- **Training Time**: Computational efficiency
- **Anomaly Score Distribution**: Separation quality

## ⚠️ Important Notes

1. **LOF and DBSCAN**: These models require `fit_predict` and may need retraining for new data
2. **Contamination Rate**: Adjust `CONTAMINATION_RATE` based on domain knowledge
3. **Feature Order**: Maintain exact feature order for production inference
4. **Null Values**: Current approach removes nulls; consider imputation for production

## 🛠️ Troubleshooting

### Issue: "Columns missing after loading"
**Solution**: Verify CSV files have exact column names as specified in `COLUMN_RENAME_MAP`

### Issue: "Model prediction shape mismatch"
**Solution**: Ensure new data has exactly 13 features in the correct order (check `feature_config.json`)

### Issue: "Different results each run"
**Solution**: Verify `RANDOM_SEED = 42` is set before any random operations

## 📚 References

- [Isolation Forest Paper](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf)
- [Local Outlier Factor](https://en.wikipedia.org/wiki/Local_outlier_factor)
- [scikit-learn Documentation](https://scikit-learn.org/stable/modules/outlier_detection.html)

## 📧 Contact

For questions or issues, please review the generated `outputs/00_SUMMARY_REPORT.txt` file first.

---

**Generated**: 2026-01-24
**Version**: 1.0
**Status**: Production-Ready ✅
