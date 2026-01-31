"""
AIET ML v4 - XGBoost Training Script
NASA-locked features, stable teacher formula, hard validation gates
"""

import os
import sys
import numpy as np
import pandas as pd
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ml_features_v4 import build_features_v4, load_feature_schema
from ml_teacher_v4 import compute_habitability_score_v4
from ml_validation_v4 import validate_model_predictions

try:
    import xgboost as xgb
    print("XGBoost version:", xgb.__version__)
except ImportError:
    print("ERROR: XGBoost not installed. Install with: pip install xgboost")
    sys.exit(1)


# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = os.path.dirname(__file__)
PLANETS_FILE = os.path.join(BASE_DIR, "exoplanets.csv")
STARS_FILE = os.path.join(BASE_DIR, "stellar_hosts.csv")

OUTPUT_DIR = os.path.join(os.path.dirname(BASE_DIR), "ml_calibration")
MODEL_PATH = os.path.join(OUTPUT_DIR, "hab_xgb_v4.json")
SCORED_CSV_PATH = os.path.join(BASE_DIR, "MLP_v4_scored.csv")

# XGBoost hyperparameters
XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "reg:squarederror",
    "random_state": 42,
    "n_jobs": -1
}


# =============================================================================
# PHASE 1 — LOAD AND PREPARE DATA
# =============================================================================

print("\n" + "="*80)
print("AIET ML v4 TRAINING - XGBoost with NASA-Locked Features")
print("="*80)

print("\n[1/7] Loading NASA data...")
planets_df = pd.read_csv(PLANETS_FILE, comment="#", low_memory=False)
stars_df = pd.read_csv(STARS_FILE, comment="#", low_memory=False)

print(f"  Loaded {len(planets_df):,} planets")
print(f"  Loaded {len(stars_df):,} star systems")

# Filter to default parameter set only
if "default_flag" in planets_df.columns:
    planets_df = planets_df[planets_df["default_flag"] == 1]
    print(f"  Filtered to {len(planets_df):,} planets with default_flag=1")

# Merge planets with their host stars
df = pd.merge(planets_df, stars_df, on="hostname", how="left", suffixes=("", "_star"))
print(f"  Merged to {len(df):,} planet-star pairs")


# =============================================================================
# PHASE 2 — BUILD FEATURES AND LABELS
# =============================================================================

print("\n[2/7] Building features and computing teacher labels...")

features_list = []
labels_list = []
planet_names = []
imputation_stats = {
    "total": 0,
    "fields": {}
}

valid_count = 0
invalid_count = 0

for idx, row in df.iterrows():
    try:
        # Build features
        planet_dict = row.to_dict()
        features, meta = build_features_v4(planet_dict, return_meta=True)
        
        # Check for NaNs
        if np.any(np.isnan(features)):
            invalid_count += 1
            continue
        
        # Compute teacher label
        teacher_result = compute_habitability_score_v4(features)
        label = teacher_result["score"]  # 0-1 scale
        
        features_list.append(features)
        labels_list.append(label)
        planet_names.append(row.get("pl_name", f"Planet_{idx}"))
        
        # Track imputation
        for field in meta["imputed_fields"]:
            imputation_stats["fields"][field] = imputation_stats["fields"].get(field, 0) + 1
        imputation_stats["total"] += len(meta["imputed_fields"])
        
        valid_count += 1
        
        if valid_count % 5000 == 0:
            print(f"  Processed {valid_count:,} planets...")
    
    except Exception as e:
        invalid_count += 1
        if invalid_count <= 3:  # Only print first few errors
            print(f"  Warning: Failed to process planet at index {idx}: {e}")

X = np.array(features_list, dtype=np.float32)
y = np.array(labels_list, dtype=np.float32)

print(f"\n  Valid planets: {valid_count:,}")
print(f"  Invalid/skipped: {invalid_count:,}")
print(f"  Feature matrix shape: {X.shape}")
print(f"  Label vector shape: {y.shape}")
print(f"\n  Imputation stats:")
print(f"    Total imputed values: {imputation_stats['total']:,}")
print(f"    Most imputed fields:")
for field, count in sorted(imputation_stats['fields'].items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"      {field}: {count:,} ({100*count/valid_count:.1f}%)")


# =============================================================================
# PHASE 3 — TRAIN/TEST SPLIT
# =============================================================================

print("\n[3/7] Splitting data...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"  Training set: {len(X_train):,} planets")
print(f"  Test set: {len(X_test):,} planets")
print(f"  Label range: [{y.min():.4f}, {y.max():.4f}]")


# =============================================================================
# PHASE 4 — TRAIN XGBOOST MODEL
# =============================================================================

print("\n[4/7] Training XGBoost model...")
print(f"  Parameters: {XGB_PARAMS}")

model = xgb.XGBRegressor(**XGB_PARAMS, early_stopping_rounds=20)
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=False
)

print(f"  Best iteration: {model.best_iteration}")
print(f"  Training complete")


# =============================================================================
# PHASE 5 — EVALUATE MODEL
# =============================================================================

print("\n[5/7] Evaluating model...")

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"\n  Training Metrics:")
print(f"    MSE: {train_mse:.6f}")
print(f"    MAE: {train_mae:.6f}")
print(f"    R²:  {train_r2:.6f}")

print(f"\n  Test Metrics:")
print(f"    MSE: {test_mse:.6f}")
print(f"    MAE: {test_mae:.6f}")
print(f"    R²:  {test_r2:.6f}")

# Feature importance
feature_schema = load_feature_schema()
feature_names = [f["name"] for f in feature_schema["features"]]
importances = model.feature_importances_

print(f"\n  Feature Importances (top 5):")
importance_pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
for name, importance in importance_pairs[:5]:
    print(f"    {name:15s}: {importance:.4f}")


# =============================================================================
# PHASE 6 — SOLAR SYSTEM VALIDATION (HARD GATE)
# =============================================================================

print("\n[6/7] Running Solar System validation gates...")

validation_passed = validate_model_predictions(
    model=model,
    feature_builder_fn=build_features_v4,
    export_dir=os.path.join(os.path.dirname(BASE_DIR), "exports")
)

if not validation_passed:
    print("\n" + "="*80)
    print("[FAIL] SOLAR SYSTEM VALIDATION FAILED")
    print("Model does not produce correct Earth-centric rankings.")
    print("Training artifacts will NOT be saved.")
    print("="*80)
    sys.exit(1)


# =============================================================================
# PHASE 7 — EXPORT MODEL ARTIFACTS
# =============================================================================

print("\n[7/7] Exporting model artifacts...")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Save XGBoost model (use get_booster() for XGBoost 3.x compatibility)
model.get_booster().save_model(MODEL_PATH)
print(f"  Model saved to: {MODEL_PATH}")

# Copy feature schema to output directory (for runtime loading)
import shutil
schema_src = os.path.join(os.path.dirname(BASE_DIR), "ml_calibration", "features_v4.json")
if os.path.exists(schema_src):
    print(f"  Feature schema already in place: {schema_src}")
else:
    print(f"  WARNING: Feature schema not found at {schema_src}")

# Export scored predictions for the full dataset
print(f"\n  Generating predictions for all planets...")
y_pred_all = model.predict(X)

# Add predictions to dataframe
df_scored = df.iloc[:len(y_pred_all)].copy()
df_scored["Habitability_Teacher_v4"] = y[:len(y_pred_all)]
df_scored["Habitability_ML_v4"] = y_pred_all

# Save scored CSV
df_scored.to_csv(SCORED_CSV_PATH, index=False)
print(f"  Scored predictions saved to: {SCORED_CSV_PATH}")

# Export training summary
summary = {
    "model_version": "v4",
    "training_date": datetime.now().isoformat(),
    "n_training_samples": len(X_train),
    "n_test_samples": len(X_test),
    "xgb_params": XGB_PARAMS,
    "best_iteration": int(model.best_iteration),
    "metrics": {
        "train": {
            "mse": float(train_mse),
            "mae": float(train_mae),
            "r2": float(train_r2)
        },
        "test": {
            "mse": float(test_mse),
            "mae": float(test_mae),
            "r2": float(test_r2)
        }
    },
    "feature_importances": {name: float(imp) for name, imp in importance_pairs},
    "imputation_stats": imputation_stats,
    "solar_system_validation": "PASSED"
}

summary_path = os.path.join(OUTPUT_DIR, "training_summary_v4.json")
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"  Training summary saved to: {summary_path}")


# =============================================================================
# TRAINING COMPLETE
# =============================================================================

print("\n" + "="*80)
print("[SUCCESS] AIET ML v4 TRAINING COMPLETE")
print("="*80)
print(f"\nModel artifacts:")
print(f"  - Model: {MODEL_PATH}")
print(f"  - Schema: {schema_src}")
print(f"  - Summary: {summary_path}")
print(f"  - Scored data: {SCORED_CSV_PATH}")
print(f"\nTest R²: {test_r2:.4f}")
print(f"Solar System validation: PASSED")
print("="*80 + "\n")
