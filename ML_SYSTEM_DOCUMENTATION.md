# AIET ML Habitability System - Complete Documentation

**Last Updated:** 2026-01-26  
**Version:** v4 (production), v3 (legacy with fixes)

---

## Table of Contents

1. [System Overview](#system-overview)
2. [ML v3 (Legacy with Critical Fixes)](#ml-v3-legacy-with-critical-fixes)
3. [ML v4 (Production-Ready)](#ml-v4-production-ready)
4. [Simulation Integration](#simulation-integration)
5. [Usage Examples](#usage-examples)
6. [Troubleshooting](#troubleshooting)

---

## System Overview

AIET uses machine learning to estimate the relative habitability of exoplanets based on physical parameters. The system has evolved through multiple versions:

- **v1-v2**: Initial PyTorch MLP models (deprecated)
- **v3**: PyTorch MLP with derived physics features (legacy, patched)
- **v4**: XGBoost with NASA-locked features (**current production**)

### Key Principles

1. **Output is NOT a probability of life** - It's a relative habitability score (Earth = 100%)
2. **NASA-first approach** - Use only parameters from NASA Exoplanet Archive or physics-derived values
3. **Reproducibility** - Same inputs always produce same outputs
4. **Validation gates** - Model must rank Solar System planets correctly

---

## ML v3 (Legacy with Critical Fixes)

### Architecture

- **Model Type**: PyTorch MLP (Multi-Layer Perceptron)
- **Input Features**: 13 (9 base + 4 derived)
- **Hidden Layers**: 64 → 32 neurons with ReLU activation
- **Output**: Single value (0-100% relative habitability)
- **Training**: Teacher-student architecture with physics-based labels

### Critical Bugs Fixed (2026-01-26)

#### Bug A: Tidal Locking Score Reversed ✅ FIXED
```python
# BEFORE (WRONG - longer periods appeared more locked)
tidal_lock_score = 1.0 / (1.0 + np.exp(-(pl_orbper - 25.0) / 5.0))

# AFTER (CORRECT - shorter periods are more locked)
tidal_lock_score = 1.0 / (1.0 + np.exp((pl_orbper - 25.0) / 5.0))
```

#### Bug B: Temperature Key Mismatch ✅ FIXED
```python
# BEFORE (WRONG - looked for non-existent key)
if "pl_temperature" in features:
    T_surf = features["pl_temperature"]

# AFTER (CORRECT - checks multiple sources)
if "temperature" in features:  # Simulation key
    T_surf = features["temperature"]
elif "pl_eqt" in features:  # NASA key
    T_surf = features["pl_eqt"]
else:
    # Fallback estimation
```

### v3 Features (13 total)

| Index | Feature | Units | Source |
|-------|---------|-------|--------|
| 0 | `pl_rade` | R⊕ | NASA or simulation |
| 1 | `pl_masse` | M⊕ | NASA or simulation |
| 2 | `pl_orbper` | days | NASA or simulation |
| 3 | `pl_orbeccen` | 0-1 | NASA or simulation |
| 4 | `pl_insol` | S⊕ | NASA or computed |
| 5 | `pl_dens` | g/cm³ | Computed from M, R |
| 6 | `v_esc_kms` | km/s | Computed from M, R |
| 7 | `tidal_lock_score` | 0-1 | Computed from period |
| 8 | **`T_surf`** | K | **Heuristic (inconsistent!)** |
| 9 | `st_teff` | K | NASA or simulation |
| 10 | `st_mass` | M☉ | NASA or simulation |
| 11 | `st_rad` | R☉ | NASA or simulation |
| 12 | `st_lum` | L☉ | NASA or computed |

**⚠️ v3 Limitations:**
- `T_surf` creates training/inference mismatch (greenhouse heuristics vary)
- Redundant features (`v_esc_kms` ≈ f(mass, radius))
- Earth normalization baked into `predict()` (not clean separation)

### v3 Files

- `src/ml_habitability.py` - Runtime calculator (with fixes)
- `ml_calibration/hab_net_v3.pth` - PyTorch model weights
- `ml_calibration/scaler_v3.joblib` - StandardScaler parameters

---

## ML v4 (Production-Ready)

### Architecture

- **Model Type**: XGBoost Regressor
- **Input Features**: 12 (all NASA-sourced or physics-derived)
- **Parameters**: 200 trees, max_depth=6, learning_rate=0.05
- **Output**: Raw score 0-1 (display as Earth=100)
- **Training**: 75,921 exoplanets from NASA Archive

### Training Performance

```
Test R²:  0.9988
Test MAE: 0.0026
Test MSE: 0.000022
```

### Solar System Validation Results

| Planet | Raw Score | Earth-Normalized | Ranking |
|--------|-----------|------------------|---------|
| **Earth** | 0.9096 | 100.0% | **#1 ✅** |
| Venus | 0.7008 | 77.0% | #2 |
| Mars | 0.6615 | 72.7% | #3 |
| Mercury | 0.5633 | 61.9% | #4 |
| Jupiter | 0.3344 | 36.8% | #5 |

**All validation gates passed:**
- ✅ Earth is top-1 among rocky planets
- ✅ Venus < 0.85 × Earth (hot planet penalty)
- ✅ Jupiter < 0.5 × Earth (gas giant penalty)

### v4 Features (12 total, NASA-locked)

| Index | Feature | Units | Imputation Rule |
|-------|---------|-------|-----------------|
| 0 | `pl_rade` | R⊕ | Median (6.0) |
| 1 | `pl_masse` | M⊕ | M-R relation if rocky, else median |
| 2 | `pl_orbper` | days | Median (10.0) |
| 3 | `pl_orbsmax` | AU | Kepler's 3rd law from period |
| 4 | `pl_orbeccen` | 0-1 | Fill with 0.0 (circular) |
| 5 | `pl_insol` | S⊕ | Compute from L/a² |
| 6 | `pl_eqt` | K | 278.5 × flux^0.25 |
| 7 | `pl_dens` | g/cm³ | Compute from M, R |
| 8 | `st_teff` | K | Spectral type median |
| 9 | `st_mass` | M☉ | Spectral type median |
| 10 | `st_rad` | R☉ | Spectral type median |
| 11 | `st_lum` | L☉ | Stefan-Boltzmann: R² × (T/5778)⁴ |

**✅ v4 Improvements:**
- **No `T_surf`** - Uses `pl_eqt` (equilibrium temp) instead
- **No redundant features** - Removed escape velocity, tidal locking score
- **Clean Earth normalization** - Separate raw (0-1) and display (0-100) modes
- **Full imputation logging** - Tracks every computed value
- **XGBoost stability** - More robust than neural networks for tabular data

### Feature Importance (Top 5)

1. **`pl_masse`** (41.1%) - Planet mass
2. **`pl_rade`** (37.7%) - Planet radius  
3. **`pl_dens`** (6.0%) - Bulk density
4. **`pl_insol`** (5.8%) - Stellar flux received
5. **`pl_eqt`** (3.6%) - Equilibrium temperature

### v4 Teacher Formula

The training labels are generated using a physics-based formula (0-1 scale):

```python
Score = 30% × flux_penalty(pl_insol, optimal=1.0, σ=0.5)
      + 20% × radius_penalty(pl_rade, optimal=1.0, σ=0.4)
      + 15% × density_penalty(pl_dens, optimal=5.5, σ=2.5)
      + 10% × mass_penalty(pl_masse, optimal=1.0, σ=2.0)
      + 10% × eccentricity_penalty(low is better)
      + 10% × stellar_type_penalty(G-type optimal)
      +  5% × period_penalty(long periods better)
```

**Key principles:**
- Primary thermal indicator: **flux** (not temperature)
- Avoids double-counting (flux OR temperature, not both at full weight)
- Penalizes gas giants aggressively (radius > 1.6 R⊕)
- Gaussian penalties around Earth-optimal values

### v4 Files

- `src/ml_habitability_v4.py` - Runtime calculator
- `src/ml_features_v4.py` - Canonical feature builder
- `src/ml_teacher_v4.py` - Teacher formula
- `src/ml_validation_v4.py` - Solar System validation gates
- `src/ml_sanity_check_v4.py` - Diagnostic tool
- `ml_calibration/hab_xgb_v4.json` - XGBoost model
- `ml_calibration/features_v4.json` - Feature schema contract
- `ml_calibration/training_summary_v4.json` - Training metadata
- `AIET_ML_inputs/train_ml_v4_xgb.py` - Training script

---

## Simulation Integration

### Key Mapping Layer

AIET simulation uses different parameter names than NASA. The integration layer (`ml_integration_v4.py`) handles this mapping:

#### AIET Simulation → NASA Schema

**Planet Parameters:**
```python
pl_rade       ← planet_body["radius"]              # Earth radii
pl_masse      ← planet_body["mass"]                # Earth masses
pl_orbper     ← planet_body["orbital_period"]      # days
pl_orbsmax    ← planet_body["semiMajorAxis"]       # AU
pl_orbeccen   ← planet_body["eccentricity"]        # unitless
pl_insol      ← planet_body["stellarFlux"]         # Earth flux
pl_eqt        ← planet_body["equilibrium_temperature"]  # Kelvin
pl_dens       ← planet_body["density"]             # g/cm³
```

**Star Parameters:**
```python
st_teff       ← star_body["temperature"]           # Kelvin
st_mass       ← star_body["mass"]                  # Solar masses
st_rad        ← star_body["radius"]                # Solar radii
st_lum        ← star_body["luminosity"]            # Solar luminosities
```

### Critical: Equilibrium vs. Surface Temperature

**⚠️ NEVER use `planet_body["temperature"]` directly for ML!**

```python
# WRONG - includes greenhouse effect (inconsistent)
pl_eqt = planet_body["temperature"]  # ❌

# CORRECT - use equilibrium temperature
pl_eqt = planet_body["equilibrium_temperature"]  # ✅

# FALLBACK - estimate if eq temp not available
if "equilibrium_temperature" not in planet_body:
    T_surf = planet_body["temperature"]
    greenhouse_offset = planet_body.get("greenhouse_offset", 33.0)
    pl_eqt = T_surf - greenhouse_offset  # Rough estimate
```

### Usage in AIET

```python
from ml_integration_v4 import sim_to_ml_features_v4, predict_with_simulation_body_v4
from ml_habitability_v4 import MLHabitabilityCalculatorV4

# Initialize ML calculator
ml_calc = MLHabitabilityCalculatorV4()

# Get simulation bodies
planet = selected_planet_body  # From AIET simulation
star = parent_star_body        # From AIET simulation

# Option 1: Manual mapping
features, diagnostics = sim_to_ml_features_v4(planet, star)
if features:
    score = ml_calc.predict(features)  # Returns 0-100
    print(f"Habitability: {score:.1f}%")
else:
    print(f"Cannot compute: {diagnostics['missing_critical']}")

# Option 2: Wrapper (recommended)
score, diagnostics = predict_with_simulation_body_v4(
    ml_calc, planet, star, return_diagnostics=True
)
if score is not None:
    print(f"Habitability: {score:.1f}%")
    print(f"Imputed fields: {diagnostics['missing_optional']}")
else:
    print(f"Failed: {diagnostics['warnings']}")
```

### Earth Normalization

The v4 system ensures Earth scores exactly 100.0:

```python
# Compute Earth reference at initialization
earth_features = get_earth_features_from_preset()  # Uses AIET preset
earth_raw_score = model.predict(earth_features)  # Raw 0-1 score

# Normalize predictions
normalized_score = (raw_score / earth_raw_score) * 100.0

# Special case: Force Earth to exactly 100.0
if preset_type == "Earth" and 99.0 < normalized_score < 101.0:
    normalized_score = 100.0
```

### Handling Missing Data

The feature builder automatically imputes missing values:

```python
features, meta = build_features_v4(planet_data)

# Check what was imputed
if meta["imputed_fields"]:
    print(f"Imputed: {meta['imputed_fields']}")
    # Example: ['pl_orbeccen', 'st_lum (Stefan-Boltzmann)', 'pl_eqt (from flux)']

# Check for warnings
if meta["warnings"]:
    print(f"Warnings: {meta['warnings']}")
```

**UI Handling:**
- If `predict()` returns `None`: Show "—" (not available)
- If score is very low (<5%): Show actual value (not a bug, planet is hostile)
- Never show "0.0%" unless genuinely computed as zero

---

## Usage Examples

### Example 1: Predict for Earth

```python
from ml_habitability_v4 import MLHabitabilityCalculatorV4

calc = MLHabitabilityCalculatorV4()

earth = {
    "pl_rade": 1.0, "pl_masse": 1.0, "pl_orbper": 365.25,
    "pl_orbsmax": 1.0, "pl_orbeccen": 0.0167, "pl_insol": 1.0,
    "pl_eqt": 255.0, "pl_dens": 5.51,
    "st_teff": 5778.0, "st_mass": 1.0, "st_rad": 1.0, "st_lum": 1.0
}

score = calc.predict(earth)
print(f"Earth: {score:.1f}%")  # Output: Earth: 100.0%
```

### Example 2: Explain Prediction

```python
explanation = calc.explain_prediction(earth)

print(f"Score: {explanation['earth_normalized_score']:.1f}%")
print("\nTop features:")
for name, importance in sorted(
    explanation['feature_importances'].items(),
    key=lambda x: x[1],
    reverse=True
)[:5]:
    value = explanation['feature_values'][name]
    print(f"  {name}: {importance:.3f} (value: {value:.3f})")

print(f"\nImputed: {explanation['imputed_fields']}")
```

### Example 3: Batch Prediction

```python
planets = [
    {"pl_rade": 1.2, "pl_masse": 1.5, ...},
    {"pl_rade": 0.8, "pl_masse": 0.6, ...},
    # ... more planets
]

scores = calc.predict_batch(planets)
for i, score in enumerate(scores):
    print(f"Planet {i+1}: {score:.1f}%")
```

### Example 4: Run Sanity Check

```python
from ml_sanity_check_v4 import run_ml_sanity_check_v4

report = run_ml_sanity_check_v4(
    ml_calculator=calc,
    planet_features=earth,  # Optional, defaults to Earth
    export_dir="exports"
)

print(f"Overall status: {report['overall_status']}")
print(f"Recommended fix: {report['recommended_fix']}")
```

---

## Troubleshooting

### Problem: All planets show 0% habitability

**Cause:** Key mapping failure - simulation keys not matching NASA schema

**Fix:**
```python
# Check diagnostics
features, diagnostics = sim_to_ml_features_v4(planet, star)
print(diagnostics["missing_critical"])
print(diagnostics["warnings"])
```

### Problem: Earth doesn't score 100%

**Cause 1:** Using wrong temperature (surface instead of equilibrium)
```python
# WRONG
features = {"pl_eqt": planet["temperature"]}  # Includes greenhouse

# CORRECT
features = {"pl_eqt": planet["equilibrium_temperature"]}
```

**Cause 2:** Earth calibration not using simulation preset
```python
# Ensure Earth reference uses AIET preset values
earth_features = get_earth_features_from_preset()
```

### Problem: Venus scores higher than Earth

**Cause:** Model not loaded correctly or training failed validation

**Fix:**
1. Check model file exists: `ml_calibration/hab_xgb_v4.json`
2. Re-run training: `python AIET_ML_inputs/train_ml_v4_xgb.py`
3. Verify Solar System validation passed during training

### Problem: Feature schema mismatch

**Symptom:** Runtime error about wrong number of features

**Fix:**
```python
# Check feature schema version
schema = load_feature_schema()
print(f"Expected features: {len(schema['features'])}")
print(f"Feature names: {[f['name'] for f in schema['features']]}")
```

### Problem: High imputation warnings

**Symptom:** Many fields being imputed/estimated

**Investigation:**
```python
features, meta = build_features_v4(planet_data)
print(f"Imputed {len(meta['imputed_fields'])}/{12} fields:")
for field in meta['imputed_fields']:
    print(f"  - {field}")
```

**Acceptable imputation:**
- `pl_orbeccen` (fill 0.0) - normal
- `st_lum` (Stefan-Boltzmann) - normal
- `pl_insol` (from L/a²) - normal
- `pl_eqt` (from flux) - normal

**Concerning imputation:**
- `pl_rade`, `pl_masse`, `st_teff`, `st_mass` - **should be provided!**

---

## Model Comparison

| Aspect | v3 (Legacy) | v4 (Production) |
|--------|-------------|-----------------|
| **Model Type** | PyTorch MLP | XGBoost |
| **Features** | 13 (4 synthetic) | 12 (all NASA) |
| **Training Data** | ~5,000 planets | 75,921 planets |
| **Test R²** | ~0.95 | 0.9988 |
| **Earth Score** | ~91% (raw) | 100.0% (normalized) |
| **Validation Gates** | None | Hard gates (fail = no export) |
| **Temperature** | T_surf (heuristic) | pl_eqt (NASA) |
| **Redundancy** | Yes (escape vel, tidal) | No |
| **Imputation Logging** | Minimal | Full audit trail |
| **Production Ready** | No (bugs fixed) | **Yes** ✅ |

---

## Summary

- **Use ML v4 for all new work** - Production-ready, validated, stable
- **v3 is legacy** - Kept for backward compatibility, critical bugs fixed
- **Always map simulation keys** - Use `ml_integration_v4.py`
- **Never use surface temperature directly** - Use equilibrium temperature
- **Check diagnostics** - Full logging of imputation and warnings
- **Earth = 100% exactly** - Calibrated reference point

**Training a new model:** Run `python AIET_ML_inputs/train_ml_v4_xgb.py`  
**Testing:** Run `python src/ml_validation_v4.py` or `python src/ml_sanity_check_v4.py`  
**Support:** Check `exports/` for diagnostic JSON reports

---

**Document Version:** 1.0  
**Last Verified:** 2026-01-26  
**Model Version:** v4.0
