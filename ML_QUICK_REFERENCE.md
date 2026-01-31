# AIET ML v4 - Quick Reference Guide

**Last Updated:** 2026-01-26

---

## 🚀 Quick Start

### Initialize ML Calculator

```python
from ml_habitability_v4 import MLHabitabilityCalculatorV4

calc = MLHabitabilityCalculatorV4()
```

### Predict from NASA Data

```python
features = {
    "pl_rade": 1.0,        # Earth radii
    "pl_masse": 1.0,       # Earth masses
    "pl_orbper": 365.25,   # days
    "pl_orbsmax": 1.0,     # AU
    "pl_orbeccen": 0.0167, # 0-1
    "pl_insol": 1.0,       # Earth flux
    "pl_eqt": 255.0,       # Kelvin (equilibrium temp)
    "pl_dens": 5.51,       # g/cm³
    "st_teff": 5778.0,     # Kelvin
    "st_mass": 1.0,        # Solar masses
    "st_rad": 1.0,         # Solar radii
    "st_lum": 1.0          # Solar luminosities
}

score = calc.predict(features)  # Returns 0-100 (Earth = 100)
```

### Predict from AIET Simulation

```python
from ml_integration_v4 import predict_with_simulation_body_v4

# Get bodies from simulation
planet = selected_planet_body
star = parent_star_body

# Predict
score, diagnostics = predict_with_simulation_body_v4(
    calc, planet, star, return_diagnostics=True
)

if score is not None:
    print(f"Habitability: {score:.1f}%")
else:
    print(f"Failed: {diagnostics['warnings']}")
```

---

## 📊 Feature Schema (12 features)

| # | Feature | Units | What it is |
|---|---------|-------|------------|
| 0 | `pl_rade` | R⊕ | Planet radius |
| 1 | `pl_masse` | M⊕ | Planet mass |
| 2 | `pl_orbper` | days | Orbital period |
| 3 | `pl_orbsmax` | AU | Semi-major axis |
| 4 | `pl_orbeccen` | 0-1 | Eccentricity |
| 5 | `pl_insol` | S⊕ | Stellar flux |
| 6 | `pl_eqt` | K | Equilibrium temp |
| 7 | `pl_dens` | g/cm³ | Bulk density |
| 8 | `st_teff` | K | Star temperature |
| 9 | `st_mass` | M☉ | Star mass |
| 10 | `st_rad` | R☉ | Star radius |
| 11 | `st_lum` | L☉ | Star luminosity |

---

## 🔑 Key Mapping (AIET → NASA)

```python
# AIET simulation keys → NASA schema keys
pl_rade       ← planet["radius"]
pl_masse      ← planet["mass"]
pl_orbper     ← planet["orbital_period"] or planet["orbper"]
pl_orbsmax    ← planet["semiMajorAxis"]
pl_orbeccen   ← planet["eccentricity"]
pl_insol      ← planet["stellarFlux"]
pl_eqt        ← planet["equilibrium_temperature"]  # NOT "temperature"!
pl_dens       ← planet["density"]

st_teff       ← star["temperature"]
st_mass       ← star["mass"]
st_rad        ← star["radius"]
st_lum        ← star["luminosity"]
```

---

## ⚠️ Common Mistakes

### ❌ DON'T: Use surface temperature
```python
features["pl_eqt"] = planet["temperature"]  # WRONG - includes greenhouse
```

### ✅ DO: Use equilibrium temperature
```python
features["pl_eqt"] = planet["equilibrium_temperature"]  # CORRECT
```

---

### ❌ DON'T: Hardcode Earth features separately
```python
earth = {"pl_rade": 1.0, "pl_masse": 1.0, ...}  # Inconsistent with preset
```

### ✅ DO: Use preset values
```python
from ml_integration_v4 import get_earth_features_from_preset
earth = get_earth_features_from_preset()  # Uses AIET preset
```

---

### ❌ DON'T: Return 0 on error
```python
try:
    score = calc.predict(features)
except:
    score = 0.0  # WRONG - hides errors
```

### ✅ DO: Return None on error
```python
try:
    score = calc.predict(features)
except Exception as e:
    print(f"ML prediction failed: {e}")
    score = None  # Show "—" in UI
```

---

## 🛠️ Diagnostic Tools

### Run Sanity Check
```bash
python src/ml_sanity_check_v4.py
```
Exports: `exports/sanity_check_v4_*.json`

### Run Solar System Validation
```bash
python src/ml_validation_v4.py
```
Checks: Earth top-1, Venus < 0.85×Earth, Jupiter < 0.5×Earth

### Test Feature Builder
```bash
python src/ml_features_v4.py
```
Tests: Earth reference vector, feature validation

### Test Integration Layer
```bash
python src/ml_integration_v4.py
```
Tests: Simulation key mapping

---

## 🔍 Debugging

### Check Feature Mapping
```python
features, diagnostics = sim_to_ml_features_v4(planet, star)

print("Success:", diagnostics["success"])
print("Missing critical:", diagnostics["missing_critical"])
print("Missing optional:", diagnostics["missing_optional"])
print("Warnings:", diagnostics["warnings"])
print("Mapped keys:", diagnostics["mapped_keys"])
```

### Get Detailed Explanation
```python
explanation = calc.explain_prediction(features)

print("Score:", explanation["earth_normalized_score"])
print("Raw score:", explanation["raw_score"])
print("Top features:", list(explanation["feature_importances"].items())[:5])
print("Imputed:", explanation["imputed_fields"])
```

### Check Imputation
```python
from ml_features_v4 import build_features_v4

features, meta = build_features_v4(planet_dict)

print(f"Imputed {len(meta['imputed_fields'])}/12 fields:")
for field in meta["imputed_fields"]:
    print(f"  - {field}")
```

---

## 📈 Expected Scores (Solar System)

| Planet | Score | Notes |
|--------|-------|-------|
| Earth | 100.0% | Reference (by design) |
| Venus | 77.0% | Hot (penalized) |
| Mars | 72.7% | Small, thin atmosphere |
| Mercury | 61.9% | Too hot, no atmosphere |
| Jupiter | 36.8% | Gas giant (penalized) |

---

## 🎯 Validation Gates

Model must pass ALL gates during training:

1. **Earth top-1** among Mercury/Venus/Earth/Mars
2. **Venus < 0.85 × Earth** (hot planet penalty)
3. **Jupiter < 0.5 × Earth** (gas giant penalty)

If any gate fails → Training exits with error (no model export)

---

## 📁 File Structure

```
AIET/
├── ml_calibration/
│   ├── hab_xgb_v4.json           # XGBoost model
│   ├── features_v4.json          # Feature schema
│   └── training_summary_v4.json  # Training metadata
│
├── src/
│   ├── ml_habitability_v4.py     # Runtime calculator
│   ├── ml_features_v4.py         # Feature builder
│   ├── ml_teacher_v4.py          # Teacher formula
│   ├── ml_validation_v4.py       # Solar System gates
│   ├── ml_sanity_check_v4.py     # Diagnostics
│   └── ml_integration_v4.py      # Simulation mapping
│
├── AIET_ML_inputs/
│   ├── train_ml_v4_xgb.py        # Training script
│   ├── exoplanets.csv            # NASA planet data
│   ├── stellar_hosts.csv         # NASA star data
│   └── MLP_v4_scored.csv         # Scored predictions
│
└── exports/
    ├── validation_v4_*.json      # Validation reports
    └── sanity_check_v4_*.json    # Diagnostic reports
```

---

## 🔄 Retrain Model

```bash
cd AIET_ML_inputs
python train_ml_v4_xgb.py
```

**Outputs:**
- `ml_calibration/hab_xgb_v4.json` (model)
- `ml_calibration/training_summary_v4.json` (metrics)
- `MLP_v4_scored.csv` (all predictions)
- `exports/validation_v4_*.json` (gates check)

**Expected output:**
```
Test R²: 0.9988
Solar System validation: PASSED
```

---

## 💡 Tips

1. **Always check diagnostics** - Don't assume prediction succeeded
2. **Log imputed fields** - Know what was computed vs. provided
3. **Use Earth preset for calibration** - Don't hardcode values
4. **Handle None gracefully** - Show "—" not "0%" in UI
5. **Validate before display** - Check score is in [0, 100] range
6. **Test with Solar System** - Mercury/Venus/Earth/Mars should rank correctly

---

## 📞 Support

**Check diagnostics:**
- `exports/sanity_check_v4_*.json` - Full diagnostic report
- `exports/validation_v4_*.json` - Solar System gate results

**Read documentation:**
- `ML_SYSTEM_DOCUMENTATION.md` - Complete technical guide
- `ML_QUICK_REFERENCE.md` - This file

**Common issues:**
- Earth not 100% → Check equilibrium_temperature vs temperature
- All zeros → Check key mapping with diagnostics
- Model not loading → Verify `hab_xgb_v4.json` exists

---

**Version:** v4.0  
**Model Training Date:** 2026-01-26  
**Test R²:** 0.9988
