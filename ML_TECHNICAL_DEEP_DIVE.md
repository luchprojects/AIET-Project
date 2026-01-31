# AIET ML System - Technical Deep Dive for AI Developers

**Target Audience:** NASA Lead AI Developer / ML Research Scientist  
**Last Updated:** January 29, 2026  
**Model Version:** XGBoost v4.1  
**Author:** AIET Development Team

---

## Executive Summary

AIET (Artificial Intelligence Exoplanet Toolkit) implements a machine learning system for predicting planetary habitability scores using a **gradient-boosted decision tree ensemble (XGBoost)** trained on real NASA Exoplanet Archive data. The model takes 12 physical/orbital features as input and outputs a continuous habitability score normalized such that **Earth = 100%**.

---

## Current Planet Habitability Scores (Solar System)

Based on latest runtime output from XGBoost v4.1:

| Planet   | Habitability Score | Surface Class | Notes |
|----------|-------------------|---------------|-------|
| **Earth**    | **100.00%** | Rocky | Reference baseline (forced normalization) |
| Mars     | 72.93% | Rocky | High score due to habitable zone proximity |
| Venus    | 57.51% | Rocky | Moderate despite extreme conditions |
| Mercury  | 2.90%  | Rocky | Too close to star, extreme temperatures |
| Jupiter  | 0.00%  | Gas Giant | No solid surface |
| Saturn   | 0.00%  | Gas Giant | No solid surface |
| Uranus   | 0.00%  | Gas Giant | No solid surface |
| Neptune  | 0.00%  | Gas Giant | No solid surface |

**Key Observation:** Mars scores surprisingly high (72.93%) because the model considers it within the habitable zone's outer boundary and detects favorable orbital/stellar flux parameters. Venus scores moderate (57.51%) despite surface hostility because stellar flux and radius fall within training distribution peaks.

---

## ML Architecture Overview

### Model Type: XGBoost (Extreme Gradient Boosting)

**Why XGBoost?**
- Handles **non-linear relationships** between planetary features
- Robust to **missing data** (critical for incomplete exoplanet observations)
- Provides **feature importance rankings** for interpretability
- Fast inference (milliseconds per prediction)
- Trained on **tabular NASA data** (not imagery/time-series)

### Model Specifications

```python
Model: XGBoost Regressor
Input: 12-dimensional feature vector
Output: Single continuous score ∈ [0, 1] (raw) → [0, 100] (normalized)
Training Data: NASA Exoplanet Archive (stellar_hosts.csv + exoplanets.csv)
Training Samples: ~5,400 confirmed exoplanets
Earth Raw Score: 0.9424 (94.24% before normalization)
Normalization Factor: 100 / 0.9424 ≈ 1.0611
```

**Architecture:**
```
Input Features (12D)
    ↓
XGBoost Ensemble (100+ decision trees)
    ↓
Raw Score (0-1)
    ↓
Earth Normalization (score / 0.9424 × 100)
    ↓
Display Score (0-100%)
```

---

## Feature Engineering (12 Features)

All features map to **NASA Exoplanet Archive naming conventions** (pl_* for planet, st_* for star).

### Planetary Features (7)

| Feature | Unit | Description | Earth Value | Valid Range |
|---------|------|-------------|-------------|-------------|
| `pl_rade` | R⊕ | Planet radius (Earth radii) | 1.0 | [0.05, 25] |
| `pl_masse` | M⊕ | Planet mass (Earth masses) | 1.0 | [1e-4, 1e4] |
| `pl_orbper` | days | Orbital period | 365.25 | [0.05, 1e7] |
| `pl_orbsmax` | AU | Semi-major axis (orbital distance) | 1.0 | [0.01, 1e5] |
| `pl_orbeccen` | — | Orbital eccentricity | 0.0167 | [0, 1) |
| `pl_insol` | S⊕ | Incident stellar flux (Earth flux units) | 1.0 | [1e-5, 1e3] |
| `pl_eqt` | K | Equilibrium temperature | 255 | [10, 2000] |

**Critical Note on Units:**  
- `pl_rade` is in **Earth radii (R⊕)**, NOT kilometers  
- `pl_masse` is in **Earth masses (M⊕)**, NOT kilograms  
- `pl_dens` (derived) is in **g/cm³**, NOT kg/m³

### Stellar Features (4)

| Feature | Unit | Description | Sun Value | Valid Range |
|---------|------|-------------|-----------|-------------|
| `st_teff` | K | Stellar effective temperature | 5800 | [2000, 50000] |
| `st_mass` | M☉ | Stellar mass (Solar masses) | 1.0 | [0.08, 100] |
| `st_rad` | R☉ | Stellar radius (Solar radii) | 1.0 | [0.1, 1000] |
| `st_lum` | L☉ | Stellar luminosity (Solar luminosities) | 1.0 | [1e-5, 1e6] |

### Derived Feature (1)

| Feature | Unit | Derivation | Earth Value |
|---------|------|------------|-------------|
| `pl_dens` | g/cm³ | mass / (4/3 × π × radius³) | 5.51 |

**Density Calculation:**
```python
# Convert Earth units to CGS
mass_g = pl_masse * 5.972e27  # Earth mass in grams
radius_cm = pl_rade * 6.371e8  # Earth radius in cm
volume_cm3 = (4/3) * π * radius_cm³
density_g_cm3 = mass_g / volume_cm3
```

---

## Training Methodology

### Teacher Model (Supervised Learning)

The model was trained using a **semi-supervised approach** with synthetic labels:

#### Teacher Scoring Function (Pseudo-Code)

```python
def calculate_teacher_score(planet, star):
    """
    Teacher function used to generate training labels.
    Combines multiple physical constraints into a composite score.
    """
    
    # 1. Habitable Zone Score (40% weight)
    hz_inner = 0.95 * sqrt(st_lum / 1.1)  # Conservative inner edge
    hz_outer = 1.37 * sqrt(st_lum / 0.53)  # Optimistic outer edge
    
    if hz_inner <= pl_orbsmax <= hz_outer:
        hz_score = 1.0  # Perfect HZ placement
    else:
        # Exponential decay outside HZ
        distance_from_hz = min(abs(pl_orbsmax - hz_inner), abs(pl_orbsmax - hz_outer))
        hz_score = exp(-distance_from_hz / 0.5)  # Decay constant = 0.5 AU
    
    # 2. Stellar Flux Score (25% weight)
    # Optimal range: 0.8-1.2 S⊕ (Earth-like insolation)
    if 0.8 <= pl_insol <= 1.2:
        flux_score = 1.0
    elif 0.5 <= pl_insol <= 2.0:
        flux_score = 0.7  # Marginal
    else:
        flux_score = 0.3  # Poor
    
    # 3. Temperature Score (20% weight)
    # Optimal: 250-300 K (liquid water range with greenhouse effect)
    if 250 <= pl_eqt <= 300:
        temp_score = 1.0
    elif 200 <= pl_eqt <= 350:
        temp_score = 0.6
    else:
        # Exponential penalty for extreme temps
        temp_deviation = max(0, abs(pl_eqt - 275) - 75)
        temp_score = exp(-temp_deviation / 100)
    
    # 4. Planet Size Score (10% weight)
    # Optimal: 0.8-1.5 R⊕ (rocky planets with atmosphere retention)
    if 0.8 <= pl_rade <= 1.5:
        size_score = 1.0
    elif 0.5 <= pl_rade <= 2.0:
        size_score = 0.7
    elif pl_rade >= 3.0:
        size_score = 0.0  # Gas giant
    else:
        size_score = 0.4  # Too small (Mars-sized)
    
    # 5. Stellar Type Score (5% weight)
    # G-type stars (5200-6000 K) are ideal
    if 5200 <= st_teff <= 6000:
        star_score = 1.0
    elif 4000 <= st_teff <= 7000:
        star_score = 0.8  # K and F-type stars
    else:
        star_score = 0.5  # M-dwarfs or hot stars
    
    # Weighted composite score
    teacher_score = (
        0.40 * hz_score +
        0.25 * flux_score +
        0.20 * temp_score +
        0.10 * size_score +
        0.05 * star_score
    )
    
    return teacher_score  # Range: [0, 1]
```

#### Teacher Score Breakdown for Solar System

| Planet | HZ (40%) | Flux (25%) | Temp (20%) | Size (10%) | Star (5%) | **Total** |
|--------|----------|------------|------------|------------|-----------|-----------|
| Mercury | 0.1 | 0.3 | 0.0 | 0.7 | 1.0 | **0.29 (29%)** |
| Venus | 0.6 | 0.7 | 0.2 | 1.0 | 1.0 | **0.61 (61%)** |
| **Earth** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | **1.00 (100%)** |
| Mars | 0.9 | 0.7 | 0.6 | 0.7 | 1.0 | **0.79 (79%)** |
| Jupiter | 0.3 | 0.3 | 0.0 | 0.0 | 1.0 | **0.20 (20%)** |

**Why Teacher ≠ Runtime Scores:**
- **Teacher scores** are synthetic labels used during training
- **Runtime scores** are XGBoost predictions that learned non-linear patterns beyond the teacher function
- The model discovered correlations the teacher didn't explicitly encode

---

## Earth Normalization Strategy

### Raw vs. Normalized Scores

```
Earth Raw Score (from XGBoost): 0.9424
Normalization Factor: 100 / 0.9424 = 1.0611
```

**All predictions are normalized:**
```python
def normalize_score(raw_score):
    return (raw_score / 0.9424) * 100
```

**Special Case - Earth Preset Forcing:**
```python
if preset_type == "Earth" and 99.0 < score_normalized < 101.0:
    score_normalized = 100.0  # Force exact 100% for UI consistency
```

This ensures the reference planet always displays **exactly 100.00%** rather than 99.74% due to floating-point precision.

---

## Surface Classification System

### Purpose
Filter out gas/ice giants that lack solid surfaces suitable for liquid water.

### Classification Rules

```python
def classify_surface(pl_rade, pl_dens):
    """
    Classify planet as rocky, giant, or unknown.
    """
    
    # Validation
    if pl_rade < 0.05 or pl_rade > 25.0:
        return "unknown"  # Unit mismatch
    if pl_dens < 0.1 or pl_dens > 20.0:
        return "unknown"  # Unit mismatch
    
    # Giant planets (no solid surface)
    if pl_rade >= 3.0:
        return "giant"  # Large radius indicates H/He envelope
    if pl_rade >= 2.0 and pl_dens <= 2.5:
        return "giant"  # Low-density "puffy" planets
    
    # Rocky planets (solid surface)
    if pl_rade <= 1.8 and pl_dens >= 3.0:
        return "rocky"  # Small + dense = rocky composition
    
    # Ambiguous (transition zone)
    return "unknown"
```

### Solar System Classifications

| Planet | Radius (R⊕) | Density (g/cm³) | Classification | Surface Applicable |
|--------|-------------|-----------------|----------------|-------------------|
| Mercury | 0.38 | 5.43 | **Rocky** | ✓ Yes |
| Venus | 0.95 | 5.24 | **Rocky** | ✓ Yes |
| Earth | 1.00 | 5.51 | **Rocky** | ✓ Yes |
| Mars | 0.53 | 3.93 | **Rocky** | ✓ Yes |
| Jupiter | 11.2 | 1.33 | **Gas Giant** | ✗ No |
| Saturn | 9.45 | 0.69 | **Gas Giant** | ✗ No |
| Uranus | 4.01 | 1.27 | **Gas Giant** | ✗ No |
| Neptune | 3.88 | 1.64 | **Gas Giant** | ✗ No |

**UI Display Modes:**
- `surface_mode="all"`: Show scores for all planets (including giants)
- `surface_mode="rocky_only"`: Show scores only for rocky planets, display "—" for giants

---

## Data Flow Architecture

### Single Source of Truth Principle

All ML predictions flow through **one canonical pipeline** to prevent inconsistencies:

```
Simulation Body Data
    ↓
planet_star_to_features_v4_canonical()  ← SINGLE ADAPTER
    ↓
Feature Validation (hard ranges)
    ↓
XGBoost Prediction (raw score 0-1)
    ↓
Earth Normalization (× 100 / 0.9424)
    ↓
Surface Classification (rocky/giant/unknown)
    ↓
Display Score (0-100% or "—")
```

### Canonical Feature Adapter

**File:** `src/ml_integration_v4.py`

```python
def planet_star_to_features_v4_canonical(planet_body, star_body):
    """
    SINGLE SOURCE OF TRUTH for simulation → ML feature conversion.
    
    Returns:
        (features_dict, meta_dict) or (None, meta_dict)
    
    Meta includes:
        - feature_sources: {feature_name: "direct" | "computed" | "imputed"}
        - validation_errors: List of range violations
        - mapping_errors: List of missing critical fields
        - warnings: Combined error/warning messages
    """
    
    features = {}
    feature_sources = {}
    validation_errors = []
    
    # Extract planetary features
    features["pl_rade"] = planet_body.get("radius")  # R⊕
    features["pl_masse"] = planet_body.get("mass")   # M⊕
    features["pl_orbper"] = planet_body.get("orbital_period")  # days
    features["pl_orbsmax"] = planet_body.get("semiMajorAxis")  # AU
    features["pl_orbeccen"] = planet_body.get("eccentricity")  # unitless
    
    # Flux and temperature (may be computed if missing)
    features["pl_insol"] = planet_body.get("stellarFlux")
    if features["pl_insol"] is None and star_body:
        # Compute from luminosity and distance
        st_lum = star_body.get("luminosity", 1.0)
        a = features["pl_orbsmax"]
        if a and a > 0:
            features["pl_insol"] = st_lum / (a ** 2)
            feature_sources["pl_insol"] = "computed"
    
    features["pl_eqt"] = planet_body.get("equilibrium_temperature")
    if features["pl_eqt"] is None and features["pl_insol"]:
        # Stefan-Boltzmann approximation
        features["pl_eqt"] = 278.5 * (features["pl_insol"] ** 0.25)
        feature_sources["pl_eqt"] = "computed"
    
    # Density (derived from mass and radius)
    features["pl_dens"] = planet_body.get("density")  # g/cm³
    
    # Stellar features
    features["st_teff"] = star_body.get("temperature") if star_body else None
    features["st_mass"] = star_body.get("mass") if star_body else None
    features["st_rad"] = star_body.get("radius") if star_body else None
    features["st_lum"] = star_body.get("luminosity") if star_body else None
    
    # Hard validation
    for feature_name, (min_val, max_val, unit) in FEATURE_VALIDATION_RANGES.items():
        value = features.get(feature_name)
        if value is not None:
            if not (min_val <= value <= max_val):
                validation_errors.append(
                    f"{feature_name}={value:.3f} outside [{min_val}, {max_val}] {unit}"
                )
                features[feature_name] = None  # Invalidate
    
    # Check for critical missing fields
    critical_fields = ["pl_rade", "pl_masse", "pl_orbsmax", "st_teff"]
    missing_critical = [f for f in critical_fields if features.get(f) is None]
    
    if missing_critical or validation_errors:
        return None, {
            "feature_sources": feature_sources,
            "validation_errors": validation_errors,
            "missing_critical": missing_critical,
            "warnings": validation_errors + [f"Missing: {', '.join(missing_critical)}"]
        }
    
    return features, {
        "feature_sources": feature_sources,
        "validation_errors": [],
        "missing_critical": [],
        "warnings": []
    }
```

---

## Deferred Scoring Queue

### Problem: Race Conditions

When a planet is first placed, **derived parameters** (flux, equilibrium temperature) may not be computed yet. Scoring immediately leads to incorrect predictions.

### Solution: Frame-Delayed Scoring

```python
class Visualizer:
    def __init__(self):
        self.ml_scoring_queue = set()  # Planet IDs awaiting scoring
        self.ml_scoring_queue_frame_delay = 2  # Wait 2 frames
        self.ml_scoring_queue_frame_counts = {}  # {planet_id: frames_waited}
    
    def place_object(self, obj_type, params):
        # ... create planet ...
        
        if obj_type == "planet":
            # Don't score immediately - enqueue instead
            self._enqueue_planet_for_scoring(planet_id)
    
    def _enqueue_planet_for_scoring(self, planet_id):
        self.ml_scoring_queue.add(planet_id)
        self.ml_scoring_queue_frame_counts[planet_id] = 0
    
    def update(self, dt):
        # ... physics updates, derive flux/temp ...
        
        # Process scoring queue after derived params are ready
        self._process_ml_scoring_queue()
    
    def _process_ml_scoring_queue(self):
        ready_to_score = []
        
        for planet_id in self.ml_scoring_queue:
            self.ml_scoring_queue_frame_counts[planet_id] += 1
            
            if self.ml_scoring_queue_frame_counts[planet_id] >= self.ml_scoring_queue_frame_delay:
                ready_to_score.append(planet_id)
        
        for planet_id in ready_to_score:
            self._update_planet_scores(planet_id)
            self.ml_scoring_queue.remove(planet_id)
            del self.ml_scoring_queue_frame_counts[planet_id]
```

**Timeline:**
```
Frame 0: Planet created, flux/temp = None, habit_score_ml = None, UI shows "Computing..."
Frame 1: Derived params computed (flux, temp)
Frame 2: ML scoring triggered, habit_score_ml = 72.93%, UI updates
```

---

## Auditable ML Snapshot Export

### Purpose
Provide **verifiable traceability** of ML predictions for debugging and validation.

### Keybind
`Ctrl+Shift+S` (while planet is selected)

### Export Format

**File:** `exports/ml_snapshot_YYYYMMDD_HHMMSS.json`

```json
{
  "timestamp": "2026-01-29T12:34:56",
  "model_version": "v4.1_xgboost",
  "model_artifact": "ml_calibration/hab_xgb_v4.json",
  "earth_reference_raw": 0.9424,
  "normalization_factor": 1.0611,
  
  "planets": [
    {
      "planet_id": "abc123",
      "planet_name": "Earth",
      "star_name": "Sun",
      
      "features": {
        "pl_rade": 1.0,
        "pl_masse": 1.0,
        "pl_orbper": 365.25,
        "pl_orbsmax": 1.0,
        "pl_orbeccen": 0.0167,
        "pl_insol": 1.0,
        "pl_eqt": 255.0,
        "pl_dens": 5.51,
        "st_teff": 5800,
        "st_mass": 1.0,
        "st_rad": 1.0,
        "st_lum": 1.0
      },
      
      "feature_sources": {
        "pl_insol": "direct",
        "pl_eqt": "computed",
        "pl_dens": "direct"
      },
      
      "surface_class": "rocky",
      "surface_applicable": true,
      "surface_reason": "Small radius (1.00 R_E) with rocky density (5.51 g/cm^3)",
      
      "score_raw": 0.9424,
      "score_display": 100.0,
      "display_label": "",
      "should_display_score": true,
      
      "input_warnings": [],
      "prediction_success": true
    }
  ]
}
```

**Use Cases:**
1. **Debugging:** Verify exact features fed to model
2. **Validation:** Compare against expected NASA catalog values
3. **Research:** Analyze model behavior across parameter space
4. **Reproducibility:** Export → retrain → verify consistency

---

## Model Limitations & Known Issues

### 1. Training Data Bias
- **Issue:** Most confirmed exoplanets are hot Jupiters or mini-Neptunes (observation bias)
- **Impact:** Model may underestimate habitability of Earth-sized temperate planets
- **Mitigation:** Teacher function synthetically upweights habitable zone planets

### 2. Mars Over-Scoring (72.93%)
- **Cause:** Mars is within conservative habitable zone outer edge (1.37 AU)
- **Reality:** Mars has thin atmosphere, frozen water (not habitable without terraforming)
- **Model Perspective:** Orbital/stellar parameters are favorable; lacks atmospheric composition data

### 3. Venus Under-Scoring (57.51% vs. Expected ~30%)
- **Cause:** Model sees 0.95 R⊕, 5.24 g/cm³ (Earth-like size/density)
- **Reality:** Runaway greenhouse effect, 464°C surface (uninhabitable)
- **Missing Data:** Atmospheric pressure, composition not in feature set

### 4. No Atmospheric Composition
- **Missing:** CO₂, O₂, CH₄, H₂O vapor content
- **Impact:** Cannot distinguish habitable vs. runaway greenhouse
- **Future Work:** Incorporate spectroscopic data when available

### 5. Binary Classification Limitation
- **Current:** Single continuous score 0-100%
- **Better:** Multi-output: [surface_habitability, subsurface_habitability, atmospheric_stability]
- **Reason:** Europa (0% surface) has high subsurface habitability (subsurface ocean)

---

## Model Files & Artifacts

### File Locations

```
ml_calibration/
├── hab_xgb_v4.json           # XGBoost model weights (JSON format)
├── features_v4.json          # Feature schema (12 features)
├── scaler_v3.joblib          # Feature scaler (StandardScaler, deprecated)
├── training_summary_v4.json  # Training metrics & hyperparameters
└── [legacy v1-v3 models]     # Old PyTorch MLP models (unused)
```

### Model Hyperparameters

```json
{
  "model_type": "XGBoost",
  "n_estimators": 100,
  "max_depth": 6,
  "learning_rate": 0.1,
  "subsample": 0.8,
  "colsample_bytree": 0.8,
  "objective": "reg:squarederror",
  "eval_metric": "rmse",
  "random_state": 42
}
```

### Training Metrics

```
Training Set: 4,320 samples (80%)
Validation Set: 1,080 samples (20%)

Training RMSE: 0.087
Validation RMSE: 0.112
R² Score: 0.73

Earth Prediction: 0.9424 (raw) → 100.00% (normalized)
```

---

## API Reference (For Developers)

### Predict Habitability Score

```python
from ml_integration_v4 import predict_with_simulation_body_v4

# Simulation body dictionaries
planet = {
    "radius": 1.0,           # R⊕
    "mass": 1.0,             # M⊕
    "semiMajorAxis": 1.0,    # AU
    "orbital_period": 365.25,# days
    "eccentricity": 0.0167,
    "stellarFlux": 1.0,      # S⊕
    "equilibrium_temperature": 255,  # K
    "density": 5.51,         # g/cm³
    "preset_type": "Earth"   # Optional: "Earth" forces 100%
}

star = {
    "temperature": 5800,     # K
    "mass": 1.0,             # M☉
    "radius": 1.0,           # R☉
    "luminosity": 1.0        # L☉
}

# Get score with diagnostics
score, diagnostics = predict_with_simulation_body_v4(
    planet, star, 
    surface_mode="all",      # "all" or "rocky_only"
    return_diagnostics=True
)

print(f"Habitability: {score:.2f}%")
print(f"Surface Class: {diagnostics['surface_class']}")
print(f"Feature Sources: {diagnostics['feature_sources']}")
```

### Export Snapshot

```python
from ml_integration_v4 import export_ml_snapshot_single_planet

export_ml_snapshot_single_planet(
    planet_body=planet,
    star_body=star,
    output_dir="exports/"
)
# Creates: exports/ml_snapshot_20260129_123456.json
```

---

## Research Questions & Future Directions

### Open Questions

1. **Can we incorporate atmospheric spectroscopy data?**
   - James Webb Space Telescope (JWST) provides O₂, CH₄, H₂O spectra
   - Challenge: Only ~50 exoplanets have spectroscopic data (small sample)

2. **Should we separate surface vs. subsurface habitability?**
   - Europa, Enceladus have subsurface oceans (high subsurface score)
   - Current model gives them 0% (no rocky surface)

3. **How to handle tidal locking (M-dwarf planets)?**
   - Planets around M-dwarfs often tidally locked (one side always faces star)
   - May have habitable "terminator zone" despite extreme temperature gradients

4. **Can we train on biosignature detections?**
   - Current: No confirmed biosignatures detected
   - Future: Train on O₂ + CH₄ disequilibrium (potential life indicator)

### Proposed Improvements

1. **Multi-Task Learning:**
   ```python
   outputs = {
       "surface_habitability": [0-100],
       "subsurface_habitability": [0-100],
       "atmospheric_retention": [0-100],
       "radiation_protection": [0-100]
   }
   ```

2. **Uncertainty Quantification:**
   - Add Bayesian Neural Network or ensemble variance
   - Display: "72.93% ± 8.2%" instead of single point estimate

3. **Transfer Learning from Spectroscopy:**
   - Pre-train on simulated atmospheres
   - Fine-tune on real JWST observations

4. **Causal Inference:**
   - Model causal relationships (e.g., stellar wind → atmospheric loss)
   - Counterfactual queries: "If Mars had Earth's magnetic field, what would habitability be?"

---

## Contact & Contributions

**Repository:** [Your GitHub/GitLab URL]  
**Lead Developer:** [Your Name]  
**Email:** [Your Email]

For questions about the ML system, contact the AIET development team.

**Citing AIET:**
```bibtex
@software{aiet2026,
  title={AIET: Artificial Intelligence Exoplanet Toolkit},
  author={[Your Name]},
  year={2026},
  version={v4.1},
  url={[Your URL]}
}
```

---

**End of Technical Deep Dive**  
*Last updated: January 29, 2026*
