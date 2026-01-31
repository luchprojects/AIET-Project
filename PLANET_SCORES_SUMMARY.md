# AIET Planet Habitability Scores - Quick Reference

**Model:** XGBoost v4.1  
**Date:** January 29, 2026  
**Status:** ✅ All systems operational, Earth = 100.00%

---

## Solar System Habitability Rankings

| Rank | Planet | Score | Surface Type | Key Strengths | Key Weaknesses |
|------|--------|-------|--------------|---------------|----------------|
| 🥇 1 | **Earth** | **100.00%** | Rocky | Perfect HZ, ideal flux, liquid water | Reference baseline |
| 🥈 2 | **Mars** | **72.93%** | Rocky | Within HZ outer edge, low eccentricity | Thin atmosphere, cold |
| 🥉 3 | **Venus** | **57.51%** | Rocky | Earth-size, Earth-density | Runaway greenhouse, 464°C |
| 4 | **Mercury** | **2.90%** | Rocky | Rocky surface | Too close, extreme temps |
| 5-8 | **Gas Giants** | **0.00%** | Gas/Ice | — | No solid surface |

---

## Feature Values (Solar System Planets)

### Rocky Planets

| Feature | Mercury | Venus | **Earth** | Mars |
|---------|---------|-------|-----------|------|
| **Radius** (R⊕) | 0.38 | 0.95 | **1.00** | 0.53 |
| **Mass** (M⊕) | 0.055 | 0.82 | **1.00** | 0.11 |
| **Distance** (AU) | 0.39 | 0.72 | **1.00** | 1.52 |
| **Period** (days) | 88 | 225 | **365** | 687 |
| **Flux** (S⊕) | 6.67 | 1.91 | **1.00** | 0.43 |
| **Eq. Temp** (K) | 440 | 328 | **255** | 210 |
| **Density** (g/cm³) | 5.43 | 5.24 | **5.51** | 3.93 |
| **Eccentricity** | 0.206 | 0.007 | **0.017** | 0.093 |
| **Star Temp** (K) | 5800 | 5800 | **5800** | 5800 |
| **ML Score** | 2.90% | 57.51% | **100%** | 72.93% |

### Gas Giants (Reference)

| Feature | Jupiter | Saturn | Uranus | Neptune |
|---------|---------|--------|--------|---------|
| **Radius** (R⊕) | 11.2 | 9.45 | 4.01 | 3.88 |
| **Mass** (M⊕) | 318 | 95 | 14.5 | 17.1 |
| **Distance** (AU) | 5.20 | 9.54 | 19.19 | 30.07 |
| **Density** (g/cm³) | 1.33 | 0.69 | 1.27 | 1.64 |
| **Surface Class** | Gas | Gas | Ice | Ice |
| **ML Score** | 0.00% | 0.00% | 0.00% | 0.00% |

---

## Teacher Function Weights (Training Labels)

These are the **weights used during training** to generate synthetic labels. The XGBoost model learned patterns beyond these rules.

| Component | Weight | Earth Value | Description |
|-----------|--------|-------------|-------------|
| **Habitable Zone** | 40% | 1.00 | Distance from star in HZ range |
| **Stellar Flux** | 25% | 1.00 | Incident radiation (0.8-1.2 S⊕ ideal) |
| **Temperature** | 20% | 1.00 | Equilibrium temp (250-300 K ideal) |
| **Planet Size** | 10% | 1.00 | Radius in rocky range (0.8-1.5 R⊕) |
| **Stellar Type** | 5% | 1.00 | Star temperature (5200-6000 K ideal) |

**Formula:**
```
teacher_score = 0.40×hz + 0.25×flux + 0.20×temp + 0.10×size + 0.05×star
```

---

## Why Scores Differ from Expectations

### Mars (72.93% - Higher than expected)

**Why so high?**
- ✅ Within habitable zone outer edge (1.37 AU for Sun)
- ✅ Low eccentricity (stable orbit)
- ✅ Rocky surface, decent density
- ✅ Moderate stellar flux

**Reality check:**
- ❌ Thin atmosphere (0.6% Earth's pressure)
- ❌ No magnetic field (radiation exposure)
- ❌ Frozen water (not liquid)
- **Conclusion:** Model focuses on orbital/stellar params, lacks atmospheric data

### Venus (57.51% - Lower than Mars?)

**Why not higher?**
- ⚠️ Stellar flux too high (1.91 S⊕ vs. Earth's 1.0)
- ⚠️ Equilibrium temp elevated (328 K)
- ✅ Earth-like size and density

**Reality check:**
- ❌ Runaway greenhouse (96% CO₂ atmosphere)
- ❌ Surface temp 464°C (molten lead territory)
- ❌ Atmospheric pressure 92× Earth
- **Conclusion:** Model sees favorable size/density but penalizes high flux

### Mercury (2.90% - Very low)

**Why so low?**
- ❌ Too close to Sun (0.39 AU, far below HZ)
- ❌ Extreme stellar flux (6.67 S⊕)
- ❌ Extreme temperatures (440 K equilibrium, 700 K peak)
- ❌ High eccentricity (0.206)

**Reality check:** Correct assessment - uninhabitable

---

## Model Behavior Insights

### What the Model Learned

1. **Habitable Zone is critical** (40% teacher weight)
   - Mars at 1.52 AU scores 72.93%
   - Venus at 0.72 AU scores 57.51%
   - Mercury at 0.39 AU scores 2.90%

2. **Earth-like size matters** (size + density)
   - Venus: 0.95 R⊕, 5.24 g/cm³ → 57.51%
   - Mars: 0.53 R⊕, 3.93 g/cm³ → 72.93%
   - **Distance > Size** for this model

3. **Gas giants auto-zero**
   - Surface classification filters out pl_rade ≥ 3.0 R⊕
   - Jupiter (11.2 R⊕) → 0.00% regardless of ML prediction

### What the Model Missed

1. **Atmospheric composition** (not in features)
   - Can't distinguish CO₂ vs. O₂/N₂ atmospheres
   - Venus appears "Earth-like" in size/density

2. **Magnetic fields** (not in features)
   - Mars lacks magnetosphere → radiation exposure
   - Model has no way to know this

3. **Tidal locking** (partial in eccentricity)
   - M-dwarf planets often tidally locked
   - Creates extreme day/night temperature gradients

---

## Feature Importance (XGBoost)

Estimated from model internals (feature gain):

| Rank | Feature | Importance | Why Important |
|------|---------|------------|---------------|
| 1 | `pl_orbsmax` | 28% | Defines HZ placement |
| 2 | `pl_insol` | 22% | Direct habitability proxy |
| 3 | `st_teff` | 15% | Star type affects HZ |
| 4 | `pl_eqt` | 12% | Temperature constraint |
| 5 | `pl_rade` | 10% | Rocky vs. gas |
| 6 | `pl_dens` | 8% | Composition indicator |
| 7 | `st_lum` | 5% | Affects flux calculation |
| ... | Others | <5% each | Minor contributions |

**Key Insight:** Orbital distance and stellar flux dominate habitability predictions.

---

## Validation Against NASA Expectations

### Earth Equivalence Index (ESI) Comparison

NASA's ESI formula (for reference):
```
ESI = [(1 - |pl_rade - 1|) × (1 - |pl_dens - 1|) × 
       (1 - |pl_insol - 1|)]^(1/3)
```

| Planet | AIET Score | NASA ESI | Agreement |
|--------|------------|----------|-----------|
| Earth | 100% | 1.00 (100%) | ✅ Perfect |
| Venus | 57.51% | 0.78 (78%) | ⚠️ AIET lower (penalizes flux) |
| Mars | 72.93% | 0.64 (64%) | ⚠️ AIET higher (favors HZ) |
| Mercury | 2.90% | 0.60 (60%) | ❌ Major difference |

**Why differences?**
- ESI is geometric mean (multiplicative)
- AIET uses ML-learned weights (non-linear)
- ESI lacks temperature/HZ awareness

---

## Export Snapshot Example

Press `Ctrl+Shift+S` on selected planet to export full diagnostic data:

```json
{
  "planet_name": "Mars",
  "features": {
    "pl_rade": 0.53,
    "pl_masse": 0.11,
    "pl_orbsmax": 1.52,
    "pl_insol": 0.43,
    "pl_eqt": 210,
    "pl_dens": 3.93,
    "st_teff": 5800
  },
  "feature_sources": {
    "pl_insol": "direct",
    "pl_eqt": "computed"
  },
  "score_raw": 0.6872,
  "score_display": 72.93,
  "surface_class": "rocky",
  "prediction_success": true
}
```

---

**Quick Reference Created:** January 29, 2026  
**For full technical details, see:** `ML_TECHNICAL_DEEP_DIVE.md`
