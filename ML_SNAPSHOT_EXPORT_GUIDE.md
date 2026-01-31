# ML Snapshot Export - Quick Reference Guide

## What is ML Snapshot Export?

An **auditable debug tool** that shows exactly what features were fed to the ML model and what score came out. Critical for verifying that sandbox display matches ML predictions.

---

## How to Use

### Export Snapshot for Current Planet

1. **Select a planet** in the sandbox (click on it)
2. **Press `Ctrl+Shift+S`**
3. **Check console** for export confirmation
4. **Open file**: `exports/ml_snapshot_<planet_name>_<timestamp>.json`

---

## Snapshot File Format

```json
{
  "timestamp": "2026-01-28T...",
  "planet": {
    "name": "Mercury",
    "id": "abc123..."
  },
  "model": {
    "version": "v4.1",
    "type": "XGBoost",
    "earth_reference_raw": 0.94236
  },
  "feature_extraction": {
    "success": true,
    "validation_errors": [],
    "mapping_errors": [],
    "warnings": []
  },
  "features": {
    "pl_rade": 0.383,
    "pl_masse": 0.055,
    "pl_orbper": 88.0,
    "pl_orbsmax": 0.387,
    "pl_orbeccen": 0.2056,
    "pl_insol": 6.67,
    "pl_eqt": 440.0,
    "pl_dens": 5.43,
    "st_teff": 5778.0,
    "st_mass": 1.0,
    "st_rad": 1.0,
    "st_lum": 1.0
  },
  "feature_sources": {
    "pl_rade": "direct",
    "pl_masse": "direct",
    "pl_orbper": "direct",
    "pl_orbsmax": "direct",
    "pl_orbeccen": "direct",
    "pl_insol": "computed",
    "pl_eqt": "direct",
    "pl_dens": "computed",
    "st_teff": "direct",
    "st_mass": "direct",
    "st_rad": "direct",
    "st_lum": "computed"
  },
  "prediction": {
    "raw_score": 0.027351,
    "displayed_score": 2.74,
    "success": true,
    "error": null
  },
  "imputed_by_feature_builder": []
}
```

---

## What Each Field Means

### `features`
The 12 features actually fed to the XGBoost model:

| Feature | Units | Meaning |
|---------|-------|---------|
| `pl_rade` | R⊕ | Planet radius (Earth radii) |
| `pl_masse` | M⊕ | Planet mass (Earth masses) |
| `pl_orbper` | days | Orbital period |
| `pl_orbsmax` | AU | Semi-major axis (distance from star) |
| `pl_orbeccen` | - | Orbital eccentricity (0-1) |
| `pl_insol` | S⊕ | Stellar flux received (Earth flux = 1.0) |
| `pl_eqt` | K | Equilibrium temperature (no atmosphere) |
| `pl_dens` | g/cm³ | Planet density |
| `st_teff` | K | Stellar effective temperature |
| `st_mass` | M☉ | Stellar mass (Solar masses) |
| `st_rad` | R☉ | Stellar radius (Solar radii) |
| `st_lum` | L☉ | Stellar luminosity (Solar luminosities) |

### `feature_sources`
How each feature was obtained:
- **`"direct"`** - Read directly from simulation body
- **`"computed"`** - Calculated from other parameters (e.g., flux = luminosity / distance²)
- **`"imputed"`** - Estimated by feature builder (rare, only if critical data missing)

### `prediction`
- **`raw_score`** (0-1): Model output before Earth normalization
- **`displayed_score`** (0-100): Normalized to Earth = 100%
  - Formula: `(raw_score / earth_reference_raw) × 100`
  - This is what shows in the UI

### `imputed_by_feature_builder`
List of any features that were imputed (estimated) by the feature builder because they were missing from the feature dict. Should be empty for well-formed planets.

---

## Verification Workflow

### Check if Sandbox Matches Snapshot

1. Place Mercury preset
2. Select Mercury
3. Note UI score (should be ~2.74%)
4. Press `Ctrl+Shift+S`
5. Open snapshot JSON
6. Compare:
   - `prediction.displayed_score` (2.74) == UI score ✅
   - `features.pl_rade` (0.383) == Mercury radius ✅
   - `features.pl_insol` (6.67) == Mercury flux ✅

**If they match:** System is working correctly ✅
**If they don't match:** Unit mismatch or race condition 🚨

---

## Common Issues & Solutions

### Issue: `validation_errors` present
**Cause:** Feature values outside physical ranges
**Fix:** Check if units are wrong (e.g., km instead of R⊕)

```json
"validation_errors": [
  "pl_rade = 2439.0000 R⊕ outside valid range [0.05, 25.00] R⊕"
]
```
→ Planet radius was passed in km (2439 km) instead of Earth radii (0.383 R⊕)

### Issue: `mapping_errors` present
**Cause:** Critical fields missing from simulation body
**Fix:** Ensure planet has radius, mass, star has temperature, mass

```json
"mapping_errors": [
  "Missing: planet.mass"
]
```

### Issue: `displayed_score` is None
**Cause:** Validation or mapping failed
**Fix:** Check `validation_errors` and `mapping_errors`

### Issue: `displayed_score` doesn't match UI
**Cause:** Race condition (planet was displayed before scoring finished)
**Fix:** Wait 2 frames after placement, then export again

---

## Debugging with Snapshots

### Problem: "Mercury shows 0% in sandbox but should be ~2.74%"

**Step 1:** Export snapshot
```
Ctrl+Shift+S
```

**Step 2:** Check `prediction.displayed_score`
- If 2.74 → UI display bug, not ML bug
- If 0.0 → Feature extraction bug

**Step 3:** Check `features` for unit mismatches
```json
"pl_rade": 2439.0  // WRONG - this is km, not R⊕
"pl_rade": 0.383   // CORRECT - this is R⊕
```

**Step 4:** Check `feature_sources` for unexpected imputations
```json
"pl_rade": "imputed"  // BAD - should be "direct"
"pl_rade": "direct"   // GOOD
```

---

## Acceptance Test with Snapshots

### Solar System Preset Verification

1. Place Sun + all 5 presets (Mercury, Venus, Earth, Mars, Jupiter)
2. For each planet:
   - Select planet
   - Press `Ctrl+Shift+S`
   - Record `prediction.displayed_score`

**Expected Results:**

| Planet | Displayed Score | Raw Score | Status |
|--------|----------------|-----------|--------|
| Mercury | 2.74% | 0.0274 | ✅ |
| Venus | 43.85% | 0.4385 | ✅ |
| Earth | 94.24% | 0.9424 | ✅ |
| Mars | 69.09% | 0.6909 | ✅ |
| Jupiter | 0.005% | 0.00005 | ✅ |

3. Compare with `validation_v4_*.json` from ML training
4. All should match within 0.01%

---

## Command Summary

| Action | Keybind | Output |
|--------|---------|--------|
| Export snapshot for selected planet | `Ctrl+Shift+S` | `exports/ml_snapshot_<planet>_<timestamp>.json` |
| ML sanity check (all planets) | `Ctrl+Shift+M` | Console output |

---

## Notes

- Snapshots are **read-only** debug tools (do not modify predictions)
- Each snapshot is timestamped (safe to export multiple times)
- Snapshots include validation status (useful for debugging unit mismatches)
- Feature sources help identify if data is computed vs direct (transparency)

---

## Example Output (Mercury)

```
[ML SNAPSHOT] Exported to: exports/ml_snapshot_Mercury_20260128_143522.json
  Planet: Mercury
  Raw Score: 0.027351
  Displayed Score: 2.74%
```

**This is your ground truth** - if this doesn't match the UI, there's a display bug, not an ML bug.
