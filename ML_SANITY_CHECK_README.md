# ML Sanity Check - Diagnostic Tool

## Purpose
Diagnose why Earth is scoring ~39% instead of a higher value in the ML habitability model.

## Features

The sanity check runs 5 comprehensive tests:

### 1. Input Sanity Check
- Validates all required inputs are present (not None/NaN)
- Logs Earth/Sun reference values
- **Status**: FAIL if any inputs are missing

### 2. Unit & Range Validation
- Checks each feature is within expected physical ranges
- Validates unit consistency (AU, days, K, Earth masses, Solar masses, etc.)
- **Status**: WARN if values are outside expected ranges (likely unit mismatch)

### 3. Feature Schema Check
- Verifies runtime feature order matches training schema
- Expected order: `pl_rade, pl_masse, pl_orbper, pl_orbeccen, pl_insol, st_teff, st_mass, st_rad, st_lum`
- **Status**: FAIL if feature names or order mismatch

### 4. Model Output Check
- Runs inference and logs raw model output
- Warns that ML score is uncalibrated (not a true probability)
- **Status**: Always WARN (interpretation caveat)

### 5. Solar System Ranking Test
- Scores Mercury, Venus, Earth, Mars, and Jupiter
- Checks if Earth ranks in top 2
- **Status**: WARN if Earth is not in top 2 (likely training/unit issue)

## Usage

### In the Application (Runtime)
Press **Ctrl+Shift+M** to run sanity check on:
- Currently selected planet (if any)
- Earth (fallback if no planet selected)

### Standalone Test Script
```bash
cd C:\Users\LuchK\AIET
python test_ml_sanity.py
```

## Output

### Console
Prints overall status and recommended fix:
```
ML SANITY CHECK COMPLETE
Overall Status: WARN
Recommended Fix: Earth ranking low - check training labels and unit consistency
```

### JSON Report
Exported to: `exports/debug_ml_<timestamp>.json`

Example structure:
```json
{
  "timestamp": "2026-01-25T...",
  "overall_status": "WARN",
  "recommended_fix": "...",
  "checks": {
    "input_sanity": { "status": "PASS", "details": {...}, "issues": [] },
    "unit_range_validation": { "status": "WARN", "details": {...}, "issues": [...] },
    "feature_schema": { "status": "PASS", "details": {...}, "issues": [] },
    "model_output": { "status": "WARN", "details": {"raw_model_output": 39.2}, "issues": [...] },
    "solar_system_ranking": {
      "status": "WARN",
      "details": {
        "scores": {"Mercury": 12.3, "Venus": 28.1, "Earth": 39.2, "Mars": 15.8, "Jupiter": 5.2},
        "ranking": [...]
      },
      "issues": ["Earth ranked #3 (expected top 2) - likely unit or label mismatch"]
    }
  }
}
```

## Common Issues & Fixes

| Status | Likely Cause | Fix |
|--------|--------------|-----|
| **Input missing/NaN** | Feature extraction failed | Check fill policy in preprocessing |
| **Values out of range** | Unit conversion error | Verify AU, days, K are in correct units |
| **Feature schema mismatch** | Feature order changed | Match runtime order to training order |
| **Earth not top 2** | Training label mismatch | Retrain model or verify label generation |

## Expected Earth Values

Reference (NASA Exoplanet Archive units):
- `pl_rade`: 1.0 R⊕
- `pl_masse`: 1.0 M⊕
- `pl_orbper`: 365.25 days
- `pl_orbeccen`: 0.0167
- `pl_insol`: 1.0 (Earth flux)
- `st_teff`: 5778 K
- `st_mass`: 1.0 M☉
- `st_rad`: 1.0 R☉
- `st_lum`: 1.0 L☉

## Next Steps

After running sanity check:

1. **Read the JSON report** in `exports/`
2. **Check "recommended_fix"** for guidance
3. **Review unit/range warnings** - most common issue
4. **Compare Earth score** to Solar System ranking
5. **If Earth < 50%**: Likely unit mismatch or training data issue

## Technical Notes

- No UI changes
- No model retraining
- Pure diagnostic tool
- Lightweight (runs in <1s)
- Safe to run anytime (read-only)
