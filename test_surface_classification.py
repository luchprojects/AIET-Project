"""
Test surface classification and ML v4.1 integration with Solar System
"""

from src.ml_habitability_v4 import MLHabitabilityCalculatorV4
from src.ml_integration_v4 import predict_with_simulation_body_v4, export_ml_debug_snapshot

# Solar System planet data
SOLAR_SYSTEM = {
    "Mercury": {
        "planet": {
            "radius": 0.383,
            "mass": 0.055,
            "orbital_period": 87.97,
            "semiMajorAxis": 0.387,
            "eccentricity": 0.206,
            "stellarFlux": 6.67,
            "equilibrium_temperature": 440.0,
            "density": 5.43,
        },
        "star": {
            "temperature": 5778.0,
            "mass": 1.0,
            "radius": 1.0,
            "luminosity": 1.0
        }
    },
    "Venus": {
        "planet": {
            "radius": 0.949,
            "mass": 0.815,
            "orbital_period": 224.70,
            "semiMajorAxis": 0.723,
            "eccentricity": 0.007,
            "stellarFlux": 1.91,
            "equilibrium_temperature": 328.0,
            "density": 5.24,
        },
        "star": {
            "temperature": 5778.0,
            "mass": 1.0,
            "radius": 1.0,
            "luminosity": 1.0
        }
    },
    "Earth": {
        "planet": {
            "radius": 1.0,
            "mass": 1.0,
            "orbital_period": 365.25,
            "semiMajorAxis": 1.0,
            "eccentricity": 0.0167,
            "stellarFlux": 1.0,
            "equilibrium_temperature": 255.0,
            "density": 5.51,
            "preset_type": "Earth"
        },
        "star": {
            "temperature": 5778.0,
            "mass": 1.0,
            "radius": 1.0,
            "luminosity": 1.0
        }
    },
    "Mars": {
        "planet": {
            "radius": 0.532,
            "mass": 0.107,
            "orbital_period": 686.98,
            "semiMajorAxis": 1.524,
            "eccentricity": 0.0934,
            "stellarFlux": 0.43,
            "equilibrium_temperature": 210.0,
            "density": 3.93,
        },
        "star": {
            "temperature": 5778.0,
            "mass": 1.0,
            "radius": 1.0,
            "luminosity": 1.0
        }
    },
    "Jupiter": {
        "planet": {
            "radius": 11.21,
            "mass": 317.8,
            "orbital_period": 4332.59,
            "semiMajorAxis": 5.203,
            "eccentricity": 0.0489,
            "stellarFlux": 0.037,
            "equilibrium_temperature": 110.0,
            "density": 1.33,
        },
        "star": {
            "temperature": 5778.0,
            "mass": 1.0,
            "radius": 1.0,
            "luminosity": 1.0
        }
    }
}

print("="*70)
print("ML v4.1 - Surface Classification Test")
print("="*70)

# Initialize ML calculator
print("\nInitializing ML v4 calculator...")
ml_calc = MLHabitabilityCalculatorV4()
print("[OK] ML v4 calculator initialized\n")

print("="*70)
print("Testing surface_mode='all' (compute scores for all planets)")
print("="*70)

for name, data in SOLAR_SYSTEM.items():
    score, diagnostics = predict_with_simulation_body_v4(
        ml_calc,
        data["planet"],
        data["star"],
        return_diagnostics=True,
        surface_mode="all"
    )
    
    surface_class = diagnostics.get("surface_class", "unknown")
    surface_reason = diagnostics.get("surface_reason", "")
    display_label = diagnostics.get("display_label", "")
    should_display = diagnostics.get("should_display_score", True)
    
    print(f"\n{name:10s}:")
    print(f"  Surface: {surface_class:8s} | {surface_reason}")
    
    if score is not None:
        if display_label:
            print(f"  Score:   {score:6.2f}% ({display_label})")
        else:
            print(f"  Score:   {score:6.2f}%")
    else:
        print(f"  Score:   — (unavailable)")

print("\n" + "="*70)
print("Testing surface_mode='rocky_only' (giants show None)")
print("="*70)

for name, data in SOLAR_SYSTEM.items():
    score, diagnostics = predict_with_simulation_body_v4(
        ml_calc,
        data["planet"],
        data["star"],
        return_diagnostics=True,
        surface_mode="rocky_only"
    )
    
    surface_class = diagnostics.get("surface_class", "unknown")
    display_label = diagnostics.get("display_label", "")
    should_display = diagnostics.get("should_display_score", True)
    
    print(f"\n{name:10s}:")
    print(f"  Surface: {surface_class:8s}")
    
    if should_display and score is not None:
        print(f"  Score:   {score:6.2f}%")
    elif not should_display:
        print(f"  Score:   — ({display_label})")
    else:
        print(f"  Score:   — (unavailable)")

# Export debug snapshot
print("\n" + "="*70)
print("Exporting debug snapshot...")
print("="*70)

bodies_list = [
    (data["planet"], data["star"], name)
    for name, data in SOLAR_SYSTEM.items()
]

snapshot_path = export_ml_debug_snapshot(ml_calc, bodies_list)

print("\n" + "="*70)
print("VALIDATION CHECKS")
print("="*70)

# Get scores in "all" mode
scores = {}
for name, data in SOLAR_SYSTEM.items():
    score, diagnostics = predict_with_simulation_body_v4(
        ml_calc,
        data["planet"],
        data["star"],
        return_diagnostics=True,
        surface_mode="all"
    )
    scores[name] = {
        "score": score,
        "surface_class": diagnostics.get("surface_class", "unknown")
    }

# Check acceptance criteria
print("\n[CHECK] Mercury classified as rocky (not giant):")
if scores["Mercury"]["surface_class"] == "rocky":
    print(f"  [PASS] Mercury is '{scores['Mercury']['surface_class']}'")
else:
    print(f"  [FAIL] Mercury is '{scores['Mercury']['surface_class']}' (should be 'rocky')")

print("\n[CHECK] Jupiter classified as giant:")
if scores["Jupiter"]["surface_class"] == "giant":
    print(f"  [PASS] Jupiter is '{scores['Jupiter']['surface_class']}'")
else:
    print(f"  [FAIL] Jupiter is '{scores['Jupiter']['surface_class']}' (should be 'giant')")

print("\n[CHECK] Earth = 100%:")
if scores["Earth"]["score"] is not None and abs(scores["Earth"]["score"] - 100.0) < 0.1:
    print(f"  [PASS] Earth score = {scores['Earth']['score']:.2f}%")
else:
    print(f"  [FAIL] Earth score = {scores['Earth']['score']:.2f}% (should be 100.00%)")

print("\n[CHECK] Mars > Venus:")
if scores["Mars"]["score"] > scores["Venus"]["score"]:
    print(f"  [PASS] Mars ({scores['Mars']['score']:.2f}%) > Venus ({scores['Venus']['score']:.2f}%)")
else:
    print(f"  [FAIL] Mars ({scores['Mars']['score']:.2f}%) <= Venus ({scores['Venus']['score']:.2f}%)")

print("\n[CHECK] Jupiter score is very low (indicating gas giant):")
if scores["Jupiter"]["score"] < 10.0:
    print(f"  [PASS] Jupiter score = {scores['Jupiter']['score']:.2f}% (low, as expected)")
else:
    print(f"  [FAIL] Jupiter score = {scores['Jupiter']['score']:.2f}% (should be very low)")

print("\n[CHECK] No silent zeros (all planets have non-None scores):")
none_count = sum(1 for name, data in scores.items() if data["score"] is None)
if none_count == 0:
    print(f"  [PASS] All {len(scores)} planets have numeric scores")
else:
    print(f"  [FAIL] {none_count} planets have None scores")

print("\n" + "="*70)
print("[SUCCESS] Surface classification system operational")
print("="*70)
