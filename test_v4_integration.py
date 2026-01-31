"""
Test ML v4 integration with AIET simulation bodies
"""

import sys
sys.path.insert(0, 'src')

from ml_habitability_v4 import MLHabitabilityCalculatorV4
from ml_integration_v4 import predict_with_simulation_body_v4

# Initialize ML v4 calculator
print("Initializing ML v4 calculator...")
try:
    calc = MLHabitabilityCalculatorV4()
    print("[OK] ML v4 calculator initialized")
except Exception as e:
    print(f"[ERROR] Failed to initialize: {e}")
    sys.exit(1)

# Test with Solar System presets (matching AIET simulation structure)
solar_system_bodies = {
    "Mercury": {
        "planet": {
            "name": "Mercury",
            "radius": 0.383, "mass": 0.055, "orbital_period": 88.0,
            "semiMajorAxis": 0.387, "eccentricity": 0.2056, "stellarFlux": 6.67,
            "equilibrium_temperature": 440.0, "temperature": 440.0,
            "greenhouse_offset": 0.0, "density": 5.43, "preset_type": "Mercury"
        },
        "star": {"temperature": 5778.0, "mass": 1.0, "radius": 1.0, "luminosity": 1.0}
    },
    "Venus": {
        "planet": {
            "name": "Venus",
            "radius": 0.949, "mass": 0.815, "orbital_period": 225.0,
            "semiMajorAxis": 0.723, "eccentricity": 0.0068, "stellarFlux": 1.91,
            "equilibrium_temperature": 237.0, "temperature": 737.0,
            "greenhouse_offset": 500.0, "density": 5.24, "preset_type": "Venus"
        },
        "star": {"temperature": 5778.0, "mass": 1.0, "radius": 1.0, "luminosity": 1.0}
    },
    "Earth": {
        "planet": {
            "name": "Earth",
            "radius": 1.0, "mass": 1.0, "orbital_period": 365.25,
            "semiMajorAxis": 1.0, "eccentricity": 0.0167, "stellarFlux": 1.0,
            "equilibrium_temperature": 255.0, "temperature": 288.0,
            "greenhouse_offset": 33.0, "density": 5.51, "preset_type": "Earth"
        },
        "star": {"temperature": 5778.0, "mass": 1.0, "radius": 1.0, "luminosity": 1.0}
    },
    "Mars": {
        "planet": {
            "name": "Mars",
            "radius": 0.532, "mass": 0.107, "orbital_period": 687.0,
            "semiMajorAxis": 1.524, "eccentricity": 0.0934, "stellarFlux": 0.43,
            "equilibrium_temperature": 200.0, "temperature": 210.0,
            "greenhouse_offset": 10.0, "density": 3.93, "preset_type": "Mars"
        },
        "star": {"temperature": 5778.0, "mass": 1.0, "radius": 1.0, "luminosity": 1.0}
    },
    "Jupiter": {
        "planet": {
            "name": "Jupiter",
            "radius": 11.2, "mass": 317.8, "orbital_period": 4333.0,
            "semiMajorAxis": 5.203, "eccentricity": 0.0484, "stellarFlux": 0.037,
            "equilibrium_temperature": 95.0, "temperature": 165.0,
            "greenhouse_offset": 70.0, "density": 1.33, "preset_type": "Jupiter"
        },
        "star": {"temperature": 5778.0, "mass": 1.0, "radius": 1.0, "luminosity": 1.0}
    }
}

print("\n" + "="*70)
print("Testing ML v4 with AIET simulation bodies:")
print("="*70)

results = {}

for planet_name, bodies in solar_system_bodies.items():
    planet_body = bodies["planet"]
    star_body = bodies["star"]
    
    score, diagnostics = predict_with_simulation_body_v4(
        calc, planet_body, star_body, return_diagnostics=True
    )
    
    results[planet_name] = score
    
    if score is not None:
        print(f"\n{planet_name:10s}: {score:6.2f}%")
        if diagnostics.get("missing_optional"):
            print(f"  Imputed: {diagnostics['missing_optional']}")
    else:
        print(f"\n{planet_name:10s}: FAILED")
        print(f"  Error: {diagnostics.get('warnings', ['Unknown'])}")

print("\n" + "="*70)
print("VALIDATION CHECK:")
print("="*70)

# Check Earth = 100
earth_score = results.get("Earth")
if earth_score and 99.5 <= earth_score <= 100.0:
    print(f"[PASS] Earth = {earth_score:.2f}% (expected 100.0)")
else:
    print(f"[FAIL] Earth = {earth_score:.2f}% (expected 100.0)")

# Check Venus < Earth
venus_score = results.get("Venus")
if venus_score and venus_score < earth_score:
    print(f"[PASS] Venus ({venus_score:.2f}%) < Earth ({earth_score:.2f}%)")
else:
    print(f"[FAIL] Venus ({venus_score:.2f}%) >= Earth ({earth_score:.2f}%)")

# Check no zeros (unless Jupiter)
all_nonzero = all(score > 0 for name, score in results.items() if score is not None)
if all_nonzero:
    print(f"[PASS] No silent zeros - all planets have nonzero scores")
else:
    zero_planets = [name for name, score in results.items() if score == 0]
    print(f"[WARN] Some planets showing 0: {zero_planets}")

print("="*70)
