"""Standalone orbit unit test"""
import numpy as np
import math
import time

DEBUG_ORBIT = True

def orbit_log(msg):
    """Debug logger for orbit diagnostics"""
    if DEBUG_ORBIT:
        print(f"[ORBIT_DBG] {time.time():.3f} | {msg}")

def run_orbit_unit_test():
    """Deterministic unit test for moon orbit mechanics"""
    orbit_log("=" * 80)
    orbit_log("STARTING ORBIT UNIT TEST")
    orbit_log("=" * 80)
    
    # Create clean test objects
    star_test = {
        "position": np.array([0.0, 0.0], dtype=float),
        "type": "star",
        "name": "StarTest"
    }
    
    planet_test = {
        "orbit_radius": 100.0,
        "orbit_angle": 0.0,
        "orbit_speed": 0.02,
        "type": "planet",
        "name": "PlanetTest",
        "parent_obj": star_test,
        "position": np.array([100.0, 0.0], dtype=float)
    }
    
    moon_test = {
        "orbit_radius": 20.0,
        "orbit_angle": 0.0,
        "orbit_speed": 0.2,
        "type": "moon",
        "name": "MoonTest",
        "parent_obj": planet_test,
        "position": planet_test["position"] + np.array([20.0, 0.0], dtype=float)
    }
    
    max_error = 0.0
    error_frames = []
    
    # Run deterministic loop
    for frame in range(1, 201):
        # Update planet first
        planet_test["orbit_angle"] += planet_test["orbit_speed"]
        planet_test["position"][0] = star_test["position"][0] + planet_test["orbit_radius"] * math.cos(planet_test["orbit_angle"])
        planet_test["position"][1] = star_test["position"][1] + planet_test["orbit_radius"] * math.sin(planet_test["orbit_angle"])
        
        # Then update moon
        moon_test["orbit_angle"] += moon_test["orbit_speed"]
        moon_test["position"][0] = planet_test["position"][0] + moon_test["orbit_radius"] * math.cos(moon_test["orbit_angle"])
        moon_test["position"][1] = planet_test["position"][1] + moon_test["orbit_radius"] * math.sin(moon_test["orbit_angle"])
        
        # Compute expected position
        expected = planet_test["position"] + np.array([
            moon_test["orbit_radius"] * math.cos(moon_test["orbit_angle"]),
            moon_test["orbit_radius"] * math.sin(moon_test["orbit_angle"])
        ])
        
        # Calculate error
        err = np.linalg.norm(moon_test["position"] - expected)
        if err > max_error:
            max_error = err
        if err > 1e-6:
            error_frames.append((frame, err))
        
        if frame <= 10 or frame % 50 == 0 or err > 1e-6:
            print(f"[UNIT_TEST] frame={frame} planet_pos={planet_test['position']} moon_pos={moon_test['position']} expected={expected} err={err:.6e}")
    
    # Summary
    orbit_log("=" * 80)
    if max_error <= 1e-6:
        orbit_log(f"UNIT TEST PASSED: max_error={max_error:.6e}")
    else:
        orbit_log(f"UNIT TEST FAILED: max_error={max_error:.6e}")
        orbit_log(f"Error frames: {error_frames[:20]}")
    orbit_log("=" * 80)
    
    return max_error <= 1e-6

if __name__ == "__main__":
    run_orbit_unit_test()

