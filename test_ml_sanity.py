#!/usr/bin/env python3
"""
Test script for ML sanity check function.
Demonstrates how to diagnose low Earth ML scores.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ml_habitability import MLHabitabilityCalculator, run_ml_sanity_check

def main():
    print("\n" + "="*70)
    print("ML SANITY CHECK TEST")
    print("="*70 + "\n")
    
    # Initialize ML calculator
    try:
        ml_calc = MLHabitabilityCalculator()
        print("✓ ML calculator initialized successfully\n")
    except Exception as e:
        print(f"✗ Failed to initialize ML calculator: {e}\n")
        return
    
    # Earth reference values (standard NASA exoplanet units)
    earth_features = {
        "pl_rade": 1.0,        # Earth radii
        "pl_masse": 1.0,       # Earth masses
        "pl_orbper": 365.25,   # days
        "pl_orbeccen": 0.0167, # unitless
        "pl_insol": 1.0,       # Earth flux units
        "st_teff": 5778.0,     # Kelvin
        "st_mass": 1.0,        # Solar masses
        "st_rad": 1.0,         # Solar radii
        "st_lum": 1.0          # Solar luminosities
    }
    
    print("Testing with Earth reference values:")
    for key, value in earth_features.items():
        print(f"  {key}: {value}")
    print()
    
    # Get prediction before sanity check
    earth_score = ml_calc.predict(earth_features)
    print(f"Earth ML Score: {earth_score:.2f}%\n")
    
    # Run comprehensive sanity check
    print("Running comprehensive sanity check...\n")
    report = run_ml_sanity_check(
        ml_calculator=ml_calc,
        planet_features=earth_features,
        export_dir="exports"
    )
    
    print("\nSummary:")
    print(f"  Overall Status: {report['overall_status']}")
    print(f"  Recommended Fix: {report['recommended_fix']}")
    
    if report.get('export_path'):
        print(f"\nDetailed report saved to: {report['export_path']}")
        print("Open this JSON file to see full diagnostic details.")
    
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
