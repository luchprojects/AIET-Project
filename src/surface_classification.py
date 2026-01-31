"""
Surface Classification for Exoplanets (ML v4.1)

Pure function approach: no side effects, no simulation coupling.
Classifies planets as rocky/giant/unknown based on radius and density.
"""

from typing import Dict, List, Any


def classify_surface(pl_rade: float, pl_dens: float) -> Dict[str, Any]:
    """
    Classify planet surface type based on radius and density.
    
    Args:
        pl_rade: Planet radius in Earth radii (R⊕)
        pl_dens: Planet density in g/cm³
    
    Returns:
        Dictionary with:
            surface_class: "rocky" | "giant" | "unknown"
            surface_applicable: bool (True if rocky surface suitable for liquid water)
            reason: str (human-readable explanation)
            warnings: list[str] (potential issues with inputs)
    
    Classification Rules:
        1. Validation: pl_rade in [0.05, 25], pl_dens in [0.1, 20]
           - Outside range -> "unknown" + warning (likely unit/key mismatch)
        
        2. Giant classification (no solid surface):
           - pl_rade >= 3.0 R_E (mini-Neptune or larger)
           - OR (pl_rade >= 2.0 R_E AND pl_dens <= 2.5 g/cm^3) (puffy/low-density)
        
        3. Rocky classification (solid surface):
           - pl_rade <= 1.8 R_E AND pl_dens >= 3.0 g/cm^3
        
        4. Unknown (ambiguous):
           - Everything else (transition zone, insufficient constraints)
    
    Examples:
        Earth:   1.0 R_E, 5.51 g/cm^3 -> rocky
        Mars:    0.532 R_E, 3.93 g/cm^3 -> rocky
        Venus:   0.95 R_E, 5.24 g/cm^3 -> rocky
        Jupiter: 11.2 R_E, 1.33 g/cm^3 -> giant
        Neptune: 3.9 R_E, 1.64 g/cm^3 -> giant
        Super-Earth: 1.5 R_E, 4.0 g/cm^3 -> rocky
        Mini-Neptune: 2.5 R_E, 2.0 g/cm^3 -> giant
    """
    
    warnings = []
    
    # =============================================================================
    # Input Validation: Check for unit mismatches or invalid data
    # =============================================================================
    
    # Check for missing/invalid inputs
    if pl_rade is None or pl_dens is None:
        return {
            "surface_class": "unknown",
            "surface_applicable": False,
            "reason": "Missing radius or density data",
            "warnings": ["pl_rade or pl_dens is None"]
        }
    
    # Check for reasonable ranges (detect unit mismatches)
    if pl_rade < 0.05 or pl_rade > 25.0:
        warnings.append(f"pl_rade={pl_rade:.2f} R_E outside expected range [0.05, 25] - possible unit mismatch")
        return {
            "surface_class": "unknown",
            "surface_applicable": False,
            "reason": f"Radius out of range ({pl_rade:.2f} R_E) - check units",
            "warnings": warnings
        }
    
    if pl_dens < 0.1 or pl_dens > 20.0:
        warnings.append(f"pl_dens={pl_dens:.2f} g/cm^3 outside expected range [0.1, 20] - possible unit mismatch")
        return {
            "surface_class": "unknown",
            "surface_applicable": False,
            "reason": f"Density out of range ({pl_dens:.2f} g/cm^3) - check units",
            "warnings": warnings
        }
    
    # =============================================================================
    # Classification Logic
    # =============================================================================
    
    # Rule 1: Giant planets (no solid surface for liquid water)
    # Large radius OR low-density "puffy" planets
    if pl_rade >= 3.0:
        return {
            "surface_class": "giant",
            "surface_applicable": False,
            "reason": f"Large radius ({pl_rade:.2f} R_E) indicates gas/ice giant",
            "warnings": warnings
        }
    
    if pl_rade >= 2.0 and pl_dens <= 2.5:
        return {
            "surface_class": "giant",
            "surface_applicable": False,
            "reason": f"Low density ({pl_dens:.2f} g/cm^3) with moderate radius ({pl_rade:.2f} R_E) indicates H/He envelope",
            "warnings": warnings
        }
    
    # Rule 2: Rocky planets (solid surface)
    # Small-to-moderate radius AND high density
    if pl_rade <= 1.8 and pl_dens >= 3.0:
        return {
            "surface_class": "rocky",
            "surface_applicable": True,
            "reason": f"Small radius ({pl_rade:.2f} R_E) with rocky density ({pl_dens:.2f} g/cm^3)",
            "warnings": warnings
        }
    
    # Rule 3: Unknown/Ambiguous (transition zone)
    # Between rocky and giant thresholds
    return {
        "surface_class": "unknown",
        "surface_applicable": False,
        "reason": f"Ambiguous: radius={pl_rade:.2f} R_E, density={pl_dens:.2f} g/cm^3 (transition zone)",
        "warnings": warnings
    }


def get_display_label(surface_class: str, surface_mode: str = "all") -> str:
    """
    Get display label for UI based on surface classification and mode.
    
    Args:
        surface_class: "rocky" | "giant" | "unknown"
        surface_mode: "all" | "rocky_only"
    
    Returns:
        Display label string for UI badge/indicator
    
    Examples:
        ("giant", "all") → "Gas/Ice Giant"
        ("giant", "rocky_only") → "Surface N/A (Gas/Ice Giant)"
        ("rocky", "all") → "" (no label needed)
        ("unknown", "all") → "Classification Uncertain"
    """
    
    if surface_class == "rocky":
        return ""  # No special label for rocky planets
    
    elif surface_class == "giant":
        if surface_mode == "rocky_only":
            return "Surface N/A (Gas/Ice Giant)"
        else:
            return "Gas/Ice Giant"
    
    elif surface_class == "unknown":
        return "Classification Uncertain"
    
    else:
        return "Unknown"


def should_display_score(surface_class: str, surface_mode: str = "all") -> bool:
    """
    Determine if numeric score should be displayed.
    
    Args:
        surface_class: "rocky" | "giant" | "unknown"
        surface_mode: "all" | "rocky_only"
    
    Returns:
        True if numeric score should be shown, False if "—" should be shown
    
    Policy:
        - surface_mode="all": Always show numeric score (even for giants)
        - surface_mode="rocky_only": Only show score for rocky planets
    """
    
    if surface_mode == "all":
        return True  # Show score for all planets
    
    elif surface_mode == "rocky_only":
        return surface_class == "rocky"  # Only show score for rocky planets
    
    else:
        return True  # Default: show score
