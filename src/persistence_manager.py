"""
AIET Persistence & Preset Architecture.

Saves and loads full simulation states (bodies + metadata) and object prefabs.
Units: AU, year; mass in Earth masses (planets/moons) or Solar masses (stars).
Does not modify the physics engine or integrator.
"""
from __future__ import annotations

import json
import math
import os
import base64
import zlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

# Use project paths for saved_systems/ and object_library/
try:
    from src.utils.paths import project_root
except ImportError:
    def project_root() -> str:
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Default G used by simulation (AU^3 / (M_sun * year^2))
G_AU = 39.478

# Directories relative to project root
SAVED_SYSTEMS_DIR = "saved_systems"
OBJECT_LIBRARY_DIR = "object_library"


def _saved_systems_path() -> str:
    path = os.path.join(project_root(), SAVED_SYSTEMS_DIR)
    os.makedirs(path, exist_ok=True)
    return path


def _object_library_path() -> str:
    path = os.path.join(project_root(), OBJECT_LIBRARY_DIR)
    os.makedirs(path, exist_ok=True)
    return path


def _body_to_serializable(body: Dict[str, Any], scale_px_per_au: float) -> Dict[str, Any]:
    """
    Convert a UI/engine body dict to a JSON-serializable body record.
    Expects body to have position_au, velocity_au (or we derive from position/velocity and orbit).
    Units: position in AU, velocity in AU/year, mass in Earth masses (planets/moons) or Solar (stars).
    """
    pos_au = body.get("position_au")
    if pos_au is None:
        pos = body.get("position")
        if pos is not None and len(pos) >= 2 and scale_px_per_au > 0:
            pos_au = [float(pos[0]) / scale_px_per_au, float(pos[1]) / scale_px_per_au]
        else:
            pos_au = [0.0, 0.0]
    if hasattr(pos_au, "tolist"):
        pos_au = pos_au.tolist()
    vel_au = body.get("velocity_au")
    if vel_au is None:
        vel_au = [0.0, 0.0]
    if hasattr(vel_au, "tolist"):
        vel_au = vel_au.tolist()

    out: Dict[str, Any] = {
        "name": str(body.get("name", "Unnamed")),
        "type": str(body.get("type", "planet")),
        "mass": float(body.get("mass", 1.0)),
        "x": float(pos_au[0]) if len(pos_au) > 0 else 0.0,
        "y": float(pos_au[1]) if len(pos_au) > 1 else 0.0,
        "vx": float(vel_au[0]) if len(vel_au) > 0 else 0.0,
        "vy": float(vel_au[1]) if len(vel_au) > 1 else 0.0,
        "radius": float(body.get("radius", body.get("actual_radius", 1.0))),
    }
    # Parent (by name) for planets/moons
    parent = body.get("parent")
    if parent:
        out["parent"] = parent
    # Optional fields for future expansion; loader will ignore unknown
    for key in ("temperature", "stellarFlux", "semiMajorAxis", "eccentricity", "orbital_period",
                "inclination", "atmosphere", "habitability_score", "greenhouse_offset",
                "density", "gravity", "base_color", "preset_type"):
        if key in body and body[key] is not None:
            val = body[key]
            if isinstance(val, (int, float, str, bool)):
                out[key] = val
            elif isinstance(val, (list, dict)):
                out[key] = val
    return out


def _serializable_to_body(rec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a loaded JSON body record into a body dict suitable for UI/engine.
    Only uses known fields; ignores unknown keys for future expansion.
    """
    body: Dict[str, Any] = {
        "name": str(rec.get("name", "Unnamed")),
        "type": str(rec.get("type", "planet")),
        "mass": float(rec.get("mass", 1.0)),
        "radius": float(rec.get("radius", 1.0)),
        "position_au": [float(rec.get("x", 0)), float(rec.get("y", 0))],
        "velocity_au": [float(rec.get("vx", 0)), float(rec.get("vy", 0))],
    }
    if "parent" in rec:
        body["parent"] = rec["parent"]
    for key in ("temperature", "stellarFlux", "semiMajorAxis", "eccentricity", "orbital_period",
                "inclination", "atmosphere", "habitability_score", "greenhouse_offset",
                "density", "gravity", "base_color", "preset_type"):
        if key in rec:
            body[key] = rec[key]
    return body


def save_current_system(
    placed_bodies: List[Dict[str, Any]],
    system_name: str,
    dt: float = 0.01,
    G: float = G_AU,
    scale_px_per_au: float = 400.0,
) -> str:
    """
    Save the current simulation state to a JSON file under saved_systems/.

    Args:
        placed_bodies: List of body dicts (must have position_au/velocity_au or position + scale).
        system_name: User-facing name (used for filename after sanitization).
        dt: Integrator timestep in years.
        G: Gravitational constant (AU^3 / (M_sun * year^2)).
        scale_px_per_au: Pixels per AU (used if position_au is missing).

    Returns:
        Path to the written file.
    """
    safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in system_name).strip() or "system"
    base = safe_name.replace(" ", "_")
    filepath = os.path.join(_saved_systems_path(), f"{base}.json")
    # Avoid overwrite collision
    counter = 0
    while os.path.exists(filepath):
        counter += 1
        filepath = os.path.join(_saved_systems_path(), f"{base}_{counter}.json")

    bodies_data: List[Dict[str, Any]] = []
    for b in placed_bodies:
        if b.get("is_destroyed", False):
            continue
        bodies_data.append(_body_to_serializable(b, scale_px_per_au))

    payload: Dict[str, Any] = {
        "system_name": system_name,
        "dt": dt,
        "G": G,
        "creation_timestamp": datetime.now(timezone.utc).isoformat(),
        "bodies": bodies_data,
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return filepath


def load_system(file_name: str) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Load a saved system from saved_systems/.

    Args:
        file_name: Basename (e.g. "three_moon_earth.json") or full path.

    Returns:
        (payload, stability_warning).
        payload: dict with keys "system_name", "dt", "G", "creation_timestamp", "bodies".
        "bodies" is a list of body dicts with position_au, velocity_au, mass, etc.
        stability_warning: Non-null if dt may be unstable (dt > T/1000 for some orbit).
    """
    if not os.path.isabs(file_name):
        filepath = os.path.join(_saved_systems_path(), file_name)
    else:
        filepath = file_name
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Saved system not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    dt = float(data.get("dt", 0.01))
    G = float(data.get("G", G_AU))
    bodies_raw = data.get("bodies", [])
    bodies: List[Dict[str, Any]] = [_serializable_to_body(rec) for rec in bodies_raw]

    # Numerical stability check: T ≈ 2π sqrt(r³ / GM); warn if dt > T/1000
    warning = _stability_warning(bodies, dt, G)

    payload: Dict[str, Any] = {
        "system_name": data.get("system_name", "Loaded"),
        "dt": dt,
        "G": G,
        "creation_timestamp": data.get("creation_timestamp", ""),
        "bodies": bodies,
    }
    return payload, warning


def _stability_warning(bodies: List[Dict[str, Any]], dt: float, G: float) -> Optional[str]:
    """If dt > T/1000 for the shortest orbital timescale, return warning string."""
    stars = [b for b in bodies if b.get("type") == "star"]
    planets = [b for b in bodies if b.get("type") == "planet"]
    if not stars or not planets:
        return None
    # Total mass of stars (in Solar masses) for GM
    M_solar = 0.0
    for s in stars:
        M_solar += float(s.get("mass", 1.0))  # stars in Solar masses
    if M_solar <= 0:
        return None
    # Smallest semi-major axis (or distance) among planets
    min_r_au = None
    for p in planets:
        a = p.get("semiMajorAxis")
        if a is not None and a > 0:
            min_r_au = min(min_r_au, float(a)) if min_r_au is not None else float(a)
        else:
            x, y = p.get("position_au", [0, 0])[:2]
            r = math.sqrt(x * x + y * y)
            if r > 0:
                min_r_au = min(min_r_au, r) if min_r_au is not None else r
    if min_r_au is None or min_r_au <= 0:
        return None
    # T = 2π sqrt(r³/(GM)); GM = G * M_solar (already in AU^3/year^2)
    T = 2.0 * math.pi * math.sqrt((min_r_au ** 3) / (G * M_solar))
    if T <= 0:
        return None
    if dt > T / 1000.0:
        return "Integrator timestep may be unstable for this system."
    return None


def list_saved_systems() -> List[str]:
    """Return list of .json basenames in saved_systems/."""
    path = _saved_systems_path()
    if not os.path.isdir(path):
        return []
    return sorted(f for f in os.listdir(path) if f.endswith(".json"))


# Optional fields copied onto object prefabs (planets, moons, stars).
_PREFAB_OPTIONAL_KEYS = (
    "default_color",
    "base_color",
    "temperature",
    "density",
    "gravity",
    "semiMajorAxis",
    "eccentricity",
    "orbital_period",
    "stellarFlux",
    "greenhouse_offset",
    "preset_type",
    "luminosity",
    "spectral_class",
    "age",
    "metallicity",
    "activity",
    "classification",
    "atmosphere_type",
    "rotation_period_days",
    "display_name",
    "parent",
    "name_locked",
)


def save_object_prefab(body: Dict[str, Any], name: Optional[str] = None) -> str:
    """
    Save a single celestial body as a prefab in object_library/.

    Args:
        body: Body dict with at least name, type, mass, radius; optional default_color, etc.
        name: Filename base (default: body["name"] sanitized).

    Returns:
        Path to the written file.
    """
    base = name or str(body.get("name", "object")).strip()
    base = "".join(c if c.isalnum() or c in " _-" else "_" for c in base).strip() or "object"
    base = base.replace(" ", "_")
    filepath = os.path.join(_object_library_path(), f"{base}.json")

    prefab: Dict[str, Any] = {
        "name": str(body.get("name", base)),
        "type": str(body.get("type", "planet")),
        "mass": float(body.get("mass", 1.0)),
        "radius": float(body.get("radius", body.get("actual_radius", 1.0))),
    }
    for key in _PREFAB_OPTIONAL_KEYS:
        if key in body and body[key] is not None:
            v = body[key]
            if isinstance(v, (int, float, str, bool, list, dict)):
                prefab[key] = v

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(prefab, f, indent=2)
    return filepath


def load_object_prefab(name: str) -> Dict[str, Any]:
    """
    Load a single object prefab from object_library/.

    Args:
        name: Basename (e.g. "super_earth.json") or name without extension.

    Returns:
        Prefab dict (name, type, mass, radius, and any optional keys).
    """
    base = name if name.endswith(".json") else f"{name}.json"
    filepath = os.path.join(_object_library_path(), base)
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Object prefab not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Return only known/used keys; ignore unknown for future expansion
    out: Dict[str, Any] = {
        "name": str(data.get("name", "Unnamed")),
        "type": str(data.get("type", "planet")),
        "mass": float(data.get("mass", 1.0)),
        "radius": float(data.get("radius", 1.0)),
    }
    for key in _PREFAB_OPTIONAL_KEYS:
        if key in data:
            out[key] = data[key]
    return out


def list_object_prefabs() -> List[str]:
    """Return list of .json basenames in object_library/."""
    path = _object_library_path()
    if not os.path.isdir(path):
        return []
    return sorted(f for f in os.listdir(path) if f.endswith(".json"))


# =============================================================================
# System Seed (portable import/export string)
# =============================================================================

_SEED_PREFIX = "AIET-SEED:v1:"


def create_system_seed(
    placed_bodies: List[Dict[str, Any]],
    system_name: str,
    dt: float = 0.01,
    G: float = G_AU,
    scale_px_per_au: float = 400.0,
    *,
    aiet_version: Optional[str] = None,
) -> str:
    """
    Create a portable "system seed" string for research reproducibility.

    The seed is a stable, self-contained encoding of the same payload saved by
    `save_current_system`, compressed and base64-url encoded.
    """
    bodies_data: List[Dict[str, Any]] = []
    for b in placed_bodies:
        if b.get("is_destroyed", False):
            continue
        bodies_data.append(_body_to_serializable(b, scale_px_per_au))

    payload: Dict[str, Any] = {
        "schema": "AIET_SYSTEM_SEED",
        "schema_version": 1,
        "aiet_version": aiet_version,
        "system_name": system_name,
        "dt": float(dt),
        "G": float(G),
        "creation_timestamp": datetime.now(timezone.utc).isoformat(),
        "bodies": bodies_data,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    packed = zlib.compress(raw, level=9)
    token = base64.urlsafe_b64encode(packed).decode("ascii")
    return f"{_SEED_PREFIX}{token}"


def load_system_seed(seed: str) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Decode a seed string back into a persistence payload (same shape as `load_system`).

    Returns (payload, stability_warning).
    """
    if not isinstance(seed, str):
        raise TypeError("seed must be a string")
    s = seed.strip()
    if s.startswith(_SEED_PREFIX):
        s = s[len(_SEED_PREFIX):]
    # tolerate accidental whitespace/newlines in copied seeds
    s = "".join(s.split())
    try:
        packed = base64.urlsafe_b64decode(s.encode("ascii"))
        raw = zlib.decompress(packed).decode("utf-8")
        data = json.loads(raw)
    except Exception as e:
        raise ValueError(f"Invalid seed: {e}") from e

    # Validate minimally; ignore unknown keys for forward compatibility
    if str(data.get("schema", "")) not in ("AIET_SYSTEM_SEED", ""):
        raise ValueError("Invalid seed schema")

    dt = float(data.get("dt", 0.01))
    G = float(data.get("G", G_AU))
    bodies_raw = data.get("bodies", [])
    bodies: List[Dict[str, Any]] = [_serializable_to_body(rec) for rec in bodies_raw]
    warning = _stability_warning(bodies, dt, G)

    payload: Dict[str, Any] = {
        "system_name": data.get("system_name", "Loaded"),
        "dt": dt,
        "G": G,
        "creation_timestamp": data.get("creation_timestamp", ""),
        "bodies": bodies,
    }
    return payload, warning
