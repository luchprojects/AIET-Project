"""Project path helpers for runtime assets and model files."""

from __future__ import annotations

import os


def project_root() -> str:
    # src/utils/paths.py -> src/utils -> src -> project root
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def font_path(filename: str) -> str:
    return os.path.join(project_root(), "src", "ui", "fonts", filename)


def model_path(filename: str = "hab_xgb.json") -> str:
    root = project_root()
    calib = os.path.join(root, "ml_calibration")
    return os.path.join(calib, filename)


def feature_schema_path(filename: str = "features.json") -> str:
    root = project_root()
    calib = os.path.join(root, "ml_calibration")
    return os.path.join(calib, filename)

