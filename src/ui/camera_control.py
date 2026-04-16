from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Camera:
    """Minimal camera state used by main_window."""

    width: int
    height: int
    zoom: float = 0.6
    zoom_min: float = 0.02
    zoom_max: float = 3.0
    offset: List[float] = field(default_factory=list)
    last_zoom_for_orbits: float = 0.6
    is_panning: bool = False
    pan_start: Optional[tuple] = None
    pan_start_offset: Optional[List[float]] = None
    camera_focus: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.offset:
            self.offset = [self.width / 2.0, self.height / 2.0]
        if not self.camera_focus:
            self.camera_focus = {
                "active": False,
                "target_body_id": None,
                "target_world_pos": None,
                "target_zoom": None,
            }
        self.last_zoom_for_orbits = self.zoom
