import pygame
import numpy as np
import threading
import time
import traceback
import math
import random
from typing import List, Tuple
from uuid import uuid4
from simulation_engine import CelestialBody, SimulationEngine
try:
    from ml_habitability import MLHabitabilityCalculator
except ImportError:
    MLHabitabilityCalculator = None

# ============================================================================
# ORBIT DEBUG INSTRUMENTATION
# ============================================================================
DEBUG_ORBIT = False
frame_trace = []  # Per-frame trace collector

def orbit_log(msg):
    """Debug logger for orbit diagnostics"""
    if DEBUG_ORBIT:
        print(f"[ORBIT_DBG] {time.time():.3f} | {msg}")

def trace(msg):
    """Add a trace message to the current frame"""
    if DEBUG_ORBIT:
        frame_trace.append(msg)

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

# Visual scaling constants (for sandbox view)
AU_TO_PX = 400            # Spread planets out significantly more
SUN_RADIUS_PX = 48        # Sun size
EARTH_RADIUS_PX = 21      # Earth size
MOON_RADIUS_PX = 6        # Moon size
MOON_ORBIT_AU = 0.00257   # Scientifically correct
MOON_ORBIT_PX = 35        # Tighter orbit to stay in Earth's "lane"

# Real-Time Simulation Constants
TIME_SCALES = [
    0.1,            # 0.1 sec / sec (slow motion)
    0.5,            # 0.5 sec / sec
    1.0,            # 1 sec / sec (default)
    60.0,           # 1 minute / sec
    3600.0,         # 1 hour / sec
    86400.0,        # 1 day / sec
    2592000.0       # 1 month / sec (Earth orbits in ~12 sec)
]
DEFAULT_SCALE_INDEX = 2  # 1.0 sec / sec

# Planet visual scaling constants (NASA-style perceptual compression)
SUN_RADIUS_R_EARTH = 109.0  # Sun radius in Earth radii (R⊕) per NASA standards
MIN_PLANET_PX = 6.0         # Minimum visual size
PLANET_SCALE_POWER = 0.5    # Size scaling
PLANET_DISTANCE_SCALE_POWER = 0.85 # Higher power pushes Earth further out to avoid Moon collisions

# Helper function to convert hex color string to RGB tuple
def hex_to_rgb(hex_string: str) -> Tuple[int, int, int]:
    """Convert hex color string (e.g., '#FDB813') to RGB tuple (e.g., (253, 184, 19))"""
    hex_string = hex_string.lstrip('#')
    return tuple(int(hex_string[i:i+2], 16) for i in (0, 2, 4))

def desaturate_color(rgb: Tuple[int, int, int], saturation: float = 0.6) -> Tuple[int, int, int]:
    """
    Desaturate an RGB color by mixing it with gray.
    
    Args:
        rgb: RGB tuple (r, g, b)
        saturation: Saturation factor (0.0 = fully gray, 1.0 = original color). Default 0.6 for slight desaturation.
    
    Returns:
        Desaturated RGB tuple
    """
    r, g, b = rgb
    # Calculate grayscale value (luminance)
    gray = int(0.299 * r + 0.587 * g + 0.114 * b)
    # Mix original color with gray
    r_desat = int(r * saturation + gray * (1.0 - saturation))
    g_desat = int(g * saturation + gray * (1.0 - saturation))
    b_desat = int(b * saturation + gray * (1.0 - saturation))
    return (r_desat, g_desat, b_desat)

# Celestial body base colors (scientifically motivated, UX-friendly)
# Colors represent dominant surface or atmospheric class, not literal imagery
CELESTIAL_BODY_COLORS = {
    "Sun": "#FDB813",
    "Mercury": "#9E9E9E",
    "Venus": "#E6C17C",
    "Earth": "#2E7FFF",
    "Mars": "#C1440E",
    "Jupiter": "#D2B48C",
    "Saturn": "#E8D8A8",
    "Uranus": "#7FDBFF",
    "Neptune": "#4169E1",
    "Moon": "#B0B0B0",
}

# Solar System Planet Presets (ordered by semi-major axis)
# All values are scientifically grounded and used as starting presets
SOLAR_SYSTEM_PLANET_PRESETS = {
    "Mercury": {
        "mass": 0.055,  # Earth masses
        "radius": 0.383,  # Earth radii (R⊕)
        "rotation_period_days": 58.6,
        "semiMajorAxis": 0.387,  # AU (Precise for 88 day period)
        "greenhouse_offset": 0.0,  # No atmosphere
        "temperature": 440.0,  # K (equilibrium temperature, no greenhouse)
        "equilibrium_temperature": 440.0,  # K
        "gravity": 3.7,  # m/s²
        "eccentricity": 0.2056,
        "orbital_period": 88.0,  # days
        "stellarFlux": 6.67,  # Earth flux units
        "density": 5.43,  # g/cm³
        "base_color": "#9E9E9E",  # Gray (rocky surface)
    },
    "Venus": {
        "mass": 0.815,
        "radius": 0.949,
        "rotation_period_days": 243.0,
        "semiMajorAxis": 0.723,  # AU (Precise for 225 day period)
        "greenhouse_offset": 500.0,  # Dense CO₂ (Runaway Greenhouse)
        "temperature": 737.0,  # K (T_eq + greenhouse)
        "equilibrium_temperature": 237.0,  # K
        "gravity": 8.87,
        "eccentricity": 0.0068,
        "orbital_period": 225.0,
        "stellarFlux": 1.91,
        "density": 5.24,
        "base_color": "#E6C17C",  # Golden yellow (sulfuric acid clouds)
    },
    "Earth": {
        "mass": 1.0,
        "radius": 1.0,
        "rotation_period_days": 1.0,
        "semiMajorAxis": 1.0,
        "greenhouse_offset": 33.0,  # Earth-like (N₂–O₂ + H₂O + CO₂)
        "temperature": 288.0,  # K
        "equilibrium_temperature": 255.0,  # K
        "gravity": 9.81,
        "eccentricity": 0.0167,
        "orbital_period": 365.25,
        "stellarFlux": 1.0,
        "density": 5.51,
        "base_color": "#2E7FFF",  # Blue (oceans and atmosphere)
    },
    "Mars": {
        "mass": 0.107,
        "radius": 0.532,
        "rotation_period_days": 1.03,
        "semiMajorAxis": 1.524,  # AU (Precise for 687 day period)
        "greenhouse_offset": 10.0,  # Thin CO₂ / N₂
        "temperature": 210.0,  # K
        "equilibrium_temperature": 200.0,  # K
        "gravity": 3.7,
        "eccentricity": 0.0934,
        "orbital_period": 687.0,
        "stellarFlux": 0.43,
        "density": 3.93,
        "base_color": "#C1440E",  # Red-orange (iron oxide surface)
    },
    "Jupiter": {
        "mass": 317.8,
        "radius": 11.2,
        "rotation_period_days": 0.41,
        "semiMajorAxis": 5.203,  # AU (Precise for 4333 day period)
        "greenhouse_offset": 70.0,  # H₂-rich
        "temperature": 165.0,  # K
        "equilibrium_temperature": 95.0,  # K
        "gravity": 24.79,
        "eccentricity": 0.0484,
        "orbital_period": 4333.0,
        "stellarFlux": 0.037,
        "density": 1.33,
        "base_color": "#D2B48C",  # Tan (ammonia clouds)
    },
    "Saturn": {
        "mass": 95.2,
        "radius": 9.5,
        "semiMajorAxis": 9.582,  # AU (Precise for 10759 day period)
        "greenhouse_offset": 70.0,  # H₂-rich
        "temperature": 134.0,  # K
        "equilibrium_temperature": 64.0,  # K
        "gravity": 10.44,
        "eccentricity": 0.0539,
        "orbital_period": 10759.0,
        "stellarFlux": 0.011,
        "density": 0.69,
        "base_color": "#E8D8A8",  # Pale yellow (ammonia ice clouds)
    },
    "Uranus": {
        "mass": 14.5,
        "radius": 4.0,
        "semiMajorAxis": 19.191,  # AU (Precise for 30687 day period)
        "greenhouse_offset": 70.0,  # H₂-rich
        "temperature": 76.0,  # K
        "equilibrium_temperature": 6.0,  # K
        "gravity": 8.69,
        "eccentricity": 0.0473,
        "orbital_period": 30687.0,
        "stellarFlux": 0.0029,
        "density": 1.27,
        "base_color": "#7FDBFF",  # Cyan (methane atmosphere)
    },
    "Neptune": {
        "mass": 17.1,
        "radius": 3.9,
        "semiMajorAxis": 30.07,  # AU (Precise for 60190 day period)
        "greenhouse_offset": 70.0,  # H₂-rich
        "temperature": 72.0,  # K
        "equilibrium_temperature": 2.0,  # K
        "gravity": 11.15,
        "eccentricity": 0.0086,
        "orbital_period": 60190.0,
        "stellarFlux": 0.0015,
        "density": 1.64,
        "base_color": "#4169E1",  # Royal blue (methane atmosphere)
    },
}

class SolarSystemVisualizer:
    def __init__(self, width: int = 1200, height: int = 800):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("AIET - Solar System Simulator")
        self.clock = pygame.time.Clock()
        self.scale = AU_TO_PX  # pixels per AU
        self.center = np.array([width/2, height/2])
        
        # Initialize fonts consistently using pygame.font.Font
        self.font = pygame.font.Font(None, 36)
        self.title_font = pygame.font.Font(None, 72)
        self.subtitle_font = pygame.font.Font(None, 24)
        self.tiny_font = pygame.font.Font(None, 18)
        self.tab_font = pygame.font.Font(None, 20)
        self.button_font = pygame.font.Font(None, 36)
        
        # Screen states
        # Home screen removed - start directly in sandbox
        self.show_home_screen = False
        self.show_simulation_builder = True  # Start directly in sandbox
        self.show_simulation = False
        self.show_customization_panel = False
        
        # Log sandbox initialization
        print("AIET sandbox initialized.")
        
        # Run orbit unit test on startup
        if DEBUG_ORBIT:
            run_orbit_unit_test()
        
        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.BLUE = (0, 0, 255)
        self.RED = (255, 0, 0)
        self.YELLOW = (255, 255, 0)
        self.PURPLE = (148, 0, 211)
        self.GRAY = (128, 128, 128)
        self.LIGHT_GRAY = (200, 200, 200)
        self.DARK_BLUE = (10, 10, 30)
        self.GRID_COLOR = (30, 30, 50)
        self.ACTIVE_TAB_COLOR = (255, 100, 100)  # Bright red for active tab
        
        # Home screen state
        self.ambient_colors = [(119, 236, 57)]  # Green color
        self.current_color_index = 0
        self.color_change_counter = 0
        self.color_change_speed = 30  # Changed from 999999 to 30 for faster color changes
        
        # Create button
        self.create_button = pygame.Rect(self.width//2 - 100, self.height//2 + 50, 200, 50)
        
        # Simulation builder tabs
        self.tab_height = 40
        self.tab_width = 150
        self.tab_margin = 10
        self.tabs = {
            "moon": pygame.Rect(self.tab_margin, self.tab_margin, self.tab_width, self.tab_height),
            "planet": pygame.Rect(self.tab_margin + self.tab_width + self.tab_margin, 
                                self.tab_margin, self.tab_width, self.tab_height),
            "star": pygame.Rect(self.tab_margin + 2*(self.tab_width + self.tab_margin), 
                              self.tab_margin, self.tab_width, self.tab_height)
        }
        self.active_tab = None
        self.space_area = pygame.Rect(0, self.tab_height + 2*self.tab_margin, 
                                    self.width, self.height - (self.tab_height + 2*self.tab_margin))
        
        # Planet preset selector (small dropdown arrow in bottom-right of Planet tab)
        self.planet_preset_arrow_size = 12
        self.planet_preset_arrow_rect = None  # Will be calculated based on Planet tab position
        self.planet_preset_dropdown_visible = False
        self.planet_preset_dropdown_rect = None  # Will be calculated when opened
        self.planet_preset_options = list(SOLAR_SYSTEM_PLANET_PRESETS.keys())  # Ordered by semi-major axis
        
        # Orbital correction animation system
        self.orbital_corrections = {}  # {body_id: {"target_radius_px": float, "start_time": float, "duration": float, "start_pos": np.array, "target_pos": np.array}}
        self.correction_animation_duration = 0.8  # seconds
        
        # Placement Trajectory Line System
        self.ARRIVAL_EPSILON = 5.0  # Pixels
        self.PLACEMENT_LINE_FADE_DURATION = 1.0  # Seconds
        
        # Preview state for placement
        self.preview_position = None  # Mouse position for preview
        self.preview_radius = None  # Preview radius based on object type
        self.placement_mode_active = False  # Track if we're in placement mode
        
        # Track placed celestial bodies
        self.placed_bodies = []
        self.bodies_by_id = {}  # ID-based registry: {body_id: body_dict} for guaranteed unique lookups
        
        try:
            self.ml_calculator = MLHabitabilityCalculator() if MLHabitabilityCalculator else None
        except Exception as e:
            print(f"Warning: Could not initialize ML calculator in Visualizer: {e}")
            self.ml_calculator = None
        self.body_counter = {"moon": 0, "planet": 0, "star": 0}
        
        # Physics parameters
        self.G = 0.5  # Gravitational constant (reduced for more stable orbits)
        self.base_time_step = 0.05  # Base simulation time step
        
        # Global Simulation Clock & Time Acceleration
        self.simulation_time_seconds = 0.0
        self.SECONDS_PER_YEAR = 365.25 * 24 * 3600
        self.SECONDS_PER_DAY = 24 * 3600
        
        self.current_scale_index = DEFAULT_SCALE_INDEX
        self.time_scale = TIME_SCALES[self.current_scale_index]
        self.paused = False
        self.last_pre_pause_scale_index = self.current_scale_index
        
        # UI Rects for time controls (initialized in _get_time_controls_layout)
        self.time_panel_rect = pygame.Rect(0, 0, 0, 0)
        self.decel_btn_rect = pygame.Rect(0, 0, 0, 0)
        self.pause_play_btn_rect = pygame.Rect(0, 0, 0, 0)
        self.accel_btn_rect = pygame.Rect(0, 0, 0, 0)
        
        # Camera state
        self.camera_zoom = 0.6
        self.camera_offset = [40.0, 160.0]
        self.camera_zoom_min = 0.01
        self.camera_zoom_max = 10.0
        self.is_panning = False
        self.pan_start = None
        self.last_zoom_for_orbits = 1.0
        
        # Camera Focus System (Click-to-Focus)
        self.camera_focus = {
            "active": False,
            "target_body_id": None,  # Follow this body during transition
            "target_world_pos": None, # Initial target if body moves
            "target_zoom": None
        }
        
        self.orbit_screen_cache = {}  # name -> (zoom, points)
        self.orbit_grid_screen_cache = {}  # name -> (zoom, points)
        self.last_middle_click_time = 0
        
        # Reset-view button (optional UX helper)
        self.reset_view_button = pygame.Rect(self.width - 140, self.tab_margin, 110, 30)
        self.orbit_points = {}  # Store orbit points for visualization
        self.orbit_history = {}  # Store orbit history for trail effect
        self.orbit_grid_points = {}  # Store grid points for orbit visualization
        
        # Spacetime grid parameters
        self.grid_size = 50  # Size of grid cells
        
        # Rotation parameters
        self.rotation_speed = 0.1  # Base rotation speed
        
        # Customization panel
        self.selected_body = None  # Direct reference to selected body (for backward compatibility)
        self.selected_body_id = None  # ID of selected body (for guaranteed correct lookup)
        self.customization_panel_width = 400  # Increased from 300 to 400 for more space
        self.customization_panel = pygame.Rect(self.width - self.customization_panel_width, 0, 
                                             self.customization_panel_width, self.height)
        self.mass_input_rect = pygame.Rect(self.width - self.customization_panel_width + 50, 120, 
                                         self.customization_panel_width - 100, 30)
        self.mass_input_active = False
        self.mass_input_text = ""
        self.mass_min = 0.1  # Minimum mass (0.1 Earth masses)
        self.mass_max = 1000.0  # Maximum mass (1000 Earth masses)
        
        # Close-info-panel rect
        self.info_panel_close_btn = pygame.Rect(0, 0, 0, 0)
        
        self.show_info_panel = False
        self.info_panel_scroll_y = 0
        self.info_panel_rect = pygame.Rect(0, 0, 0, 0)
        self.info_icon_rect = pygame.Rect(0, 0, 0, 0)
        
        # Planet reference dropdown properties
        self.planet_dropdown_rect = pygame.Rect(self.width - self.customization_panel_width + 50, 120, 
                                              self.customization_panel_width - 100, 30)
        self.planet_dropdown_active = False
        self.planet_dropdown_options = [
            ("Mercury", 0.055),  # Mass in Earth masses
            ("Venus", 0.815),
            ("Earth", 1.0),
            ("Mars", 0.107),
            ("Jupiter", 317.8),
            ("Saturn", 95.2),
            ("Uranus", 14.5),
            ("Neptune", 17.1),
            ("Custom", None)  # Added custom mass option
        ]
        self.planet_dropdown_selected = None  # No planet preselected - user must choose
        self.planet_dropdown_visible = False
        self.show_custom_mass_input = False
        
        # Planet age dropdown properties
        self.planet_age_dropdown_rect = pygame.Rect(self.width - self.customization_panel_width + 50, 180,
                                                  self.customization_panel_width - 100, 30)
        self.planet_age_dropdown_active = False
        self.planet_age_dropdown_options = [
            ("Solar System Planets (4.6 Gyr)", 4.6),
            ("Custom", None)
        ]
        self.planet_age_dropdown_selected = "Solar System Planets (4.6 Gyr)"
        self.planet_age_dropdown_visible = False
        self.show_custom_age_input = False
        
        # Planet radius dropdown properties
        self.planet_radius_dropdown_rect = pygame.Rect(self.width - self.customization_panel_width + 50, 240,
                                                     self.customization_panel_width - 100, 30)
        self.planet_radius_dropdown_active = False
        self.planet_radius_dropdown_options = [
            ("Mercury", 0.38),
            ("Venus", 0.95),
            ("Earth", 1.0),
            ("Mars", 0.53),
            ("Jupiter", 11.2),
            ("Saturn", 9.5),
            ("Uranus", 4.0),
            ("Neptune", 3.9),
            ("Custom", None)
        ]
        self.planet_radius_dropdown_selected = "Earth"
        self.planet_radius_dropdown_visible = False
        self.show_custom_radius_input = False
        self.radius_input_active = False
        self.radius_input_text = ""
        
        # Planet temperature dropdown properties
        self.planet_temperature_dropdown_rect = pygame.Rect(self.width - self.customization_panel_width + 50, 300,
                                                          self.customization_panel_width - 100, 30)
        self.planet_temperature_dropdown_active = False
        self.planet_temperature_dropdown_options = [
            ("Mercury", 440),
            ("Venus", 737),
            ("Earth", 288),
            ("Mars", 210),
            ("Jupiter", 165),
            ("Saturn", 134),
            ("Uranus", 76),
            ("Neptune", 72),
            ("Custom", None)
        ]
        self.planet_temperature_dropdown_selected = "Earth"
        self.planet_temperature_dropdown_visible = False
        self.show_custom_planet_temperature_input = False
        
        # Planet atmospheric composition / greenhouse type dropdown properties
        self.planet_atmosphere_dropdown_rect = pygame.Rect(self.width - self.customization_panel_width + 50, 360,
                                                          self.customization_panel_width - 100, 30)
        self.planet_atmosphere_dropdown_active = False
        self.planet_atmosphere_dropdown_options = [
            ("Mercury-like (None)", 0.0),
            ("Venus-like (Thick)", 500.0),
            ("Earth-like (Moderate)", 33.0),
            ("Mars-like (Thin)", 10.0),
            ("Gas Giant–like (Very Thick)", 70.0),
            ("Custom", None)
        ]
        self.planet_atmosphere_dropdown_selected = "Earth-like (Moderate)"  # Default to Earth-like
        self.planet_atmosphere_dropdown_visible = False
        self.show_custom_atmosphere_input = False
        self.planet_atmosphere_input_text = ""
        
        # Planet gravity dropdown properties
        self.planet_gravity_dropdown_rect = pygame.Rect(self.width - self.customization_panel_width + 50, 420,
                                                      self.customization_panel_width - 100, 30)
        self.planet_gravity_dropdown_active = False
        self.planet_gravity_dropdown_options = [
            ("Mercury", 3.7),
            ("Venus", 8.87),
            ("Earth", 9.81),
            ("Mars", 3.7),
            ("Jupiter", 24.79),
            ("Saturn", 10.44),
            ("Uranus", 8.69),
            ("Neptune", 11.15),
            ("Custom", None)
        ]
        self.planet_gravity_dropdown_selected = "Earth"  # Default to Earth
        self.planet_gravity_dropdown_visible = False
        self.show_custom_planet_gravity_input = False
        self.planet_gravity_input_text = ""
        
        # Moon reference dropdown properties
        self.moon_dropdown_rect = pygame.Rect(self.width - self.customization_panel_width + 50, 120,
                                            self.customization_panel_width - 100, 30)
        self.moon_dropdown_active = False
        self.moon_dropdown_options = [
            ("Deimos", 2.0e-8, "M☾"),
            ("Phobos", 1.5e-7, "M☾"),
            ("Enceladus", 0.0015, "M☾"),
            ("Europa", 0.0073, "M☾"),
            ("Callisto", 0.0107, "M☾"),
            ("Titan", 0.0184, "M☾"),
            ("Ganymede", 0.0202, "M☾"),
            ("Moon", 1.0, "M☾"),
            ("Custom", None, "M☾")
        ]
        self.moon_dropdown_selected = "Moon"
        self.moon_dropdown_visible = False
        self.show_custom_moon_mass_input = False
        self.moon_mass_input_text = ""
        self.show_custom_moon_radius_input = False
        self.moon_radius_input_text = ""
        self.moon_gravity_input_text = ""
        self.moon_age_dropdown_rect = pygame.Rect(self.width - self.customization_panel_width + 50, 180,
                                                self.customization_panel_width - 100, 30)
        self.moon_age_dropdown_active = False
        self.moon_age_dropdown_options = [
            ("Solar System Moons (4.6 Gyr)", 4.6),
            ("Custom", None)
        ]
        self.moon_age_dropdown_selected = "Solar System Moons (4.6 Gyr)"
        self.moon_age_dropdown_visible = False
        self.show_custom_moon_age_input = False
        
        # Moon radius dropdown properties
        self.moon_radius_dropdown_rect = pygame.Rect(self.width - self.customization_panel_width + 50, 240,
                                                   self.customization_panel_width - 100, 30)
        self.moon_radius_dropdown_active = False
        self.moon_radius_dropdown_options = [
            ("Deimos", 6.2),
            ("Phobos", 11.3),
            ("Enceladus", 252.1),
            ("Europa", 1560.8),
            ("Moon", 1737.4),
            ("Callisto", 2410.3),
            ("Titan", 2574.7),
            ("Ganymede", 2634.1),
            ("Custom", None)
        ]
        self.moon_radius_dropdown_selected = "Moon"
        self.moon_radius_dropdown_visible = False
        self.show_custom_moon_radius_input = False
        
        # Moon orbital distance dropdown properties
        self.moon_orbital_distance_dropdown_rect = pygame.Rect(self.width - self.customization_panel_width + 50, 300,
                                                             self.customization_panel_width - 100, 30)
        self.moon_orbital_distance_dropdown_active = False
        self.moon_orbital_distance_dropdown_options = [
            ("Phobos", 9378),
            ("Deimos", 23460),
            ("Enceladus", 237950),
            ("Moon", 384400),
            ("Europa", 670900),
            ("Ganymede", 1070400),
            ("Titan", 1221870),
            ("Callisto", 1882700),
            ("Custom", None)
        ]
        self.moon_orbital_distance_dropdown_selected = "Moon"
        self.moon_orbital_distance_dropdown_visible = False
        self.show_custom_moon_orbital_distance_input = False
        
        # Moon orbital period dropdown properties
        self.moon_orbital_period_dropdown_rect = pygame.Rect(self.width - self.customization_panel_width + 50, 360,
                                                           self.customization_panel_width - 100, 30)
        self.moon_orbital_period_dropdown_active = False
        self.moon_orbital_period_dropdown_options = [
            ("Phobos", 0.32),
            ("Deimos", 1.26),
            ("Enceladus", 1.37),
            ("Europa", 3.55),
            ("Ganymede", 7.15),
            ("Titan", 15.95),
            ("Callisto", 16.69),
            ("Moon", 27.3),
            ("Custom", None)
        ]
        self.moon_orbital_period_dropdown_selected = "Moon"
        self.moon_orbital_period_dropdown_visible = False
        self.show_custom_moon_orbital_period_input = False
        
        # Moon surface temperature dropdown properties
        self.moon_temperature_dropdown_rect = pygame.Rect(self.width - self.customization_panel_width + 50, 420,
                                                        self.customization_panel_width - 100, 30)
        self.moon_temperature_dropdown_active = False
        self.moon_temperature_dropdown_options = [
            ("Enceladus", 75),
            ("Titan", 94),
            ("Europa", 102),
            ("Ganymede", 110),
            ("Callisto", 134),
            ("Moon", 220),
            ("Custom", None)
        ]
        self.moon_temperature_dropdown_selected = "Moon"
        self.moon_temperature_dropdown_visible = False
        self.show_custom_moon_temperature_input = False
        
        # Moon surface gravity dropdown properties
        self.moon_gravity_dropdown_rect = pygame.Rect(self.width - self.customization_panel_width + 50, 480,
                                                    self.customization_panel_width - 100, 30)
        self.moon_gravity_dropdown_active = False
        self.moon_gravity_dropdown_options = [
            ("Deimos", 0.003),
            ("Phobos", 0.0057),
            ("Enceladus", 0.113),
            ("Callisto", 1.235),
            ("Europa", 1.314),
            ("Titan", 1.352),
            ("Ganymede", 1.428),
            ("Moon", 1.62),
            ("Custom", None)
        ]
        self.moon_gravity_dropdown_selected = "Moon"
        self.moon_gravity_dropdown_visible = False
        self.show_custom_moon_gravity_input = False
        self.moon_gravity_input_text = ""
        self.close_button_size = 20
        self.close_button = pygame.Rect(self.width - self.close_button_size - 10, 10, 
                                      self.close_button_size, self.close_button_size)
        
        # Orbit toggle UI elements (only for planets and moons)
        # Position these near the bottom of the customization panel
        self.orbit_toggle_y = 700  # Y position for orbit toggles
        self.orbit_enabled_checkbox = pygame.Rect(self.width - self.customization_panel_width + 50, self.orbit_toggle_y, 20, 20)
        self.last_revolution_checkbox = pygame.Rect(self.width - self.customization_panel_width + 50, self.orbit_toggle_y + 30, 20, 20)
        
        # Age input properties
        self.age_input_rect = pygame.Rect(self.width - self.customization_panel_width + 50, 180, 
                                        self.customization_panel_width - 100, 30)
        self.age_input_active = False
        self.age_input_text = ""
        self.age_min = 0.0  # 0 Gyr
        self.age_max = 10.0  # 10 Gyr
        
        # Luminosity input properties
        self.luminosity_dropdown_rect = pygame.Rect(self.width - self.customization_panel_width + 50, 300,
                                                  self.customization_panel_width - 100, 30)
        self.luminosity_dropdown_active = False
        self.luminosity_dropdown_options = [
            ("Red Dwarf", 0.04),
            ("Orange Dwarf", 0.15),
            ("G-type Main Sequence (Sun)", 1.00),
            ("Bright F-type Star", 2.00),
            ("Blue A-type Star", 25.00),
            ("B-type Giant", 10000.00),
            ("O-type Supergiant", 100000.00),
            ("Custom", None)
        ]
        self.luminosity_dropdown_selected = "G-type Main Sequence (Sun)"  # Default to Sun
        self.luminosity_dropdown_visible = False
        self.show_custom_luminosity_input = False
        self.luminosity_input_active = False
        self.luminosity_input_text = ""
        self.luminosity_min = 0.0  # Minimum luminosity
        self.luminosity_max = 1000000.0  # Maximum luminosity (for very massive stars)
        
        # Dropdown menu properties
        self.dropdown_surface = None
        self.dropdown_rect = None
        self.dropdown_options_rects = []
        self.dropdown_background_color = (255, 255, 255)  # Pure white
        self.dropdown_border_color = (200, 200, 200)  # Light gray for borders
        self.dropdown_hover_color = (240, 240, 240)  # Slightly darker for hover
        self.dropdown_text_color = (0, 0, 0)  # Black text
        self.dropdown_padding = 5  # Padding around text
        self.dropdown_option_height = 30  # Height of each option
        self.dropdown_border_width = 1  # Width of option borders
        
        # Spectral Class (Temperature) dropdown properties - merged from spectral and temperature dropdowns
        self.spectral_class_dropdown_rect = pygame.Rect(self.width - self.customization_panel_width + 50, 240, 
                                                     self.customization_panel_width - 100, 30)
        self.spectral_class_dropdown_active = False
        self.spectral_class_dropdown_options = [
            ("M-type (Red, ~3,000 K)", 3000, (255, 0, 0)),
            ("K-type (Orange, ~4,500 K)", 4500, (255, 165, 0)),
            ("G-type (Yellow, Sun, ~5,800 K)", 5800, (255, 255, 0)),
            ("F-type (Yellow-white, ~7,500 K)", 7500, (255, 255, 224)),
            ("A-type (White, ~10,000 K)", 10000, (255, 255, 255)),
            ("B-type (Blue-white, ~20,000 K)", 20000, (173, 216, 230)),
            ("O-type (Blue, ~40,000 K)", 40000, (0, 0, 255)),
            ("Custom", None, None)
        ]
        self.spectral_class_dropdown_selected = "G-type (Yellow, Sun, ~5,800 K)"  # Default to Sun
        self.spectral_class_dropdown_visible = False
        self.show_custom_temperature_input = False
        self.temperature_input_active = False
        self.temperature_input_text = ""
        self.temperature_min = 2000  # Minimum temperature (K)
        self.temperature_max = 50000  # Maximum temperature (K)

        # Star mass dropdown properties
        self.star_mass_dropdown_rect = pygame.Rect(self.width - self.customization_panel_width + 50, 120,
                                                 self.customization_panel_width - 100, 30)
        self.star_mass_dropdown_active = False
        self.star_mass_dropdown_options = [
            ("Red Dwarf (0.08 M☉, hydrogen-burning limit)", 0.08),
            ("Low-mass Star (0.5 M☉)", 0.5),
            ("Sun-like Star (1.0 M☉)", 1.0),
            ("Intermediate Star (3.0 M☉)", 3.0),
            ("Massive Star (10.0 M☉)", 10.0),
            ("Custom", None)
        ]
        self.star_mass_dropdown_selected = "Sun-like Star (1.0 M☉)"  # Default to Sun
        self.star_mass_dropdown_visible = False
        self.show_custom_star_mass_input = False
    
        # Star age dropdown properties
        self.star_age_dropdown_rect = pygame.Rect(self.width - self.customization_panel_width + 50, 180,
                                                self.customization_panel_width - 100, 30)
        self.star_age_dropdown_active = False
        self.star_age_dropdown_options = [
            ("1.0 (Gyr)", 1.0),
            ("4.6 (Gyr) (Sun)", 4.6),
            ("7.0 (Gyr)", 7.0),
            ("Custom", None)
        ]
        self.star_age_dropdown_selected = "4.6 (Gyr) (Sun)"  # Default to Sun
        self.star_age_dropdown_visible = False
        self.show_custom_star_age_input = False
    
        # Star radius dropdown properties
        self.radius_dropdown_rect = pygame.Rect(self.width - self.customization_panel_width + 50, 360,
                                              self.customization_panel_width - 100, 30)
        self.radius_dropdown_active = False
        self.radius_dropdown_options = [
            ("M-type", 0.3),
            ("K-type", 0.8),
            ("G-type (Sun)", 1.0),
            ("F-type", 1.3),
            ("A-type", 2.0),
            ("B-type", 5.0),
            ("O-type", 10.0),
            ("Custom", None)
        ]
        self.radius_dropdown_selected = "G-type (Sun)"  # Default to Sun
        self.radius_dropdown_visible = False
        self.show_custom_radius_input = False
        self.radius_input_active = False
        self.radius_input_text = ""
        self.radius_min = 0.1  # Minimum radius (R☉)
        self.radius_max = 100.0  # Maximum radius (R☉)
    
        # Star activity level dropdown properties
        self.activity_dropdown_rect = pygame.Rect(self.width - self.customization_panel_width + 50, 420,
                                                self.customization_panel_width - 100, 30)
        self.activity_dropdown_active = False
        self.activity_dropdown_options = [
            ("Low", 0.25),
            ("Moderate (Sun)", 0.5),
            ("High", 0.75),
            ("Very High", 1.0),
            ("Custom", None)
        ]
        self.activity_dropdown_selected = "Moderate (Sun)"  # Default to Sun
        self.activity_dropdown_visible = False
        self.show_custom_activity_input = False
        self.activity_input_active = False
        self.activity_input_text = ""
        self.activity_min = 0.0  # Minimum activity level
        self.activity_max = 1.0  # Maximum activity level

        # Star metallicity dropdown properties
        self.metallicity_dropdown_rect = pygame.Rect(self.width - self.customization_panel_width + 50, 480,
                                                   self.customization_panel_width - 100, 30)
        self.metallicity_dropdown_active = False
        self.metallicity_dropdown_options = [
            ("-0.5 (Metal-poor)", -0.5),
            ("0.0 (Sun)", 0.0),
            ("+0.3 (Metal-rich)", 0.3),
            ("Custom", None)
        ]
        self.metallicity_dropdown_selected = "0.0 (Sun)"  # Default to Sun
        self.metallicity_dropdown_visible = False
        self.show_custom_metallicity_input = False
        self.metallicity_input_active = False
        self.metallicity_input_text = ""
        self.metallicity_min = -1.0  # Minimum metallicity [Fe/H]
        self.metallicity_max = 1.0  # Maximum metallicity [Fe/H]
        
        # Planet orbital distance (semi-major axis) dropdown properties
        self.planet_orbital_distance_dropdown_rect = pygame.Rect(self.width - self.customization_panel_width + 50, 480, self.customization_panel_width - 100, 30)
        self.planet_orbital_distance_dropdown_active = False
        self.planet_orbital_distance_dropdown_options = [
            ("Mercury", 0.387),
            ("Venus", 0.723),
            ("Earth", 1.0),
            ("Mars", 1.524),
            ("Jupiter", 5.203),
            ("Saturn", 9.582),
            ("Uranus", 19.191),
            ("Neptune", 30.07),
            ("Custom", None)
        ]
        self.planet_orbital_distance_dropdown_selected = "Earth"
        self.planet_orbital_distance_dropdown_visible = False
        self.show_custom_orbital_distance_input = False
        self.orbital_distance_input_active = False
        self.orbital_distance_input_text = ""
        self.orbital_distance_min = 0.01
        self.orbital_distance_max = 1000.0
        
        # Planet orbital eccentricity dropdown properties
        self.planet_orbital_eccentricity_dropdown_rect = pygame.Rect(self.width - self.customization_panel_width + 50, 540, self.customization_panel_width - 100, 30)
        self.planet_orbital_eccentricity_dropdown_active = False
        self.planet_orbital_eccentricity_dropdown_options = [
            ("Circular Orbit", 0.0),
            ("Mercury", 0.2056),
            ("Venus", 0.0068),
            ("Earth", 0.0167),
            ("Mars", 0.0934),
            ("Jupiter", 0.0484),
            ("Saturn", 0.0539),
            ("Uranus", 0.0473),
            ("Neptune", 0.0086),
            ("Custom", None)
        
        ]
        self.planet_orbital_eccentricity_dropdown_selected = "Earth"
        self.planet_orbital_eccentricity_dropdown_visible = False
        self.show_custom_orbital_eccentricity_input = False
        self.orbital_eccentricity_input_active = False
        self.orbital_eccentricity_input_text = ""
        self.orbital_eccentricity_min = 0.0
        self.orbital_eccentricity_max = 1.0
        
        # Planet orbital period dropdown properties
        self.planet_orbital_period_dropdown_rect = pygame.Rect(self.width - self.customization_panel_width + 50, 600, self.customization_panel_width - 100, 30)
        self.planet_orbital_period_dropdown_active = False
        self.planet_orbital_period_dropdown_options = [
            ("Mercury", 88),
            ("Venus", 225),
            ("Earth", 365.25),
            ("Mars", 687),
            ("Jupiter", 4333),
            ("Saturn", 10759),
            ("Uranus", 30687),
            ("Neptune", 60190),
            ("Custom", None)
        ]
        self.planet_orbital_period_dropdown_selected = "Earth"
        self.planet_orbital_period_dropdown_visible = False
        self.show_custom_orbital_period_input = False
        self.orbital_period_input_active = False
        self.orbital_period_input_text = ""
        self.orbital_period_min = 1.0
        self.orbital_period_max = 100000.0
        
        # Planet stellar flux dropdown properties
        self.planet_stellar_flux_dropdown_rect = pygame.Rect(self.width - self.customization_panel_width + 50, 660, self.customization_panel_width - 100, 30)
        self.planet_stellar_flux_dropdown_active = False
        self.planet_stellar_flux_dropdown_options = [
            ("Mercury", 6.67),
            ("Venus", 1.91),
            ("Earth", 1.0),
            ("Mars", 0.43),
            ("Jupiter", 0.037),
            ("Saturn", 0.011),
            ("Uranus", 0.0029),
            ("Neptune", 0.0015),
            ("Custom", None)
        ]
        self.planet_stellar_flux_dropdown_selected = "Earth"
        self.planet_stellar_flux_dropdown_visible = False
        self.show_custom_stellar_flux_input = False
        self.stellar_flux_input_active = False
        self.stellar_flux_input_text = ""
        self.stellar_flux_min = 0.001
        self.stellar_flux_max = 100.0
        
        # Planet density dropdown properties
        self.planet_density_dropdown_rect = pygame.Rect(self.width - self.customization_panel_width + 50, 720, self.customization_panel_width - 100, 30)
        self.planet_density_dropdown_active = False
        self.planet_density_dropdown_options = [
            ("Saturn", 0.69),
            ("Jupiter", 1.33),
            ("Neptune", 1.64),
            ("Uranus", 1.27),
            ("Mars", 3.93),
            ("Mercury", 5.43),
            ("Venus", 5.24),
            ("Earth", 5.51),
            ("Custom", None)
        ]
        self.planet_density_dropdown_selected = "Earth"
        self.planet_density_dropdown_visible = False
        self.show_custom_planet_density_input = False
        self.planet_density_input_text = ""
        self.planet_density_min = 0.1
        self.planet_density_max = 20.0
        
        # --- NEW CUSTOM INPUT MODAL STATE ---
        self.pending_custom_field = None      # e.g., "mass", "radius", "semi_major_axis", "star_mass"
        self.pending_custom_body_id = None    # ID of the body being edited
        self.pending_custom_value = None      # Validated value to be applied
        self.show_custom_modal = False        # Whether to show the modal
        self.active_tooltip = None           # Current tooltip text to render
        self.climate_info_icon_rect = None   # Rect for climate/axial info (i) icon
        self.climate_info_pinned = False     # Whether the climate tooltip is pinned open
        self.custom_modal_text = ""           # Current text in the input box
        self.custom_modal_error = None        # Error message if validation fails
        self.custom_modal_title = ""          # Title of the modal
        self.custom_modal_helper = ""         # Helper text with bounds
        self.custom_modal_min = 0.0           # Physical minimum
        self.custom_modal_max = 100.0         # Physical maximum
        self.custom_modal_unit = ""           # Unit label (e.g., M⊕, AU, M☉)
        self.previous_dropdown_selection = None # Store selection before 'Custom' was chosen
        self.apply_btn_rect = pygame.Rect(0,0,0,0)  # Initialize to avoid hasattr issues
        self.cancel_btn_rect = pygame.Rect(0,0,0,0) # Initialize to avoid hasattr issues
        # ------------------------------------
        
        # Initialize sandbox with auto-spawn (after all properties are initialized)
        self.initSandbox()
    
    def _slider_from_scale(self, scale: float) -> float:
        """Map a time scale (0–5) to a normalized slider position (0–1) using piecewise segments."""
        scale = max(0.0, min(5.0, scale))
        points = [
            (0.0, 0.0),
            (0.1, 0.1),
            (0.25, 0.25),
            (0.5, 1.0),
            (0.75, 2.0),
            (1.0, 5.0),
        ]
        for i in range(1, len(points)):
            x0, y0 = points[i - 1]
            x1, y1 = points[i]
            if y0 <= scale <= y1:
                if y1 == y0:
                    return x0
                t = (scale - y0) / (y1 - y0)
                return x0 + t * (x1 - x0)
        return 1.0
    
    def _scale_from_slider(self, slider_pos: float) -> float:
        """Map a normalized slider position (0–1) to time scale (0–5) using piecewise segments."""
        slider_pos = max(0.0, min(1.0, slider_pos))
        points = [
            (0.0, 0.0),
            (0.1, 0.1),
            (0.25, 0.25),
            (0.5, 1.0),
            (0.75, 2.0),
            (1.0, 5.0),
        ]
        for i in range(1, len(points)):
            x0, y0 = points[i - 1]
            x1, y1 = points[i]
            if x0 <= slider_pos <= x1:
                if x1 == x0:
                    return y0
                t = (slider_pos - x0) / (x1 - x0)
                return y0 + t * (y1 - y0)
        return 5.0
    
    def _get_time_controls_layout(self):
        """Compute rects and positions for the unified Time Control Box (bottom center)."""
        box_width = 450
        box_height = 100
        box_x = (self.width - box_width) // 2
        box_y = self.height - box_height - 20
        self.time_panel_rect = pygame.Rect(box_x, box_y, box_width, box_height)
        
        btn_size = 40
        gap = 25
        center_x = box_x + box_width // 2
        btn_y = box_y + 55
        
        self.pause_play_btn_rect = pygame.Rect(center_x - btn_size // 2, btn_y, btn_size, btn_size)
        self.decel_btn_rect = pygame.Rect(self.pause_play_btn_rect.left - btn_size - gap, btn_y, btn_size, btn_size)
        self.accel_btn_rect = pygame.Rect(self.pause_play_btn_rect.right + gap, btn_y, btn_size, btn_size)
        
        return {
            "panel_rect": self.time_panel_rect,
            "decel_rect": self.decel_btn_rect,
            "pause_play_rect": self.pause_play_btn_rect,
            "accel_rect": self.accel_btn_rect
        }
    
    def world_to_screen(self, pos):
        """Convert world coordinates to screen coordinates using camera."""
        return [
            pos[0] * self.camera_zoom + self.camera_offset[0],
            pos[1] * self.camera_zoom + self.camera_offset[1],
        ]
    
    def screen_to_world(self, pos):
        # Instrumentation: Coordinate transform
        if DEBUG_ORBIT:
            trace(f"COORD_TRANSFORM screen_to_world pos={pos}")
        """Convert screen coordinates to world coordinates using camera."""
        return [
            (pos[0] - self.camera_offset[0]) / self.camera_zoom,
            (pos[1] - self.camera_offset[1]) / self.camera_zoom,
        ]
    
    def reset_camera(self):
        """Reset camera to default view (zoom=0.6, centered in visible area). Does not affect physics or object positions."""
        self.camera_zoom = 0.6
        self.camera_offset = [40.0, 160.0]
        # Clear orbit caches so they recalculate at new zoom
        self.orbit_screen_cache.clear()
        self.orbit_grid_screen_cache.clear()
        self.last_zoom_for_orbits = self.camera_zoom
        # Cancel any active focus
        self.camera_focus["active"] = False

    def update_camera(self):
        """Smoothly interpolate camera zoom and offset towards target focus"""
        if not self.camera_focus["active"]:
            return
            
        target_zoom = self.camera_focus["target_zoom"]
        target_body_id = self.camera_focus["target_body_id"]
        
        # Get target world position (follow body if it exists)
        if target_body_id and target_body_id in self.bodies_by_id:
            target_world_pos = self.bodies_by_id[target_body_id]["position"]
            self.camera_focus["target_world_pos"] = target_world_pos.copy()
        else:
            target_world_pos = self.camera_focus["target_world_pos"]
            
        if target_world_pos is None or target_zoom is None:
            self.camera_focus["active"] = False
            return
            
        # Smooth interpolation factor (tunable)
        INTERP_FACTOR = 0.08
        
        # 1. Smoothly interpolate zoom
        zoom_diff = target_zoom - self.camera_zoom
        self.camera_zoom += zoom_diff * INTERP_FACTOR
        
        # 2. Smoothly interpolate offset to center target_world_pos
        # target_offset = screen_center - (world_pos * zoom)
        screen_center_x = self.width / 2
        screen_center_y = self.height / 2
        
        target_offset_x = screen_center_x - target_world_pos[0] * self.camera_zoom
        target_offset_y = screen_center_y - target_world_pos[1] * self.camera_zoom
        
        offset_diff_x = target_offset_x - self.camera_offset[0]
        offset_diff_y = target_offset_y - self.camera_offset[1]
        
        self.camera_offset[0] += offset_diff_x * INTERP_FACTOR
        self.camera_offset[1] += offset_diff_y * INTERP_FACTOR
        
        # Snap to target if very close
        if abs(zoom_diff) < 0.005 and math.hypot(offset_diff_x, offset_diff_y) < 0.5:
            self.camera_zoom = target_zoom
            self.camera_offset[0] = target_offset_x
            self.camera_offset[1] = target_offset_y
            self.camera_focus["active"] = False
            
        # Invalidate orbit caches during movement
        self.orbit_screen_cache.clear()
        self.orbit_grid_screen_cache.clear()
        self.last_zoom_for_orbits = self.camera_zoom
    
    def _cached_screen_points(self, name, points, cache_dict):
        """Transform a list of world-space points to screen-space with zoom caching."""
        if not points:
            return []
        cached = cache_dict.get(name)
        if cached:
            prev_zoom, cached_pts = cached
            if abs(self.camera_zoom - prev_zoom) / max(prev_zoom, 1e-6) <= 0.02:
                return cached_pts
        screen_pts = [np.array(self.world_to_screen(p)) for p in points]
        cache_dict[name] = (self.camera_zoom, screen_pts)
        return screen_pts
    
    def _any_dropdown_active(self) -> bool:
        """Return True if any dropdown overlay is active/visible."""
        return (
            self.planet_preset_dropdown_visible
            or self.planet_dropdown_visible
            or self.moon_dropdown_visible
            or self.star_mass_dropdown_visible
            or self.luminosity_dropdown_visible
            or self.planet_age_dropdown_visible
            or self.star_age_dropdown_visible
            or self.moon_age_dropdown_visible
            or self.moon_radius_dropdown_visible
            or self.moon_orbital_distance_dropdown_visible
            or self.moon_orbital_period_dropdown_visible
            or self.moon_temperature_dropdown_visible
            or self.moon_gravity_dropdown_visible
            or self.spectral_class_dropdown_visible
            or self.radius_dropdown_visible
            or self.activity_dropdown_visible
            or self.metallicity_dropdown_visible
            or self.planet_radius_dropdown_visible
            or self.planet_temperature_dropdown_visible
            or self.planet_atmosphere_dropdown_visible
            or self.planet_gravity_dropdown_visible
            or self.planet_orbital_distance_dropdown_visible
            or self.planet_orbital_eccentricity_dropdown_visible
            or self.planet_orbital_period_dropdown_visible
            or self.planet_stellar_flux_dropdown_visible
            or self.planet_density_dropdown_visible
        )
    
    def handle_time_controls_input(self, event, mouse_pos) -> bool:
        """Handle mouse interaction with the unified Time Control Box."""
        if self._any_dropdown_active():
            return False
        
        layout = self._get_time_controls_layout()
        
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            # 1. Decelerate (⏪)
            if layout["decel_rect"].collidepoint(mouse_pos):
                if self.current_scale_index > 0:
                    self.current_scale_index -= 1
                    self.time_scale = TIME_SCALES[self.current_scale_index]
                    self.paused = False  # Resume on speed change
                return True
            
            # 2. Accelerate (⏩)
            if layout["accel_rect"].collidepoint(mouse_pos):
                if self.current_scale_index < len(TIME_SCALES) - 1:
                    self.current_scale_index += 1
                    self.time_scale = TIME_SCALES[self.current_scale_index]
                    self.paused = False  # Resume on speed change
                return True
            
            # 3. Pause/Play (⏸ / ▶)
            if layout["pause_play_rect"].collidepoint(mouse_pos):
                if self.paused:
                    # Resume
                    self.paused = False
                    self.current_scale_index = self.last_pre_pause_scale_index
                    self.time_scale = TIME_SCALES[self.current_scale_index]
                else:
                    # Pause
                    self.last_pre_pause_scale_index = self.current_scale_index
                    self.paused = True
                    self.time_scale = 0.0
                return True
            
            # Also consume clicks on the panel background
            if layout["panel_rect"].collidepoint(mouse_pos):
                return True
        
        return False
    
    def _format_time_scale(self, scale):
        """Format the simulation speed for display."""
        if scale == 0: return "Paused"
        if scale < 1.0: return f"1s = {scale:.1f}s"
        if scale < 60: return f"1s = {int(scale)}s"
        if scale < 3600: return f"1s = {int(scale/60)} min"
        if scale < 86400: return f"1s = {int(scale/3600)} hour"
        if scale < 2592000: return f"1s = {int(scale/86400)} day"
        return f"1s = {int(scale/2592000)} month"

    def draw_time_controls(self, surface):
        """Draw the unified Time Control Box with explicit scaling and global clock."""
        layout = self._get_time_controls_layout()
        panel_rect = layout["panel_rect"]
        
        # 1. Draw Box Background
        bg_surface = pygame.Surface(panel_rect.size, pygame.SRCALPHA)
        bg_surface.fill((0, 0, 0, 180))  # Semi-transparent black
        surface.blit(bg_surface, panel_rect.topleft)
        pygame.draw.rect(surface, self.WHITE, panel_rect, 2, border_radius=10)
        
        # 2. Draw Speed and Clock Info
        # Speed Mapping
        active_scale = TIME_SCALES[self.current_scale_index] if not self.paused else 0.0
        speed_mapping = self._format_time_scale(active_scale)
        speed_text = self.subtitle_font.render(f"Speed: {speed_mapping}", True, self.YELLOW)
        surface.blit(speed_text, (panel_rect.left + 20, panel_rect.top + 10))
        
        # Clock
        total_seconds = int(self.simulation_time_seconds)
        years = total_seconds // int(self.SECONDS_PER_YEAR)
        remaining = total_seconds % int(self.SECONDS_PER_YEAR)
        days = remaining // int(self.SECONDS_PER_DAY)
        remaining %= int(self.SECONDS_PER_DAY)
        h = remaining // 3600
        m = (remaining % 3600) // 60
        s = remaining % 60
        clock_str = f"Time: Year {years} | Day {days} | {h:02d}:{m:02d}:{s:02d}"
        clock_text = self.subtitle_font.render(clock_str, True, self.WHITE)
        surface.blit(clock_text, (panel_rect.left + 20, panel_rect.top + 32))
        
        # 3. Draw Controls: ⏪   ⏸ / ▶   ⏩
        # Decel (⏪)
        d_rect = layout["decel_rect"]
        pygame.draw.rect(surface, self.GRAY, d_rect, border_radius=5)
        d_label = self.font.render("<<", True, self.WHITE)
        surface.blit(d_label, d_label.get_rect(center=d_rect.center))
        
        # Pause/Play (⏸ / ▶)
        p_rect = layout["pause_play_rect"]
        pygame.draw.rect(surface, self.ACTIVE_TAB_COLOR if self.paused else self.GRAY, p_rect, border_radius=5)
        p_label_str = ">" if self.paused else "||"
        p_label = self.font.render(p_label_str, True, self.WHITE)
        surface.blit(p_label, p_label.get_rect(center=p_rect.center))
        
        # Accel (⏩)
        a_rect = layout["accel_rect"]
        pygame.draw.rect(surface, self.GRAY, a_rect, border_radius=5)
        a_label = self.font.render(">>", True, self.WHITE)
        surface.blit(a_label, a_label.get_rect(center=a_rect.center))
    
    def draw_reset_button(self):
        """Draw the Reset View button in screen space (not affected by camera)."""
        mouse_pos = pygame.mouse.get_pos()
        is_hovering = self.reset_view_button.collidepoint(mouse_pos)
        
        # Button background color (brighter on hover)
        if is_hovering:
            bg_color = (230, 230, 230, 200)
        else:
            bg_color = (200, 200, 200, 180)
        
        # Draw button background
        button_surface = pygame.Surface((self.reset_view_button.width, self.reset_view_button.height), pygame.SRCALPHA)
        button_surface.fill(bg_color)
        self.screen.blit(button_surface, self.reset_view_button.topleft)
        
        # Draw white border
        pygame.draw.rect(self.screen, (255, 255, 255), self.reset_view_button, 2, border_radius=6)
        
        # Draw button text
        label = self.subtitle_font.render("Reset View", True, (255, 255, 255))
        text_rect = label.get_rect(center=self.reset_view_button.center)
        self.screen.blit(label, text_rect)
    
    def draw_scale_indicator(self):
        """Draw a zoom-aware distance scale indicator in the bottom-left corner."""
        # Calculate pixels per AU at current zoom
        pixels_per_au = self.scale * self.camera_zoom
        
        # Target width range: 80 to 120 pixels for the scale bar
        target_min = 80
        target_max = 120
        
        # Standard astronomical unit scales (1, 2, 5 pattern for readability)
        candidate_scales = []
        for i in range(-5, 6): # from 10^-5 to 10^5 AU
            base = 10**i
            candidate_scales.extend([base, base * 2, base * 5])
        candidate_scales.sort()
        
        # Find the best scale that fits our target pixel width
        selected_au = candidate_scales[0]
        for au in candidate_scales:
            width_px = au * pixels_per_au
            if width_px >= target_min:
                selected_au = au
                # If this scale is already too big, check if the previous one was better
                if width_px > target_max:
                    prev_au = candidate_scales[max(0, candidate_scales.index(au)-1)]
                    prev_width = prev_au * pixels_per_au
                    target_mid = (target_min + target_max) / 2
                    if abs(prev_width - target_mid) < abs(width_px - target_mid):
                        selected_au = prev_au
                break
            selected_au = au
            
        width_px = selected_au * pixels_per_au
        
        # Position: Bottom-left corner with margins
        margin_x = 25
        margin_y = 25
        
        # Adjust Y position to be above the instruction text if it's visible at the bottom
        y_pos = self.height - margin_y - 45
        x_start = margin_x
        
        # Prepare label text
        if selected_au >= 1000:
            label_text = f"{selected_au:,.0f} AU"
        elif selected_au >= 1:
            label_text = f"{selected_au:g} AU"
        else:
            # Use scientific notation for very small values, or just :g
            label_text = f"{selected_au:g} AU"
            
        # UI Rendering
        tick_height = 6
        label_surface = self.subtitle_font.render(label_text, True, self.WHITE)
        label_rect = label_surface.get_rect(midtop=(x_start + width_px/2, y_pos + tick_height + 4))
        
        # Draw background for readability
        bg_rect = label_rect.inflate(16, 8)
        scale_rect = pygame.Rect(x_start - 8, y_pos - tick_height - 4, width_px + 16, tick_height * 2 + 8)
        bg_rect = bg_rect.union(scale_rect)
        
        bg_surface = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
        bg_surface.fill((0, 0, 0, 140)) # Semi-transparent black
        self.screen.blit(bg_surface, bg_rect.topleft)
        
        # Draw scale bar (white line with end ticks)
        line_color = self.WHITE
        line_thickness = 2
        pygame.draw.line(self.screen, line_color, (x_start, y_pos), (x_start + width_px, y_pos), line_thickness)
        pygame.draw.line(self.screen, line_color, (x_start, y_pos - tick_height), (x_start, y_pos + tick_height), line_thickness)
        pygame.draw.line(self.screen, line_color, (x_start + width_px, y_pos - tick_height), (x_start + width_px, y_pos + tick_height), line_thickness)
        
        # Draw label
        self.screen.blit(label_surface, label_rect)

    def calculate_hitbox_radius(self, obj_type: str, visual_radius: float) -> float:
        """
        Calculate hitbox radius for a celestial object based on its type and visual radius.
        
        Args:
            obj_type: "star", "planet", or "moon"
            visual_radius: The visual radius in pixels
            
        Returns:
            Hitbox radius in pixels (visual_radius * scale_factor)
        """
        scale_factors = {
            "star": 1.3,
            "planet": 1.8,
            "moon": 6.0  # Increased significantly for better clickability
        }
        scale_factor = scale_factors.get(obj_type, 1.0)
        hitbox_radius = visual_radius * scale_factor
        
        # Ensure a minimum hitbox size for moons (minimum 15px radius / 30px diameter)
        if obj_type == "moon":
            return max(hitbox_radius, 15.0)
        return hitbox_radius
    
    def calculate_star_visual_radius(self, star_body: dict) -> float:
        """
        Calculate visual radius in pixels for a star based on its stellar radius (R☉).
        Formula: star_radius_px = SUN_RADIUS_PX * stellar_radius_solar
        
        Args:
            star_body: Star body dictionary with "radius" in Solar radii (R☉)
            
        Returns:
            Visual radius in pixels
        """
        stellar_radius_solar = star_body.get("radius", 1.0)
        # Authoritative Formula: star_radius_px = BASE_STAR_RADIUS_PX * stellar_radius_solar
        return SUN_RADIUS_PX * stellar_radius_solar

    def get_visual_orbit_radius(self, semi_major_axis_au: float) -> float:
        """
        Calculate visual orbit radius in pixels using perceptual scaling.
        This matches the scaling method used for planet radii (perceptual power law).
        
        Formula: visual_radius_px = AU_TO_PX * semi_major_axis_au ^ PLANET_DISTANCE_SCALE_POWER
        
        Args:
            semi_major_axis_au: Orbital distance in Astronomical Units (AU)
            
        Returns:
            Visual orbit radius in pixels for rendering
        """
        return float(AU_TO_PX * math.pow(max(0.0, semi_major_axis_au), PLANET_DISTANCE_SCALE_POWER))

    def get_visual_moon_orbit_radius(self, semi_major_axis_au: float) -> float:
        """
        Calculate visual orbit radius for a moon using perceptual scaling.
        Matches the planet scaling method (perceptual power law) but normalized to Moon standards.
        
        Formula: visual_radius_px = MOON_ORBIT_PX * (semi_major_axis_au / MOON_ORBIT_AU) ^ PLANET_DISTANCE_SCALE_POWER
        
        Args:
            semi_major_axis_au: Orbital distance in Astronomical Units (AU)
            
        Returns:
            Visual orbit radius in pixels relative to parent planet
        """
        if semi_major_axis_au <= 0:
            return 0.0
        # Normalize to MOON_ORBIT_AU so Earth-Moon distance is always MOON_ORBIT_PX
        normalization = semi_major_axis_au / MOON_ORBIT_AU
        return float(MOON_ORBIT_PX * math.pow(normalization, PLANET_DISTANCE_SCALE_POWER))

    def calculate_planet_visual_radius(self, planet_body: dict, placed_bodies: list) -> float:
        """
        Calculate normalized visual radius for a planet using NASA-style perceptual scaling.
        
        This function normalizes planet radius relative to parent star and applies
        square-root scaling to preserve ordering while compressing range. This ensures
        Jupiter (11 R⊕) always appears smaller than the Sun (109 R⊕) while maintaining
        scientifically accurate relative ratios.
        
        Args:
            planet_body: Planet body dictionary with "radius" in Earth radii (R⊕)
            placed_bodies: List of all placed bodies to find parent star
            
        Returns:
            Visual radius in pixels (clamped between 4 and star_radius_px * 0.85)
        """
        # Get planet radius in Earth radii (R⊕)
        planet_radius_R_earth = planet_body.get("radius", 1.0)
        
        # New Perceptual Scaling Formula: map Earth (1.0 R⊕) to EARTH_RADIUS_PX (21px)
        # Use a gentler power than 0.5 to keep range compressed but distinguishable
        visual_radius = EARTH_RADIUS_PX * math.pow(planet_radius_R_earth, 0.4)
        
        # Find parent star for max clamping
        parent_star = planet_body.get("parent_obj")
        if parent_star is None:
            # Fallback: find nearest star
            stars = [b for b in placed_bodies if b.get("type") == "star"]
            if stars:
                planet_pos = planet_body.get("position", np.array([0, 0]))
                parent_star = min(stars, key=lambda s: np.linalg.norm(s.get("position", np.array([0, 0])) - planet_pos))
        
        if parent_star:
            star_visual_radius = self.calculate_star_visual_radius(parent_star)
        else:
            # Fallback to Sun-sized star visual radius (48px)
            star_visual_radius = SUN_RADIUS_PX
            
        # Clamp: min=MIN_PLANET_PX, max=star_radius_px * 0.85
        # Relaxed clamp allows planets to be larger and more visible in sandbox
        visual_radius = max(MIN_PLANET_PX, min(visual_radius, star_visual_radius * 0.85))
        
        return visual_radius
    
    def calculate_moon_visual_radius(self, moon_body: dict, placed_bodies: list) -> float:
        """
        Calculate normalized visual radius for a moon relative to its parent planet.
        
        Moons are stored in pixels (legacy), but we need to scale them relative to
        their parent planet to maintain correct size relationships.
        
        Args:
            moon_body: Moon body dictionary with "radius" in pixels and "parent_obj" reference
            placed_bodies: List of all placed bodies to find parent planet
            
        Returns:
            Visual radius in pixels (scaled relative to parent planet)
        """
        # Get moon's stored radius in pixels (legacy format)
        moon_radius_px = moon_body.get("radius", MOON_RADIUS_PX)
        
        # Find parent planet
        parent_planet = moon_body.get("parent_obj")
        if parent_planet is None:
            # Fallback: find nearest planet
            planets = [b for b in placed_bodies if b.get("type") == "planet"]
            if planets:
                moon_pos = moon_body.get("position", np.array([0, 0]))
                parent_planet = min(planets, key=lambda p: np.linalg.norm(p.get("position", np.array([0, 0])) - moon_pos))
        
        if parent_planet:
            # Get parent planet's visual radius (already normalized and scaled)
            parent_visual_radius = self.calculate_planet_visual_radius(parent_planet, placed_bodies)
            
            # Calculate moon radius in Earth radii
            # Use actual_radius in km if available, otherwise use default (Earth's Moon = 1737.4 km)
            EARTH_RADIUS_KM = 6371.0  # Earth's radius in km
            moon_actual_radius_km = moon_body.get("actual_radius", 1737.4)  # Default to Earth's Moon
            moon_radius_R_earth = moon_actual_radius_km / EARTH_RADIUS_KM
            
            # Get parent planet radius in Earth radii
            parent_radius_R_earth = parent_planet.get("radius", 1.0)  # Parent radius in R⊕
            
            # Calculate moon size relative to parent planet using perceptual scaling
            # Use a higher power (0.75) than planets (0.5) to keep moons from looking too large
            moon_relative_size = math.pow(moon_radius_R_earth / parent_radius_R_earth, 0.75)
            visual_radius = parent_visual_radius * moon_relative_size
            
            # Ensure moon is always smaller than parent and at least 3px for visibility
            visual_radius = max(3, min(visual_radius, parent_visual_radius * 0.9))
        else:
            # No parent planet found, use default moon size (but smaller than default planet)
            visual_radius = max(3, min(moon_radius_px, 6))
        
        return visual_radius
    
    def apply_planet_preset(self, preset_name: str):
        """
        Apply a Solar System planet preset to the currently selected planet.
        Uses update_selected_body_property to ensure per-object updates with no shared state.
        
        Args:
            preset_name: Name of the preset (e.g., "Mercury", "Earth", "Mars")
        """
        if not self.selected_body_id or self.selected_body_id not in self.bodies_by_id:
            print(f"ERROR: Cannot apply preset - no planet selected")
            return False
        
        if preset_name not in SOLAR_SYSTEM_PLANET_PRESETS:
            print(f"ERROR: Unknown preset name: {preset_name}")
            return False
        
        # Get canonical body from registry
        body = self.bodies_by_id[self.selected_body_id]
        
        if body.get("type") != "planet":
            print(f"ERROR: Preset can only be applied to planets")
            return False
        
        # Get preset data (deep copy to avoid shared references)
        preset = SOLAR_SYSTEM_PLANET_PRESETS[preset_name]
        
        # Apply all preset parameters using update_selected_body_property
        # This ensures per-object updates with no shared state
        self.update_selected_body_property("mass", preset["mass"], "mass")
        self.update_selected_body_property("radius", preset["radius"], "radius")
        self.update_selected_body_property("semiMajorAxis", preset["semiMajorAxis"], "semiMajorAxis")
        self.update_selected_body_property("greenhouse_offset", preset["greenhouse_offset"], "greenhouse_offset")
        self.update_selected_body_property("equilibrium_temperature", preset["equilibrium_temperature"], "equilibrium_temperature")
        self.update_selected_body_property("temperature", preset["temperature"], "temperature")
        self.update_selected_body_property("gravity", preset["gravity"], "gravity")
        self.update_selected_body_property("eccentricity", preset["eccentricity"], "eccentricity")
        self.update_selected_body_property("orbital_period", preset["orbital_period"], "orbital_period")
        self.update_selected_body_property("stellarFlux", preset["stellarFlux"], "stellarFlux")
        if "density" in preset:
            self.update_selected_body_property("density", preset["density"], "density")
        if "base_color" in preset:
            self.update_selected_body_property("base_color", preset["base_color"], "base_color")
        
        # CRITICAL: Update position using centralized function
        parent_star = body.get("parent_obj")
        if parent_star is None and body.get("parent"):
            parent_star = next((b for b in self.placed_bodies if b["name"] == body.get("parent")), None)
        if parent_star:
            self.compute_planet_position(body, parent_star)
        
        # Regenerate orbit grid for visualization
        self.generate_orbit_grid(body)
        self.clear_orbit_points(body)
        
        # Sync all UI dropdowns to reflect new values
        # This ensures UI controls are in sync with the planet's actual values
        self._sync_all_dropdown_labels(body)
        
        print(f"Applied preset '{preset_name}' to planet {body.get('name')}")
        return True
    
    def _perform_registry_update(self, body_id, key, value, debug_name=""):
        """
        Internal function to perform the actual dictionary update with all isolation checks.
        """
        if not body_id:
            print(f"ERROR: Cannot update {key} - no body id provided")
            return False
            
        if body_id not in self.bodies_by_id:
            print(f"ERROR: Body id {body_id} not found in registry")
            return False
            
        body = self.bodies_by_id[body_id]
        body_dict_id_before = id(body)
        
        # Isolation and shared-state checks
        if body.get("id") != body_id:
            raise AssertionError(f"CRITICAL: Body dict id mismatch! Expected {body_id}, got {body.get('id')}")
            
        # Verify body dict is unique in placed_bodies
        for other_body in self.placed_bodies:
            if other_body.get("id") != body_id and id(other_body) == body_dict_id_before:
                raise AssertionError(f"CRITICAL: Found shared dict for body {body_id} and {other_body.get('id')}")

        # Convert value to float if it's numeric
        if isinstance(value, (int, float)) or (hasattr(value, 'item') and not isinstance(value, str)):
            if hasattr(value, 'item'):
                value = float(value.item())
            else:
                value = float(value)
                
        # Perform update
        body[key] = value
        
        # Verify dict identity remains same
        if id(body) != body_dict_id_before:
            raise AssertionError(f"CRITICAL: Body dict identity changed during update!")
            
        return True

    def update_selected_body_property(self, key, value, debug_name=""):
        """
        Legacy wrapper for update_selected_body_property.
        Redirects to apply_parameter_change for physical and orbital parameters.
        """
        if key in ["mass", "density", "gravity", "radius", "actual_radius", "temperature", "luminosity", "semiMajorAxis", "orbital_period", "eccentricity", "greenhouse_offset"]:
            return self.apply_parameter_change(self.selected_body_id, key, value)
            
        # For non-physical parameters, use the raw update
        return self._perform_registry_update(self.selected_body_id, key, value, debug_name)

    def apply_parameter_change(self, body_id, parameter, value):
        """
        CENTRALIZED: Enforce Physical Parameter -> Sandbox Mapping
        Ensures that changing mass, density, and surface gravity produces
        scientifically correct and isolated effects.
        """
        if not body_id or body_id not in self.bodies_by_id:
            return False
            
        body = self.bodies_by_id[body_id]
        
        # 1. Validation & Pre-processing
        if hasattr(value, 'item'):
            value = float(value.item())
        else:
            try:
                value = float(value)
            except (ValueError, TypeError):
                return False

        # 2. Apply correctly based on parameter
        mass_changed = False
        if parameter == "mass":
            # Direct Mass Update
            self._perform_registry_update(body_id, "mass", value)
            mass_changed = True
            
            # --- NASA MAIN SEQUENCE COUPLING (STRICT SIMULATION) ---
            # If it's a star, changing mass automatically snaps it to the Main Sequence
            if body.get("type") == "star":
                M = value
                # 1. Luminosity-Mass Relation (Piecewise)
                if M < 0.43:
                    L = 0.23 * (M ** 2.3)
                elif M < 2.0:
                    L = M ** 4.0
                elif M < 55.0:
                    L = 1.4 * (M ** 3.5)
                else:
                    L = 32000.0 * M
                
                # 2. Radius-Mass Relation (Piecewise)
                if M < 1.0:
                    R = M ** 0.8
                else:
                    R = M ** 0.57
                
                # 3. Temperature (Derived from Stefan-Boltzmann)
                # T = T_sun * (L / R^2)^0.25
                T = 5800.0 * (L / (R ** 2)) ** 0.25
                
                # Update registry with Main Sequence values
                self._perform_registry_update(body_id, "luminosity", float(L))
                self._perform_registry_update(body_id, "radius", float(R))
                self._perform_registry_update(body_id, "temperature", float(T))
                
                # Trigger visual updates for the star
                self._update_derived_parameters(body)
                self._update_visual_size(body)
            
            # --- NASA PLANETARY MASS-RADIUS COUPLING ---
            elif body.get("type") == "planet":
                M = value # Earth masses
                # Using NASA Exoplanet Archive inspired power laws
                if M < 1.0:
                    # Small sub-Earths: slightly less compressed
                    R = M ** 0.3
                elif M < 10.0:
                    # Super-Earths/Terrestrial: R = M^0.27
                    R = M ** 0.27
                elif M < 318.0:
                    # Neptunes to Jupiters: Rapid growth as gas envelopes thicken
                    # Log-linear interpolation between 10M (approx 2.5R) and 318M (11.2R)
                    R = 2.5 * (M / 10.0) ** 0.45
                else:
                    # Brown Dwarfs / Super-Jupiters: Radius starts to shrink due to degeneracy
                    # R decreases slightly as mass increases
                    R = 11.2 * (M / 318.0) ** -0.08
                
                self._perform_registry_update(body_id, "radius", float(R))
                self._update_derived_parameters(body)
                self._update_visual_size(body)
            
            # --- NASA MOON MASS-RADIUS COUPLING ---
            elif body.get("type") == "moon":
                M = value # Earth masses
                # Moons are typically rock/ice mixtures. 
                # Scaling follows R = M^0.3 for small icy/rocky bodies
                R = M ** 0.3
                self._perform_registry_update(body_id, "radius", float(R))
                self._update_derived_parameters(body)
                self._update_visual_size(body)
            # -------------------------------------------------------
            
        elif parameter == "density":
            # Density Change -> Recompute Mass, Keep Radius
            self._perform_registry_update(body_id, "density", value)
            radius = body.get("actual_radius", body.get("radius", 1.0))
            # mass (M_earth) = density / 5.51 * (radius_R_earth)^3
            new_mass = (value / 5.51) * (radius ** 3)
            self._perform_registry_update(body_id, "mass", new_mass)
            mass_changed = True
            
        elif parameter == "gravity":
            # Gravity Change -> Recompute Mass, Keep Radius
            self._perform_registry_update(body_id, "gravity", value)
            radius = body.get("actual_radius", body.get("radius", 1.0))
            # mass (M_earth) = gravity / 9.81 * (radius_R_earth)^2
            new_mass = (value / 9.81) * (radius ** 2)
            self._perform_registry_update(body_id, "mass", new_mass)
            mass_changed = True
            
        elif parameter == "radius" or parameter == "actual_radius":
            # Radius Change (Geometric)
            self._perform_registry_update(body_id, parameter, value)
            # Update density and gravity based on new radius (keep mass constant)
            self._update_derived_parameters(body)
            # Update visual size
            self._update_visual_size(body)

        elif parameter == "temperature":
            # Temperature Change -> Recompute Luminosity if Star
            self._perform_registry_update(body_id, "temperature", value)
            if body.get("type") == "star":
                self._update_derived_parameters(body)

        elif parameter == "luminosity":
            # Luminosity Change -> Recompute Radius, Keep Temperature
            self._perform_registry_update(body_id, "luminosity", value)
            if body.get("type") == "star":
                # R = sqrt(L / (T/T_sun)^4)
                temp = body.get("temperature", 5800.0)
                new_radius = np.sqrt(value / ((temp / 5800.0) ** 4))
                self._perform_registry_update(body_id, "radius", new_radius)
                # Sync other parameters affected by radius
                self._update_derived_parameters(body)
                self._update_visual_size(body)

        elif parameter == "semiMajorAxis":
            # Orbital Distance Change
            self._perform_registry_update(body_id, "semiMajorAxis", value)
            # Enforce coupling to update Period
            self._apply_kepler_coupling(body, "semiMajorAxis")
            # Recompute physics (orbital speed, etc.)
            self.recompute_orbit_parameters(body, force_recompute=True)
            # CRITICAL: Update stellar flux and temperature when AU changes
            # This ensures ML model receives updated insolation values
            if body.get("type") in ["planet", "moon"]:
                self._update_stellar_flux(body)

        elif parameter == "orbital_period":
            # Orbital Period Change
            self._perform_registry_update(body_id, "orbital_period", value)
            # Enforce coupling to update AU
            self._apply_kepler_coupling(body, "orbital_period")
            # Recompute physics (orbital speed, etc.)
            self.recompute_orbit_parameters(body, force_recompute=True)
            # CRITICAL: Update stellar flux and temperature when orbital period changes
            # (which may have changed AU via coupling)
            if body.get("type") in ["planet", "moon"]:
                self._update_stellar_flux(body)

        elif parameter == "eccentricity":
            # Eccentricity Change
            self._perform_registry_update(body_id, "eccentricity", value)
            # Recompute physics (orbital speed, etc.)
            self.recompute_orbit_parameters(body, force_recompute=True)
            # CRITICAL: Update stellar flux and temperature when eccentricity changes
            # (affects seasonal flux variation and temperature swings)
            if body.get("type") in ["planet", "moon"]:
                self._update_stellar_flux(body)

        elif parameter == "greenhouse_offset":
            # Atmosphere / Greenhouse Change
            self._perform_registry_update(body_id, "greenhouse_offset", value)
            # Recalculate surface temperature: T_surface = T_eq + greenhouse_offset
            t_eq = body.get("equilibrium_temperature", 255.0)
            self._perform_registry_update(body_id, "temperature", float(t_eq + value))

        elif parameter == "age":
            # Age Change
            self._perform_registry_update(body_id, "age", value)
            if body.get("type") == "star":
                # NASA Gyrochronology: Older stars spin slower and are less active
                if value < 1.0: activity = "Hyperactive"
                elif value < 3.0: activity = "Active"
                elif value < 6.0: activity = "Moderate"
                else: activity = "Quiet"
                # Store directly in body and sync labels later
                body["activity"] = activity
                # Update visual feedback if needed
                self._apply_visual_parameter_feedback(body)

        if mass_changed:
            # Recompute other derived values (density or gravity)
            self._update_derived_parameters(body)
            # Recompute physics (orbital speed, etc.)
            self.recompute_orbit_parameters(body, force_recompute=True)
            
            # CRITICAL: Propagate mass change to children using Kepler coupling
            # Star mass change affects planets; Planet mass change affects moons
            for b in self.placed_bodies:
                if b.get("parent") == body.get("name") or b.get("parent_id") == body_id:
                    old_au = b.get("semiMajorAxis", 0.0)
                    self._apply_kepler_coupling(b, "parent_mass")
                    self.recompute_orbit_parameters(b, force_recompute=True)
                    
                    # Trigger visual correction if child AU changed due to star mass change
                    new_au = b.get("semiMajorAxis", 0.0)
                    if abs(new_au - old_au) > 1e-7:
                        self._start_orbital_correction(b, new_au)
                    else:
                        # Even if AU didn't change (driver=AU), we need to regenerate grid 
                        # because parent mass change affects orbital speed/rendering
                        if b["type"] != "star":
                            self.generate_orbit_grid(b)
                            self.clear_orbit_points(b)
                    
                    # CRITICAL: Sync child dropdown labels too!
                    self._update_stellar_flux(b)
                    self._sync_all_dropdown_labels(b)

        # 3. Apply Visual Feedback (tints, glows, etc.)
        self._apply_visual_parameter_feedback(body)
        
        # 4. Propagate star property changes (that affect luminosity) to children's flux
        # (Kepler coupling for orbits is already handled above)
        if body.get("type") == "star" and parameter in ["mass", "luminosity", "temperature", "radius", "actual_radius"]:
             for b in self.placed_bodies:
                if b.get("parent") == body.get("name") or b.get("parent_id") == body_id:
                    self._update_stellar_flux(b)
                    self._sync_all_dropdown_labels(b)

        # 5. Sync dropdown labels to reflect derived changes
        self._sync_all_dropdown_labels(body)
        
        # 5. Sync selected_body reference
        if self.selected_body_id == body_id:
            self.selected_body = body
            
        # 6. CRITICAL: Recalculate habitability AFTER all parameter updates complete
        # This includes all cascading updates (mass→radius→density, AU→flux→temp, etc.)
        # The ML model receives ALL current parameters, ensuring it reflects the complete
        # physical state after all dependent variables have been updated.
        if body.get("type") == "planet":
            # Temporary set selected body to this body to reuse _update_planet_scores
            old_selected = self.selected_body
            self.selected_body = body
            # _update_planet_scores reads ALL current parameters fresh from body dict
            self._update_planet_scores()
            self.selected_body = old_selected
        elif body.get("type") == "star":
            # When star properties change, recalculate for all child planets
            for child in self.placed_bodies:
                if child.get("type") == "planet" and (child.get("parent") == body.get("name") or child.get("parent_id") == body_id):
                    old_selected = self.selected_body
                    self.selected_body = child
                    # _update_planet_scores reads ALL current parameters fresh from body dict
                    self._update_planet_scores()
                    self.selected_body = old_selected
                    
        return True

    def _update_derived_parameters(self, body):
        """Update density and gravity based on current mass and radius"""
        mass = body.get("mass", 1.0)
        radius = body.get("actual_radius", body.get("radius", 1.0))
        
        if radius > 0:
            if body.get("type") == "star":
                # Star derived parameters
                # Density: rho = rho_sun * (mass/r^3) where rho_sun ~ 1.41 g/cm³
                body["density"] = 1.41 * (mass / (radius ** 3))
                # Gravity: g = g_sun * (mass/r^2) where g_sun ~ 274 m/s²
                body["gravity"] = 274.0 * (mass / (radius ** 2))
                
                # Luminosity: Stefan-Boltzmann Law in solar units: L = R^2 * (T/T_sun)^4
                # T_sun is ~5800 K in this codebase
                temp = body.get("temperature", 5800.0)
                luminosity = (radius ** 2) * ((temp / 5800.0) ** 4)
                body["luminosity"] = luminosity
                
                # CRITICAL: Update habitable zone when luminosity changes
                body["hz_surface"] = self.create_habitable_zone(body)
            else:
                # Planet/Moon derived parameters
                # Update density: rho = rho_earth * (mass/r^3)
                body["density"] = 5.51 * (mass / (radius ** 3))
                # Update gravity: g = g_earth * (mass/r^2)
                body["gravity"] = 9.81 * (mass / (radius ** 2))
                
                # --- NASA ATMOSPHERIC RETENTION & ESCAPE VELOCITY ---
                # ve = sqrt(2GM/R)
                # In Earth units: ve = 11.2 km/s * sqrt(mass_earth / radius_earth)
                v_escape = 11.186 * np.sqrt(max(0, mass / radius))
                body["escape_velocity"] = v_escape
                
                # Thermal Velocity check (approximate for water vapor/gases)
                # If ve < 6 * v_thermal, atmosphere is lost over geological time
                t_surf = body.get("temperature", 288.0)
                v_thermal = 0.157 * np.sqrt(t_surf / 18.0) # v_rms for H2O in km/s
                
                if v_escape < 6.0 * v_thermal:
                    body["atmosphere_stripped"] = True
                    # Automatically reduce greenhouse if it's too high for the gravity
                    if body.get("greenhouse_offset", 0.0) > 50.0:
                        body["greenhouse_offset"] = 10.0 # Force thin atmosphere
                else:
                    body["atmosphere_stripped"] = False
                
                # --- HILL SPHERE STABILITY (Triggered by Planet Mass Change) ---
                if body.get("type") == "planet" and body.get("hill_radius_au"):
                    # Check stability of any orbiting moons
                    r_hill_au = body["hill_radius_au"]
                    for b in self.placed_bodies:
                        if b.get("parent_id") == body.get("id"):
                            m_au = b.get("semiMajorAxis", 0.0)
                            if m_au > 0.5 * r_hill_au:
                                b["is_unstable"] = True
                            else:
                                b["is_unstable"] = False
                            self._sync_all_dropdown_labels(b)
                # ----------------------------------------------------------------
                
        # Always attempt to update stellar flux if it's a planet or moon
        if body.get("type") != "star":
            self._update_stellar_flux(body)

    def _update_stellar_flux(self, body):
        """
        Update Stellar Flux (F) for a planet/moon based on parent star's luminosity (L) 
        and orbital distance (d).
        Formula: Mean F = L / (4 * pi * a^2 * sqrt(1 - e^2))
        Also updates the planet's equilibrium temperature based on NASA thermal models.
        """
        if not body or body.get("type") == "star":
            return
            
        # Get parent star's luminosity
        # Traverse up to find the ultimate star (handles moons orbiting planets)
        curr = body
        parent_star = None
        while curr:
            p = curr.get("parent_obj")
            if not p and curr.get("parent"):
                p = next((b for b in self.placed_bodies if b["name"] == curr.get("parent")), None)
            if p and p.get("type") == "star":
                parent_star = p
                break
            curr = p
            if not curr: break
            
        if parent_star and parent_star.get("type") == "star":
            L = parent_star.get("luminosity", 1.0)
            
            # Use planet's distance for moon's flux
            # If body is a moon, find its parent planet's semiMajorAxis
            dist_obj = body
            parent_p = None
            if body.get("type") == "moon":
                parent_p = body.get("parent_obj")
                if not parent_p and body.get("parent"):
                    parent_p = next((b for b in self.placed_bodies if b["name"] == body.get("parent")), None)
                if parent_p and parent_p.get("type") == "planet":
                    dist_obj = parent_p
            
            a = dist_obj.get("semiMajorAxis", 1.0)
            e = float(body.get("eccentricity", 0.0))
            
            if a > 0:
                # 1. Update Mean Stellar Flux (Earth units)
                flux_mean = L / (4 * np.pi * (a ** 2) * np.sqrt(max(0.01, 1 - e ** 2)))
                body["stellarFlux"] = float(flux_mean)
                
                # 2. Peak and Min Flux (Perihelion and Aphelion)
                flux_max = L / (4 * np.pi * (a ** 2) * (1 - e) ** 2)
                flux_min = L / (4 * np.pi * (a ** 2) * (1 + e) ** 2)
                
                # 3. Seasonal Flux Variation Amplitude (%)
                variation_pct = ((flux_max - flux_min) / (2 * flux_mean)) * 100
                body["seasonal_flux_variation"] = float(variation_pct)
                
                # 4. Climate Regime
                if variation_pct < 5: regime = "Stable"
                elif variation_pct < 25: regime = "Seasonal"
                else: regime = "Extreme"
                body["climate_regime"] = regime
                
                # 5. Update UI label for flux
                body["planet_stellar_flux_dropdown_selected"] = f"Flux: {self._format_value(flux_mean, 'EFU')}"
                
                # 6. Update Equilibrium Temperature (NASA Planetary Bond Albedo formula)
                density = body.get("density", 5.5)
                albedo = 0.4 if density < 2.0 else 0.15
                t_eq_mean = 255.0 * (L / (a ** 2)) ** 0.25 * ((1.0 - albedo) / 0.7) ** 0.25
                body["equilibrium_temperature"] = float(t_eq_mean)
                
                # 7. Seasonal Temperature Swing (Delta T)
                t_max = t_eq_mean * (flux_max / flux_mean) ** 0.25
                t_min = t_eq_mean * (flux_min / flux_mean) ** 0.25
                body["seasonal_temp_swing"] = float(t_max - t_min)
                
                # --- NEW: TIDAL HEATING CONTRIBUTION ---
                # NASA Proxy: T_tidal ~ C * (M_parent^0.5 * R_body^1.25 * e^0.5) / a^1.5
                parent_obj = body.get("parent_obj")
                if not parent_obj and body.get("parent"):
                    parent_obj = next((b for b in self.placed_bodies if b["name"] == body.get("parent")), None)
                
                if parent_obj:
                    # M_parent in Earth masses
                    if parent_obj.get("type") == "star":
                        m_p = parent_obj.get("mass", 1.0) * 333000.0
                    else:
                        m_p = parent_obj.get("mass", 1.0)
                    
                    r_b = body.get("actual_radius", body.get("radius", 1.0))
                    
                    # Compute tidal heating delta (K)
                    if a > 0 and e > 0:
                        t_tidal = 0.1 * ((m_p**0.5) * (r_b**1.25) * (e**0.5)) / (a**1.5)
                        t_tidal = min(2000.0, t_tidal) # Safety clamp
                    else:
                        t_tidal = 0.0
                else:
                    t_tidal = 0.0
                
                body["tidal_heating_delta"] = float(t_tidal)
                # ---------------------------------------

                # 8. Update Surface Temperature (T_eq + Greenhouse Effect + Tidal Heating)
                greenhouse = body.get("greenhouse_offset", 0.0)
                body["temperature"] = float(t_eq_mean + greenhouse + t_tidal)
                
                # --- NASA TIDAL LOCKING CHECK ---
                locking_radius = 0.2 * (L ** 0.5)
                is_locked = a < locking_radius
                body["is_tidally_locked"] = is_locked
                
                # --- NASA HILL SPHERE STABILITY ---
                if body.get("type") == "planet":
                    m_planet = body.get("mass", 1.0)
                    M_star = parent_star.get("mass", 1.0)
                    r_hill_au = a * (m_planet / (3.0 * 333000.0 * M_star))**(1/3)
                    body["hill_radius_au"] = float(r_hill_au)
                    
                    # Check stability of any orbiting moons
                    for b in self.placed_bodies:
                        if b.get("parent_id") == body.get("id"):
                            m_au = b.get("semiMajorAxis", 0.0)
                            if m_au > 0.5 * r_hill_au:
                                b["is_unstable"] = True
                            else:
                                b["is_unstable"] = False
                
                # --- NEW: AXIAL TILT STABILITY INFERENCE ---
                # Factor 1: Stabilizing Moon
                # NASA rule: A large moon (mass > 1% of planet) within 20% of Hill radius stabilizes axial tilt
                has_stabilizing_moon = False
                if body.get("type") == "planet":
                    m_planet = body.get("mass", 1.0)
                    r_hill = body.get("hill_radius_au", 0.0)
                    for b in self.placed_bodies:
                        if b.get("parent_id") == body.get("id"):
                            m_moon = b.get("mass", 0.0)
                            d_moon = b.get("semiMajorAxis", 0.0)
                            # Earth mass units for m_moon and m_planet
                            if m_moon > 0.01 * m_planet and d_moon < 0.2 * r_hill:
                                has_stabilizing_moon = True
                                break
                
                # Inference Logic
                if is_locked:
                    stability = "Locked / Low"
                    contrast = 0.1 # Minimal
                elif has_stabilizing_moon and e < 0.1:
                    stability = "Stable"
                    contrast = 0.3 # Earth-like
                elif not has_stabilizing_moon and (e > 0.05 or body.get("type") == "moon"):
                    stability = "Marginal"
                    contrast = 0.6 # Mars-like
                elif e > 0.2:
                    stability = "Chaotic"
                    contrast = 0.9 # Extreme exoplanet
                else:
                    stability = "Marginal"
                    contrast = 0.5
                
                body["axial_stability"] = stability
                body["seasonal_contrast_score"] = float(contrast)
                
                # Trigger habitability update
                if body.get("type") == "planet":
                    old_selected = self.selected_body
                    self.selected_body = body
                    self._update_planet_scores()
                    self.selected_body = old_selected
                
                # --- NEW: DISTANCE-SCALED ATMOSPHERIC EROSION ---
                # NASA Proxy: Mass Loss Rate ~ F_XUV / v_esc^2
                # F_XUV ~ Activity / a^2
                activity_map = {"Quiet": 1.0, "Moderate": 5.0, "Active": 25.0, "Hyperactive": 100.0}
                act_val = activity_map.get(parent_star.get("activity", "Quiet"), 1.0)
                
                v_esc = body.get("escape_velocity", 11.186)
                # Erosion intensity: scale relative to Earth (act=5 (moderate sun), a=1, v_esc=11.2)
                # Earth erosion index ~ 5 / 125 ~ 0.04
                erosion_index = (act_val / (a ** 2)) / (max(0.1, v_esc) ** 2)
                
                # Calibration:
                # Earth: index 0.04 -> Stable
                # Hot Jupiter: act=25, a=0.05, v_esc=60 -> (25/0.0025)/3600 = 10000/3600 = 2.7 -> Stable
                # M-dwarf Super-Earth: act=100, a=0.05, v_esc=15 -> (100/0.0025)/225 = 40000/225 = 177 -> Rapidly Stripped
                
                if erosion_index < 5.0:
                    erosion_status = "Stable"
                elif erosion_index < 100.0:
                    erosion_status = "Slowly Eroding"
                else:
                    erosion_status = "Rapidly Stripped"
                
                body["atmospheric_erosion_status"] = erosion_status
                # ---------------------------------------------
                # ---------------------------------------------

                # --- NEW: MAGNETIC FIELD INFERENCE ---
                # NASA Proxy: Field strength is a function of mass and rotation
                # Venus/Mercury are weak because of slow rotation or small cores
                
                # Default rotation: 1.0 day (Earth-like)
                # Venus-like: 243 days
                # Mercury-like: 58 days
                rot_days = body.get("rotation_period_days", 1.0)
                if is_locked:
                    rot_days = body.get("orbital_period", 365.25)
                
                # Special cases for Venus/Mercury presets
                preset_type = body.get("preset_type", "")
                if preset_type == "Venus" and "rotation_period_days" not in body:
                    rot_days = 243.0
                elif preset_type == "Mercury" and "rotation_period_days" not in body:
                    rot_days = 58.0
                
                # Magnetosphere Index ~ Mass^0.5 / RotationPeriod
                # Earth: 1.0 / 1.0 = 1.0
                # Venus: 1.0 / 243.0 = 0.004
                # Mercury: 0.1^0.5 / 58 = 0.3 / 58 = 0.005
                # Jupiter: 317^0.5 / 0.4 = 17.8 / 0.4 = 44.5
                mag_index = (max(0.01, body.get("mass", 1.0)) ** 0.5) / max(0.1, rot_days)
                
                if mag_index > 0.5:
                    mag_status = "Strong"
                elif mag_index > 0.01:
                    mag_status = "Weak"
                else:
                    mag_status = "None"
                
                body["magnetic_field_status"] = mag_status
                # Shielding warning if field is weak/none
                body["is_shielded"] = mag_status == "Strong"
                # -------------------------------------

                # Sync UI labels to reflect new calculated temperature/flux
                self._sync_all_dropdown_labels(body)
                
                return True
        return False

    def _update_visual_size(self, body):
        """Update hitboxes and visual references when radius changes"""
        if body["type"] == "star":
            visual_radius = self.calculate_star_visual_radius(body)
            for p in self.placed_bodies:
                if p.get("type") == "planet" and (p.get("parent") == body.get("name") or p.get("parent_id") == body.get("id")):
                    p_visual_radius = self.calculate_planet_visual_radius(p, self.placed_bodies)
                    p["hitbox_radius"] = float(self.calculate_hitbox_radius("planet", p_visual_radius))
        elif body["type"] == "planet":
            visual_radius = self.calculate_planet_visual_radius(body, self.placed_bodies)
            for m in self.placed_bodies:
                if m.get("type") == "moon" and (m.get("parent") == body.get("name") or m.get("parent_id") == body.get("id")):
                    m_visual_radius = self.calculate_moon_visual_radius(m, self.placed_bodies)
                    m["radius"] = float(m_visual_radius)
                    m["hitbox_radius"] = float(self.calculate_hitbox_radius("moon", m_visual_radius))
        else: # moon
            visual_radius = self.calculate_moon_visual_radius(body, self.placed_bodies)
            body["radius"] = float(visual_radius)
        
        body["hitbox_radius"] = float(self.calculate_hitbox_radius(body["type"], visual_radius))

    def _apply_visual_parameter_feedback(self, body):
        """Apply color tints and glows based on physical parameters"""
        if body.get("type") != "planet":
            return
            
        # Density Tints
        # <1.5: Gas giant (Pale blue)
        # 1.5–3: Ice (Cyan)
        # 3–5.5: Rocky (Neutral gray)
        # >6: Iron-rich (Dark slate)
        density = body.get("density", 5.51)
        if density < 1.5:
            body["density_tint"] = (173, 216, 230) # Pale Blue
        elif density < 3.0:
            body["density_tint"] = (0, 255, 255)   # Cyan
        elif density < 5.5:
            body["density_tint"] = (128, 128, 128) # Neutral Gray
        else:
            body["density_tint"] = (47, 79, 79)    # Dark Slate Gray
            
        # Gravity Visuals (Terminator shading / Atmosphere glow strength)
        # Higher gravity -> slightly stronger atmosphere/shading
        gravity = body.get("gravity", 9.81)
        body["gravity_visual_strength"] = min(2.0, gravity / 9.81)

        # --- NASA DENSITY-COLOR CORRELATION ---
        # Low density planets are likely gas giants, high density are rocky/metallic
        if not body.get("base_color_custom"): # Only auto-update if user hasn't set a custom color
            density = body.get("density", 5.51)
            if density < 2.0:
                # Gas giant colors: Tans, Pinks, Blues
                body["base_color"] = "#D2B48C" if density > 1.0 else "#E8D8A8"
            elif density < 4.0:
                # Icy/Rocky mix: Whites, Grays
                body["base_color"] = "#B0B0B0"
            else:
                # High density rock/iron: Dark Grays, Reds
                body["base_color"] = "#9E9E9E" if density < 7.0 else "#7F7F7F"
        # --------------------------------------

    def _sync_all_dropdown_labels(self, body):
        """
        Sync all dropdown selection labels with current body values.
        Ensures UI controls reflect derived changes across all physical and orbital parameters.
        """
        if body.get("type") == "planet":
            # 1. Mass label (planet_dropdown_selected)
            mass = body.get("mass", 1.0)
            found = False
            for name, val in self.planet_dropdown_options:
                if val is not None and abs(val - mass) < 1e-4:
                    body["planet_dropdown_selected"] = name
                    found = True
                    break
            if not found:
                body["planet_dropdown_selected"] = f"Mass: {self._format_value(mass, 'M⊕')}"

            # 2. Radius label (planet_radius_dropdown_selected)
            radius = body.get("actual_radius", body.get("radius", 1.0))
            found = False
            for name, val in self.planet_radius_dropdown_options:
                if val is not None and abs(val - radius) < 1e-3:
                    body["planet_radius_dropdown_selected"] = name
                    found = True
                    break
            if not found:
                body["planet_radius_dropdown_selected"] = f"Radius: {self._format_value(radius, 'R🜨')}"

            # 3. Density label (planet_density_dropdown_selected)
            density = body.get("density", 5.51)
            found = False
            for name, val in self.planet_density_dropdown_options:
                if val is not None and abs(val - density) < 1e-3:
                    body["planet_density_dropdown_selected"] = name
                    found = True
                    break
            if not found:
                body["planet_density_dropdown_selected"] = f"Density: {self._format_value(density, 'g/cm³')}"

            # 4. Gravity label (planet_gravity_dropdown_selected)
            gravity = body.get("gravity", 9.81)
            found = False
            for name, val in self.planet_gravity_dropdown_options:
                if val is not None and abs(val - gravity) < 1e-3:
                    body["planet_gravity_dropdown_selected"] = name
                    found = True
                    break
            if not found:
                body["planet_gravity_dropdown_selected"] = f"Gravity: {self._format_value(gravity, 'm/s²')}"

            # 5. Temperature label
            temp = body.get("temperature", 288)
            found = False
            for name, val in self.planet_temperature_dropdown_options:
                if val is not None and abs(val - temp) < 1.0:
                    body["planet_temperature_dropdown_selected"] = name
                    found = True
                    break
            if not found:
                body["planet_temperature_dropdown_selected"] = f"Temp: {self._format_value(temp, 'K')}"

            # 6. Atmosphere label
            offset = body.get("greenhouse_offset", 33.0)
            erosion = body.get("atmospheric_erosion_status", "Stable")
            
            if body.get("atmosphere_stripped"):
                body["planet_atmosphere_dropdown_selected"] = "Stripped (Low Gravity)"
            elif erosion != "Stable":
                body["planet_atmosphere_dropdown_selected"] = f"Atmosphere ({erosion})"
            else:
                found = False
                for name, val in self.planet_atmosphere_dropdown_options:
                    if val is not None and abs(val - offset) < 0.1:
                        body["planet_atmosphere_dropdown_selected"] = name
                        found = True
                        break
                if not found:
                    body["planet_atmosphere_dropdown_selected"] = "Custom"

            # 7. Semi-Major Axis label
            au = body.get("semiMajorAxis", 1.0)
            if body.get("is_tidally_locked"):
                body["planet_orbital_distance_dropdown_selected"] = f"{self._format_value(au, 'AU')} (Tidally Locked)"
            else:
                found = False
                for name, val in self.planet_orbital_distance_dropdown_options:
                    if val is not None and abs(val - au) < 0.02:
                        body["planet_orbital_distance_dropdown_selected"] = name
                        found = True
                        break
                if not found:
                    body["planet_orbital_distance_dropdown_selected"] = f"Dist: {self._format_value(au, 'AU')}"

            # 8. Eccentricity label
            ecc = body.get("eccentricity", 0.0167)
            found = False
            for name, val in self.planet_orbital_eccentricity_dropdown_options:
                if val is not None and abs(val - ecc) < 0.005:
                    body["planet_orbital_eccentricity_dropdown_selected"] = name
                    found = True
                    break
            if not found:
                body["planet_orbital_eccentricity_dropdown_selected"] = f"Ecc: {ecc:.3f}"

            # 9. Orbital Period label
            period = body.get("orbital_period", 365.25)
            found = False
            for name, val in self.planet_orbital_period_dropdown_options:
                if val is not None and abs(val - period) < 1.0:
                    body["planet_orbital_period_dropdown_selected"] = name
                    found = True
                    break
            if not found:
                body["planet_orbital_period_dropdown_selected"] = f"Period: {self._format_value(period, 'days', allow_scientific=False)}"

            # 10. Stellar Flux label
            flux = body.get("stellarFlux", 1.0)
            found = False
            for name, val in self.planet_stellar_flux_dropdown_options:
                if val is not None and abs(val - flux) < 0.001:
                    body["planet_stellar_flux_dropdown_selected"] = name
                    found = True
                    break
            if not found:
                body["planet_stellar_flux_dropdown_selected"] = f"Flux: {self._format_value(flux, 'EFU')}"
                
            # Sync class-level globals if this is the selected body
            if body.get("id") == self.selected_body_id:
                self.planet_dropdown_selected = body.get("planet_dropdown_selected")
                self.planet_radius_dropdown_selected = body.get("planet_radius_dropdown_selected")
                self.planet_density_dropdown_selected = body.get("planet_density_dropdown_selected")
                self.planet_gravity_dropdown_selected = body.get("planet_gravity_dropdown_selected")
                self.planet_temperature_dropdown_selected = body.get("planet_temperature_dropdown_selected")
                self.planet_atmosphere_dropdown_selected = body.get("planet_atmosphere_dropdown_selected")
                self.planet_orbital_distance_dropdown_selected = body.get("planet_orbital_distance_dropdown_selected")
                self.planet_orbital_eccentricity_dropdown_selected = body.get("planet_orbital_eccentricity_dropdown_selected")
                self.planet_orbital_period_dropdown_selected = body.get("planet_orbital_period_dropdown_selected")
                self.planet_stellar_flux_dropdown_selected = body.get("planet_stellar_flux_dropdown_selected")
                
        elif body.get("type") == "moon":
            # Moon mass
            mass = body.get("mass", 0.0123)
            found = False
            for option_data in self.moon_dropdown_options:
                name = option_data[0]
                val = option_data[1]
                if val is not None and abs(val - mass) < 1e-5:
                    body["moon_dropdown_selected"] = name
                    found = True
                    break
            if not found:
                body["moon_dropdown_selected"] = f"Mass: {self._format_value(mass, 'M☾')}"
                
            # Moon gravity
            gravity = body.get("gravity", 1.62)
            found = False
            for name, val in self.moon_gravity_dropdown_options:
                if val is not None and abs(val - gravity) < 1e-3:
                    body["moon_gravity_dropdown_selected"] = name
                    found = True
                    break
            if not found:
                body["moon_gravity_dropdown_selected"] = f"Gravity: {self._format_value(gravity, 'm/s²')}"
                
            # Moon orbital distance label
            au = body.get("semiMajorAxis", MOON_ORBIT_AU)
            if body.get("is_unstable"):
                body["moon_orbital_distance_dropdown_selected"] = "Unstable (Hill Limit)"
            elif body.get("is_tidally_locked"):
                body["moon_orbital_distance_dropdown_selected"] = "Tidally Locked"
            else:
                found = False
                for name, val in self.moon_orbital_distance_dropdown_options:
                    if val is not None and abs(val - au) < 0.0001:
                        body["moon_orbital_distance_dropdown_selected"] = name
                        found = True
                        break
                if not found:
                    # Convert AU to km for moon distance display
                    AU_TO_KM = 149597870.7
                    dist_km = au * AU_TO_KM
                    body["moon_orbital_distance_dropdown_selected"] = f"Dist: {self._format_value(dist_km, 'km')}"

            # Moon orbital period label
            period = body.get("orbital_period", 27.3)
            found = False
            for name, val in self.moon_orbital_period_dropdown_options:
                if val is not None and abs(val - period) < 0.2:
                    body["moon_orbital_period_dropdown_selected"] = name
                    found = True
                    break
            if not found:
                body["moon_orbital_period_dropdown_selected"] = f"Period: {self._format_value(period, 'days')}"
                
            # Sync class-level globals
            if body.get("id") == self.selected_body_id:
                self.moon_dropdown_selected = body.get("moon_dropdown_selected")
                self.moon_gravity_dropdown_selected = body.get("moon_gravity_dropdown_selected")
                self.moon_orbital_distance_dropdown_selected = body.get("moon_orbital_distance_dropdown_selected")
                self.moon_orbital_period_dropdown_selected = body.get("moon_orbital_period_dropdown_selected")

        elif body.get("type") == "star":
            # Star mass
            mass = body.get("mass", 1.0)
            # Ensure mass is a Python float
            if hasattr(mass, 'item'):
                mass = float(mass.item())
            else:
                mass = float(mass)
                
            found = False
            for name, val in self.star_mass_dropdown_options:
                if val is not None and abs(val - mass) < 0.001:
                    body["star_mass_dropdown_selected"] = name
                    found = True
                    break
            if not found:
                body["star_mass_dropdown_selected"] = f"Mass: {mass:.2f} M☉"
                
            # Star radius
            radius = body.get("radius", 1.0)
            found = False
            for name, val in self.radius_dropdown_options:
                if val is not None and abs(val - radius) < 0.01:
                    body["radius_dropdown_selected"] = name
                    found = True
                    break
            if not found:
                body["radius_dropdown_selected"] = f"Radius: {radius:.2f} R☉"
                
            # Star temperature
            temp = body.get("temperature", 5800.0)
            found = False
            for name, val, color in self.spectral_class_dropdown_options:
                if val is not None and abs(val - temp) < 10:
                    body["spectral_class_dropdown_selected"] = name
                    found = True
                    break
            if not found:
                body["spectral_class_dropdown_selected"] = f"Temp: {temp:.0f} K"
                
            # Star activity
            activity = body.get("activity", "Moderate")
            found = False
            for name, val in self.activity_dropdown_options:
                if name == activity:
                    body["activity_dropdown_selected"] = name
                    found = True
                    break
            if not found:
                body["activity_dropdown_selected"] = activity
                
            # Star luminosity
            lum = body.get("luminosity", 1.0)
            found = False
            for name, val in self.luminosity_dropdown_options:
                if val is not None and abs(val - lum) < 0.01:
                    body["luminosity_dropdown_selected"] = name
                    found = True
                    break
            if not found:
                body["luminosity_dropdown_selected"] = f"Luminosity: {lum:.2f} L☉"

            # Sync class-level globals
            if body.get("id") == self.selected_body_id:
                self.star_mass_dropdown_selected = body.get("star_mass_dropdown_selected")
                self.radius_dropdown_selected = body.get("radius_dropdown_selected")
                self.spectral_class_dropdown_selected = body.get("spectral_class_dropdown_selected")
                self.luminosity_dropdown_selected = body.get("luminosity_dropdown_selected")

    def get_selected_body(self):
        """
        Get the selected body from the ID registry to ensure we're working with the correct object.
        This prevents shared-state bugs by always using the canonical registry entry.
        """
        if self.selected_body_id and self.selected_body_id in self.bodies_by_id:
            body = self.bodies_by_id[self.selected_body_id]
            # Update selected_body reference to match registry
            self.selected_body = body
            return body
        return None
    
    def _perform_registry_update(self, body_id, key, value, debug_name=""):
        """
        Internal function to perform the actual dictionary update with all isolation checks.
        """
        if not body_id:
            print(f"ERROR: Cannot update {key} - no body id provided")
            return False
            
        if body_id not in self.bodies_by_id:
            print(f"ERROR: Body id {body_id} not found in registry")
            return False
            
        body = self.bodies_by_id[body_id]
        body_dict_id_before = id(body)
        
        # Isolation and shared-state checks
        if body.get("id") != body_id:
            raise AssertionError(f"CRITICAL: Body dict id mismatch! Expected {body_id}, got {body.get('id')}")
            
        # Verify body dict is unique in placed_bodies
        for other_body in self.placed_bodies:
            if other_body.get("id") != body_id and id(other_body) == body_dict_id_before:
                raise AssertionError(f"CRITICAL: Found shared dict for body {body_id} and {other_body.get('id')}")

        # Convert value to float if it's numeric
        if isinstance(value, (int, float)) or (hasattr(value, 'item') and not isinstance(value, str)):
            if hasattr(value, 'item'):
                value = float(value.item())
            else:
                value = float(value)
                
        # Perform update
        body[key] = value
        
        # Verify dict identity remains same
        if id(body) != body_dict_id_before:
            raise AssertionError(f"CRITICAL: Body dict identity changed during update!")
            
        return True

    def update_selected_body_property(self, key, value, debug_name=""):
        """
        Update a property on the selected body using the ID registry.
        This ensures we're always updating the correct object and prevents shared-state bugs.
        Automatically triggers orbit recalculation for physics-affecting parameters.
        """
        # Call the centralized handler for physical parameters
        if key in ["mass", "density", "gravity", "radius", "actual_radius", "temperature", "luminosity"]:
            return self.apply_parameter_change(self.selected_body_id, key, value)
            
        # For other parameters, perform direct update with isolation checks
        if not self._perform_registry_update(self.selected_body_id, key, value, debug_name):
            return False
        
        if not self.selected_body_id:
            print(f"ERROR: Cannot update {key} - no body selected")
            return False
            
        # CRITICAL: Get body ONLY from registry, never use self.selected_body directly
        if self.selected_body_id not in self.bodies_by_id:
            print(f"ERROR: Selected body id {self.selected_body_id} not found in registry")
            return False
            
        body = self.bodies_by_id[self.selected_body_id]
        
        # CRITICAL: Verify this body is unique - no other body shares the same dict
        body_dict_id = id(body)
        shared_with = []
        for other_body in self.placed_bodies:
            other_id = other_body.get("id")
            other_dict_id = id(other_body)
            if other_id != self.selected_body_id and other_dict_id == body_dict_id:
                shared_with.append((other_id, other_body.get("name")))
                print(f"CRITICAL ERROR: Body {self.selected_body_id[:8]} shares dict with body {other_id[:8]} ({other_body.get('name')})! dict_id={body_dict_id}")
        
        if shared_with:
            error_msg = f"SHARED_DICT_DETECTED: Selected body {self.selected_body_id} shares dict with: {shared_with}"
            print(error_msg)
            raise AssertionError(error_msg)
        
        # CRITICAL: Verify the body in registry matches the body in placed_bodies
        body_in_list = next((b for b in self.placed_bodies if b.get("id") == self.selected_body_id), None)
        if body_in_list is None:
            raise AssertionError(f"CRITICAL: Selected body {self.selected_body_id} not found in placed_bodies!")
        if id(body_in_list) != body_dict_id:
            print(f"WARNING: Body {self.selected_body_id} in registry has different dict_id than in placed_bodies!")
            print(f"  Registry dict_id: {body_dict_id}")
            print(f"  List dict_id: {id(body_in_list)}")
            # Use the one from placed_bodies to be safe
            body = body_in_list
            body_dict_id = id(body)
        
        # CRITICAL: Before update, store all body values to detect cross-body mutations
        old_masses = {}
        if key == "mass":
            print(f"\n{'='*60}")
            print(f"MASS UPDATE REQUESTED")
            print(f"  Selected body_id: {self.selected_body_id}")
            print(f"  Body from registry dict_id: {body_dict_id}")
            print(f"  Body name: {body.get('name', 'unknown')}")
            print(f"  Body current mass: {body.get('mass', 0.0):.6f}")
            print(f"  New mass value: {value:.6f}")
            print(f"\n=== ALL PLANETS BEFORE UPDATE ===")
            for b in self.placed_bodies:
                if b.get("type") == "planet":
                    body_id = b.get("id", "no-id")
                    old_mass = b.get("mass", 0.0)
                    old_masses[body_id] = old_mass
                    b_dict_id = id(b)
                    is_selected = (body_id == self.selected_body_id)
                    marker = " <-- SELECTED" if is_selected else ""
                    print(f"  Body {body_id[:8]} name={b.get('name')} mass={old_mass:.6f} dict_id={b_dict_id}{marker}")
                    if b_dict_id == body_dict_id and body_id != self.selected_body_id:
                        raise AssertionError(f"CRITICAL: Found duplicate dict_id! Body {body_id} has same dict as selected body {self.selected_body_id}")
            print(f"{'='*60}\n")
        
        if body:
            # Convert value to float if it's numeric to prevent numpy array issues
            if isinstance(value, (int, float)) or (hasattr(value, 'item') and not isinstance(value, str)):
                if hasattr(value, 'item'):
                    value = float(value.item())
                else:
                    value = float(value)
            
            # CRITICAL: Update ONLY this specific body dict
            old_value = body.get(key)
            
            # CRITICAL: Double-check we're updating the right body before mutation
            if body.get("id") != self.selected_body_id:
                raise AssertionError(f"CRITICAL: Body dict id mismatch! Expected {self.selected_body_id}, got {body.get('id')}")
            
            # Verify body dict is still unique before updating
            body_dict_id_before = id(body)
            for other_body in self.placed_bodies:
                if other_body.get("id") != self.selected_body_id and id(other_body) == body_dict_id_before:
                    raise AssertionError(f"CRITICAL: About to update shared dict! Body {self.selected_body_id} shares dict with {other_body.get('id')}")
            
            # NOW update the value
            body[key] = value
            
            # If radius or actual_radius changed, update hitbox_radius to match new visual size
            if key == "radius" or key == "actual_radius":
                if body["type"] == "star":
                    visual_radius = self.calculate_star_visual_radius(body)
                    # ALSO update hitboxes of all planets orbiting this star, as their max clamp depends on star size
                    for p in self.placed_bodies:
                        if p.get("type") == "planet" and (p.get("parent") == body.get("name") or p.get("parent_id") == body.get("id")):
                            p_visual_radius = self.calculate_planet_visual_radius(p, self.placed_bodies)
                            p["hitbox_radius"] = float(self.calculate_hitbox_radius("planet", p_visual_radius))
                elif body["type"] == "planet":
                    visual_radius = self.calculate_planet_visual_radius(body, self.placed_bodies)
                    # ALSO update visual radius and hitboxes of all moons orbiting this planet
                    for m in self.placed_bodies:
                        if m.get("type") == "moon" and (m.get("parent") == body.get("name") or m.get("parent_id") == body.get("id")):
                            m_visual_radius = self.calculate_moon_visual_radius(m, self.placed_bodies)
                            m["radius"] = float(m_visual_radius)
                            m["hitbox_radius"] = float(self.calculate_hitbox_radius("moon", m_visual_radius))
                else:  # moon
                    visual_radius = self.calculate_moon_visual_radius(body, self.placed_bodies)
                    # Sync legacy visual radius field for moons to ensure real-time UI updates
                    body["radius"] = float(visual_radius)
                body["hitbox_radius"] = float(self.calculate_hitbox_radius(body["type"], visual_radius))
            
            # Verify the update only affected this body
            body_dict_id_after = id(body)
            if body_dict_id_before != body_dict_id_after:
                raise AssertionError(f"CRITICAL: Body dict id changed during update! {body_dict_id_before} -> {body_dict_id_after}")
            
            # CRITICAL: Verify the update only affected the intended body
            if key == "mass" and debug_name:
                print(f"\n=== AFTER MASS UPDATE ===")
                print(f"  Updated body_id={self.selected_body_id[:8]} dict_id={body_dict_id} old_mass={old_value:.6f} new_mass={value:.6f}")
                changed_bodies = []
                for b in self.placed_bodies:
                    if b.get("type") == "planet":
                        body_id = b.get("id", "no-id")
                        current_mass = b.get("mass", 0.0)
                        old_mass = old_masses.get(body_id, current_mass)
                        print(f"  Body {body_id[:8]} name={b.get('name')} old_mass={old_mass:.6f} new_mass={current_mass:.6f} dict_id={id(b)}")
                        # Check if this body's mass changed when it shouldn't have
                        if body_id != self.selected_body_id:
                            if abs(current_mass - old_mass) > 1e-6:
                                changed_bodies.append((body_id, b.get("name"), id(b), old_mass, current_mass))
                                print(f"  *** ERROR: Body {b.get('name')} mass changed from {old_mass:.6f} to {current_mass:.6f} but was NOT selected!")
                
                if changed_bodies:
                    print(f"\n*** CRITICAL ERROR: {len(changed_bodies)} other bodies changed mass!")
                    for other_id, other_name, other_dict_id, old_m, new_m in changed_bodies:
                        print(f"  - Body {other_id[:8]} ({other_name}) dict_id={other_dict_id} {old_m:.6f} -> {new_m:.6f}")
                        if other_dict_id == body_dict_id:
                            raise AssertionError(f"SHARED_STATE DETECTED: Body {other_id} shares dict with selected body {self.selected_body_id}! dict_id={body_dict_id}")
                    raise AssertionError(f"MASS_UPDATE_BUG: {len(changed_bodies)} non-selected bodies had their mass changed!")
            
            if debug_name:
                print(f"DEBUG: Updated {debug_name} for body id={self.selected_body_id}, {key}={value}, position_id={id(body['position'])}, body_id={id(body)}")
            
            # Update selected_body reference to match registry (for backward compatibility)
            self.selected_body = body
            
            # CRITICAL: Recompute orbit parameters when physics-affecting parameters change
            # This ensures each body has independent physics calculations
            # CRITICAL: radius/actual_radius is NOT a physics-affecting key
            # Radius affects visual size only, never physics (orbit, mass, AU, etc.)
            physics_affecting_keys = ["mass", "semiMajorAxis", "orbit_radius", "eccentricity", "orbital_period"]
            if key in physics_affecting_keys:
                # Capture AU before coupling to detect side-effect changes
                au_before_coupling = body.get("semiMajorAxis", 0.0)
                
                # Apply Kepler coupling for AU/Period
                if key in ["semiMajorAxis", "orbital_period"]:
                    self._apply_kepler_coupling(body, key)

                # CRITICAL: If semiMajorAxis changed (either directly or via coupling), trigger position correction
                current_au = body.get("semiMajorAxis", 0.0)
                if abs(current_au - au_before_coupling) > 1e-7 or key == "semiMajorAxis":
                    self._start_orbital_correction(body, current_au)
                
                self.recompute_orbit_parameters(body, force_recompute=True)
                # Regenerate orbit grid for visualization
                if body["type"] != "star":
                    self.generate_orbit_grid(body)
                    self.clear_orbit_points(body)
            
            # Sync dropdown labels to reflect any coupled changes
            self._sync_all_dropdown_labels(body)

            # CRITICAL: Assert no shared state after parameter update
            self._assert_no_shared_state()
            
            # After every parameter write, verify that no bodies share parameter containers
            self.debug_verify_body_references(source=f"update_selected_body_property:{debug_name or key}")
            return True
        return False

    def debug_verify_body_references(self, source: str = ""):
        """
        Debug helper: print memory ids for each body's core containers to detect shared references.
        This is a direct implementation of the 'no shared state' verification requirement.
        """
        if not self.placed_bodies:
            return

        print("DEBUG_REF_CHECK", source or "unknown")
        seen_body_ids = set()
        seen_position_ids = set()
        seen_orbit_points_ids = set()

        for body in self.placed_bodies:
            bid = body.get("id", "<no-id>")
            body_id_obj = id(body)
            pos_id = id(body.get("position"))
            orbit_points_id = id(body.get("orbit_points"))

            print(
                "BODY_REF",
                bid,
                "body_id", body_id_obj,
                "position_id", pos_id,
                "orbit_points_id", orbit_points_id,
            )

            # Simple duplicate detection to surface any accidental sharing
            if body_id_obj in seen_body_ids:
                print("WARNING: Duplicate body dict reference detected", bid)
            if pos_id in seen_position_ids:
                print("WARNING: Shared position reference detected", bid)
            if orbit_points_id in seen_orbit_points_ids:
                print("WARNING: Shared orbit_points reference detected", bid)

            seen_body_ids.add(body_id_obj)
            seen_position_ids.add(pos_id)
            seen_orbit_points_ids.add(orbit_points_id)
    
    def create_planet_from_preset(self, preset_name: str, spawn_position: np.ndarray, body_id: str = None) -> dict:
        """
        Create a planet from a preset with all parameters deep-copied at creation.
        This is the authoritative planet creation function - all preset values are applied here.
        
        Args:
            preset_name: Name of the preset (e.g., "Mars", "Earth")
            spawn_position: Initial visual position (will be corrected to orbit radius)
            body_id: Optional body ID (will be generated if None)
        
        Returns:
            Complete planet body dictionary with all preset parameters
        """
        if preset_name not in SOLAR_SYSTEM_PLANET_PRESETS:
            raise ValueError(f"Unknown preset: {preset_name}")
        
        if body_id is None:
            body_id = str(uuid4())
        
        # Deep copy preset data
        preset = SOLAR_SYSTEM_PLANET_PRESETS[preset_name]
        
        # Get parent star for orbit calculation
        stars = [b for b in self.placed_bodies if b["type"] == "star"]
        parent_star = None
        if stars:
            parent_star = min(stars, key=lambda s: np.linalg.norm(s["position"] - spawn_position))
        
        # Calculate correct orbital position from AU using visual scaling
        orbit_radius_au = preset.get("semiMajorAxis", 1.0)
        orbit_radius_px = self.get_visual_orbit_radius(orbit_radius_au)
        orbit_angle = float(random.uniform(0, 2 * np.pi))
        
        # Target position based on physics (AU)
        if parent_star:
            target_position = np.array([
                parent_star["position"][0] + orbit_radius_px * np.cos(orbit_angle),
                parent_star["position"][1] + orbit_radius_px * np.sin(orbit_angle)
            ], dtype=float)
        else:
            # No star yet, use spawn position as target
            target_position = np.array(spawn_position, dtype=float)
        
        # Create body with preset values - start at spawn position (will animate)
        body = self._create_new_body_dict(
            obj_type="planet",
            body_id=body_id,
            position=np.array(spawn_position, dtype=float),  # Start at click position (will animate to target)
            default_name=preset_name,
            default_mass=preset["mass"],
            default_age=4.6,  # Default age
            default_radius=preset["radius"]  # In Earth radii (R⊕)
        )
        
        # Add all preset parameters (deep copied)
        body.update({
            "gravity": float(preset.get("gravity", 9.81)),
            "semiMajorAxis": float(orbit_radius_au),  # Authoritative AU value
            "orbit_radius_au": float(orbit_radius_au),  # Store AU explicitly
            "eccentricity": float(preset.get("eccentricity", 0.0167)),
            "orbital_period": float(preset.get("orbital_period", 365.25)),
            "stellarFlux": float(preset.get("stellarFlux", 1.0)),
            "temperature": float(preset.get("temperature", 288.0)),
            "equilibrium_temperature": float(preset.get("equilibrium_temperature", 255.0)),
            "greenhouse_offset": float(preset.get("greenhouse_offset", 33.0)),
            "rotation_period_days": float(preset.get("rotation_period_days", 1.0)),
            "density": float(preset.get("density", 5.51)),
            "base_color": str(preset.get("base_color", CELESTIAL_BODY_COLORS.get(preset_name, "#2E7FFF"))),  # Per-object color from preset
            "planet_dropdown_selected": preset_name,
            "orbit_angle": float(orbit_angle),
            "visual_position": np.array(spawn_position, dtype=float).copy(),  # Temporary visual position
            "target_position": target_position.copy(),  # Physics-determined position
            "is_correcting_orbit": True,  # Flag to trigger animation
            "is_placing": True,  # Track placement lifecycle
            "placement_line": {
                "start": np.array(spawn_position, dtype=float).copy(),
                "end": target_position.copy(),
                "alpha": 255,
                "fade_start_time": None,
                "fade_duration": self.PLACEMENT_LINE_FADE_DURATION
            }
        })
        
        # Set parent if star exists
        if parent_star:
            body["parent"] = parent_star["name"]
            body["parent_id"] = parent_star["id"]
            body["parent_obj"] = parent_star
        
        # Initialize orbit correction animation
        self.orbital_corrections[body_id] = {
            "target_radius_px": orbit_radius_px,
            "start_time": time.time(),
            "duration": self.correction_animation_duration,
            "start_pos": np.array(spawn_position, dtype=float).copy(),
            "target_pos": target_position.copy(),
            "position_history": [np.array(spawn_position, dtype=float).copy()]  # Track position history for trail
        }
        
        # Log creation
        print(f"SPAWN_PLANET | id={body_id[:8]} | preset={preset_name} | mass={preset['mass']} | AU={orbit_radius_au} | radius={preset['radius']} R⊕")
        
        return body
    
    def _create_new_body_dict(self, obj_type: str, body_id: str, position: np.ndarray,
                              default_name: str, default_mass: float, default_age: float,
                              default_radius: float) -> dict:
        """
        Factory function that creates a completely new body dictionary with no shared references.
        Every nested structure (lists, dicts, arrays) is a new instance.
        
        This is CRITICAL to prevent shared state bugs where changing one body affects others.
        """
        # CRITICAL: For planets, radius is stored in Earth radii (R⊕), not pixels
        # For stars, radius is stored in Solar radii (R☉)
        # For moons, radius is stored in pixels (legacy)
        if obj_type == "planet":
            # Store in Earth radii: default_radius is already in R⊕ (1.0 = Earth's radius)
            body_radius = float(default_radius) if default_radius > 0 else 1.0
        elif obj_type == "star":
            # Store in Solar radii (R☉): default_radius is already in R☉
            # Handle potential legacy values (if SUN_RADIUS_PX=32 was passed, convert to 1.0)
            if default_radius >= 32.0:
                body_radius = float(default_radius) / SUN_RADIUS_PX
            else:
                body_radius = float(default_radius) if default_radius > 0 else 1.0
        else:
            # Moons: store in pixels (legacy behavior)
            body_radius = float(default_radius)
        
        # Determine default base_color based on type and name
        default_base_color = None
        if obj_type == "star":
            # Stars default to Sun color
            default_base_color = CELESTIAL_BODY_COLORS.get("Sun", "#FDB813")
        elif obj_type == "moon":
            # Moons default to Moon color
            default_base_color = CELESTIAL_BODY_COLORS.get("Moon", "#B0B0B0")
        elif obj_type == "planet":
            # Planets: try to match by name, otherwise default to Earth color
            default_base_color = CELESTIAL_BODY_COLORS.get(default_name, CELESTIAL_BODY_COLORS.get("Earth", "#2E7FFF"))
        
        # Create new body dict with all primitive values
        body = {
            "id": str(body_id),  # Unique identifier
            "type": str(obj_type),  # Ensure string, not shared reference
            "position": np.array(position, dtype=float).copy(),  # NEW array instance
            "velocity": np.array([0.0, 0.0], dtype=float).copy(),  # NEW array instance
            "radius": body_radius,  # Scalar: R⊕ for planets, pixels for stars/moons
            # Hitbox will be computed from visual radius on render
            "hitbox_radius": 0.0,  # Will be computed when needed
            "name": str(default_name),  # NEW string instance
            "mass": float(default_mass),  # Scalar (Solar masses for stars, Earth masses for planets/moons)
            "base_color": str(default_base_color),  # Hex color string (per-object, no shared state)
            "parent": None,  # Will be set later
            "parent_id": None,  # Will be set later
            "parent_obj": None,  # Will be set later
            "orbit_radius": 0.0,  # Scalar
            "orbit_angle": float(random.uniform(0, 2 * np.pi)) if obj_type != "star" else 0.0,  # Random phase for planets/moons
            "orbit_speed": 0.0,  # Scalar
            "rotation_angle": 0.0,  # Scalar
            "rotation_speed": float(self.rotation_speed * (1.0 if obj_type == "planet" else 2.0 if obj_type == "moon" else 0.0)),  # Scalar
            "rotation_period_days": 1.0,  # Default to 1 day (Earth-like)
            "age": float(default_age),  # Scalar
            "habit_score": 0.0,  # Scalar
            "orbit_points": [],  # NEW list instance - CRITICAL: must be new list
            "max_orbit_points": 2000,  # Scalar
            "orbit_enabled": True,  # Scalar
            "orbital_driver": "semi_major_axis",  # Default driver for Kepler coupling
        }
        
        # Verify all nested structures are new instances
        assert isinstance(body["position"], np.ndarray), "position must be numpy array"
        assert isinstance(body["velocity"], np.ndarray), "velocity must be numpy array"
        assert isinstance(body["orbit_points"], list), "orbit_points must be list"
        
        # CRITICAL: Verify this body dict is completely unique
        body_dict_id = id(body)
        for existing_body in self.placed_bodies:
            existing_dict_id = id(existing_body)
            if existing_dict_id == body_dict_id:
                raise AssertionError(f"FACTORY_ERROR: New body {body_id} shares dict with existing body {existing_body.get('id')}! dict_id={body_dict_id}")
        
        # Verify orbit_points list is unique
        orbit_points_id = id(body["orbit_points"])
        for existing_body in self.placed_bodies:
            if "orbit_points" in existing_body:
                existing_orbit_points_id = id(existing_body["orbit_points"])
                if existing_orbit_points_id == orbit_points_id:
                    raise AssertionError(f"FACTORY_ERROR: New body {body_id} shares orbit_points list with existing body {existing_body.get('id')}!")
        
        return body
    
    def _assert_no_shared_state(self):
        """
        Hard assertion to detect shared state between bodies.
        Raises AssertionError if any bodies share memory references.
        """
        if len(self.placed_bodies) < 2:
            return
        
        # Assert all body dicts are unique
        body_ids = [id(b) for b in self.placed_bodies]
        assert len(body_ids) == len(set(body_ids)), \
            f"SHARED_STATE_DETECTED: {len(body_ids) - len(set(body_ids))} bodies share dict references"
        
        # Assert all position arrays are unique
        position_ids = [id(b.get("position")) for b in self.placed_bodies if "position" in b]
        assert len(position_ids) == len(set(position_ids)), \
            f"SHARED_STATE_DETECTED: {len(position_ids) - len(set(position_ids))} bodies share position arrays"
        
        # Assert all velocity arrays are unique
        velocity_ids = [id(b.get("velocity")) for b in self.placed_bodies if "velocity" in b]
        assert len(velocity_ids) == len(set(velocity_ids)), \
            f"SHARED_STATE_DETECTED: {len(velocity_ids) - len(set(velocity_ids))} bodies share velocity arrays"
        
        # Assert all orbit_points lists are unique
        orbit_points_ids = [id(b.get("orbit_points")) for b in self.placed_bodies if "orbit_points" in b]
        assert len(orbit_points_ids) == len(set(orbit_points_ids)), \
            f"SHARED_STATE_DETECTED: {len(orbit_points_ids) - len(set(orbit_points_ids))} bodies share orbit_points lists"
        
        # Assert all hz_surface objects are unique (if they exist)
        hz_surface_ids = [id(b.get("hz_surface")) for b in self.placed_bodies if "hz_surface" in b and b.get("hz_surface") is not None]
        if len(hz_surface_ids) > 1:
            assert len(hz_surface_ids) == len(set(hz_surface_ids)), \
                f"SHARED_STATE_DETECTED: {len(hz_surface_ids) - len(set(hz_surface_ids))} bodies share hz_surface objects"
        
        print(f"ASSERTION_PASSED: All {len(self.placed_bodies)} bodies have unique memory references")
    
    def place_object(self, obj_type: str, params: dict = None):
        """
        Place a celestial object using the same logic as user clicks.
        This ensures full parameter lists, event bindings, orbit initialization, and physics behavior.
        
        Args:
            obj_type: "star", "planet", or "moon"
            params: Optional dict with parameters like:
                - "name": Custom name (defaults to "Sun", "Earth", "Moon")
                - "semi_major_axis": For planets, orbital distance in AU (defaults to 1.0)
                - Other type-specific parameters
        """
        if params is None:
            params = {}
        
        # Set active tab to the object type
        self.active_tab = obj_type
        
        # Increment body counter
        self.body_counter[obj_type] += 1
        
        # Set default values based on body type
        if obj_type == "star":
            # Sun-like defaults
            default_mass = 1.0  # Solar masses
            default_age = 4.6  # Gyr
            default_spectral = "G-type (Yellow, Sun)"
            default_luminosity = 1.0  # Solar luminosities
            default_name = params.get("name", "Sun")
            default_radius = 1.0  # Solar radii (R☉)
            
            # Calculate position (center of screen for star)
            position = np.array([self.width/2, self.height/2], dtype=float)
        elif obj_type == "planet":
            # Earth-like defaults
            default_mass = 1.0  # Earth masses
            default_age = 4.6  # Gyr
            default_name = params.get("name", "Earth")
            # For planets, default_radius is in Earth radii (R⊕), not pixels
            default_radius = 1.0  # 1.0 R⊕ = Earth's radius
            
            # Calculate position based on semi_major_axis using visual scaling
            semi_major_axis = params.get("semi_major_axis", 1.0)
            orbit_radius_px = self.get_visual_orbit_radius(semi_major_axis)
            position = np.array([self.width/2 + orbit_radius_px, self.height/2], dtype=float)
        else:  # moon
            # Luna-like defaults
            default_mass = 1.0  # Earth's Moon mass (1 lunar mass)
            default_age = 4.6  # Gyr
            default_name = params.get("name", "Moon")
            default_radius = MOON_RADIUS_PX  # Slightly enlarged for visibility
            
            # Calculate position based on semi_major_axis (relative to parent planet)
            semi_major_axis = params.get("semi_major_axis", 0.00257)  # Moon's distance in AU
            # Find parent planet (Earth) to position moon relative to it
            parent_planet = next((b for b in self.placed_bodies if b["name"] == "Earth" and b["type"] == "planet"), None)
            if parent_planet:
                # Position moon at the right of the planet
                position = np.array([parent_planet["position"][0] + MOON_ORBIT_PX, parent_planet["position"][1]], dtype=float)
            else:
                # Fallback: position relative to center
                position = np.array([self.width/2 + MOON_ORBIT_PX, self.height/2], dtype=float)
        
        # Create unique ID for this body to ensure independence
        body_id = str(uuid4())
        
        # CRITICAL: Use factory function to create body with NO shared references
        body = self._create_new_body_dict(
            obj_type=obj_type,
            body_id=body_id,
            position=position,
            default_name=default_name,
            default_mass=default_mass,
            default_age=default_age,
            default_radius=default_radius
        )
        
        # Add planet-specific attributes
        if obj_type == "planet":
            # Default equilibrium temperature (Earth's T_eq ~255K)
            default_T_eq = 255.0
            # Default atmosphere offset (Earth-like: +33K)
            default_greenhouse_offset = 33.0
            default_T_surface = default_T_eq + default_greenhouse_offset  # 288K
            semi_major_axis = params.get("semi_major_axis", 1.0)
            
            # CRITICAL: Initialize orbit_angle randomly to ensure independent orbital phase
            # This prevents planets from stacking at the same phase
            body["orbit_angle"] = float(random.uniform(0, 2 * np.pi))
            
            body.update({
                "gravity": float(9.81),  # Earth's gravity in m/s²
                "semiMajorAxis": float(semi_major_axis),  # Orbital distance (AU)
                "eccentricity": float(0.0167),  # Default orbital eccentricity (Earth-like)
                "orbital_period": float(365.25),  # Default orbital period (days) - Earth's value
                "stellarFlux": float(1.0),  # Default stellar flux (Earth units)
                "temperature": float(default_T_surface),  # Surface temperature with greenhouse effect
                "equilibrium_temperature": float(default_T_eq),  # Equilibrium temperature
                "greenhouse_offset": float(default_greenhouse_offset),  # Greenhouse offset
                "planet_dropdown_selected": "Earth",  # CRITICAL: Per-body dropdown state, not global
                "orbital_driver": "semi_major_axis",  # Default driver
            })
            # Apply initial coupling to ensure consistency
            self._apply_kepler_coupling(body, "semiMajorAxis")
        
        # Add star-specific attributes
        if obj_type == "star":
            # Use 5800K as default Sun temperature for consistency with dropdowns
            default_T = params.get("temperature", 5800.0)
            body.update({
                "luminosity": float(default_luminosity),
                "temperature": float(default_T),
                "star_temperature": float(default_T),  # Keep for backward compatibility
                "star_color": (253, 184, 19),  # Yellow color for G-type star (tuple, immutable) - matches base_color
                "base_color": str(CELESTIAL_BODY_COLORS.get(default_name, CELESTIAL_BODY_COLORS.get("Sun", "#FDB813"))),  # Ensure base_color is set
            })
            # Ensure luminosity is calculated from the start
            self._update_derived_parameters(body)
            # Create habitable zone for the star (already called in _update_derived_parameters, but safe to keep)
            body["hz_surface"] = self.create_habitable_zone(body)
        
        # Add moon-specific attributes
        if obj_type == "moon":
            semi_major_axis = params.get("semi_major_axis", 0.00257)
            # Find parent planet to set orbit relative to it
            parent_planet = next((b for b in self.placed_bodies if b["name"] == "Earth" and b["type"] == "planet"), None)
            if parent_planet:
                # Calculate orbit radius from parent planet's position
                orbit_radius = np.linalg.norm(position - parent_planet["position"])
                # Ensure minimum orbit radius
                if orbit_radius < MOON_ORBIT_PX:
                    orbit_radius = MOON_ORBIT_PX
                
                # Calculate initial orbit angle from position
                dx = position[0] - parent_planet["position"][0]
                dy = position[1] - parent_planet["position"][1]
                orbit_angle = np.arctan2(dy, dx)
                
                # Calculate orbital speed based on orbital period (default 27.3 days)
                orbital_period = body.get("orbital_period", 27.3)
                omega_real = (2 * np.pi * 365.25) / orbital_period
                orbit_speed = float(omega_real)
                
                # CRITICAL: Immediately recalculate moon position from planet + orbit offset
                # This ensures the moon starts at the correct position relative to the planet
                moon_offset_x = orbit_radius * np.cos(orbit_angle)
                moon_offset_y = orbit_radius * np.sin(orbit_angle)
                position[0] = parent_planet["position"][0] + moon_offset_x
                position[1] = parent_planet["position"][1] + moon_offset_y
                
                # Set initial velocity for circular orbit
                v = orbit_speed * orbit_radius
                velocity = np.array([-v * np.sin(orbit_angle), v * np.cos(orbit_angle)])
                
                body.update({
                    "actual_radius": float(1737.4),  # Actual radius in km (The Moon) - for dropdown logic
                    "radius": float(default_radius),  # Visual radius in pixels for display
                    "semiMajorAxis": float(semi_major_axis),  # Orbital distance in AU
                    "orbit_radius": float(orbit_radius),  # Orbital distance in pixels (calculated from position)
                    "orbit_angle": float(orbit_angle),  # Initial orbit angle
                    "orbit_speed": float(orbit_speed),  # Orbital speed
                    "velocity": velocity.copy(),  # Initial velocity - ensure independent copy
                    "parent": parent_planet["name"],  # Set parent explicitly
                    "parent_id": parent_planet["id"],  # Use UUID for parent lookup
                    "parent_obj": parent_planet,  # Set permanent parent reference
                    "temperature": float(220),  # Surface temperature in K (Earth's Moon)
                    "gravity": float(1.62),  # Surface gravity in m/s² (Earth's Moon)
                    "orbital_period": float(27.3),  # Orbital period in days (Earth's Moon)
                    "orbital_driver": "semi_major_axis",  # Default driver
                })
                # Apply initial coupling to ensure consistency
                self._apply_kepler_coupling(body, "semiMajorAxis")

            else:
                body.update({
                    "actual_radius": float(1737.4),
                    "radius": float(default_radius),
                    "semiMajorAxis": float(semi_major_axis),
                    "orbit_radius": float(MOON_ORBIT_PX),  # Fallback orbital distance
                    "temperature": float(220),
                    "gravity": float(1.62),
                    "orbital_period": float(27.3),
                    "orbital_driver": "semi_major_axis",
                })
        
        self.placed_bodies.append(body)
        # Register body by ID for guaranteed unique lookups
        self.bodies_by_id[body_id] = body
        
        # Calculate visual radius for hitbox based on type
        if obj_type == "star":
            visual_radius = self.calculate_star_visual_radius(body)
        elif obj_type == "planet":
            visual_radius = self.calculate_planet_visual_radius(body, self.placed_bodies)
        else:  # moon
            visual_radius = self.calculate_moon_visual_radius(body, self.placed_bodies)
            
        # Update hitbox_radius to match visual radius
        body["hitbox_radius"] = float(self.calculate_hitbox_radius(obj_type, visual_radius))
        
        # Trigger initial habitability calculation
        if obj_type == "planet":
            self.selected_body = body
            self._update_planet_scores()
            self.selected_body = None  # Clear temp selection
        
        # CRITICAL: Hard assertion to detect shared state immediately
        self._assert_no_shared_state()
        
        # Debug: Verify independence
        print(f"DEBUG: Created body id={body_id}, name={body['name']}, mass={body['mass']}, position_id={id(body['position'])}, params_id={id(body)}")
        # Hard verification: ensure no bodies share core containers after creation
        self.debug_verify_body_references(source="place_object")
        # Note: orbit_points is now stored in the body dict itself, not in self.orbit_points
        # Keeping self.orbit_points for backward compatibility during transition, but it will be removed
        
        # Set dropdown selections to match defaults
        if obj_type == "star":
            self.star_mass_dropdown_selected = "Sun-like Star (1.0 M☉)"
            self.star_age_dropdown_selected = "4.6 (Gyr) (Sun)"
            self.spectral_dropdown_selected = "G-type (Yellow, Sun) (5,778 K)"
            self.luminosity_dropdown_selected = "G-type Main Sequence (Sun)"
            self.temperature_dropdown_selected = "G-type (Sun) (5,800 K)"
            self.radius_dropdown_selected = "G-type (Sun)"
            self.activity_dropdown_selected = "Moderate (Sun)"
            self.metallicity_dropdown_selected = "0.0 (Sun)"
        elif obj_type == "planet":
            # REMOVED: self.planet_dropdown_selected = "Earth"  # Now stored per-body in body["planet_dropdown_selected"]
            self.planet_age_dropdown_selected = "Solar System Planets (4.6 Gyr)"
            self.planet_gravity_dropdown_selected = "Earth"
            self.planet_atmosphere_dropdown_selected = "Earth-like (Moderate)"
            self.planet_orbital_distance_dropdown_selected = "Earth"
            self.planet_orbital_eccentricity_dropdown_selected = "Earth"
            self.planet_orbital_period_dropdown_selected = "Earth"
            self.planet_stellar_flux_dropdown_selected = "Earth"
        else:  # moon
            self.moon_dropdown_selected = "Moon"
            self.moon_age_dropdown_selected = "Moon"
            self.moon_radius_dropdown_selected = "Moon"
            self.moon_orbital_distance_dropdown_selected = "Moon"
            self.moon_orbital_period_dropdown_selected = "Moon"
            self.moon_temperature_dropdown_selected = "Moon"
            self.moon_gravity_dropdown_selected = "Moon"
        
        # Generate orbit grid for planets and moons
        # For moons, only generate orbit if parent is already set (from above)
        # For planets, always generate orbit
        if obj_type == "planet":
            self.generate_orbit_grid(body)
        elif obj_type == "moon" and body.get("parent"):
            # Moon's parent is already set, generate orbit grid for visualization
            self.generate_orbit_grid(body)
        
        # Automatically start simulation when at least one star and one planet are placed
        stars = [b for b in self.placed_bodies if b["type"] == "star"]
        planets = [b for b in self.placed_bodies if b["type"] == "planet"]
        
        if len(stars) > 0 and len(planets) > 0:
            self.show_simulation_builder = False
            self.show_simulation = True
            # Clear any selected body and active tab when simulation starts for better UX
            self.selected_body = None
            self.show_customization_panel = False
            self.active_tab = None
            self.clear_preview()  # Clear preview when simulation starts
            # Initialize all orbits when simulation starts
            self.initialize_all_orbits()
    
    def auto_spawn_default_system(self):
        """Spawn default Sun–Earth–Moon using user-placement logic after init."""
        print("🌍 Auto-placing Sun–Earth–Moon system...")
        # Use the *exact same* functions that user clicks trigger
        self.place_object("star", {"name": "Sun"})
        self.place_object("planet", {"name": "Earth", "semi_major_axis": 1.0})
        self.place_object("moon", {"name": "Moon", "semi_major_axis": 0.00257})
    
    def initSandbox(self):
        """Initialize the sandbox with scene setup and auto-spawn default system."""
        # Scene setup is already done in __init__
        # Delay spawn until everything ready
        threading.Timer(1.0, self.auto_spawn_default_system).start()
    
    def render_info_panel(self):
        """Render the derived physics info panel."""
        if not self.show_info_panel or not self.selected_body:
            return

        body = self.selected_body
        panel_width = 350
        panel_height = 500
        padding = 20
        
        # Position panel relative to info icon
        self.info_panel_rect = pygame.Rect(self.width - panel_width - 50, 50, panel_width, panel_height)
        
        # Draw shadow
        shadow_rect = self.info_panel_rect.copy()
        shadow_rect.move_ip(4, 4)
        pygame.draw.rect(self.screen, (20, 20, 20, 100), shadow_rect, border_radius=10)
        
        # Draw main panel
        pygame.draw.rect(self.screen, (40, 40, 60), self.info_panel_rect, border_radius=10)
        pygame.draw.rect(self.screen, self.WHITE, self.info_panel_rect, 2, border_radius=10)
        
        # Draw Title
        title_surf = self.font.render("Derived Physics", True, self.WHITE)
        title_rect = title_surf.get_rect(midtop=(self.info_panel_rect.centerx, self.info_panel_rect.top + 15))
        self.screen.blit(title_surf, title_rect)
        
        # Close button X
        self.info_panel_close_btn = pygame.Rect(self.info_panel_rect.right - 30, self.info_panel_rect.top + 10, 20, 20)
        pygame.draw.line(self.screen, self.WHITE, (self.info_panel_close_btn.left, self.info_panel_close_btn.top), (self.info_panel_close_btn.right, self.info_panel_close_btn.bottom), 2)
        pygame.draw.line(self.screen, self.WHITE, (self.info_panel_close_btn.left, self.info_panel_close_btn.bottom), (self.info_panel_close_btn.right, self.info_panel_close_btn.top), 2)

        curr_y = self.info_panel_rect.top + 60
        
        def draw_section_header(text, y):
            header = self.subtitle_font.render(text, True, (150, 200, 255))
            self.screen.blit(header, (self.info_panel_rect.left + padding, y))
            return y + 25

        def draw_stat(label, value, y, unit=""):
            label_surf = self.tiny_font.render(label, True, (200, 200, 200))
            val_text = f"{value} {unit}" if unit else str(value)
            val_surf = self.tiny_font.render(val_text, True, self.WHITE)
            self.screen.blit(label_surf, (self.info_panel_rect.left + padding, y))
            self.screen.blit(val_surf, (self.info_panel_rect.right - padding - val_surf.get_width(), y))
            return y + 20

        if body.get("type") == "planet" or body.get("type") == "moon":
            # --- Climate & Seasons ---
            curr_y = draw_section_header("Climate & Seasons", curr_y)
            stability = body.get("axial_stability", "Unknown")
            curr_y = draw_stat("Axial Stability:", stability, curr_y)
            # Small explanation for stability
            expl = "Stable tilt" if stability == "Stable" else "Chaotic variation" if stability == "Chaotic" else "No axial tilt"
            expl_surf = self.tiny_font.render(f"({expl})", True, (150, 150, 150))
            self.screen.blit(expl_surf, (self.info_panel_rect.left + padding + 10, curr_y))
            curr_y += 20
            
            flux_var = body.get("seasonal_flux_variation", 0.0)
            curr_y = draw_stat("Seasonal Flux Var:", f"{flux_var:.1f}", curr_y, "%")
            
            temp_swing = body.get("seasonal_temp_swing", 0.0)
            curr_y = draw_stat("Seasonal Temp Swing:", f"{temp_swing:.1f}", curr_y, "K")
            
            # Seasonal Contrast Bar
            curr_y = draw_stat("Seasonal Contrast:", "", curr_y)
            contrast = body.get("seasonal_contrast_score", 0.5)
            bar_rect = pygame.Rect(self.info_panel_rect.left + padding, curr_y, panel_width - 2*padding, 10)
            pygame.draw.rect(self.screen, (60, 60, 80), bar_rect)
            contrast_rect = pygame.Rect(bar_rect.left, bar_rect.top, int(bar_rect.width * contrast), bar_rect.height)
            pygame.draw.rect(self.screen, (255, 100, 100), contrast_rect)
            curr_y += 30

            # --- Thermal Breakdown ---
            curr_y = draw_section_header("Thermal Breakdown", curr_y)
            t_eq = body.get("equilibrium_temperature", 255.0)
            curr_y = draw_stat("Equilibrium Temp:", f"{t_eq:.1f}", curr_y, "K")
            
            greenhouse = body.get("greenhouse_offset", 0.0)
            curr_y = draw_stat("Greenhouse Effect:", f"+{greenhouse:.1f}", curr_y, "K")
            
            tidal = body.get("tidal_heating_delta", 0.0)
            curr_y = draw_stat("Tidal Heating:", f"+{tidal:.1f}", curr_y, "K")
            curr_y += 10

            # --- Atmospheric Physics ---
            curr_y = draw_section_header("Atmospheric Physics", curr_y)
            mag = body.get("magnetic_field_status", "None")
            curr_y = draw_stat("Magnetosphere:", mag, curr_y)
            
            erosion = body.get("atmospheric_erosion_status", "Stable")
            curr_y = draw_stat("Erosion Status:", erosion, curr_y)
            # Small explanation for erosion
            expl = "Atmosphere is protected" if erosion == "Stable" else "Losing atmosphere slowly" if erosion == "Slowly Eroding" else "Atmosphere being stripped"
            expl_surf = self.tiny_font.render(f"({expl})", True, (150, 150, 150))
            self.screen.blit(expl_surf, (self.info_panel_rect.left + padding + 10, curr_y))
            curr_y += 20

    def handle_events(self) -> bool:
        """Handle pygame events. Returns False if the window should close."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            # --- CUSTOM MODAL EVENT HANDLING (BLOCKS OTHER INPUT) ---
            if self.show_custom_modal:
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if hasattr(self, 'apply_btn_rect') and self.apply_btn_rect.collidepoint(event.pos):
                        if not self.custom_modal_error and self.custom_modal_text:
                            self.apply_custom_value()
                        continue
                    if hasattr(self, 'cancel_btn_rect') and self.cancel_btn_rect.collidepoint(event.pos):
                        self.cancel_custom_modal()
                        continue
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.cancel_custom_modal()
                    elif event.key == pygame.K_RETURN:
                        if not self.custom_modal_error and self.custom_modal_text:
                            self.apply_custom_value()
                    elif event.key == pygame.K_BACKSPACE:
                        self.custom_modal_text = self.custom_modal_text[:-1]
                        self.validate_custom_modal_input()
                    else:
                        # Allow numeric input
                        if event.unicode.isdigit() or event.unicode in ".-e+":
                            self.custom_modal_text += event.unicode
                            self.validate_custom_modal_input()
                continue
            
            # --- INFO PANEL EVENT HANDLING ---
            if self.show_info_panel:
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self.info_panel_close_btn.collidepoint(event.pos):
                        self.show_info_panel = False
                        continue
                    # Also close if clicking outside the panel
                    if not self.info_panel_rect.collidepoint(event.pos):
                        self.show_info_panel = False
                        # DON'T continue, allow the click to interact with other things if intended
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    self.show_info_panel = False
                    continue
            # ---------------------------------
            # --------------------------------------------------------
            
            # --- MODAL DROPDOWN HANDLING (TOP LAYER, EXCLUSIVE CAPTURE) ---
            if self._any_dropdown_active():
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    # Attempt to handle selection from the dropdown menu area
                    if self._handle_dropdown_selection(event.pos):
                        continue # Click handled (option selected)
                    
                    # If we reach here, it's a click outside the active dropdown menu area.
                    # Modal rule: click anywhere else closes it and is consumed.
                    self.close_all_dropdowns()
                    continue
                
                # Block all other mouse interactions while modal dropdown is open
                if event.type in [pygame.MOUSEBUTTONUP, pygame.MOUSEMOTION, pygame.MOUSEWHEEL]:
                    continue

            # Handle Reset View button click (UI only, screen space) - check before other handlers
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.reset_view_button.collidepoint(event.pos):
                    self.reset_camera()
                    continue  # Consume event to prevent other handlers
                
                # Climate/axial info icon click handling (pin/unpin tooltip)
                # Use a slightly inflated hitbox so it's easy to click.
                if hasattr(self, "climate_info_icon_rect") and self.climate_info_icon_rect:
                    expanded_info_rect = self.climate_info_icon_rect.inflate(6, 6)
                    if expanded_info_rect.collidepoint(event.pos):
                        # Toggle pinned state when clicking the (i) icon
                        self.climate_info_pinned = not getattr(self, "climate_info_pinned", False)
                        continue  # Consume click so it doesn't trigger other UI beneath
                    else:
                        # If clicking anywhere else, unpin the tooltip (if it was pinned)
                        if getattr(self, "climate_info_pinned", False):
                            self.climate_info_pinned = False
            
            # Time control interactions (⏪ ⏸/▶ ⏩) – handle before other UI (simulation only)
            handled_tc = False
            if (
                self.show_simulation
                and event.type == pygame.MOUSEBUTTONDOWN
                and event.button == 1
                and hasattr(event, "pos")
            ):
                layout = self._get_time_controls_layout()
                if (
                    layout["panel_rect"].collidepoint(event.pos)
                ):
                    if self.handle_time_controls_input(event, event.pos):
                        handled_tc = True
            if handled_tc:
                continue
            
            # Camera controls (pan/zoom)
            if event.type == pygame.MOUSEWHEEL:
                # Cancel focus mode on manual zoom
                self.camera_focus["active"] = False
                
                mouse_x, mouse_y = pygame.mouse.get_pos()
                world_before = self.screen_to_world((mouse_x, mouse_y))
                if event.y > 0:
                    self.camera_zoom *= 1.1
                elif event.y < 0:
                    self.camera_zoom /= 1.1
                self.camera_zoom = max(self.camera_zoom_min, min(self.camera_zoom_max, self.camera_zoom))
                # Adjust offset so zoom is centered on cursor
                self.camera_offset[0] = mouse_x - world_before[0] * self.camera_zoom
                self.camera_offset[1] = mouse_y - world_before[1] * self.camera_zoom
                # Invalidate orbit screen caches when zoom changes significantly (>2%)
                if abs(self.camera_zoom - self.last_zoom_for_orbits) / max(self.last_zoom_for_orbits, 1e-6) > 0.02:
                    self.orbit_screen_cache.clear()
                    self.orbit_grid_screen_cache.clear()
                    self.last_zoom_for_orbits = self.camera_zoom
                continue
            
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                # Right-click start pan
                self.is_panning = True
                self.pan_start = event.pos
                # Cancel focus mode on manual pan
                self.camera_focus["active"] = False
                continue
            if event.type == pygame.MOUSEBUTTONUP and event.button == 3:
                self.is_panning = False
                self.pan_start = None
                continue
            if event.type == pygame.MOUSEMOTION and self.is_panning and self.pan_start:
                dx = event.pos[0] - self.pan_start[0]
                dy = event.pos[1] - self.pan_start[1]
                self.camera_offset[0] += dx
                self.camera_offset[1] += dy
                self.pan_start = event.pos
                continue
            
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 2:
                now = pygame.time.get_ticks()
                if now - self.last_middle_click_time < 300:
                    # Double middle-click: reset view
                    self.reset_camera()
                self.last_middle_click_time = now
                continue

            if event.type == pygame.MOUSEBUTTONDOWN:
                # Home screen removed - start directly in sandbox
                # if self.show_home_screen:
                #     # Check if click is within the create button area
                #     button_center_x = self.width//2
                #     button_center_y = self.height*3/4
                #     button_width = 200
                #     button_height = 50
                #     button_rect = pygame.Rect(
                #         button_center_x - button_width//2,
                #         button_center_y - button_height//2,
                #         button_width,
                #         button_height
                #     )
                #     if button_rect.collidepoint(event.pos):
                #         self.show_home_screen = False
                #         self.show_simulation_builder = True
                if self.show_simulation_builder or self.show_simulation:
                    # Check if click is in the customization panel
                    if self.show_customization_panel and self.customization_panel.collidepoint(event.pos):
                        print(f'DEBUG: Mouse click at {event.pos}')
                        print(f'DEBUG: Orbital distance rect: {self.planet_orbital_distance_dropdown_rect}')
                        # Check if close button was clicked
                        if self.close_button.collidepoint(event.pos):
                            self.show_customization_panel = False
                            self.selected_body = None
                            self.selected_body_id = None
                            self.mass_input_active = False
                            self.planet_dropdown_active = False
                            self.planet_dropdown_visible = False
                            self.luminosity_input_active = False
                        # Handle planet dropdown FIRST (only for planets) - check before checkboxes
                        if (self.selected_body and self.selected_body.get('type') == 'planet' and 
                              self.planet_dropdown_rect.collidepoint(event.pos)):
                            print(f'DEBUG: Planet dropdown clicked at {event.pos}')
                            self.close_all_dropdowns()
                            self.close_all_dropdowns()
                            self.planet_dropdown_active = True
                            self.mass_input_active = False
                            self.planet_dropdown_visible = True
                            self.create_dropdown_surface()
                        # Handle moon dropdown (only for moons) - check before checkboxes
                        elif (self.selected_body and self.selected_body.get('type') == 'moon' and 
                              self.moon_dropdown_rect.collidepoint(event.pos)):
                            print(f'DEBUG: Moon dropdown clicked at {event.pos}')
                            self.close_all_dropdowns()
                            self.close_all_dropdowns()
                            self.moon_dropdown_active = True
                            self.mass_input_active = False
                            self.moon_dropdown_visible = True
                            self.create_dropdown_surface()
                        # Handle orbit toggle checkboxes (only for planets and moons) - after dropdowns
                        # Only check if click is actually on a checkbox, not just any planet/moon click
                        elif (self.selected_body and self.selected_body.get('type') in ['planet', 'moon'] and
                              (self.orbit_enabled_checkbox.collidepoint(event.pos) or 
                               self.last_revolution_checkbox.collidepoint(event.pos))):
                            if self.orbit_enabled_checkbox.collidepoint(event.pos):
                                # Toggle orbit enabled
                                body = self.get_selected_body()
                                if body:
                                    body["orbit_enabled"] = not body.get("orbit_enabled", True)
                            elif self.last_revolution_checkbox.collidepoint(event.pos):
                                # Toggle last revolution only
                                current_max = self.selected_body.get("max_orbit_points", 2000)
                                body = self.get_selected_body()
                                if body:
                                    if current_max < 1000:
                                        # Switch to full history
                                        body["max_orbit_points"] = 2000
                                    else:
                                        # Switch to last revolution only (~600 points)
                                        body["max_orbit_points"] = 600
                                    # Clear existing points when switching modes
                                    self.clear_orbit_points(body)
                        # Check if clicked on a planet option (for placement when no planet selected, or for editing selected planet)
                        elif self.planet_dropdown_visible:
                            # Handle dropdown clicks for placement (no planet selected) or editing (planet selected)
                            if not self.selected_body or self.selected_body.get('type') != 'planet':
                                # Placement mode: user is selecting a planet to place
                                dropdown_y = self.planet_dropdown_rect.bottom
                                for i, (planet_name, mass) in enumerate(self.planet_dropdown_options):
                                    option_rect = pygame.Rect(
                                        self.planet_dropdown_rect.left,
                                        dropdown_y + i * 30,
                                        self.planet_dropdown_rect.width,
                                        30
                                    )
                                    if option_rect.collidepoint(event.pos):
                                        if planet_name == "Custom":
                                            # For custom, clear selection (explicit state change)
                                            self.planet_dropdown_selected = None
                                            self.clear_preview()
                                        else:
                                            # Toggle selection - if already selected, deselect it
                                            if self.planet_dropdown_selected == planet_name:
                                                # Deselect: clear selection and preview
                                                self.planet_dropdown_selected = None
                                                self.clear_preview()
                                                print(f"Deselected planet for placement: {planet_name}")
                                            else:
                                                # Select: Store the selected planet for placement
                                                self.planet_dropdown_selected = planet_name
                                                print(f"Selected planet for placement: {planet_name}")
                                                # Set preview radius once when planet is selected (stable per object type)
                                                if planet_name in SOLAR_SYSTEM_PLANET_PRESETS:
                                                    preset = SOLAR_SYSTEM_PLANET_PRESETS[planet_name]
                                                    planet_radius_re = preset["radius"]  # In Earth radii
                                                    # Use updated perceptual scaling for preview (matches calculate_planet_visual_radius)
                                                    visual_radius = EARTH_RADIUS_PX * math.pow(planet_radius_re, 0.4)
                                                    # Authoritative clamp: min=MIN_PLANET_PX, max=star_radius_px * 0.85
                                                    # For preview, assume Sun-sized star if no star placed
                                                    self.preview_radius = int(max(MIN_PLANET_PX, min(visual_radius, SUN_RADIUS_PX * 0.85)))
                                                else:
                                                    self.preview_radius = int(MIN_PLANET_PX)  # New minimum Earth size
                                                # Initialize preview position immediately at mouse cursor
                                                self.preview_position = pygame.mouse.get_pos()
                                                # Activate placement mode - preview position will be updated every frame
                                                self.placement_mode_active = True
                                        # Keep dropdown open to show selection, but recreate surface to show highlight
                                        self.create_dropdown_surface()
                                        # Prevent event from continuing to space area handler
                                        continue
                            else:
                                # Editing mode: user is editing a selected planet
                                if self.selected_body and self.selected_body.get('type') == 'planet':
                                    dropdown_y = self.planet_dropdown_rect.bottom
                                    for i, (planet_name, mass) in enumerate(self.planet_dropdown_options):
                                        option_rect = pygame.Rect(
                                            self.planet_dropdown_rect.left,
                                            dropdown_y + i * 30,
                                            self.planet_dropdown_rect.width,
                                            30
                                        )
                                        if option_rect.collidepoint(event.pos):
                                            print(f"PLANET DROPDOWN CLICK (path 1): planet_name={planet_name}, mass={mass}, selected_body_id={self.selected_body_id}", flush=True)
                                            if planet_name == "Custom":
                                                # Trigger custom modal instead of showing inline input
                                                body = self.get_selected_body()
                                                self.previous_dropdown_selection = body.get("planet_dropdown_selected") if body else None
                                                
                                                self.show_custom_modal = True
                                                self.custom_modal_title = "Set Custom Planet Mass"
                                                self.custom_modal_helper = "Enter mass in Earth masses (M⊕). Allowed: 0.01 – 500"
                                                self.custom_modal_min = 0.01
                                                self.custom_modal_max = 500.0
                                                self.custom_modal_unit = "M⊕"
                                                self.pending_custom_field = "mass"
                                                self.pending_custom_body_id = self.selected_body_id
                                                
                                                self.custom_modal_text = ""
                                                self.validate_custom_modal_input()
                                            else:
                                                # CRITICAL: Use update_selected_body_property to ensure we update ONLY the selected body
                                                # This prevents cross-body mutations by always using ID-based registry lookup
                                                if self.selected_body_id:
                                                    # Convert mass to float if needed
                                                    if hasattr(mass, 'item'):
                                                        mass_value = float(mass.item())
                                                    else:
                                                        mass_value = float(mass)
                                                    print(f"ABOUT TO CALL update_selected_body_property (path 1) with mass={mass_value}", flush=True)
                                                    # Use the safe update function that verifies body isolation
                                                    if self.update_selected_body_property("mass", mass_value, "mass"):
                                                        print(f"DEBUG: Updated mass via registry for body id={self.selected_body_id}, new_mass={mass_value}", flush=True)
                                                        # CRITICAL: Store dropdown selection in the body's dict, not global state
                                                        body = self.get_selected_body()
                                                        if body:
                                                            body["planet_dropdown_selected"] = planet_name
                                                    else:
                                                        print(f"ERROR: update_selected_body_property returned False!", flush=True)
                                                else:
                                                    print(f"ERROR: No selected_body_id in path 1!", flush=True)
                                                self.show_custom_mass_input = False
                                            # REMOVED: self.planet_dropdown_selected = planet_name  # This was global, now stored per-body
                                            self.planet_age_dropdown_selected = "Solar System Planets (4.6 Gyr)"
                                            self.planet_gravity_dropdown_selected = "Earth"
                                            self.planet_dropdown_visible = False
                                            self.planet_dropdown_active = False
                                            break
                            # Only close dropdown if clicking outside both the dropdown and its options
                            if (self.planet_dropdown_visible and not 
                                (self.planet_dropdown_rect.collidepoint(event.pos) or 
                                 any(pygame.Rect(
                                     self.planet_dropdown_rect.left,
                                     self.planet_dropdown_rect.bottom + i * 30,
                                     self.planet_dropdown_rect.width,
                                     30
                                 ).collidepoint(event.pos)
                                 for i in range(len(self.planet_dropdown_options))))):
                                self.planet_dropdown_visible = False
                                self.mass_input_active = False
                                self.planet_dropdown_active = False
                        # Handle planet age dropdown (only for planets)
                        elif (self.selected_body and self.selected_body.get('type') == 'planet' and 
                              self.planet_age_dropdown_rect.collidepoint(event.pos)):
                            self.close_all_dropdowns()
                            self.planet_age_dropdown_active = True
                            self.age_input_active = False
                            self.planet_age_dropdown_visible = True
                            self.create_dropdown_surface()
                        # Check if clicked on a planet age option (only for planets)
                        elif (self.selected_body and self.selected_body.get('type') == 'planet' and 
                              self.planet_age_dropdown_visible):
                            dropdown_y = self.planet_age_dropdown_rect.bottom
                            for i, (age_name, age) in enumerate(self.planet_age_dropdown_options):
                                option_rect = pygame.Rect(
                                    self.planet_age_dropdown_rect.left,
                                    dropdown_y + i * 30,
                                    self.planet_age_dropdown_rect.width,
                                    30
                                )
                                if option_rect.collidepoint(event.pos):
                                    if age_name == "Custom":
                                        # Trigger custom modal for planet age
                                        body = self.get_selected_body()
                                        self.previous_dropdown_selection = body.get("planet_age_dropdown_selected") if body else None
                                        
                                        self.show_custom_modal = True
                                        self.custom_modal_title = "Set Custom Planet Age"
                                        self.custom_modal_helper = "Enter age in Gigayears (Gyr). Allowed: 0.1 – 13.8"
                                        self.custom_modal_min = 0.1
                                        self.custom_modal_max = 13.8
                                        self.custom_modal_unit = "Gyr"
                                        self.pending_custom_field = "age"
                                        self.pending_custom_body_id = self.selected_body_id
                                        
                                        self.custom_modal_text = ""
                                        self.validate_custom_modal_input()
                                    else:
                                        # Ensure we're updating the correct selected body
                                        self.update_selected_body_property("age", age, "age")
                                        self.show_custom_age_input = False
                                    self.planet_age_dropdown_selected = age_name
                                    self.planet_age_dropdown_visible = False
                                    break
                            # Only close dropdown if clicking outside both the dropdown and its options
                            if (self.planet_age_dropdown_visible and not 
                                (self.planet_age_dropdown_rect.collidepoint(event.pos) or 
                                 any(pygame.Rect(
                                     self.planet_age_dropdown_rect.left,
                                     self.planet_age_dropdown_rect.bottom + i * 30,
                                     self.planet_age_dropdown_rect.width,
                                     30
                                 ).collidepoint(event.pos)
                                 for i in range(len(self.planet_age_dropdown_options))))):
                                self.planet_age_dropdown_visible = False
                                self.age_input_active = False
                                self.planet_age_dropdown_active = False
                        # Handle planet radius dropdown (only for planets)
                        elif (self.selected_body and self.selected_body.get('type') == 'planet' and 
                              self.planet_radius_dropdown_rect.collidepoint(event.pos)):
                            self.close_all_dropdowns()
                            self.planet_radius_dropdown_active = True
                            self.planet_radius_dropdown_visible = True
                            self.create_dropdown_surface()
                        # Check if clicked on a planet radius option (only for planets)
                        elif (self.selected_body and self.selected_body.get('type') == 'planet' and 
                              self.planet_radius_dropdown_visible):
                            dropdown_y = self.planet_radius_dropdown_rect.bottom
                            for i, (radius_name, radius) in enumerate(self.planet_radius_dropdown_options):
                                option_rect = pygame.Rect(
                                    self.planet_radius_dropdown_rect.left,
                                    dropdown_y + i * 30,
                                    self.planet_radius_dropdown_rect.width,
                                    30
                                )
                                if option_rect.collidepoint(event.pos):
                                    if radius_name == "Custom":
                                        # Trigger custom modal for planet radius
                                        body = self.get_selected_body()
                                        self.previous_dropdown_selection = body.get("planet_radius_dropdown_selected") if body else None
                                        
                                        self.show_custom_modal = True
                                        self.custom_modal_title = "Set Custom Planet Radius"
                                        self.custom_modal_helper = "Enter radius in Earth radii (R⊕). Allowed: 0.1 – 20"
                                        self.custom_modal_min = 0.1
                                        self.custom_modal_max = 20.0
                                        self.custom_modal_unit = "R⊕"
                                        self.pending_custom_field = "radius"
                                        self.pending_custom_body_id = self.selected_body_id
                                        
                                        self.custom_modal_text = ""
                                        self.validate_custom_modal_input()
                                    else:
                                        # CRITICAL: radius is in Earth radii (R⊕), store directly
                                        # No conversion needed - radius dropdown value IS in R⊕
                                        body = self.get_selected_body()
                                        if body:
                                            # Store radius in R⊕ (Earth radii)
                                            self.update_selected_body_property("radius", radius, "radius")
                                            # Hitbox will be computed from visual radius on render
                                            # No physics updates - radius affects visual only
                                        self.show_custom_radius_input = False
                                    self.planet_radius_dropdown_selected = radius_name
                                    self.planet_radius_dropdown_visible = False
                                    self.planet_radius_dropdown_active = False
                                    break
                            # Only close dropdown if clicking outside both the dropdown and its options
                            if (self.planet_radius_dropdown_visible and not 
                                (self.planet_radius_dropdown_rect.collidepoint(event.pos) or 
                                 any(pygame.Rect(
                                     self.planet_radius_dropdown_rect.left,
                                     self.planet_radius_dropdown_rect.bottom + i * 30,
                                     self.planet_radius_dropdown_rect.width,
                                     30
                                 ).collidepoint(event.pos)
                                 for i in range(len(self.planet_radius_dropdown_options))))):
                                self.planet_radius_dropdown_visible = False
                                self.planet_radius_dropdown_active = False
                        # Handle planet temperature dropdown (only for planets)
                        elif (self.selected_body and self.selected_body.get('type') == 'planet' and 
                              self.planet_temperature_dropdown_rect.collidepoint(event.pos)):
                            self.close_all_dropdowns()
                            self.planet_temperature_dropdown_active = True
                            self.planet_temperature_dropdown_visible = True
                            self.create_dropdown_surface()
                        # Check if clicked on a planet temperature option (only for planets)
                        elif (self.selected_body and self.selected_body.get('type') == 'planet' and 
                              self.planet_temperature_dropdown_visible):
                            dropdown_y = self.planet_temperature_dropdown_rect.bottom
                            for i, (temp_name, temp) in enumerate(self.planet_temperature_dropdown_options):
                                option_rect = pygame.Rect(
                                    self.planet_temperature_dropdown_rect.left,
                                    dropdown_y + i * 30,
                                    self.planet_temperature_dropdown_rect.width,
                                    30
                                )
                                if option_rect.collidepoint(event.pos):
                                    if temp_name == "Custom":
                                        # Trigger custom modal for planet temperature
                                        body = self.get_selected_body()
                                        self.previous_dropdown_selection = body.get("planet_temperature_dropdown_selected") if body else None
                                        
                                        self.show_custom_modal = True
                                        self.custom_modal_title = "Set Planet Temperature"
                                        self.custom_modal_helper = "Enter equilibrium temperature in Kelvin (K). Allowed: 1 – 5000"
                                        self.custom_modal_min = 1.0
                                        self.custom_modal_max = 5000.0
                                        self.custom_modal_unit = "K"
                                        self.pending_custom_field = "temperature"
                                        self.pending_custom_body_id = self.selected_body_id
                                        
                                        self.custom_modal_text = ""
                                        self.validate_custom_modal_input()
                                    else:
                                        # When temperature is set via dropdown, set it as equilibrium temperature
                                        # (assuming no greenhouse effect is applied yet)
                                        body = self.get_selected_body()
                                        if body:
                                            self.update_selected_body_property("equilibrium_temperature", temp, "equilibrium_temperature")
                                            # Apply current greenhouse offset if set
                                            if 'greenhouse_offset' in body:
                                                delta_t = body.get('greenhouse_offset', 0.0)
                                                self.update_selected_body_property("temperature", temp + delta_t, "temperature")
                                            else:
                                                self.update_selected_body_property("temperature", temp, "temperature")
                                        self.show_custom_planet_temperature_input = False
                                        self._update_planet_scores()
                                    self.planet_temperature_dropdown_selected = temp_name
                                    self.planet_temperature_dropdown_visible = False
                                    self.planet_temperature_dropdown_active = False
                                    break
                            # Only close dropdown if clicking outside both the dropdown and its options
                            if (self.planet_temperature_dropdown_visible and not 
                                (self.planet_temperature_dropdown_rect.collidepoint(event.pos) or 
                                 any(pygame.Rect(
                                     self.planet_temperature_dropdown_rect.left,
                                     self.planet_temperature_dropdown_rect.bottom + i * 30,
                                     self.planet_temperature_dropdown_rect.width,
                                     30
                                 ).collidepoint(event.pos)
                                 for i in range(len(self.planet_temperature_dropdown_options))))):
                                self.planet_temperature_dropdown_visible = False
                                self.planet_temperature_dropdown_active = False
                        # Handle planet atmospheric composition dropdown (only for planets)
                        elif (self.selected_body and self.selected_body.get('type') == 'planet' and 
                              self.planet_atmosphere_dropdown_rect.collidepoint(event.pos)):
                            self.close_all_dropdowns()
                            self.planet_atmosphere_dropdown_active = True
                            self.planet_atmosphere_dropdown_visible = True
                            self.create_dropdown_surface()
                        # Check if clicked on an atmospheric composition option (only for planets)
                        elif (self.selected_body and self.selected_body.get('type') == 'planet' and 
                              self.planet_atmosphere_dropdown_visible):
                            dropdown_y = self.planet_atmosphere_dropdown_rect.bottom
                            for i, (atm_name, delta_t) in enumerate(self.planet_atmosphere_dropdown_options):
                                option_rect = pygame.Rect(
                                    self.planet_atmosphere_dropdown_rect.left,
                                    dropdown_y + i * 30,
                                    self.planet_atmosphere_dropdown_rect.width,
                                    30
                                )
                                if option_rect.collidepoint(event.pos):
                                    if atm_name == "Custom":
                                        # Trigger custom modal for planet atmosphere (greenhouse offset)
                                        body = self.get_selected_body()
                                        self.previous_dropdown_selection = body.get("planet_atmosphere_dropdown_selected") if body else None
                                        
                                        self.show_custom_modal = True
                                        self.custom_modal_title = "Set Atmosphere Effect"
                                        self.custom_modal_helper = "Enter greenhouse warming (ΔT) in Kelvin (K). Allowed: 0 – 1000"
                                        self.custom_modal_min = 0.0
                                        self.custom_modal_max = 1000.0
                                        self.custom_modal_unit = "K"
                                        self.pending_custom_field = "atmosphere"
                                        self.pending_custom_body_id = self.selected_body_id
                                        
                                        self.custom_modal_text = ""
                                        self.validate_custom_modal_input()
                                    else:
                                        # Calculate new surface temperature: T_surface = T_eq + ΔT_greenhouse
                                        # If T_eq is not set, use current temperature as T_eq
                                        if 'equilibrium_temperature' not in self.selected_body:
                                            # Use current temperature as equilibrium if not set
                                            current_temp = self.selected_body.get('temperature', 255)  # Default to ~Earth's T_eq
                                            self.update_selected_body_property("equilibrium_temperature", current_temp, "equilibrium_temperature")
                                        T_eq = self.selected_body.get('equilibrium_temperature', 255)
                                        T_surface = T_eq + delta_t
                                        # Update temperature and greenhouse offset via the per-object registry
                                        self.update_selected_body_property("temperature", T_surface, "temperature")
                                        self.update_selected_body_property("greenhouse_offset", delta_t, "greenhouse_offset")
                                        self.show_custom_atmosphere_input = False
                                        # Update scores (f_T and H) - will be handled by score calculation
                                        self._update_planet_scores()
                                    self.planet_atmosphere_dropdown_selected = atm_name
                                    self.planet_atmosphere_dropdown_visible = False
                                    self.planet_atmosphere_dropdown_active = False
                                    break
                            # Only close dropdown if clicking outside both the dropdown and its options
                            if (self.planet_atmosphere_dropdown_visible and not 
                                (self.planet_atmosphere_dropdown_rect.collidepoint(event.pos) or 
                                 any(pygame.Rect(
                                     self.planet_atmosphere_dropdown_rect.left,
                                     self.planet_atmosphere_dropdown_rect.bottom + i * 30,
                                     self.planet_atmosphere_dropdown_rect.width,
                                     30
                                 ).collidepoint(event.pos)
                                 for i in range(len(self.planet_atmosphere_dropdown_options))))):
                                self.planet_atmosphere_dropdown_visible = False
                                self.planet_atmosphere_dropdown_active = False
                        # Handle planet gravity dropdown (only for planets)
                        elif (self.selected_body and self.selected_body.get('type') == 'planet' and 
                              self.planet_gravity_dropdown_rect.collidepoint(event.pos)):
                            self.close_all_dropdowns()
                            self.planet_gravity_dropdown_active = True
                            self.planet_gravity_dropdown_visible = True
                            self.create_dropdown_surface()
                        # Handle planet orbital distance dropdown (only for planets)
                        elif (self.selected_body and self.selected_body.get('type') == 'planet' and 
                              self.planet_orbital_distance_dropdown_rect.collidepoint(event.pos)):
                            print('DEBUG: Orbital distance dropdown clicked')
                            self.close_all_dropdowns()
                            self.planet_orbital_distance_dropdown_active = True
                            self.planet_orbital_distance_dropdown_visible = True
                            self.create_dropdown_surface()
                        # Handle planet orbital eccentricity dropdown (only for planets)
                        elif (self.selected_body and self.selected_body.get('type') == 'planet' and 
                              self.planet_orbital_eccentricity_dropdown_rect.collidepoint(event.pos)):
                            print('DEBUG: Orbital eccentricity dropdown clicked')
                            self.close_all_dropdowns()
                            self.planet_orbital_eccentricity_dropdown_active = True
                            self.planet_orbital_eccentricity_dropdown_visible = True
                            self.create_dropdown_surface()
                        # Handle planet orbital period dropdown (only for planets)
                        elif (self.selected_body and self.selected_body.get('type') == 'planet' and 
                              self.planet_orbital_period_dropdown_rect.collidepoint(event.pos)):
                            print('DEBUG: Orbital period dropdown clicked')
                            self.close_all_dropdowns()
                            self.planet_orbital_period_dropdown_active = True
                            self.planet_orbital_period_dropdown_visible = True
                            self.create_dropdown_surface()
                        # Check if clicked on a planet gravity option (only for planets)
                        elif (self.selected_body and self.selected_body.get('type') == 'planet' and 
                              self.planet_gravity_dropdown_visible):
                            dropdown_y = self.planet_gravity_dropdown_rect.bottom
                            for i, (gravity_name, gravity) in enumerate(self.planet_gravity_dropdown_options):
                                option_rect = pygame.Rect(
                                    self.planet_gravity_dropdown_rect.left,
                                    dropdown_y + i * 30,
                                    self.planet_gravity_dropdown_rect.width,
                                    30
                                )
                                if option_rect.collidepoint(event.pos):
                                    if gravity_name == "Custom":
                                        # Trigger custom modal for planet gravity
                                        body = self.get_selected_body()
                                        self.previous_dropdown_selection = body.get("planet_gravity_dropdown_selected") if body else None
                                        
                                        self.show_custom_modal = True
                                        self.custom_modal_title = "Set Surface Gravity"
                                        self.custom_modal_helper = "Changing gravity will recompute mass to preserve radius. Range: 0.1 – 500 m/s²"
                                        self.custom_modal_min = 0.1
                                        self.custom_modal_max = 500.0
                                        self.custom_modal_unit = "m/s²"
                                        self.pending_custom_field = "gravity"
                                        self.pending_custom_body_id = self.selected_body_id
                                        
                                        self.custom_modal_text = ""
                                        self.validate_custom_modal_input()
                                    else:
                                        self.update_selected_body_property("gravity", gravity, "gravity")
                                        self.show_custom_planet_gravity_input = False
                                    self.planet_gravity_dropdown_selected = gravity_name
                                    self.planet_gravity_dropdown_visible = False
                                    self.planet_gravity_dropdown_active = False
                                    break
                            # Only close dropdown if clicking outside both the dropdown and its options
                            if (self.planet_gravity_dropdown_visible and not 
                                (self.planet_gravity_dropdown_rect.collidepoint(event.pos) or 
                                 any(pygame.Rect(
                                     self.planet_gravity_dropdown_rect.left,
                                     self.planet_gravity_dropdown_rect.bottom + i * 30,
                                     self.planet_gravity_dropdown_rect.width,
                                     30
                                 ).collidepoint(event.pos)
                                 for i in range(len(self.planet_gravity_dropdown_options))))):
                                self.planet_gravity_dropdown_visible = False
                                self.planet_gravity_dropdown_active = False
                        # Check if clicked on a planet orbital distance option (only for planets)
                        elif (self.selected_body and self.selected_body.get('type') == 'planet' and 
                              self.planet_orbital_distance_dropdown_visible):
                            dropdown_y = self.planet_orbital_distance_dropdown_rect.bottom
                            for i, (dist_name, dist) in enumerate(self.planet_orbital_distance_dropdown_options):
                                option_rect = pygame.Rect(
                                    self.planet_orbital_distance_dropdown_rect.left,
                                    dropdown_y + i * 30,
                                    self.planet_orbital_distance_dropdown_rect.width,
                                    30
                                )
                                if option_rect.collidepoint(event.pos):
                                    if dist_name == "Custom":
                                        # Trigger custom modal for planet orbital distance
                                        body = self.get_selected_body()
                                        self.previous_dropdown_selection = body.get("planet_orbital_distance_dropdown_selected") if body else None
                                        
                                        self.show_custom_modal = True
                                        self.custom_modal_title = "Set Orbital Distance"
                                        self.custom_modal_helper = "Enter orbital distance in AU. Allowed: 0.01 – 100"
                                        self.custom_modal_min = 0.01
                                        self.custom_modal_max = 100.0
                                        self.custom_modal_unit = "AU"
                                        self.pending_custom_field = "semi_major_axis"
                                        self.pending_custom_body_id = self.selected_body_id
                                        
                                        self.custom_modal_text = ""
                                        self.validate_custom_modal_input()
                                    else:
                                        body = self.get_selected_body()
                                        if body:
                                            self.update_selected_body_property("semiMajorAxis", dist, "semiMajorAxis")
                                            # CRITICAL: Update position using centralized function
                                            # This ensures position is derived from semiMajorAxis via visual scaling
                                            parent_star = next((b for b in self.placed_bodies if b["name"] == body.get("parent")), None)
                                            if parent_star is None and body.get("parent_obj"):
                                                parent_star = body["parent_obj"]
                                            if parent_star:
                                                trace(f"PRE_WRITE {body['name']} pos={body['position'].copy()} source=handle_events_planet_distance")
                                                self.compute_planet_position(body, parent_star)
                                                trace(f"POST_WRITE {body['name']} pos={body['position'].copy()} source=handle_events_planet_distance")
                                            self.generate_orbit_grid(body)
                                        self.show_custom_orbital_distance_input = False
                                    self.planet_orbital_distance_dropdown_selected = dist_name
                                    self.planet_orbital_distance_dropdown_visible = False
                                    self.planet_orbital_distance_dropdown_active = False
                                    break
                            # Only close dropdown if clicking outside both the dropdown and its options
                            if (self.planet_orbital_distance_dropdown_visible and not 
                                (self.planet_orbital_distance_dropdown_rect.collidepoint(event.pos) or 
                                 any(pygame.Rect(
                                     self.planet_orbital_distance_dropdown_rect.left,
                                     self.planet_orbital_distance_dropdown_rect.bottom + i * 30,
                                     self.planet_orbital_distance_dropdown_rect.width,
                                     30
                                 ).collidepoint(event.pos)
                                 for i in range(len(self.planet_orbital_distance_dropdown_options))))):
                                self.planet_orbital_distance_dropdown_visible = False
                                self.planet_orbital_distance_dropdown_active = False
                        # Check if clicked on a planet orbital eccentricity option (only for planets)
                        elif (self.selected_body and self.selected_body.get('type') == 'planet' and 
                              self.planet_orbital_eccentricity_dropdown_visible):
                            dropdown_y = self.planet_orbital_eccentricity_dropdown_rect.bottom
                            for i, (ecc_name, ecc) in enumerate(self.planet_orbital_eccentricity_dropdown_options):
                                option_rect = pygame.Rect(
                                    self.planet_orbital_eccentricity_dropdown_rect.left,
                                    dropdown_y + i * 30,
                                    self.planet_orbital_eccentricity_dropdown_rect.width,
                                    30
                                )
                                if option_rect.collidepoint(event.pos):
                                    if ecc_name == "Custom":
                                        self.show_custom_orbital_eccentricity_input = True
                                        self.orbital_eccentricity_input_active = True
                                        self.orbital_eccentricity_input_text = f"{self.selected_body.get('eccentricity', 0.0167):.3f}"
                                    else:
                                        self.update_selected_body_property("eccentricity", ecc, "eccentricity")
                                        self.show_custom_orbital_eccentricity_input = False
                                    self.planet_orbital_eccentricity_dropdown_selected = ecc_name
                                    self.planet_orbital_eccentricity_dropdown_visible = False
                                    self.planet_orbital_eccentricity_dropdown_active = False
                                    break
                            # Only close dropdown if clicking outside both the dropdown and its options
                            if (self.planet_orbital_eccentricity_dropdown_visible and not 
                                (self.planet_orbital_eccentricity_dropdown_rect.collidepoint(event.pos) or 
                                 any(pygame.Rect(
                                     self.planet_orbital_eccentricity_dropdown_rect.left,
                                     self.planet_orbital_eccentricity_dropdown_rect.bottom + i * 30,
                                     self.planet_orbital_eccentricity_dropdown_rect.width,
                                     30
                                 ).collidepoint(event.pos)
                                 for i in range(len(self.planet_orbital_eccentricity_dropdown_options))))):
                                self.planet_orbital_eccentricity_dropdown_visible = False
                                self.planet_orbital_eccentricity_dropdown_active = False
                        # Check if clicked on a planet orbital period option (only for planets)
                        elif (self.selected_body and self.selected_body.get('type') == 'planet' and 
                              self.planet_orbital_period_dropdown_visible):
                            dropdown_y = self.planet_orbital_period_dropdown_rect.bottom
                            for i, (period_name, period) in enumerate(self.planet_orbital_period_dropdown_options):
                                option_rect = pygame.Rect(
                                    self.planet_orbital_period_dropdown_rect.left,
                                    dropdown_y + i * 30,
                                    self.planet_orbital_period_dropdown_rect.width,
                                    30
                                )
                                if option_rect.collidepoint(event.pos):
                                    if period_name == "Custom":
                                        self.show_custom_orbital_period_input = True
                                        self.orbital_period_input_active = True
                                        self.orbital_period_input_text = f"{self.selected_body.get('orbital_period', 365.25):.0f}"
                                    else:
                                        self.update_selected_body_property("orbital_period", period, "orbital_period")
                                        self.show_custom_orbital_period_input = False
                                    self.planet_orbital_period_dropdown_selected = period_name
                                    self.planet_orbital_period_dropdown_visible = False
                                    self.planet_orbital_period_dropdown_active = False
                                    break
                            # Only close dropdown if clicking outside both the dropdown and its options
                            if (self.planet_orbital_period_dropdown_visible and not 
                                (self.planet_orbital_period_dropdown_rect.collidepoint(event.pos) or 
                                 any(pygame.Rect(
                                     self.planet_orbital_period_dropdown_rect.left,
                                     self.planet_orbital_period_dropdown_rect.bottom + i * 30,
                                     self.planet_orbital_period_dropdown_rect.width,
                                     30
                                 ).collidepoint(event.pos)
                                 for i in range(len(self.planet_orbital_period_dropdown_options))))):
                                                            self.planet_orbital_period_dropdown_visible = False
                            self.planet_orbital_period_dropdown_active = False
                        # Handle planet stellar flux dropdown (only for planets)
                        elif (self.selected_body and self.selected_body.get('type') == 'planet' and 
                              self.planet_stellar_flux_dropdown_rect.collidepoint(event.pos)):
                            print('DEBUG: Stellar flux dropdown clicked')
                            self.close_all_dropdowns()
                            self.close_all_dropdowns()
                            self.planet_stellar_flux_dropdown_active = True
                            self.planet_stellar_flux_dropdown_visible = True
                            # Deactivate other dropdowns
                            self.planet_dropdown_active = False
                            self.planet_dropdown_visible = False
                            self.planet_age_dropdown_active = False
                            self.planet_age_dropdown_visible = False
                            self.planet_radius_dropdown_active = False
                            self.planet_radius_dropdown_visible = False
                            self.planet_temperature_dropdown_active = False
                            self.planet_temperature_dropdown_visible = False
                            self.planet_gravity_dropdown_active = False
                            self.planet_gravity_dropdown_visible = False
                            self.planet_orbital_distance_dropdown_active = False
                            self.planet_orbital_distance_dropdown_visible = False
                            self.planet_orbital_eccentricity_dropdown_active = False
                            self.planet_orbital_eccentricity_dropdown_visible = False
                            self.planet_orbital_period_dropdown_active = False
                            self.planet_orbital_period_dropdown_visible = False
                            self.create_dropdown_surface()
                        # Handle planet density dropdown (only for planets)
                        elif (self.selected_body and self.selected_body.get('type') == 'planet' and 
                              self.planet_density_dropdown_rect.collidepoint(event.pos)):
                            print('DEBUG: Density dropdown clicked')
                            self.close_all_dropdowns()
                            self.close_all_dropdowns()
                            self.planet_density_dropdown_active = True
                            self.planet_density_dropdown_visible = True
                            # Deactivate other dropdowns
                            self.planet_dropdown_active = False
                            self.planet_dropdown_visible = False
                            self.planet_age_dropdown_active = False
                            self.planet_age_dropdown_visible = False
                            self.planet_radius_dropdown_active = False
                            self.planet_radius_dropdown_visible = False
                            self.planet_temperature_dropdown_active = False
                            self.planet_temperature_dropdown_visible = False
                            self.planet_gravity_dropdown_active = False
                            self.planet_gravity_dropdown_visible = False
                            self.planet_orbital_distance_dropdown_active = False
                            self.planet_orbital_distance_dropdown_visible = False
                            self.planet_orbital_eccentricity_dropdown_active = False
                            self.planet_orbital_eccentricity_dropdown_visible = False
                            self.planet_orbital_period_dropdown_active = False
                            self.planet_orbital_period_dropdown_visible = False
                            self.planet_stellar_flux_dropdown_active = False
                            self.planet_stellar_flux_dropdown_visible = False
                            self.create_dropdown_surface()
                        # Check if clicked on a planet stellar flux option (only for planets)
                        elif (self.selected_body and self.selected_body.get('type') == 'planet' and 
                              self.planet_stellar_flux_dropdown_visible):
                            dropdown_y = self.planet_stellar_flux_dropdown_rect.bottom
                            for i, (flux_name, flux) in enumerate(self.planet_stellar_flux_dropdown_options):
                                option_rect = pygame.Rect(
                                    self.planet_stellar_flux_dropdown_rect.left,
                                    dropdown_y + i * 30,
                                    self.planet_stellar_flux_dropdown_rect.width,
                                    30
                                )
                                if option_rect.collidepoint(event.pos):
                                    if flux_name == "Custom":
                                        self.show_custom_stellar_flux_input = True
                                        self.stellar_flux_input_active = True
                                        self.stellar_flux_input_text = f"{self.selected_body.get('stellarFlux', 1.0):.3f}"
                                    else:
                                        self.update_selected_body_property("stellarFlux", flux, "stellarFlux")
                                        self.show_custom_stellar_flux_input = False
                                    self.planet_stellar_flux_dropdown_selected = flux_name
                                    self.planet_stellar_flux_dropdown_visible = False
                                    self.planet_stellar_flux_dropdown_active = False
                                    break
                            # Only close dropdown if clicking outside both the dropdown and its options
                            if (self.planet_stellar_flux_dropdown_visible and not 
                                (self.planet_stellar_flux_dropdown_rect.collidepoint(event.pos) or 
                                 any(pygame.Rect(
                                     self.planet_stellar_flux_dropdown_rect.left,
                                     self.planet_stellar_flux_dropdown_rect.bottom + i * 30,
                                     self.planet_stellar_flux_dropdown_rect.width,
                                     30
                                 ).collidepoint(event.pos)
                                 for i in range(len(self.planet_stellar_flux_dropdown_options))))):
                                self.planet_stellar_flux_dropdown_visible = False
                                self.planet_stellar_flux_dropdown_active = False
                        # Check if clicked on a planet density option (only for planets)
                        elif (self.selected_body and self.selected_body.get('type') == 'planet' and 
                              self.planet_density_dropdown_visible):
                            dropdown_y = self.planet_density_dropdown_rect.bottom
                            for i, (density_name, density) in enumerate(self.planet_density_dropdown_options):
                                option_rect = pygame.Rect(
                                    self.planet_density_dropdown_rect.left,
                                    dropdown_y + i * 30,
                                    self.planet_density_dropdown_rect.width,
                                    30
                                )
                                if option_rect.collidepoint(event.pos):
                                    if density_name == "Custom":
                                        # Trigger custom modal for planet density
                                        body = self.get_selected_body()
                                        self.previous_dropdown_selection = body.get("planet_density_dropdown_selected") if body else None
                                        
                                        self.show_custom_modal = True
                                        self.custom_modal_title = "Set Custom Planet Density"
                                        self.custom_modal_helper = "Enter density in g/cm³. Earth = 5.51. Range: 0.5 – 25.0"
                                        self.custom_modal_min = 0.5
                                        self.custom_modal_max = 25.0
                                        self.custom_modal_unit = "g/cm³"
                                        self.pending_custom_field = "density"
                                        self.pending_custom_body_id = self.selected_body_id
                                        
                                        self.custom_modal_text = ""
                                        self.validate_custom_modal_input()
                                    else:
                                        self.update_selected_body_property("density", density, "density")
                                    self.planet_density_dropdown_selected = density_name
                                    self.planet_density_dropdown_visible = False
                                    self.planet_density_dropdown_active = False
                                    break
                            # Only close dropdown if clicking outside both the dropdown and its options
                            if (self.planet_density_dropdown_visible and not 
                                (self.planet_density_dropdown_rect.collidepoint(event.pos) or 
                                 any(pygame.Rect(
                                     self.planet_density_dropdown_rect.left,
                                     self.planet_density_dropdown_rect.bottom + i * 30,
                                     self.planet_density_dropdown_rect.width,
                                     30
                                 ).collidepoint(event.pos)
                                 for i in range(len(self.planet_density_dropdown_options))))):
                                self.planet_density_dropdown_visible = False
                                self.planet_density_dropdown_active = False
                        # Handle moon dropdown (only for moons)
                        elif (self.selected_body and self.selected_body.get('type') == 'moon' and 
                              self.moon_dropdown_rect.collidepoint(event.pos)):
                            self.close_all_dropdowns()
                            self.moon_dropdown_active = True
                            self.mass_input_active = False
                            self.moon_dropdown_visible = True
                            self.create_dropdown_surface()
                        # Handle moon age dropdown (only for moons)
                        elif (self.selected_body and self.selected_body.get('type') == 'moon' and 
                              self.moon_age_dropdown_rect.collidepoint(event.pos)):
                            self.close_all_dropdowns()
                            self.moon_age_dropdown_active = True
                            self.age_input_active = False
                            self.moon_age_dropdown_visible = True
                            self.create_dropdown_surface()
                        # Handle moon radius dropdown (only for moons)
                        elif (self.selected_body and self.selected_body.get('type') == 'moon' and 
                              self.moon_radius_dropdown_rect.collidepoint(event.pos)):
                            self.close_all_dropdowns()
                            self.close_all_dropdowns()
                            self.moon_radius_dropdown_active = True
                            self.moon_radius_dropdown_visible = True
                            self.create_dropdown_surface()
                        # Handle moon orbital distance dropdown (only for moons)
                        elif (self.selected_body and self.selected_body.get('type') == 'moon' and 
                              self.moon_orbital_distance_dropdown_rect.collidepoint(event.pos)):
                            self.close_all_dropdowns()
                            self.close_all_dropdowns()
                            self.moon_orbital_distance_dropdown_active = True
                            self.moon_orbital_distance_dropdown_visible = True
                            self.create_dropdown_surface()
                        # Handle moon orbital period dropdown (only for moons)
                        elif (self.selected_body and self.selected_body.get('type') == 'moon' and 
                              self.moon_orbital_period_dropdown_rect.collidepoint(event.pos)):
                            self.close_all_dropdowns()
                            self.close_all_dropdowns()
                            self.moon_orbital_period_dropdown_active = True
                            self.moon_orbital_period_dropdown_visible = True
                            self.create_dropdown_surface()
                        # Handle moon temperature dropdown (only for moons)
                        elif (self.selected_body and self.selected_body.get('type') == 'moon' and 
                              self.moon_temperature_dropdown_rect.collidepoint(event.pos)):
                            self.close_all_dropdowns()
                            self.close_all_dropdowns()
                            self.moon_temperature_dropdown_active = True
                            self.moon_temperature_dropdown_visible = True
                            self.create_dropdown_surface()
                        # Handle moon gravity dropdown (only for moons)
                        elif (self.selected_body and self.selected_body.get('type') == 'moon' and 
                              self.moon_gravity_dropdown_rect.collidepoint(event.pos)):
                            self.close_all_dropdowns()
                            self.close_all_dropdowns()
                            self.moon_gravity_dropdown_active = True
                            self.moon_gravity_dropdown_visible = True
                            self.create_dropdown_surface()
                        # Check if clicked on a moon option (only for moons)
                        elif (self.selected_body and self.selected_body.get('type') == 'moon' and 
                              self.moon_dropdown_visible):
                            dropdown_y = self.moon_dropdown_rect.bottom
                            for i, option_data in enumerate(self.moon_dropdown_options):
                                if len(option_data) == 3:
                                    moon_name, mass, unit = option_data
                                else:
                                    moon_name, mass = option_data
                                    unit = None
                                option_rect = pygame.Rect(
                                    self.moon_dropdown_rect.left,
                                    dropdown_y + i * 30,
                                    self.moon_dropdown_rect.width,
                                    30
                                )
                                if option_rect.collidepoint(event.pos):
                                    if moon_name == "Custom":
                                        # Trigger custom modal for moon mass
                                        body = self.get_selected_body()
                                        self.previous_dropdown_selection = body.get("moon_dropdown_selected") if body else None
                                        
                                        self.show_custom_modal = True
                                        self.custom_modal_title = "Set Custom Moon Mass"
                                        self.custom_modal_helper = "Enter mass in Lunar masses (M☾). Allowed: 0.0001 – 10"
                                        self.custom_modal_min = 0.0001
                                        self.custom_modal_max = 10.0
                                        self.custom_modal_unit = "M☾"
                                        self.pending_custom_field = "mass"
                                        self.pending_custom_body_id = self.selected_body_id
                                        
                                        self.custom_modal_text = ""
                                        self.validate_custom_modal_input()
                                    else:
                                        # Ensure we're updating the correct selected body and mass is stored as a Python float
                                        if self.selected_body:
                                            # Convert kg to lunar masses if needed (1 M☾ = 7.35e22 kg)
                                            if unit == "kg":
                                                mass_value = mass / 7.35e22  # Convert kg to M☾
                                            else:
                                                mass_value = mass
                                            # Ensure mass is stored as a Python float, not a numpy array/scalar
                                            if hasattr(mass_value, 'item'):
                                                mass_value = float(mass_value.item())
                                            else:
                                                mass_value = float(mass_value)
                                            self.update_selected_body_property("mass", mass_value, "mass")
                                        self.show_custom_moon_mass_input = False
                                        # Update the text input to show the actual value, preserving scientific notation
                                        if mass is not None:
                                            if abs(mass) < 0.001 or abs(mass) >= 1000:
                                                self.mass_input_text = f"{mass:.2e}"
                                            else:
                                                self.mass_input_text = f"{mass:.6f}".rstrip('0').rstrip('.')
                                    self.moon_dropdown_selected = moon_name
                                    self.moon_dropdown_visible = False
                                    if self.selected_body["type"] != "star":
                                        self.generate_orbit_grid(self.selected_body)
                                    break
                            # Only close dropdown if clicking outside both the dropdown and its options
                            if (self.moon_dropdown_visible and not 
                                (self.moon_dropdown_rect.collidepoint(event.pos) or 
                                 any(pygame.Rect(
                                     self.moon_dropdown_rect.left,
                                     self.moon_dropdown_rect.bottom + i * 30,
                                     self.moon_dropdown_rect.width,
                                     30
                                 ).collidepoint(event.pos)
                                 for i in range(len(self.moon_dropdown_options))))):
                                self.moon_dropdown_visible = False
                                self.mass_input_active = False
                                self.moon_dropdown_active = False
                        # Check if clicked on a moon age option (only for moons)
                        elif (self.selected_body and self.selected_body.get('type') == 'moon' and 
                              self.moon_age_dropdown_visible):
                            dropdown_y = self.moon_age_dropdown_rect.bottom
                            for i, (age_name, age) in enumerate(self.moon_age_dropdown_options):
                                option_rect = pygame.Rect(
                                    self.moon_age_dropdown_rect.left,
                                    dropdown_y + i * 30,
                                    self.moon_age_dropdown_rect.width,
                                    30
                                )
                                if option_rect.collidepoint(event.pos):
                                    if age_name == "Custom":
                                        # Trigger custom modal for moon age
                                        body = self.get_selected_body()
                                        self.previous_dropdown_selection = body.get("moon_age_dropdown_selected") if body else None
                                        
                                        self.show_custom_modal = True
                                        self.custom_modal_title = "Set Moon Age"
                                        self.custom_modal_helper = "Enter age in Gigayears (Gyr). Allowed: 0.01 – 10.0"
                                        self.custom_modal_min = 0.01
                                        self.custom_modal_max = 10.0
                                        self.custom_modal_unit = "Gyr"
                                        self.pending_custom_field = "age"
                                        self.pending_custom_body_id = self.selected_body_id
                                        
                                        self.custom_modal_text = ""
                                        self.validate_custom_modal_input()
                                    else:
                                        # Ensure age is stored as a Python float, not a numpy array/scalar
                                        if hasattr(age, 'item'):
                                            self.update_selected_body_property("age", age, "age")
                                        else:
                                            self.update_selected_body_property("age", age, "age")
                                    self.moon_age_dropdown_selected = age_name
                                    self.moon_age_dropdown_visible = False
                                    break
                            # Only close dropdown if clicking outside both the dropdown and its options
                            if (self.moon_age_dropdown_visible and not 
                                (self.moon_age_dropdown_rect.collidepoint(event.pos) or 
                                 any(pygame.Rect(
                                     self.moon_age_dropdown_rect.left,
                                     self.moon_age_dropdown_rect.bottom + i * 30,
                                     self.moon_age_dropdown_rect.width,
                                     30
                                 ).collidepoint(event.pos)
                                 for i in range(len(self.moon_age_dropdown_options))))):
                                self.moon_age_dropdown_visible = False
                                self.age_input_active = False
                                self.moon_age_dropdown_active = False
                        # Check if clicked on a moon radius option (only for moons)
                        elif (self.selected_body and self.selected_body.get('type') == 'moon' and 
                              self.moon_radius_dropdown_visible):
                            dropdown_y = self.moon_radius_dropdown_rect.bottom
                            for i, (radius_name, radius) in enumerate(self.moon_radius_dropdown_options):
                                option_rect = pygame.Rect(
                                    self.moon_radius_dropdown_rect.left,
                                    dropdown_y + i * 30,
                                    self.moon_radius_dropdown_rect.width,
                                    30
                                )
                                if option_rect.collidepoint(event.pos):
                                    if radius_name == "Custom":
                                        # Trigger custom modal for moon radius
                                        body = self.get_selected_body()
                                        self.previous_dropdown_selection = body.get("moon_radius_dropdown_selected") if body else None
                                        
                                        self.show_custom_modal = True
                                        self.custom_modal_title = "Set Custom Moon Radius"
                                        self.custom_modal_helper = "Enter radius in kilometers (km). Allowed: 1 – 10000"
                                        self.custom_modal_min = 1
                                        self.custom_modal_max = 10000
                                        self.custom_modal_unit = "km"
                                        self.pending_custom_field = "radius"
                                        self.pending_custom_body_id = self.selected_body_id
                                        
                                        self.custom_modal_text = ""
                                        self.validate_custom_modal_input()
                                    else:
                                        # Update the moon's physical radius in km
                                        # Visual radius is recomputed in real-time by calculate_moon_visual_radius
                                        body = self.get_selected_body()
                                        if body:
                                            self.update_selected_body_property("actual_radius", float(radius), "actual_radius")
                                        self.show_custom_moon_radius_input = False
                                    self.moon_radius_dropdown_selected = radius_name
                                    self.moon_radius_dropdown_visible = False
                                    self.moon_radius_dropdown_active = False
                                    break
                            # Only close dropdown if clicking outside both the dropdown and its options
                            if (self.moon_radius_dropdown_visible and not 
                                (self.moon_radius_dropdown_rect.collidepoint(event.pos) or 
                                 any(pygame.Rect(
                                     self.moon_radius_dropdown_rect.left,
                                     self.moon_radius_dropdown_rect.bottom + i * 30,
                                     self.moon_radius_dropdown_rect.width,
                                     30
                                 ).collidepoint(event.pos)
                                 for i in range(len(self.moon_radius_dropdown_options))))):
                                self.moon_radius_dropdown_visible = False
                                self.moon_radius_dropdown_active = False
                        # Check if clicked on a moon orbital distance option (only for moons)
                        elif (self.selected_body and self.selected_body.get('type') == 'moon' and 
                              self.moon_orbital_distance_dropdown_visible):
                            dropdown_y = self.moon_orbital_distance_dropdown_rect.bottom
                            for i, (distance_name, distance) in enumerate(self.moon_orbital_distance_dropdown_options):
                                option_rect = pygame.Rect(
                                    self.moon_orbital_distance_dropdown_rect.left,
                                    dropdown_y + i * 30,
                                    self.moon_orbital_distance_dropdown_rect.width,
                                    30
                                )
                                if option_rect.collidepoint(event.pos):
                                    if distance_name == "Custom":
                                        # Trigger custom modal for moon orbital distance
                                        body = self.get_selected_body()
                                        self.previous_dropdown_selection = body.get("moon_orbital_distance_dropdown_selected") if body else None
                                        
                                        self.show_custom_modal = True
                                        self.custom_modal_title = "Set Orbital Distance"
                                        self.custom_modal_helper = "Enter orbital distance in kilometers (km). Allowed: 1000 – 2000000"
                                        self.custom_modal_min = 1000
                                        self.custom_modal_max = 2000000
                                        self.custom_modal_unit = "km"
                                        self.pending_custom_field = "semi_major_axis"
                                        self.pending_custom_body_id = self.selected_body_id
                                        
                                        self.custom_modal_text = ""
                                        self.validate_custom_modal_input()
                                    else:
                                        # Update the moon's semi-major axis in AU (converting from km)
                                        body = self.get_selected_body()
                                        if body:
                                            KM_TO_AU = 6.68459e-9
                                            distance_au = float(distance) * KM_TO_AU
                                            self.update_selected_body_property("semiMajorAxis", distance_au, "semiMajorAxis")
                                            # Visual orbit_radius and physics are recomputed automatically
                                            self.recompute_orbit_parameters(body, force_recompute=True)
                                            self.clear_orbit_points(body)
                                        # Regenerate orbit grid with new radius
                                        self.generate_orbit_grid(self.selected_body)
                                        self.show_custom_moon_orbital_distance_input = False
                                        # Update the text input to show the actual value, preserving scientific notation
                                        if distance is not None:
                                            if abs(distance) < 0.001 or abs(distance) >= 1000:
                                                self.orbital_distance_input_text = f"{distance:.2e}"
                                            else:
                                                self.orbital_distance_input_text = f"{distance:.2f}"
                                    self.moon_orbital_distance_dropdown_selected = distance_name
                                    self.moon_orbital_distance_dropdown_visible = False
                                    self.moon_orbital_distance_dropdown_active = False
                                    break
                            # Only close dropdown if clicking outside both the dropdown and its options
                            if (self.moon_orbital_distance_dropdown_visible and not 
                                (self.moon_orbital_distance_dropdown_rect.collidepoint(event.pos) or 
                                 any(pygame.Rect(
                                     self.moon_orbital_distance_dropdown_rect.left,
                                     self.moon_orbital_distance_dropdown_rect.bottom + i * 30,
                                     self.moon_orbital_distance_dropdown_rect.width,
                                     30
                                 ).collidepoint(event.pos)
                                 for i in range(len(self.moon_orbital_distance_dropdown_options))))):
                                self.moon_orbital_distance_dropdown_visible = False
                                self.moon_orbital_distance_dropdown_active = False
                        # Check if clicked on a moon orbital period option (only for moons)
                        elif (self.selected_body and self.selected_body.get('type') == 'moon' and 
                              self.moon_orbital_period_dropdown_visible):
                            dropdown_y = self.moon_orbital_period_dropdown_rect.bottom
                            for i, (period_name, period) in enumerate(self.moon_orbital_period_dropdown_options):
                                option_rect = pygame.Rect(
                                    self.moon_orbital_period_dropdown_rect.left,
                                    dropdown_y + i * 30,
                                    self.moon_orbital_period_dropdown_rect.width,
                                    30
                                )
                                if option_rect.collidepoint(event.pos):
                                    if period_name == "Custom":
                                        self.show_custom_moon_orbital_period_input = True
                                        self.orbital_period_input_active = True
                                        self.orbital_period_input_text = f"{self.selected_body.get('orbital_period', 27.3):.0f}"
                                    else:
                                        self.update_selected_body_property("orbital_period", period, "orbital_period")
                                        self.show_custom_moon_orbital_period_input = False
                                    self.moon_orbital_period_dropdown_selected = period_name
                                    self.moon_orbital_period_dropdown_visible = False
                                    self.moon_orbital_period_dropdown_active = False
                                    break
                            # Only close dropdown if clicking outside both the dropdown and its options
                            if (self.moon_orbital_period_dropdown_visible and not 
                                (self.moon_orbital_period_dropdown_rect.collidepoint(event.pos) or 
                                 any(pygame.Rect(
                                     self.moon_orbital_period_dropdown_rect.left,
                                     self.moon_orbital_period_dropdown_rect.bottom + i * 30,
                                     self.moon_orbital_period_dropdown_rect.width,
                                     30
                                 ).collidepoint(event.pos)
                                 for i in range(len(self.moon_orbital_period_dropdown_options))))):
                                self.moon_orbital_period_dropdown_visible = False
                                self.moon_orbital_period_dropdown_active = False
                        # Check if clicked on a moon temperature option (only for moons)
                        elif (self.selected_body and self.selected_body.get('type') == 'moon' and 
                              self.moon_temperature_dropdown_visible):
                            dropdown_y = self.moon_temperature_dropdown_rect.bottom
                            for i, (temp_name, temp) in enumerate(self.moon_temperature_dropdown_options):
                                option_rect = pygame.Rect(
                                    self.moon_temperature_dropdown_rect.left,
                                    dropdown_y + i * 30,
                                    self.moon_temperature_dropdown_rect.width,
                                    30
                                )
                                if option_rect.collidepoint(event.pos):
                                    if temp_name == "Custom":
                                        # Trigger custom modal for moon temperature
                                        body = self.get_selected_body()
                                        self.previous_dropdown_selection = body.get("moon_temperature_dropdown_selected") if body else None
                                        
                                        self.show_custom_modal = True
                                        self.custom_modal_title = "Set Moon Temperature"
                                        self.custom_modal_helper = "Enter surface temperature in Kelvin (K). Allowed: 1 – 1000"
                                        self.custom_modal_min = 1.0
                                        self.custom_modal_max = 1000.0
                                        self.custom_modal_unit = "K"
                                        self.pending_custom_field = "temperature"
                                        self.pending_custom_body_id = self.selected_body_id
                                        
                                        self.custom_modal_text = ""
                                        self.validate_custom_modal_input()
                                    else:
                                        self.update_selected_body_property("temperature", temp, "temperature")
                                        self.show_custom_moon_temperature_input = False
                                    self.moon_temperature_dropdown_selected = temp_name
                                    self.moon_temperature_dropdown_visible = False
                                    self.moon_temperature_dropdown_active = False
                                    break
                            # Only close dropdown if clicking outside both the dropdown and its options
                            if (self.moon_temperature_dropdown_visible and not 
                                (self.moon_temperature_dropdown_rect.collidepoint(event.pos) or 
                                 any(pygame.Rect(
                                     self.moon_temperature_dropdown_rect.left,
                                     self.moon_temperature_dropdown_rect.bottom + i * 30,
                                     self.moon_temperature_dropdown_rect.width,
                                     30
                                 ).collidepoint(event.pos)
                                 for i in range(len(self.moon_temperature_dropdown_options))))):
                                self.moon_temperature_dropdown_visible = False
                                self.moon_temperature_dropdown_active = False
                        # Check if clicked on a moon gravity option (only for moons)
                        elif (self.selected_body and self.selected_body.get('type') == 'moon' and 
                              self.moon_gravity_dropdown_visible):
                            dropdown_y = self.moon_gravity_dropdown_rect.bottom
                            for i, (gravity_name, gravity) in enumerate(self.moon_gravity_dropdown_options):
                                option_rect = pygame.Rect(
                                    self.moon_gravity_dropdown_rect.left,
                                    dropdown_y + i * 30,
                                    self.moon_gravity_dropdown_rect.width,
                                    30
                                )
                                if option_rect.collidepoint(event.pos):
                                    if gravity_name == "Custom":
                                        # Trigger custom modal for moon gravity
                                        body = self.get_selected_body()
                                        self.previous_dropdown_selection = body.get("moon_gravity_dropdown_selected") if body else None
                                        
                                        self.show_custom_modal = True
                                        self.custom_modal_title = "Set Moon Gravity"
                                        self.custom_modal_helper = "Enter surface gravity in m/s². Allowed: 0.01 – 10"
                                        self.custom_modal_min = 0.01
                                        self.custom_modal_max = 10.0
                                        self.custom_modal_unit = "m/s²"
                                        self.pending_custom_field = "gravity"
                                        self.pending_custom_body_id = self.selected_body_id
                                        
                                        self.custom_modal_text = ""
                                        self.validate_custom_modal_input()
                                    else:
                                        # Update surface gravity via per-object registry
                                        self.update_selected_body_property("gravity", gravity, "gravity")
                                        self.show_custom_moon_gravity_input = False
                                        # Update the text input to show the actual value, preserving scientific notation
                                        if gravity is not None:
                                            if abs(gravity) < 0.001 or abs(gravity) >= 1000:
                                                self.moon_gravity_input_text = f"{gravity:.2e}"
                                            else:
                                                self.moon_gravity_input_text = f"{gravity:.3f}".rstrip('0').rstrip('.')
                                    self.moon_gravity_dropdown_selected = gravity_name
                                    self.moon_gravity_dropdown_visible = False
                                    self.moon_gravity_dropdown_active = False
                                    break
                            # Only close dropdown if clicking outside both the dropdown and its options
                            if (self.moon_gravity_dropdown_visible and not 
                                (self.moon_gravity_dropdown_rect.collidepoint(event.pos) or 
                                 any(pygame.Rect(
                                     self.moon_gravity_dropdown_rect.left,
                                     self.moon_gravity_dropdown_rect.bottom + i * 30,
                                     self.moon_gravity_dropdown_rect.width,
                                     30
                                 ).collidepoint(event.pos)
                                 for i in range(len(self.moon_gravity_dropdown_options))))):
                                self.moon_gravity_dropdown_visible = False
                                self.moon_gravity_dropdown_active = False
                        # Handle luminosity dropdown (only for stars)
                        elif (self.selected_body and self.selected_body.get('type') == 'star' and 
                              self.luminosity_dropdown_rect.collidepoint(event.pos)):
                            self.close_all_dropdowns()
                            self.luminosity_dropdown_active = True
                            self.luminosity_input_active = False
                            self.luminosity_dropdown_visible = True
                            self.create_dropdown_surface()
                        # Check if clicked on a luminosity option (only for stars)
                        elif (self.selected_body and self.selected_body.get('type') == 'star' and 
                              self.luminosity_dropdown_visible):
                            dropdown_y = self.luminosity_dropdown_rect.bottom
                            for i, (luminosity_name, luminosity) in enumerate(self.luminosity_dropdown_options):
                                option_rect = pygame.Rect(
                                    self.luminosity_dropdown_rect.left,
                                    dropdown_y + i * 30,
                                    self.luminosity_dropdown_rect.width,
                                    30
                                )
                                if option_rect.collidepoint(event.pos):
                                    if luminosity_name == "Custom":
                                        self.show_custom_luminosity_input = True
                                        self.luminosity_input_active = True
                                        self.luminosity_input_text = self._format_value(self.selected_body.get('luminosity', 1.0), '', for_dropdown=False)
                                    else:
                                        self.update_selected_body_property("luminosity", luminosity, "luminosity")
                                        # Update habitable zone when luminosity changes
                                        self.selected_body["hz_surface"] = self.create_habitable_zone(self.selected_body)
                                        self.show_custom_luminosity_input = False
                                    self.luminosity_dropdown_selected = luminosity_name
                                    self.luminosity_dropdown_visible = False
                                    break
                        # Only close luminosity dropdown if clicking outside both the dropdown and its options
                        if (self.luminosity_dropdown_visible and not 
                            (self.luminosity_dropdown_rect.collidepoint(event.pos) or 
                             any(pygame.Rect(
                                 self.luminosity_dropdown_rect.left,
                                 self.luminosity_dropdown_rect.bottom + i * 30,
                                 self.luminosity_dropdown_rect.width,
                                 30
                             ).collidepoint(event.pos)
                             for i in range(len(self.luminosity_dropdown_options))))):
                            self.luminosity_dropdown_visible = False
                            self.luminosity_input_active = False
                            self.luminosity_dropdown_active = False
                        # Handle spectral class dropdown (only for stars) - merged from spectral and temperature dropdowns
                        if self.selected_body and self.selected_body.get('type') == 'star' and self.spectral_class_dropdown_rect.collidepoint(event.pos):
                            self.close_all_dropdowns()
                            self.spectral_class_dropdown_active = True
                            self.spectral_class_dropdown_visible = True
                            self.create_dropdown_surface()
                        # Handle star mass dropdown (only for stars)
                        elif (self.selected_body and self.selected_body.get('type') == 'star' and 
                              self.star_mass_dropdown_rect.collidepoint(event.pos)):
                            self.close_all_dropdowns()
                            self.star_mass_dropdown_active = True
                            self.mass_input_active = False
                            self.star_mass_dropdown_visible = True
                            self.create_dropdown_surface()
                        # Handle star age dropdown (only for stars)
                        elif (self.selected_body and self.selected_body.get('type') == 'star' and 
                              self.star_age_dropdown_rect.collidepoint(event.pos)):
                            self.close_all_dropdowns()
                            self.star_age_dropdown_active = True
                            self.age_input_active = False
                            self.star_age_dropdown_visible = True
                            self.create_dropdown_surface()
                        # Check if clicked on a star age option (only for stars)
                        elif (self.selected_body and self.selected_body.get('type') == 'star' and 
                              self.star_age_dropdown_visible):
                            dropdown_y = self.star_age_dropdown_rect.bottom
                            for i, (age_name, age) in enumerate(self.star_age_dropdown_options):
                                option_rect = pygame.Rect(
                                    self.star_age_dropdown_rect.left,
                                    dropdown_y + i * 30,
                                    self.star_age_dropdown_rect.width,
                                    30
                                )
                                if option_rect.collidepoint(event.pos):
                                    if age_name == "Custom":
                                        self.show_custom_star_age_input = True
                                        self.age_input_active = True
                                        self.age_input_text = self._format_value(self.selected_body.get('age', 0.0), '', for_dropdown=False)
                                    else:
                                        # CRITICAL: Use update_selected_body_property to ensure we update ONLY the selected body
                                        if hasattr(age, 'item'):
                                            age_value = float(age.item())
                                        else:
                                            age_value = float(age)
                                        self.update_selected_body_property("age", age_value, "age")
                                        self.show_custom_star_age_input = False
                                    self.star_age_dropdown_selected = age_name
                                    self.star_age_dropdown_visible = False
                                    break
                            # Only close dropdown if clicking outside both the dropdown and its options
                            if (self.star_age_dropdown_visible and not 
                                (self.star_age_dropdown_rect.collidepoint(event.pos) or 
                                 any(pygame.Rect(
                                     self.star_age_dropdown_rect.left,
                                     self.star_age_dropdown_rect.bottom + i * 30,
                                     self.star_age_dropdown_rect.width,
                                     30
                                 ).collidepoint(event.pos)
                                 for i in range(len(self.star_age_dropdown_options))))):
                                self.star_age_dropdown_visible = False
                                self.age_input_active = False
                                self.star_age_dropdown_active = False
                        # Check if clicked on a star mass option (only for stars)
                        elif (self.selected_body and self.selected_body.get('type') == 'star' and 
                              self.star_mass_dropdown_visible):
                            dropdown_y = self.star_mass_dropdown_rect.bottom
                            for i, (mass_name, mass) in enumerate(self.star_mass_dropdown_options):
                                option_rect = pygame.Rect(
                                    self.star_mass_dropdown_rect.left,
                                    dropdown_y + i * 30,
                                    self.star_mass_dropdown_rect.width,
                                    30
                                )
                                if option_rect.collidepoint(event.pos):
                                    if mass_name == "Custom":
                                        # Trigger custom modal for star mass
                                        body = self.get_selected_body()
                                        self.previous_dropdown_selection = body.get("star_mass_dropdown_selected") if body else None
                                        
                                        self.show_custom_modal = True
                                        self.custom_modal_title = "Set Custom Star Mass"
                                        self.custom_modal_helper = "Enter stellar mass in Solar masses (M☉). Allowed: 0.08 – 150"
                                        self.custom_modal_min = 0.08
                                        self.custom_modal_max = 150.0
                                        self.custom_modal_unit = "M☉"
                                        self.pending_custom_field = "mass"
                                        self.pending_custom_body_id = self.selected_body_id
                                        
                                        self.custom_modal_text = ""
                                        self.validate_custom_modal_input()
                                    else:
                                        # Ensure we're updating the correct selected body and mass is stored as a Python float
                                        if self.selected_body:
                                            mass_value = mass # Store in Solar masses directly
                                            # Ensure mass is stored as a Python float, not a numpy array/scalar
                                            if hasattr(mass_value, 'item'):
                                                mass_value = float(mass_value.item())
                                            else:
                                                mass_value = float(mass_value)
                                            self.update_selected_body_property("mass", mass_value, "mass")
                                        self.show_custom_star_mass_input = False
                                    self.star_mass_dropdown_selected = mass_name
                                    self.star_mass_dropdown_visible = False
                                    self.star_mass_dropdown_active = False
                                    break
                            # Only close dropdown if clicking outside both the dropdown and its options
                            if (self.star_mass_dropdown_visible and not 
                                (self.star_mass_dropdown_rect.collidepoint(event.pos) or 
                                 any(pygame.Rect(
                                     self.star_mass_dropdown_rect.left,
                                     self.star_mass_dropdown_rect.bottom + i * 30,
                                     self.star_mass_dropdown_rect.width,
                                     30
                                 ).collidepoint(event.pos)
                                 for i in range(len(self.star_mass_dropdown_options))))):
                                self.star_mass_dropdown_visible = False
                                self.mass_input_active = False
                                self.star_mass_dropdown_active = False
                        # Handle spectral class dropdown selection (merged from spectral and temperature dropdowns)
                        elif self.spectral_class_dropdown_visible:
                            # First check if click is within the spectral class dropdown area
                            dropdown_area = pygame.Rect(
                                self.spectral_class_dropdown_rect.left,
                                self.spectral_class_dropdown_rect.top,
                                self.spectral_class_dropdown_rect.width,
                                self.spectral_class_dropdown_rect.height + len(self.spectral_class_dropdown_options) * self.dropdown_option_height
                            )
                            
                            if dropdown_area.collidepoint(event.pos):
                                # Convert click position to be relative to the dropdown surface
                                relative_y = event.pos[1] - self.spectral_class_dropdown_rect.bottom
                                option_index = relative_y // self.dropdown_option_height
                                
                                if 0 <= option_index < len(self.spectral_class_dropdown_options):
                                    name, temp, color = self.spectral_class_dropdown_options[option_index]
                                    if temp is not None and color is not None:
                                        if self.selected_body:
                                            self.update_selected_body_property("temperature", temp, "temperature")
                                            self.update_selected_body_property("star_color", color, "star_color")
                                        self.spectral_class_dropdown_selected = name
                                    else:  # Custom option
                                        self.show_custom_temperature_input = True
                                        self.temperature_input_active = True
                                        self.temperature_input_text = self._format_value(self.selected_body.get('temperature', 5800), '', for_dropdown=False)
                                    self.spectral_class_dropdown_visible = False
                                    self.spectral_class_dropdown_active = False
                            else:
                                # Click outside the spectral class dropdown area
                                self.spectral_class_dropdown_visible = False
                                self.spectral_class_dropdown_active = False
                        # Handle radius dropdown (only for stars)
                        elif (self.selected_body and self.selected_body.get('type') == 'star' and 
                              self.radius_dropdown_rect.collidepoint(event.pos)):
                            self.close_all_dropdowns()
                            self.radius_dropdown_active = True
                            self.radius_input_active = False
                            self.radius_dropdown_visible = True
                            self.create_dropdown_surface()
                        # Handle activity level dropdown (only for stars)
                        elif (self.selected_body and self.selected_body.get('type') == 'star' and 
                              self.activity_dropdown_rect.collidepoint(event.pos)):
                            self.close_all_dropdowns()
                            self.activity_dropdown_active = True
                            self.activity_input_active = False
                            self.activity_dropdown_visible = True
                            self.create_dropdown_surface()
                        elif (self.selected_body and self.selected_body.get('type') == 'star' and 
                              self.metallicity_dropdown_rect.collidepoint(event.pos)):
                            self.close_all_dropdowns()
                            self.metallicity_dropdown_active = True
                            self.metallicity_input_active = False
                            self.metallicity_dropdown_visible = True
                            self.create_dropdown_surface()
                        # Handle dropdown option selection
                        if (self.planet_dropdown_visible or self.moon_dropdown_visible or 
                            self.star_mass_dropdown_visible or self.luminosity_dropdown_visible or
                            self.planet_age_dropdown_visible or self.star_age_dropdown_visible or
                            self.moon_age_dropdown_visible or self.moon_radius_dropdown_visible or
                            self.moon_orbital_distance_dropdown_visible or self.moon_orbital_period_dropdown_visible or
                            self.moon_temperature_dropdown_visible or self.moon_gravity_dropdown_visible or
                            self.spectral_class_dropdown_visible or self.radius_dropdown_visible or
                            self.activity_dropdown_visible or self.metallicity_dropdown_visible or
                            self.planet_orbital_distance_dropdown_visible or self.planet_orbital_eccentricity_dropdown_visible or
                            self.planet_orbital_period_dropdown_visible):
                            
                            # Check if click is within any dropdown option
                            for i, option_rect in enumerate(self.dropdown_options_rects):
                                # Convert option_rect to screen coordinates
                                screen_option_rect = pygame.Rect(
                                    self.dropdown_rect.left,
                                    self.dropdown_rect.top + i * self.dropdown_option_height,
                                    option_rect.width,
                                    option_rect.height
                                )
                                
                                if screen_option_rect.collidepoint(event.pos):
                                    if self.planet_dropdown_visible:
                                        if i < len(self.planet_dropdown_options):
                                            name, value = self.planet_dropdown_options[i]
                                            print(f"PLANET DROPDOWN CLICK: name={name}, value={value}, selected_body_id={self.selected_body_id}", flush=True)
                                            # Check if in placement mode (no planet selected) or editing mode (planet selected)
                                            if not self.selected_body_id or not self.selected_body or self.selected_body.get('type') != 'planet':
                                                # Placement mode: Store selected planet for placement
                                                if name == "Custom":
                                                    self.planet_dropdown_selected = None
                                                    self.clear_preview()
                                                else:
                                                    self.planet_dropdown_selected = name
                                                    print(f"Selected planet for placement: {name}")
                                                    # Set preview radius once when planet is selected (stable per object type)
                                                    if name in SOLAR_SYSTEM_PLANET_PRESETS:
                                                        preset = SOLAR_SYSTEM_PLANET_PRESETS[name]
                                                        planet_radius_re = preset["radius"]  # In Earth radii
                                                        # Use updated perceptual scaling for preview (matches calculate_planet_visual_radius)
                                                        visual_radius = EARTH_RADIUS_PX * math.pow(planet_radius_re, 0.4)
                                                        # Authoritative clamp: min=MIN_PLANET_PX, max=star_radius_px * 0.85
                                                        # For preview, assume Sun-sized star if no star placed
                                                        self.preview_radius = int(max(MIN_PLANET_PX, min(visual_radius, SUN_RADIUS_PX * 0.85)))
                                                    else:
                                                        self.preview_radius = int(MIN_PLANET_PX)  # New minimum Earth size
                                                    # Activate placement mode - preview position will be updated every frame
                                                    self.placement_mode_active = True
                                                # Keep dropdown open to show selection, but recreate surface to show highlight
                                                self.create_dropdown_surface()
                                            else:
                                                # Editing mode: Update selected planet's mass
                                                if value is not None:
                                                    # Ensure we're updating the correct selected body and mass is stored as a Python float
                                                    # Ensure mass is stored as a Python float, not a numpy array/scalar
                                                    mass_value = float(value) if not hasattr(value, 'item') else float(value.item())
                                                    print(f"ABOUT TO CALL update_selected_body_property with mass={mass_value}", flush=True)
                                                    if self.update_selected_body_property("mass", mass_value, "mass"):
                                                        # CRITICAL: Store dropdown selection in the body's dict, not global state
                                                        body = self.get_selected_body()
                                                        if body:
                                                            body["planet_dropdown_selected"] = name
                                                else:
                                                    self.show_custom_planet_mass_input = True
                                                    self.planet_mass_input_active = True
                                                # REMOVED: self.planet_dropdown_selected = name  # This was global, now stored per-body
                                                self.planet_age_dropdown_selected = "Solar System Planets (4.6 Gyr)"
                                                self.planet_gravity_dropdown_selected = "Earth"
                                                self.planet_dropdown_visible = False
                                                self.planet_dropdown_active = False
                                    elif self.moon_dropdown_visible:
                                        if i < len(self.moon_dropdown_options):
                                            option_data = self.moon_dropdown_options[i]
                                            if len(option_data) == 3:
                                                name, value, unit = option_data
                                            else:
                                                name, value = option_data
                                                unit = None
                                            if value is not None:
                                                # Ensure we're updating the correct selected body and mass is stored as a Python float
                                                if self.selected_body:
                                                    # Convert kg to lunar masses if needed (1 M☾ = 7.35e22 kg)
                                                    if unit == "kg":
                                                        mass_value = value / 7.35e22  # Convert kg to M☾
                                                    else:
                                                        mass_value = value
                                                    # CRITICAL: Use update_selected_body_property to ensure we update ONLY the selected body
                                                    if hasattr(mass_value, 'item'):
                                                        mass_float = float(mass_value.item())
                                                    else:
                                                        mass_float = float(mass_value)
                                                    self.update_selected_body_property("mass", mass_float, "mass")
                                            else:
                                                self.show_custom_moon_mass_input = True
                                                self.moon_mass_input_active = True
                                            self.moon_dropdown_selected = name
                                            self.moon_dropdown_visible = False
                                            self.moon_dropdown_active = False
                                    elif self.moon_age_dropdown_visible:
                                        if i < len(self.moon_age_dropdown_options):
                                            name, value = self.moon_age_dropdown_options[i]
                                            if value is not None:
                                                # Ensure age is stored as a Python float, not a numpy array/scalar
                                                self.update_selected_body_property("age", value, "age")
                                            else:
                                                # Trigger custom modal for moon age
                                                body = self.get_selected_body()
                                                self.previous_dropdown_selection = body.get("moon_age_dropdown_selected") if body else None
                                                
                                                self.show_custom_modal = True
                                                self.custom_modal_title = "Set Moon Age"
                                                self.custom_modal_helper = "Enter age in Gigayears (Gyr). Allowed: 0.01 – 10.0"
                                                self.custom_modal_min = 0.01
                                                self.custom_modal_max = 10.0
                                                self.custom_modal_unit = "Gyr"
                                                self.pending_custom_field = "age"
                                                self.pending_custom_body_id = self.selected_body_id
                                                
                                                self.custom_modal_text = ""
                                                self.validate_custom_modal_input()
                                            self.moon_age_dropdown_selected = name
                                            self.moon_age_dropdown_visible = False
                                            self.moon_age_dropdown_active = False
                                    elif self.moon_radius_dropdown_visible:
                                        if i < len(self.moon_radius_dropdown_options):
                                            name, value = self.moon_radius_dropdown_options[i]
                                            if value is not None:
                                                # Update the moon's physical radius in km
                                                # Visual radius is recomputed in real-time by calculate_moon_visual_radius
                                                body = self.get_selected_body()
                                                if body:
                                                    self.update_selected_body_property("actual_radius", float(value), "actual_radius")
                                                self.show_custom_moon_radius_input = False
                                            else:
                                                self.show_custom_moon_radius_input = True
                                            self.moon_radius_dropdown_selected = name
                                            self.moon_radius_dropdown_visible = False
                                            self.moon_radius_dropdown_active = False
                                    elif self.moon_orbital_distance_dropdown_visible:
                                        if i < len(self.moon_orbital_distance_dropdown_options):
                                            name, value = self.moon_orbital_distance_dropdown_options[i]
                                            if value is not None:
                                                # Update the moon's semi-major axis in AU (converting from km)
                                                body = self.get_selected_body()
                                                if body:
                                                    KM_TO_AU = 6.68459e-9
                                                    distance_au = float(value) * KM_TO_AU
                                                    self.update_selected_body_property("semiMajorAxis", distance_au, "semiMajorAxis")
                                                    self.clear_orbit_points(body)
                                                # Regenerate orbit grid with new radius
                                                self.generate_orbit_grid(self.selected_body)
                                            else:
                                                self.show_custom_moon_orbital_distance_input = True
                                            self.moon_orbital_distance_dropdown_selected = name
                                            self.moon_orbital_distance_dropdown_visible = False
                                            self.moon_orbital_distance_dropdown_active = False
                                    elif self.star_mass_dropdown_visible:
                                        if i < len(self.star_mass_dropdown_options):
                                            name, value = self.star_mass_dropdown_options[i]
                                            if value is not None:
                                                # Ensure we're updating the correct selected body and mass is stored as a Python float
                                                if self.selected_body:
                                                    # Ensure mass is stored as a Python float, not a numpy array/scalar
                                                    self.update_selected_body_property("mass", value, "mass")
                                            else:
                                                self.show_custom_star_mass_input = True
                                                self.star_mass_input_active = True
                                            self.star_mass_dropdown_selected = name
                                            self.star_mass_dropdown_visible = False
                                            self.star_mass_dropdown_active = False
                                    elif self.luminosity_dropdown_visible:
                                        if i < len(self.luminosity_dropdown_options):
                                            name, value = self.luminosity_dropdown_options[i]
                                            if value is not None:
                                                self.update_selected_body_property("luminosity", value, "luminosity")
                                            else:
                                                self.show_custom_luminosity_input = True
                                                self.luminosity_input_active = True
                                            self.luminosity_dropdown_selected = name
                                            self.luminosity_dropdown_visible = False
                                            self.luminosity_dropdown_active = False
                                    elif self.planet_age_dropdown_visible:
                                        if i < len(self.planet_age_dropdown_options):
                                            name, value = self.planet_age_dropdown_options[i]
                                            if value is not None:
                                                # Ensure we're updating the correct selected body
                                                if self.selected_body:
                                                    # Ensure age is stored as a Python float, not a numpy array/scalar
                                                    if hasattr(value, 'item'):
                                                        self.update_selected_body_property("age", value, "age")
                                                    else:
                                                        self.update_selected_body_property("age", value, "age")
                                            else:
                                                self.show_custom_planet_age_input = True
                                                self.planet_age_input_active = True
                                            self.planet_age_dropdown_selected = name
                                            self.planet_age_dropdown_visible = False
                                            self.planet_age_dropdown_active = False
                                    elif self.star_age_dropdown_visible:
                                        if i < len(self.star_age_dropdown_options):
                                            name, value = self.star_age_dropdown_options[i]
                                            if value is not None:
                                                # Ensure age is stored as a Python float, not a numpy array/scalar
                                                self.update_selected_body_property("age", value, "age")
                                            else:
                                                self.show_custom_star_age_input = True
                                                self.star_age_input_active = True
                                            self.star_age_dropdown_selected = name
                                            self.star_age_dropdown_visible = False
                                            self.star_age_dropdown_active = False
                                    elif self.radius_dropdown_visible:
                                        if i < len(self.radius_dropdown_options):
                                            name, value = self.radius_dropdown_options[i]
                                            if value is not None:
                                                body = self.get_selected_body()
                                                if body:
                                                    self.update_selected_body_property("radius", value, "radius")
                                            else:
                                                self.show_custom_radius_input = True
                                                self.radius_input_active = True
                                            self.radius_dropdown_selected = name
                                            self.radius_dropdown_visible = False
                                            self.radius_dropdown_active = False
                                    elif self.activity_dropdown_visible:
                                        if i < len(self.activity_dropdown_options):
                                            name, value = self.activity_dropdown_options[i]
                                            if value is not None:
                                                self.update_selected_body_property("activity", value, "activity")
                                            else:
                                                self.show_custom_activity_input = True
                                                self.activity_input_active = True
                                            self.activity_dropdown_selected = name
                                            self.activity_dropdown_visible = False
                                            self.activity_dropdown_active = False
                                    elif self.metallicity_dropdown_visible:
                                        if i < len(self.metallicity_dropdown_options):
                                            name, value = self.metallicity_dropdown_options[i]
                                            if value is not None:
                                                self.update_selected_body_property("metallicity", value, "metallicity")
                                            else:
                                                self.show_custom_metallicity_input = True
                                                self.metallicity_input_active = True
                                            self.metallicity_dropdown_selected = name
                                            self.metallicity_dropdown_visible = False
                                            self.metallicity_dropdown_active = False
                                    elif self.planet_orbital_distance_dropdown_visible:
                                        if i < len(self.planet_orbital_distance_dropdown_options):
                                            name, value = self.planet_orbital_distance_dropdown_options[i]
                                            if value is not None:
                                                self.update_selected_body_property("semiMajorAxis", value, "semiMajorAxis")
                                                # CRITICAL: Update position using centralized function
                                                parent_star = next((b for b in self.placed_bodies if b["name"] == self.selected_body.get("parent")), None)
                                                body = self.get_selected_body()
                                                if parent_star is None and body and body.get("parent_obj"):
                                                    parent_star = body["parent_obj"]
                                                if parent_star and body:
                                                    # Use centralized function to ensure position is derived from semiMajorAxis via visual scaling
                                                    self.compute_planet_position(body, parent_star)
                                                if body:
                                                    self.generate_orbit_grid(body)
                                                self.show_custom_orbital_distance_input = False
                                            else:
                                                self.show_custom_orbital_distance_input = True
                                                self.orbital_distance_input_active = True
                                                self.orbital_distance_input_text = f"{self.selected_body.get('semiMajorAxis', 1.0):.2f}"
                                            self.planet_orbital_distance_dropdown_selected = name
                                            self.planet_orbital_distance_dropdown_visible = False
                                            self.planet_orbital_distance_dropdown_active = False
                                            break
                                    elif self.planet_orbital_eccentricity_dropdown_visible:
                                        if i < len(self.planet_orbital_eccentricity_dropdown_options):
                                            name, value = self.planet_orbital_eccentricity_dropdown_options[i]
                                            if value is not None:
                                                self.update_selected_body_property("eccentricity", value, "eccentricity")
                                                self.show_custom_orbital_eccentricity_input = False
                                            else:
                                                self.show_custom_orbital_eccentricity_input = True
                                                self.orbital_eccentricity_input_active = True
                                                self.orbital_eccentricity_input_text = f"{self.selected_body.get('eccentricity', 0.0167):.3f}"
                                            self.planet_orbital_eccentricity_dropdown_selected = name
                                            self.planet_orbital_eccentricity_dropdown_visible = False
                                            self.planet_orbital_eccentricity_dropdown_active = False
                                            break
                                    elif self.planet_orbital_period_dropdown_visible:
                                        if i < len(self.planet_orbital_period_dropdown_options):
                                            name, value = self.planet_orbital_period_dropdown_options[i]
                                            if value is not None:
                                                self.update_selected_body_property("orbital_period", value, "orbital_period")
                                                self.show_custom_orbital_period_input = False
                                            else:
                                                self.show_custom_orbital_period_input = True
                                                self.orbital_period_input_active = True
                                                self.orbital_period_input_text = f"{self.selected_body.get('orbital_period', 365.25):.0f}"
                                            self.planet_orbital_period_dropdown_selected = name
                                            self.planet_orbital_period_dropdown_visible = False
                                            self.planet_orbital_period_dropdown_active = False
                                            break
                    else:
                        # Handle planet preset selector arrow click
                        # Allow opening dropdown when Planet tab is active (even if no planet selected)
                        if (self.planet_preset_arrow_rect and 
                            self.planet_preset_arrow_rect.collidepoint(event.pos) and
                            self.active_tab == "planet"):
                            # Toggle preset dropdown
                            self.planet_preset_dropdown_visible = not self.planet_preset_dropdown_visible
                            print(f"DEBUG: Arrow clicked! Preset dropdown toggled. Visible: {self.planet_preset_dropdown_visible}, Active tab: {self.active_tab}, Arrow rect: {self.planet_preset_arrow_rect}, Click pos: {event.pos}")
                        
                        # Handle planet preset dropdown option clicks
                        elif (self.planet_preset_dropdown_visible and 
                              self.planet_preset_dropdown_rect and
                              self.planet_preset_dropdown_rect.collidepoint(event.pos)):
                            # Calculate which option was clicked
                            relative_y = event.pos[1] - self.planet_preset_dropdown_rect.top
                            option_index = relative_y // 28
                            
                            if 0 <= option_index < len(self.planet_preset_options):
                                preset_name = self.planet_preset_options[option_index]
                                # Check if a planet is selected (editing mode) or not (placement mode)
                                if self.selected_body and self.selected_body.get('type') == 'planet':
                                    # Editing mode: Apply preset to selected planet
                                    self.apply_planet_preset(preset_name)
                                    # Close dropdown
                                    self.planet_preset_dropdown_visible = False
                                else:
                                    # Placement mode: Toggle selection - if already selected, deselect it
                                    if self.planet_dropdown_selected == preset_name:
                                        # Deselect: clear selection and preview
                                        self.planet_dropdown_selected = None
                                        self.clear_preview()
                                        print(f"Deselected planet preset: {preset_name}")
                                    else:
                                        # Select: Store selected preset for planet placement
                                        self.planet_dropdown_selected = preset_name
                                        print(f"Selected planet preset for placement: {preset_name}")
                                        # Set preview radius once when planet is selected (stable per object type)
                                        if preset_name in SOLAR_SYSTEM_PLANET_PRESETS:
                                            preset = SOLAR_SYSTEM_PLANET_PRESETS[preset_name]
                                            planet_radius_re = preset["radius"]  # In Earth radii
                                            # Use updated perceptual scaling for preview (matches calculate_planet_visual_radius)
                                            visual_radius = EARTH_RADIUS_PX * math.pow(planet_radius_re, 0.4)
                                            # Authoritative clamp: min=MIN_PLANET_PX, max=star_radius_px * 0.85
                                            # For preview, assume Sun-sized star if no star placed
                                            self.preview_radius = int(max(MIN_PLANET_PX, min(visual_radius, SUN_RADIUS_PX * 0.85)))
                                        else:
                                            self.preview_radius = int(MIN_PLANET_PX)  # New minimum Earth size
                                        # Initialize preview position immediately at mouse cursor
                                        self.preview_position = pygame.mouse.get_pos()
                                        # Activate placement mode - preview position will be updated every frame
                                        self.placement_mode_active = True
                                    # Keep dropdown open to show selection (will be highlighted on next render)
                                    # Dropdown will close when clicking outside or when placing planet
                        
                        # Close preset dropdown if clicking outside
                        elif (self.planet_preset_dropdown_visible and
                              self.planet_preset_dropdown_rect and
                              not self.planet_preset_dropdown_rect.collidepoint(event.pos) and
                              (not self.planet_preset_arrow_rect or not self.planet_preset_arrow_rect.collidepoint(event.pos))):
                            self.planet_preset_dropdown_visible = False
                        
                        # Handle tab clicks (but exclude preset arrow area)
                        for tab_name, tab_rect in self.tabs.items():
                            # Skip if clicking on preset arrow
                            if (tab_name == "planet" and 
                                self.planet_preset_arrow_rect and 
                                self.planet_preset_arrow_rect.collidepoint(event.pos)):
                                continue  # Let preset arrow handler deal with it
                            
                            if tab_rect.collidepoint(event.pos):
                                # Special handling for Planet tab - toggle preset dropdown
                                if tab_name == "planet":
                                    self.active_tab = "planet"
                                    # Toggle planet preset dropdown menu (close if already open, open if closed)
                                    if self.planet_preset_dropdown_visible:
                                        self.planet_preset_dropdown_visible = False
                                    else:
                                        self.planet_preset_dropdown_visible = True
                                    # Clear selected body when switching to planet tab
                                    self.selected_body = None
                                    self.selected_body_id = None
                                    self.show_customization_panel = False
                                    # Clear preview when switching to planet tab (explicit state change)
                                    if not self.planet_dropdown_selected:
                                        self.clear_preview()
                                    self.mass_input_active = False
                                    # Also close the customization panel dropdown if it was open
                                    self.planet_dropdown_active = False
                                    self.planet_dropdown_visible = False
                                else:
                                    # For other tabs, toggle active tab - if clicking the same tab, deactivate it
                                    if self.active_tab == tab_name:
                                        self.active_tab = None
                                        # Clear preview when deactivating tab (explicit state change)
                                        if not self.planet_dropdown_selected:
                                            self.clear_preview()
                                    else:
                                        self.active_tab = tab_name
                                        # Activate placement mode for stars and moons
                                        if tab_name in ["star", "moon"]:
                                            self.start_placement_mode(tab_name)
                                    # Clear selected body when changing tabs
                                    self.selected_body = None
                                    self.selected_body_id = None
                                    self.show_customization_panel = False
                                    # Clear preview when switching tabs (explicit state change)
                                    if not self.planet_dropdown_selected:
                                        self.clear_preview()
                                    self.mass_input_active = False
                                    self.planet_dropdown_active = False
                                    self.planet_dropdown_visible = False
                                    # Close preset dropdown when switching tabs
                                    self.planet_preset_dropdown_visible = False
                                break
                        
                        # Handle space area clicks
                        space_area = pygame.Rect(0, self.tab_height + 2*self.tab_margin, 
                                              self.width, 
                                              self.height - (self.tab_height + 2*self.tab_margin))
                        
                        # Skip selection while panning
                        if self.is_panning:
                            continue
                        
                        # Check if click is on a celestial body (using hitbox for easier selection)
                        clicked_body = None
                        for body in self.placed_bodies:
                            body_screen = self.world_to_screen(body["position"])
                            # CRITICAL: Compute hitbox from visual radius
                            # For planets, use NASA-style normalized visual radius
                            # For moons, scale relative to parent planet
                            if body["type"] == "planet":
                                visual_radius = self.calculate_planet_visual_radius(body, self.placed_bodies)
                            elif body["type"] == "moon":
                                visual_radius = self.calculate_moon_visual_radius(body, self.placed_bodies)
                            else:
                                # Stars: calculate visual radius using authoritative formula
                                visual_radius = self.calculate_star_visual_radius(body)
                            
                            # Calculate hitbox with scale factor for clickability
                            hitbox_radius_px = self.calculate_hitbox_radius(body["type"], visual_radius)
                            body_hitbox_radius = hitbox_radius_px * self.camera_zoom
                            if (event.pos[0] - body_screen[0])**2 + (event.pos[1] - body_screen[1])**2 <= body_hitbox_radius**2:
                                clicked_body = body
                                break
                        
                        if clicked_body:
                            # CRITICAL: Verify clicked body is in registry and get canonical reference
                            clicked_id = clicked_body.get("id")
                            if not clicked_id:
                                print(f"ERROR: Clicked body has no ID! name={clicked_body.get('name')}")
                                continue
                            
                            # Get canonical body from registry to ensure we're using the right reference
                            if clicked_id not in self.bodies_by_id:
                                print(f"ERROR: Clicked body id {clicked_id} not in registry!")
                                continue
                            
                            canonical_body = self.bodies_by_id[clicked_id]
                            
                            # Verify clicked body and canonical body are the same object
                            if id(clicked_body) != id(canonical_body):
                                print(f"WARNING: Clicked body dict_id={id(clicked_body)} differs from registry dict_id={id(canonical_body)}")
                                print(f"  Using canonical body from registry")
                            
                            # Select the clicked body (by both direct reference and stable ID)
                            self.selected_body = canonical_body
                            self.selected_body_id = clicked_id
                            
                            # Trigger Click-to-Focus camera mode
                            self.camera_focus["active"] = True
                            self.camera_focus["target_body_id"] = clicked_id
                            self.camera_focus["target_world_pos"] = canonical_body["position"].copy()
                            
                            if canonical_body["type"] == "star":
                                self.camera_focus["target_zoom"] = 0.6
                            elif canonical_body["type"] == "planet":
                                self.camera_focus["target_zoom"] = 1.0
                            elif canonical_body["type"] == "moon":
                                self.camera_focus["target_zoom"] = 1.5
                            
                            print(f"SELECTED body_id={clicked_id[:8]} name={canonical_body.get('name')} dict_id={id(canonical_body)}")
                            
                            self.show_customization_panel = True
                            self.mass_input_active = False
                            self.planet_dropdown_active = False
                            self.planet_dropdown_visible = False
                            
                            # CRITICAL: Initialize dropdown selection based on body's current mass (per-body state)
                            if canonical_body.get('type') == 'planet':
                                # Find which dropdown option matches the body's current mass
                                body_mass = canonical_body.get('mass', 1.0)
                                if hasattr(body_mass, 'item'):
                                    body_mass = float(body_mass.item())
                                else:
                                    body_mass = float(body_mass)
                                
                                # Find matching option
                                matching_option = None
                                for name, value in self.planet_dropdown_options:
                                    if value is not None and abs(float(value) - body_mass) < 0.01:  # Small tolerance for float comparison
                                        matching_option = name
                                        break
                                
                                # Initialize per-body dropdown selection
                                if matching_option:
                                    canonical_body["planet_dropdown_selected"] = matching_option
                                else:
                                    # Mass doesn't match any preset, use "Custom"
                                    canonical_body["planet_dropdown_selected"] = "Custom"
                            # --- Planet age dropdown selection logic ---
                            if self.selected_body.get('type') == 'planet':
                                age = self.selected_body.get('age', 4.6)
                                # Ensure age is a Python float, not a numpy array/scalar
                                if hasattr(age, 'item'):
                                    age = float(age.item())
                                else:
                                    age = float(age)
                                found = False
                                for name, preset_age in self.planet_age_dropdown_options:
                                    if preset_age is not None and abs(preset_age - age) < 0.1:
                                        self.planet_age_dropdown_selected = name
                                        found = True
                                        break
                                if not found:
                                    self.planet_age_dropdown_selected = "Custom"
                            
                            # --- Planet temperature dropdown selection logic ---
                            if self.selected_body.get('type') == 'planet':
                                temperature = self.selected_body.get('temperature', 288)
                                # Ensure temperature is a Python float, not a numpy array/scalar
                                if hasattr(temperature, 'item'):
                                    temperature = float(temperature.item())
                                else:
                                    temperature = float(temperature)
                                found = False
                                for name, preset_temp in self.planet_temperature_dropdown_options:
                                    if preset_temp is not None and abs(preset_temp - temperature) < 1:
                                        self.planet_temperature_dropdown_selected = name
                                        found = True
                                        break
                                if not found:
                                    self.planet_temperature_dropdown_selected = "Custom"
                            
                            # --- Planet radius dropdown selection logic ---
                            if self.selected_body.get('type') == 'planet':
                                # CRITICAL: For planets, radius is stored in Earth radii (R⊕)
                                radius_earth = self.selected_body.get('radius', 1.0)
                                # Ensure radius is a Python float
                                if hasattr(radius_earth, 'item'):
                                    radius_earth = float(radius_earth.item())
                                else:
                                    radius_earth = float(radius_earth)
                                found = False
                                for name, preset_radius in self.planet_radius_dropdown_options:
                                    if preset_radius is not None and abs(preset_radius - radius_earth) < 0.01:
                                        self.planet_radius_dropdown_selected = name
                                        found = True
                                        break
                                if not found:
                                    self.planet_radius_dropdown_selected = "Custom"
                            
                            # --- Planet atmosphere dropdown selection logic ---
                            if self.selected_body.get('type') == 'planet':
                                greenhouse_offset = self.selected_body.get('greenhouse_offset', 33.0)
                                # Ensure greenhouse_offset is a Python float
                                if hasattr(greenhouse_offset, 'item'):
                                    greenhouse_offset = float(greenhouse_offset.item())
                                else:
                                    greenhouse_offset = float(greenhouse_offset)
                                found = False
                                for name, preset_offset in self.planet_atmosphere_dropdown_options:
                                    if preset_offset is not None and abs(preset_offset - greenhouse_offset) < 0.1:
                                        self.planet_atmosphere_dropdown_selected = name
                                        found = True
                                        break
                                if not found:
                                    self.planet_atmosphere_dropdown_selected = "Custom"
                            
                            # --- Planet gravity dropdown selection logic ---
                            if self.selected_body.get('type') == 'planet':
                                gravity = self.selected_body.get('gravity', 9.81)
                                # Ensure gravity is a Python float
                                if hasattr(gravity, 'item'):
                                    gravity = float(gravity.item())
                                else:
                                    gravity = float(gravity)
                                found = False
                                for name, preset_gravity in self.planet_gravity_dropdown_options:
                                    if preset_gravity is not None and abs(preset_gravity - gravity) < 0.01:
                                        self.planet_gravity_dropdown_selected = name
                                        found = True
                                        break
                                if not found:
                                    self.planet_gravity_dropdown_selected = "Custom"
                            
                            # --- Planet orbital distance (semi-major axis) dropdown selection logic ---
                            if self.selected_body.get('type') == 'planet':
                                semi_major_axis = self.selected_body.get('semiMajorAxis', 1.0)
                                # Ensure semiMajorAxis is a Python float
                                if hasattr(semi_major_axis, 'item'):
                                    semi_major_axis = float(semi_major_axis.item())
                                else:
                                    semi_major_axis = float(semi_major_axis)
                                found = False
                                for name, preset_distance in self.planet_orbital_distance_dropdown_options:
                                    if preset_distance is not None and abs(preset_distance - semi_major_axis) < 0.01:
                                        self.planet_orbital_distance_dropdown_selected = name
                                        found = True
                                        break
                                if not found:
                                    self.planet_orbital_distance_dropdown_selected = "Custom"
                            
                            # --- Planet orbital eccentricity dropdown selection logic ---
                            if self.selected_body.get('type') == 'planet':
                                eccentricity = self.selected_body.get('eccentricity', 0.0167)
                                # Ensure eccentricity is a Python float
                                if hasattr(eccentricity, 'item'):
                                    eccentricity = float(eccentricity.item())
                                else:
                                    eccentricity = float(eccentricity)
                                found = False
                                for name, preset_ecc in self.planet_orbital_eccentricity_dropdown_options:
                                    if preset_ecc is not None and abs(preset_ecc - eccentricity) < 0.001:
                                        self.planet_orbital_eccentricity_dropdown_selected = name
                                        found = True
                                        break
                                if not found:
                                    self.planet_orbital_eccentricity_dropdown_selected = "Custom"
                            
                            # --- Planet orbital period dropdown selection logic ---
                            if self.selected_body.get('type') == 'planet':
                                orbital_period = self.selected_body.get('orbital_period', 365.25)
                                # Ensure orbital_period is a Python float
                                if hasattr(orbital_period, 'item'):
                                    orbital_period = float(orbital_period.item())
                                else:
                                    orbital_period = float(orbital_period)
                                found = False
                                for name, preset_period in self.planet_orbital_period_dropdown_options:
                                    if preset_period is not None and abs(preset_period - orbital_period) < 0.1:
                                        self.planet_orbital_period_dropdown_selected = name
                                        found = True
                                        break
                                if not found:
                                    self.planet_orbital_period_dropdown_selected = "Custom"
                            
                            # --- Planet stellar flux dropdown selection logic ---
                            if self.selected_body.get('type') == 'planet':
                                stellar_flux = self.selected_body.get('stellarFlux', 1.0)
                                # Ensure stellarFlux is a Python float
                                if hasattr(stellar_flux, 'item'):
                                    stellar_flux = float(stellar_flux.item())
                                else:
                                    stellar_flux = float(stellar_flux)
                                found = False
                                for name, preset_flux in self.planet_stellar_flux_dropdown_options:
                                    if preset_flux is not None and abs(preset_flux - stellar_flux) < 0.001:
                                        self.planet_stellar_flux_dropdown_selected = name
                                        found = True
                                        break
                                if not found:
                                    self.planet_stellar_flux_dropdown_selected = "Custom"
                            
                            # --- Planet density dropdown selection logic ---
                            if self.selected_body.get('type') == 'planet':
                                density = self.selected_body.get('density', 5.51)  # Default to Earth's density if not set
                                # Ensure density is a Python float
                                if hasattr(density, 'item'):
                                    density = float(density.item())
                                else:
                                    density = float(density)
                                found = False
                                for name, preset_density in self.planet_density_dropdown_options:
                                    if preset_density is not None and abs(preset_density - density) < 0.01:
                                        self.planet_density_dropdown_selected = name
                                        found = True
                                        break
                                if not found:
                                    self.planet_density_dropdown_selected = "Custom"
                            
                            # --- Star mass dropdown selection logic ---
                            if self.selected_body.get('type') == 'star':
                                mass_solar = self.selected_body.get('mass', 1.0)  # Already in Solar masses
                                # Ensure mass is a Python float
                                if hasattr(mass_solar, 'item'):
                                    mass_solar = float(mass_solar.item())
                                else:
                                    mass_solar = float(mass_solar)
                                
                                found = False
                                for name, preset_mass in self.star_mass_dropdown_options:
                                    if preset_mass is not None and abs(preset_mass - mass_solar) < 0.01:
                                        self.star_mass_dropdown_selected = name
                                        found = True
                                        break
                                if not found:
                                    self.star_mass_dropdown_selected = "Custom"
                            
                            # --- Star temperature (spectral class) dropdown selection logic ---
                            if self.selected_body.get('type') == 'star':
                                # Stars use "temperature" for updates, but may have "star_temperature" for initialization
                                temperature = self.selected_body.get('temperature', self.selected_body.get('star_temperature', 5778))
                                # Ensure temperature is a Python float
                                if hasattr(temperature, 'item'):
                                    temperature = float(temperature.item())
                                else:
                                    temperature = float(temperature)
                                found = False
                                for name, preset_temp, preset_color in self.spectral_class_dropdown_options:
                                    if preset_temp is not None and abs(preset_temp - temperature) < 50:  # 50K tolerance
                                        self.spectral_class_dropdown_selected = name
                                        found = True
                                        break
                                if not found:
                                    self.spectral_class_dropdown_selected = "Custom"
                            
                            # --- Star luminosity dropdown selection logic ---
                            if self.selected_body.get('type') == 'star':
                                luminosity = self.selected_body.get('luminosity', 1.0)
                                # Ensure luminosity is a Python float
                                if hasattr(luminosity, 'item'):
                                    luminosity = float(luminosity.item())
                                else:
                                    luminosity = float(luminosity)
                                found = False
                                for name, preset_lum in self.luminosity_dropdown_options:
                                    if preset_lum is not None and abs(preset_lum - luminosity) < 0.01:
                                        self.luminosity_dropdown_selected = name
                                        found = True
                                        break
                                if not found:
                                    self.luminosity_dropdown_selected = "Custom"
                            
                            # --- Star radius dropdown selection logic ---
                            if self.selected_body.get('type') == 'star':
                                radius_solar = self.selected_body.get('radius', 1.0)
                                # Ensure radius_solar is a Python float
                                if hasattr(radius_solar, 'item'):
                                    radius_solar = float(radius_solar.item())
                                else:
                                    radius_solar = float(radius_solar)
                                found = False
                                for name, preset_radius in self.radius_dropdown_options:
                                    if preset_radius is not None and abs(preset_radius - radius_solar) < 0.1:
                                        self.radius_dropdown_selected = name
                                        found = True
                                        break
                                if not found:
                                    self.radius_dropdown_selected = "Custom"
                            
                            # --- Star activity level dropdown selection logic ---
                            if self.selected_body.get('type') == 'star':
                                activity = self.selected_body.get('activity', 0.5)  # Default to Sun's moderate activity
                                # Ensure activity is a Python float
                                if hasattr(activity, 'item'):
                                    activity = float(activity.item())
                                else:
                                    activity = float(activity)
                                found = False
                                for name, preset_activity in self.activity_dropdown_options:
                                    if preset_activity is not None and abs(preset_activity - activity) < 0.01:
                                        self.activity_dropdown_selected = name
                                        found = True
                                        break
                                if not found:
                                    self.activity_dropdown_selected = "Custom"
                            
                            # --- Star metallicity dropdown selection logic ---
                            if self.selected_body.get('type') == 'star':
                                metallicity = self.selected_body.get('metallicity', 0.0)  # Default to Sun's metallicity
                                # Ensure metallicity is a Python float
                                if hasattr(metallicity, 'item'):
                                    metallicity = float(metallicity.item())
                                else:
                                    metallicity = float(metallicity)
                                found = False
                                for name, preset_metallicity in self.metallicity_dropdown_options:
                                    if preset_metallicity is not None and abs(preset_metallicity - metallicity) < 0.01:
                                        self.metallicity_dropdown_selected = name
                                        found = True
                                        break
                                if not found:
                                    self.metallicity_dropdown_selected = "Custom"
                            
                            # --- Moon age dropdown selection logic ---
                            elif self.selected_body.get('type') == 'moon':
                                age = self.selected_body.get('age', 4.6)
                                # Ensure age is a Python float, not a numpy array/scalar
                                if hasattr(age, 'item'):
                                    age = float(age.item())
                                else:
                                    age = float(age)
                                found = False
                                for name, preset_age in self.moon_age_dropdown_options:
                                    if preset_age == age:
                                        self.moon_age_dropdown_selected = name
                                        found = True
                                        break
                                if not found:
                                    self.moon_age_dropdown_selected = "Custom"
                                
                                # --- Moon radius dropdown selection logic ---
                                radius = self.selected_body.get('actual_radius', 1737.4)
                                found = False
                                for name, preset_radius in self.moon_radius_dropdown_options:
                                    if preset_radius == radius:
                                        self.moon_radius_dropdown_selected = name
                                        found = True
                                        break
                                if not found:
                                    self.moon_radius_dropdown_selected = "Custom"
                                
                                # --- Moon orbital distance dropdown selection logic ---
                                semi_major_axis = self.selected_body.get('semiMajorAxis', 0.00257)
                                # Convert AU to km for dropdown comparison
                                AU_TO_KM = 149597870.7
                                distance_km = semi_major_axis * AU_TO_KM
                                found = False
                                for name, preset_distance in self.moon_orbital_distance_dropdown_options:
                                    if preset_distance is not None and abs(preset_distance - distance_km) < 100:  # Larger tolerance for km
                                        self.moon_orbital_distance_dropdown_selected = name
                                        found = True
                                        break
                                if not found:
                                    self.moon_orbital_distance_dropdown_selected = "Custom"
                                
                                # --- Moon orbital period dropdown selection logic ---
                                orbital_period = self.selected_body.get('orbital_period', 27.3)
                                found = False
                                for name, preset_period in self.moon_orbital_period_dropdown_options:
                                    if preset_period is not None and abs(preset_period - orbital_period) < 0.1:
                                        self.moon_orbital_period_dropdown_selected = name
                                        found = True
                                        break
                                if not found:
                                    self.moon_orbital_period_dropdown_selected = "Moon"  # Default to Moon instead of Custom
                                
                                # --- Moon temperature dropdown selection logic ---
                                temperature = self.selected_body.get('temperature', 220)
                                found = False
                                for name, preset_temp in self.moon_temperature_dropdown_options:
                                    if preset_temp is not None and abs(preset_temp - temperature) < 1:
                                        self.moon_temperature_dropdown_selected = name
                                        found = True
                                        break
                                if not found:
                                    self.moon_temperature_dropdown_selected = "Moon"  # Default to Moon instead of Custom
                                
                                # --- Moon mass dropdown selection logic ---
                                mass = self.selected_body.get('mass', 1.0)
                                # Ensure mass is a Python float, not a numpy array/scalar
                                if hasattr(mass, 'item'):
                                    mass = float(mass.item())
                                else:
                                    mass = float(mass)
                                found = False
                                for option_data in self.moon_dropdown_options:
                                    if len(option_data) == 3:
                                        moon_name, preset_mass, unit = option_data
                                    else:
                                        moon_name, preset_mass = option_data
                                        unit = None
                                    if preset_mass is not None:
                                        # Convert preset_mass to lunar masses for comparison
                                        if unit == "kg":
                                            preset_mass_lunar = preset_mass / 7.35e22  # Convert kg to M☾
                                        else:
                                            preset_mass_lunar = preset_mass
                                        if abs(preset_mass_lunar - mass) < 0.0001:  # Small tolerance for float comparison
                                            self.moon_dropdown_selected = moon_name
                                            found = True
                                            break
                                if not found:
                                    self.moon_dropdown_selected = "Custom"
                                
                                # --- Moon gravity dropdown selection logic ---
                                gravity = self.selected_body.get('gravity', 1.62)  # Default to Moon's gravity
                                # Ensure gravity is a Python float
                                if hasattr(gravity, 'item'):
                                    gravity = float(gravity.item())
                                else:
                                    gravity = float(gravity)
                                found = False
                                for name, preset_gravity in self.moon_gravity_dropdown_options:
                                    if preset_gravity is not None and abs(preset_gravity - gravity) < 0.001:
                                        self.moon_gravity_dropdown_selected = name
                                        found = True
                                        break
                                if not found:
                                    self.moon_gravity_dropdown_selected = "Custom"
                        elif self.active_tab and space_area.collidepoint(event.pos):
                            # Check if click was on a dropdown option - if so, don't place
                            clicked_on_dropdown_option = False
                            
                            # Check planet dropdown options
                            if self.planet_dropdown_visible:
                                dropdown_y = self.planet_dropdown_rect.bottom
                                for i, (planet_name, mass) in enumerate(self.planet_dropdown_options):
                                    option_rect = pygame.Rect(
                                        self.planet_dropdown_rect.left,
                                        dropdown_y + i * 30,
                                        self.planet_dropdown_rect.width,
                                        30
                                    )
                                    if option_rect.collidepoint(event.pos):
                                        clicked_on_dropdown_option = True
                                        break
                            
                            # Check preset dropdown options
                            if not clicked_on_dropdown_option and self.planet_preset_dropdown_visible and self.planet_preset_dropdown_rect:
                                if self.planet_preset_dropdown_rect.collidepoint(event.pos):
                                    clicked_on_dropdown_option = True
                            
                            # Check other dropdown options (using dropdown_options_rects)
                            if not clicked_on_dropdown_option and hasattr(self, 'dropdown_options_rects') and self.dropdown_options_rects:
                                for i, option_rect in enumerate(self.dropdown_options_rects):
                                    if hasattr(self, 'dropdown_rect') and self.dropdown_rect:
                                        screen_option_rect = pygame.Rect(
                                            self.dropdown_rect.left,
                                            self.dropdown_rect.top + i * self.dropdown_option_height,
                                            option_rect.width,
                                            option_rect.height
                                        )
                                        if screen_option_rect.collidepoint(event.pos):
                                            clicked_on_dropdown_option = True
                                            break
                            
                            # Only place if not clicking on a dropdown option
                            if not clicked_on_dropdown_option:
                                # Clear preview when placing object (explicit state change)
                                self.clear_preview()
                                
                                # Create a new celestial body at the click position
                                self.body_counter[self.active_tab] += 1
                                
                                # Set default values based on body type
                                if self.active_tab == "star":
                                    # Sun-like defaults
                                    default_mass = 1.0  # Solar masses
                                    default_age = 4.6  # Gyr
                                    default_spectral = "G-type (Yellow, Sun)"
                                    default_luminosity = 1.0  # Solar luminosities
                                    default_name = "Sun"
                                    default_radius = 1.0  # Solar radii (R☉)
                                elif self.active_tab == "planet":
                                    # Require explicit planet selection - no default
                                    selected_planet_name = self.planet_dropdown_selected if hasattr(self, 'planet_dropdown_selected') else None
                                    
                                    # Debug output
                                    print(f"DEBUG: Attempting to place planet. selected_planet_name={selected_planet_name}, planet_dropdown_selected={getattr(self, 'planet_dropdown_selected', 'NOT_SET')}")
                                    
                                    # Prevent placement if no planet is selected
                                    if not selected_planet_name or selected_planet_name not in SOLAR_SYSTEM_PLANET_PRESETS:
                                        print(f"Please select a planet from the dropdown before placing (selected: {selected_planet_name})")
                                        # Skip the rest of this block - don't create the body
                                        # Set defaults to None to skip body creation
                                        default_mass = None
                                        default_radius = None
                                        default_name = None
                                        default_age = None
                                    else:
                                        # Get preset parameters
                                        preset = SOLAR_SYSTEM_PLANET_PRESETS[selected_planet_name]
                                        default_mass = preset["mass"]
                                        default_radius = preset["radius"]  # In Earth radii (R⊕)
                                        default_name = selected_planet_name
                                        default_age = 4.6  # Gyr (can be customized later)
                                else:  # moon
                                    # Luna-like defaults
                                    default_mass = 1.0  # Earth's Moon mass (1 lunar mass)
                                    default_age = 4.6  # Gyr
                                    default_name = "Moon"
                                    default_radius = MOON_RADIUS_PX  # Slightly enlarged for visibility
                                
                                # Only create body if we have valid defaults (i.e., a planet was selected for planets, or for stars/moons)
                                if (self.active_tab != "planet" or (default_mass is not None and default_name is not None)):
                                    world_click = self.screen_to_world(event.pos)
                                    
                                    # Create unique ID for this body to ensure independence
                                    body_id = str(uuid4())
                                    
                                    # CRITICAL: For planets, use authoritative preset-based creation
                                    if self.active_tab == "planet":
                                        # Get the selected planet name (already validated above)
                                        selected_planet_name = self.planet_dropdown_selected if hasattr(self, 'planet_dropdown_selected') else None
                                        
                                        # Use authoritative preset-based creation function
                                        if selected_planet_name and selected_planet_name in SOLAR_SYSTEM_PLANET_PRESETS:
                                            body = self.create_planet_from_preset(selected_planet_name, world_click, body_id)
                                            
                                            # Generate orbit grid for visualization (based on target position)
                                            self.generate_orbit_grid(body)
                                            self.clear_orbit_points(body)
                                            
                                            # Body is created with all preset values - skip standard planet attribute setup
                                            # Continue to registration and placement logic below
                                        else:
                                            # Should not happen (validation above), but skip if it does
                                            continue
                                    else:
                                        # For stars and moons, use standard creation
                                        # CRITICAL: Use factory function to create body with NO shared references
                                        body = self._create_new_body_dict(
                                            obj_type=self.active_tab,
                                            body_id=body_id,
                                            position=world_click,
                                            default_name=default_name,
                                            default_mass=default_mass,
                                            default_age=default_age,
                                            default_radius=default_radius
                                        )
                                    
                                    # Add star-specific attributes
                                    if self.active_tab == "star":
                                        body.update({
                                            "luminosity": float(default_luminosity),
                                            "star_temperature": float(5778),  # Sun's temperature in Kelvin
                                            "star_color": (253, 184, 19),  # Yellow color for G-type star (tuple, immutable) - matches base_color
                                            "base_color": str(CELESTIAL_BODY_COLORS.get(default_name, CELESTIAL_BODY_COLORS.get("Sun", "#FDB813"))),  # Ensure base_color is set
                                        })
                                        # Create habitable zone for the star
                                        body["hz_surface"] = self.create_habitable_zone(body)
                                    
                                    # Add moon-specific attributes
                                    if self.active_tab == "moon":
                                        body.update({
                                            "actual_radius": 1737.4,  # Actual radius in km (The Moon) - for dropdown logic
                                            "radius": default_radius,  # Visual radius in pixels for display
                                            "temperature": 220,  # Surface temperature in K (Earth's Moon)
                                            "gravity": 1.62,  # Surface gravity in m/s² (Earth's Moon)
                                            "orbital_period": 27.3,  # Orbital period in days (Earth's Moon)
                                        })
                                    
                                    # Ensure the new body is completely independent
                                    # Clear any selected body to avoid confusion when placing new objects
                                    self.selected_body = None
                                    self.selected_body_id = None
                                    self.show_customization_panel = False
                                    
                                    # Close dropdowns after successful placement
                                    if self.active_tab == "planet":
                                        self.planet_dropdown_visible = False
                                        self.planet_dropdown_active = False
                                        self.planet_preset_dropdown_visible = False
                                    # Clear selection and preview after successful placement (explicit state change)
                                    self.planet_dropdown_selected = None
                                    self.clear_preview()
                                    
                                    self.placed_bodies.append(body)
                                    # Register body by ID for guaranteed unique lookups
                                    self.bodies_by_id[body_id] = body
                                    
                                    # Calculate visual radius for hitbox based on type
                                    if body["type"] == "star":
                                        visual_radius = self.calculate_star_visual_radius(body)
                                    elif body["type"] == "planet":
                                        visual_radius = self.calculate_planet_visual_radius(body, self.placed_bodies)
                                    else:  # moon
                                        visual_radius = self.calculate_moon_visual_radius(body, self.placed_bodies)
                                        
                                    # Update hitbox_radius to match visual radius
                                    body["hitbox_radius"] = float(self.calculate_hitbox_radius(body["type"], visual_radius))
                                    
                                    # CRITICAL: Hard assertion to detect shared state immediately
                                    self._assert_no_shared_state()
                                    
                                    # Debug: Verify independence
                                    print(f"DEBUG: Created body id={body_id}, name={body['name']}, mass={body['mass']}, position_id={id(body['position'])}, params_id={id(body)}")
                                    # Hard verification: ensure no bodies share core containers after creation
                                    self.debug_verify_body_references(source="handle_events_place_body")
                                    
                                    # For moons, immediately find nearest planet and set up orbit
                                    if self.active_tab == "moon":
                                        planets = [b for b in self.placed_bodies if b["type"] == "planet"]
                                        if planets:
                                            # Find nearest planet to the moon's cursor position
                                            nearest_planet = min(planets, key=lambda p: np.linalg.norm(p["position"] - body["position"]))
                                            # Calculate orbit radius in AU from cursor position using reverse visual scaling
                                            # Formula: AU = MOON_ORBIT_AU * (orbit_radius_px / MOON_ORBIT_PX)^2
                                            orbit_radius_px = np.linalg.norm(nearest_planet["position"] - body["position"])
                                            # Ensure minimum orbit radius
                                            if orbit_radius_px < MOON_ORBIT_PX * 0.5:
                                                orbit_radius_px = MOON_ORBIT_PX * 0.5
                                            
                                            orbit_radius_au = MOON_ORBIT_AU * (orbit_radius_px / MOON_ORBIT_PX)**2
                                            
                                            # Set parent and orbital distance in AU
                                            body["parent"] = nearest_planet["name"]
                                            body["parent_id"] = nearest_planet["id"]  # Use UUID for parent lookup
                                            body["parent_obj"] = nearest_planet  # Set permanent parent reference for faster lookups
                                            body["semiMajorAxis"] = float(orbit_radius_au)
                                            body["orbit_radius"] = float(self.get_visual_moon_orbit_radius(orbit_radius_au))
                                            
                                            # Calculate initial orbit angle from cursor position
                                            dx = body["position"][0] - nearest_planet["position"][0]
                                            dy = body["position"][1] - nearest_planet["position"][1]
                                            body["orbit_angle"] = np.arctan2(dy, dx)
                                            
                                            # Calculate orbital speed based on orbital period (default 27.3 days)
                                            # We no longer apply MOON_SPEED_FACTOR to prevent aliasing/flickering
                                            orbital_period = body.get("orbital_period", 27.3)
                                            omega_real = (2 * np.pi * 365.25) / orbital_period
                                            body["orbit_speed"] = float(omega_real)
                                            
                                            # CRITICAL: Immediately recalculate moon position from planet + orbit offset
                                            # This ensures the moon starts at the correct position relative to the planet
                                            moon_offset_x = orbit_radius_px * np.cos(body["orbit_angle"])
                                            moon_offset_y = orbit_radius_px * np.sin(body["orbit_angle"])
                                            # Instrumentation: Pre-write
                                            trace(f"PRE_WRITE {body['name']} pos={body['position'].copy()} source=handle_events_moon_placement")
                                            body["position"][0] = nearest_planet["position"][0] + moon_offset_x
                                            body["position"][1] = nearest_planet["position"][1] + moon_offset_y
                                            # Ensure position is float array
                                            body["position"] = np.array(body["position"], dtype=float)
                                            # Instrumentation: Post-write
                                            trace(f"POST_WRITE {body['name']} pos={body['position'].copy()} source=handle_events_moon_placement")
                                        
                                        # Debug output
                                        print(f"DEBUG: Moon {body['name']} placed:")
                                        print(f"  parent={nearest_planet['name']}, orbit_radius={orbit_radius_px:.2f}")
                                        print(f"  orbit_angle={body['orbit_angle']:.4f}, orbit_speed={body['orbit_speed']:.6f}")
                                        print(f"  planet_pos=({nearest_planet['position'][0]:.2f}, {nearest_planet['position'][1]:.2f})")
                                        print(f"  moon_pos=({body['position'][0]:.2f}, {body['position'][1]:.2f})")
                                        
                                        # Set initial velocity for circular orbit
                                        v = body["orbit_speed"] * body["orbit_radius"]
                                        body["velocity"] = np.array([-v * np.sin(body["orbit_angle"]), v * np.cos(body["orbit_angle"])]).copy()  # Ensure independent copy
                                        
                                        # Generate orbit grid for visualization
                                        self.generate_orbit_grid(body)
                                    else:
                                        # No planets available yet, moon will be set up later in update_physics
                                        pass
                                # Note: orbit_points is now stored in the body dict itself, not in self.orbit_points
                                
                                # Set dropdown selections to match defaults
                                if self.active_tab == "star":
                                    self.star_mass_dropdown_selected = "Sun-like Star (1.0 M☉)"
                                    self.star_age_dropdown_selected = "4.6 (Gyr) (Sun)"
                                    self.spectral_dropdown_selected = "G-type (Yellow, Sun) (5,778 K)"
                                    self.luminosity_dropdown_selected = "G-type Main Sequence (Sun)"
                                    self.temperature_dropdown_selected = "G-type (Sun) (5,800 K)"
                                    self.radius_dropdown_selected = "G-type (Sun)"
                                    self.activity_dropdown_selected = "Moderate (Sun)"
                                    self.metallicity_dropdown_selected = "0.0 (Sun)"
                                elif self.active_tab == "planet":
                                    # Store selected planet name in body (already done above)
                                    # Sync UI dropdowns to match preset values
                                    selected_planet_name = body.get("planet_dropdown_selected", "Earth")
                                    if selected_planet_name in SOLAR_SYSTEM_PLANET_PRESETS:
                                        preset = SOLAR_SYSTEM_PLANET_PRESETS[selected_planet_name]
                                        # Sync gravity dropdown
                                        gravity = preset.get("gravity", 9.81)
                                        found = False
                                        for name, preset_gravity in self.planet_gravity_dropdown_options:
                                            if preset_gravity is not None and abs(preset_gravity - gravity) < 0.01:
                                                self.planet_gravity_dropdown_selected = name
                                                found = True
                                                break
                                        if not found:
                                            self.planet_gravity_dropdown_selected = "Custom"
                                        
                                        # Sync atmosphere dropdown
                                        greenhouse_offset = preset.get("greenhouse_offset", 33.0)
                                        found = False
                                        for name, preset_offset in self.planet_atmosphere_dropdown_options:
                                            if preset_offset is not None and abs(preset_offset - greenhouse_offset) < 0.1:
                                                self.planet_atmosphere_dropdown_selected = name
                                                found = True
                                                break
                                        if not found:
                                            self.planet_atmosphere_dropdown_selected = "Custom"
                                        
                                        # Sync orbital distance dropdown
                                        semi_major_axis = preset.get("semiMajorAxis", 1.0)
                                        found = False
                                        for name, preset_distance in self.planet_orbital_distance_dropdown_options:
                                            if preset_distance is not None and abs(preset_distance - semi_major_axis) < 0.01:
                                                self.planet_orbital_distance_dropdown_selected = name
                                                found = True
                                                break
                                        if not found:
                                            self.planet_orbital_distance_dropdown_selected = "Custom"
                                        
                                        # Sync other dropdowns similarly
                                        self.planet_age_dropdown_selected = "Solar System Planets (4.6 Gyr)"  # Default age
                                        self.planet_orbital_eccentricity_dropdown_selected = "Earth"  # Can be enhanced
                                        self.planet_orbital_period_dropdown_selected = "Earth"  # Can be enhanced
                                        self.planet_stellar_flux_dropdown_selected = "Earth"  # Can be enhanced
                                    else:
                                        # Default Earth values
                                        self.planet_age_dropdown_selected = "Solar System Planets (4.6 Gyr)"
                                        self.planet_gravity_dropdown_selected = "Earth"
                                        self.planet_atmosphere_dropdown_selected = "Earth-like (Moderate)"
                                        self.planet_orbital_distance_dropdown_selected = "Earth"
                                        self.planet_orbital_eccentricity_dropdown_selected = "Earth"
                                        self.planet_orbital_period_dropdown_selected = "Earth"
                                        self.planet_stellar_flux_dropdown_selected = "Earth"
                                elif self.active_tab == "moon":  # moon
                                    self.moon_dropdown_selected = "Moon"
                                    self.moon_age_dropdown_selected = "Moon"
                                    self.moon_radius_dropdown_selected = "Moon"
                                    self.moon_orbital_distance_dropdown_selected = "Moon"
                                    self.moon_orbital_period_dropdown_selected = "Moon"
                                    self.moon_temperature_dropdown_selected = "Moon"
                                    self.moon_gravity_dropdown_selected = "Moon"
                                
                                # Automatically start simulation when at least one star and one planet are placed
                                stars = [b for b in self.placed_bodies if b["type"] == "star"]
                                planets = [b for b in self.placed_bodies if b["type"] == "planet"]
                                
                                if len(stars) > 0 and len(planets) > 0:
                                    print(f"DEBUG: Starting simulation. Selected body: {self.selected_body}")
                                    print(f"DEBUG: show_customization_panel before: {self.show_customization_panel}")
                                    self.show_simulation_builder = False
                                    self.show_simulation = True
                                    # Clear any selected body and active tab when simulation starts for better UX
                                    self.selected_body = None
                                    self.selected_body_id = None
                                    self.show_customization_panel = False
                                    self.active_tab = None
                                self.clear_preview()  # Clear preview when simulation starts
                                # Initialize all orbits when simulation starts
                                self.initialize_all_orbits()
                                print(f"DEBUG: show_customization_panel after: {self.show_customization_panel}")
                        else:
                            # Clicked empty space, deselect body
                            self.selected_body = None
                            self.selected_body_id = None
                            self.show_customization_panel = False
                            self.mass_input_active = False
                            self.planet_dropdown_active = False
                            self.planet_dropdown_visible = False
                elif self.show_simulation:
                    # Check if click is in the space area (not in the tab area)
                    space_area = pygame.Rect(0, self.tab_height + 2*self.tab_margin, 
                                          self.width, 
                                          self.height - (self.tab_height + 2*self.tab_margin))
                    if space_area.collidepoint(event.pos):
                        if event.button == 1:  # Left click to return to builder
                            self.show_simulation = False
                            self.show_simulation_builder = True
            elif event.type == pygame.MOUSEBUTTONUP:
                # Don't close dropdown on mouse up
                self.mass_input_active = False
                self.planet_dropdown_active = False
            elif event.type == pygame.MOUSEMOTION:
                # Update mass slider if dragging
                if self.mass_input_active and self.selected_body:
                    # Constrain to slider bounds
                    x_pos = max(self.mass_input_rect.left, min(event.pos[0], self.mass_input_rect.right))
                    self.mass_input_text = f"{x_pos - self.mass_input_rect.left}/{self.mass_input_rect.width}"
            elif event.type == pygame.KEYDOWN:
                if self.mass_input_active and self.selected_body:
                    if event.key == pygame.K_RETURN:
                        # Try to convert input to float and validate
                        new_mass = self._parse_input_value(self.mass_input_text)
                        if new_mass is not None:
                            if self.selected_body.get('type') == 'moon':
                                # For moons, check if input is in kg or M☾
                                input_text = self.mass_input_text.strip().lower()
                                if 'kg' in input_text:
                                    # Convert kg to lunar masses (1 M☾ = 7.35e22 kg)
                                    lunar_mass = new_mass / 7.35e22
                                else:
                                    # Assume M☾ if no unit specified
                                    lunar_mass = new_mass
                                
                                if self.mass_min <= lunar_mass <= self.mass_max:
                                    # Ensure we're updating the correct selected body and mass is stored as a Python float
                                    if self.selected_body:
                                        # Ensure mass is stored as a Python float, not a numpy array/scalar
                                        if hasattr(lunar_mass, 'item'):
                                            self.update_selected_body_property("mass", lunar_mass, "mass")
                                        else:
                                            self.update_selected_body_property("mass", lunar_mass, "mass")
                                # Regenerate orbit grid for non-star bodies after mass change
                                if self.selected_body and self.selected_body.get("type") != "star":
                                    self.generate_orbit_grid(self.selected_body)
                            else:
                                if self.mass_min <= new_mass <= self.mass_max:
                                    # Ensure we're updating the correct selected body and mass is stored as a Python float
                                    if self.selected_body:
                                        # Ensure mass is stored as a Python float, not a numpy array/scalar
                                        if hasattr(new_mass, 'item'):
                                            self.update_selected_body_property("mass", new_mass, "mass")
                                        else:
                                            self.update_selected_body_property("mass", new_mass, "mass")
                                    if self.selected_body and self.selected_body.get("type") != "star":
                                        self.generate_orbit_grid(self.selected_body)
                                self.mass_input_active = False
                        else:
                            # Invalid input, keep current value
                            if self.selected_body.get('type') == 'moon':
                                # CRITICAL: Read mass from registry
                                body = self.get_selected_body()
                                if body and body.get('type') == 'moon':
                                    lunar_mass = body.get('mass', 1.0)
                                    self.mass_input_text = self._format_value(lunar_mass, '', for_dropdown=False)
                                else:
                                    self.mass_input_text = "1.0"
                            else:
                                # CRITICAL: Read mass from registry
                                body = self.get_selected_body()
                                if body:
                                    self.mass_input_text = self._format_value(body.get('mass', 1.0), '', for_dropdown=False)
                                else:
                                    self.mass_input_text = "1.0"
                    elif event.key == pygame.K_BACKSPACE:
                        self.mass_input_text = self.mass_input_text[:-1]
                    elif event.key == pygame.K_ESCAPE:
                        self.mass_input_active = False
                        # CRITICAL: Read mass from registry
                        body = self.get_selected_body()
                        if body:
                            if body.get('type') == 'moon':
                                # For moons, mass is already in Lunar masses
                                lunar_mass = body.get('mass', 1.0)
                                self.mass_input_text = self._format_value(lunar_mass, '', for_dropdown=False)
                            else:
                                self.mass_input_text = self._format_value(body.get('mass', 1.0), '', for_dropdown=False)
                        else:
                            self.mass_input_text = "1.0"
                    elif event.unicode.isnumeric() or event.unicode == '.' or event.unicode.lower() == 'e' or event.unicode == '-' or event.unicode == '+' or event.unicode.lower() in ['k', 'g', 'm', '☾']:
                        # Allow numbers, decimal point, scientific notation (e, E), signs, and unit characters
                        if self.selected_body.get('type') == 'moon':
                            # For moons, allow scientific notation and unit characters for very small masses
                            self.mass_input_text += event.unicode
                        else:
                            # For other bodies, only allow numbers and decimal point
                            if event.unicode == '.':
                                if '.' in self.mass_input_text or len(self.mass_input_text) == 0:
                                    pass  # Don't allow multiple decimals or starting with a decimal
                                else:
                                    self.mass_input_text += event.unicode
                            elif event.unicode.lower() == 'e' or event.unicode == '-' or event.unicode == '+':
                                pass  # Don't allow scientific notation for non-moons
                            else:
                                # Prevent leading zeros unless followed by a decimal
                                if self.mass_input_text == "0" and event.unicode != '.':
                                    pass
                                else:
                                    self.mass_input_text += event.unicode
                if self.luminosity_input_active and self.selected_body and self.selected_body.get('type') == 'star':
                    if event.key == pygame.K_RETURN:
                        # Try to convert input to float and validate
                        new_luminosity = self._parse_input_value(self.luminosity_input_text)
                        if new_luminosity is not None and self.luminosity_min <= new_luminosity <= self.luminosity_max:
                            self.update_selected_body_property("luminosity", new_luminosity, "luminosity")
                            # Update habitable zone when luminosity changes
                            body = self.get_selected_body()
                            if body:
                                body["hz_surface"] = self.create_habitable_zone(body)
                            self.luminosity_input_active = False
                        else:
                            # Invalid input, keep current value
                            self.luminosity_input_text = self._format_value(self.selected_body.get('luminosity', 1.0), '', for_dropdown=False)
                    elif event.key == pygame.K_BACKSPACE:
                        self.luminosity_input_text = self.luminosity_input_text[:-1]
                    elif event.key == pygame.K_ESCAPE:
                        self.luminosity_input_active = False
                        self.luminosity_input_text = self._format_value(self.selected_body.get('luminosity', 1.0), '', for_dropdown=False)
                    elif event.unicode.isnumeric() or event.unicode == '.':
                        # Only allow numbers and a single decimal point
                        if event.unicode == '.':
                            if '.' in self.luminosity_input_text or len(self.luminosity_input_text) == 0:
                                pass  # Don't allow multiple decimals or starting with a decimal
                            else:
                                self.luminosity_input_text += event.unicode
                        else:
                            # Prevent leading zeros unless followed by a decimal
                            if self.luminosity_input_text == "0" and event.unicode != '.':
                                pass
                            else:
                                self.luminosity_input_text += event.unicode
                if self.temperature_input_active and self.selected_body and self.selected_body.get('type') == 'star':
                    if event.key == pygame.K_RETURN:
                        temp = self._parse_input_value(self.temperature_input_text)
                        if temp is not None and self.temperature_min <= temp <= self.temperature_max:
                            if self.selected_body_id:
                                self.update_selected_body_property("temperature", temp, "temperature")
                            self.temperature_dropdown_selected = f"Custom ({temp:.0f} K)"
                            self.temperature_input_text = ""
                            self.temperature_input_active = False
                            self.show_custom_temperature_input = False
                    elif event.key == pygame.K_BACKSPACE:
                        self.temperature_input_text = self.temperature_input_text[:-1]
                    elif event.unicode.isnumeric() or event.unicode == '.':
                        self.temperature_input_text += event.unicode
                if self.metallicity_input_active and self.selected_body and self.selected_body.get('type') == 'star':
                    if event.key == pygame.K_RETURN:
                        metallicity = self._parse_input_value(self.metallicity_input_text)
                        if metallicity is not None and self.metallicity_min <= metallicity <= self.metallicity_max:
                            self.update_selected_body_property("metallicity", metallicity, "metallosity")
                            self.metallicity_input_text = ""
                            self.metallicity_input_active = False
                            self.show_custom_metallicity_input = False
                    elif event.key == pygame.K_BACKSPACE:
                        self.metallicity_input_text = self.metallicity_input_text[:-1]
                    elif event.unicode.isnumeric() or event.unicode == '.':
                        self.metallicity_input_text += event.unicode
                if self.show_custom_radius_input and self.selected_body:
                    if event.key == pygame.K_RETURN:
                        radius = self._parse_input_value(self.radius_input_text)
                        if radius is not None:
                            if self.selected_body.get('type') == 'planet' and 0.1 <= radius <= 20.0:
                                # Planet: radius in Earth radii (R⊕)
                                self.update_selected_body_property("radius", radius, "radius")
                                self.radius_input_text = ""
                                self.show_custom_radius_input = False
                                self.radius_input_active = False
                            elif self.selected_body.get('type') == 'star' and self.radius_min <= radius <= self.radius_max:
                                # Star: radius in Solar radii (R☉)
                                self.update_selected_body_property("radius", radius, "radius")
                                self.radius_input_text = ""
                                self.show_custom_radius_input = False
                                self.radius_input_active = False
                    elif event.key == pygame.K_BACKSPACE:
                        self.radius_input_text = self.radius_input_text[:-1]
                    elif event.key == pygame.K_ESCAPE:
                        self.show_custom_radius_input = False
                        self.radius_input_active = False
                        self.radius_input_text = ""
                    elif event.unicode.isnumeric() or event.unicode == '.':
                        self.radius_input_text += event.unicode
                if self.show_custom_planet_gravity_input and self.selected_body and self.selected_body.get('type') == 'planet':
                    if event.key == pygame.K_RETURN:
                        gravity = self._parse_input_value(self.planet_gravity_input_text)
                        if gravity is not None and 0.1 <= gravity <= 100.0:  # Reasonable gravity range
                            self.update_selected_body_property("gravity", gravity, "gravity")
                            self.planet_gravity_input_text = ""
                            self.show_custom_planet_gravity_input = False
                    elif event.key == pygame.K_BACKSPACE:
                        self.planet_gravity_input_text = self.planet_gravity_input_text[:-1]
                    elif event.key == pygame.K_ESCAPE:
                        self.show_custom_planet_gravity_input = False
                        self.planet_gravity_input_text = ""
                    elif event.unicode.isnumeric() or event.unicode == '.':
                        self.planet_gravity_input_text += event.unicode
                if self.show_custom_atmosphere_input and self.selected_body and self.selected_body.get('type') == 'planet':
                    if event.key == pygame.K_RETURN:
                        delta_t = self._parse_input_value(self.planet_atmosphere_input_text)
                        if delta_t is not None and -273.15 <= delta_t <= 1000.0:  # Reasonable range for ΔT
                            # Calculate new surface temperature: T_surface = T_eq + ΔT_greenhouse
                            if 'equilibrium_temperature' not in self.selected_body:
                                # Use current temperature as equilibrium if not set
                                current_temp = self.selected_body.get('temperature', 255)
                                self.selected_body['equilibrium_temperature'] = current_temp
                            T_eq = self.selected_body.get('equilibrium_temperature', 255)
                            T_surface = T_eq + delta_t
                            # CRITICAL: Use update_selected_body_property to ensure we update ONLY the selected body
                            self.update_selected_body_property("temperature", T_surface, "temperature")
                            self.update_selected_body_property("greenhouse_offset", delta_t, "greenhouse_offset")
                            self.planet_atmosphere_dropdown_selected = "Custom"
                            self.planet_atmosphere_input_text = ""
                            self.show_custom_atmosphere_input = False
                            # Update scores (f_T and H)
                            self._update_planet_scores()
                    elif event.key == pygame.K_BACKSPACE:
                        self.planet_atmosphere_input_text = self.planet_atmosphere_input_text[:-1]
                    elif event.key == pygame.K_ESCAPE:
                        self.show_custom_atmosphere_input = False
                        self.planet_atmosphere_input_text = ""
                    elif event.unicode.isnumeric() or event.unicode == '.' or event.unicode == '-' or event.unicode == '+':
                        # Allow numbers, decimal point, and signs for ΔT
                        self.planet_atmosphere_input_text += event.unicode
                if self.show_custom_moon_radius_input and self.selected_body and self.selected_body.get('type') == 'moon':
                    if event.key == pygame.K_RETURN:
                        radius = self._parse_input_value(self.moon_radius_input_text)
                        if radius is not None and 1 <= radius <= 10000:
                            self.update_selected_body_property("actual_radius", radius, "actual_radius")
                            self.moon_radius_input_text = ""
                            self.show_custom_moon_radius_input = False
                    elif event.key == pygame.K_BACKSPACE:
                        self.moon_radius_input_text = self.moon_radius_input_text[:-1]
                    elif event.key == pygame.K_ESCAPE:
                        self.show_custom_moon_radius_input = False
                        self.moon_radius_input_text = ""
                    elif event.unicode.isnumeric() or event.unicode == '.':
                        self.moon_radius_input_text += event.unicode

                if self.show_custom_moon_gravity_input and self.selected_body and self.selected_body.get('type') == 'moon':
                    if event.key == pygame.K_RETURN:
                        gravity = self._parse_input_value(self.moon_gravity_input_text)
                        if gravity is not None and 0.001 <= gravity <= 100.0:  # Reasonable gravity range for moons
                            self.update_selected_body_property("gravity", gravity, "gravity")
                            self.moon_gravity_input_text = ""
                            self.show_custom_moon_gravity_input = False
                    elif event.key == pygame.K_BACKSPACE:
                        self.moon_gravity_input_text = self.moon_gravity_input_text[:-1]
                    elif event.key == pygame.K_ESCAPE:
                        self.show_custom_moon_gravity_input = False
                        self.moon_gravity_input_text = ""
                    elif event.unicode.isnumeric() or event.unicode == '.' or event.unicode.lower() == 'e' or event.unicode == '-' or event.unicode == '+':
                        # Allow numbers, decimal point, scientific notation (e, E), and signs
                        self.moon_gravity_input_text += event.unicode
                if self.show_custom_moon_orbital_distance_input and self.selected_body and self.selected_body.get('type') == 'moon':
                    if event.key == pygame.K_RETURN:
                        distance_km = self._parse_input_value(self.orbital_distance_input_text)
                        if distance_km is not None and 1000 <= distance_km <= 10000000:  # Reasonable orbital distance range for moons (km)
                            # Convert km to AU
                            AU_TO_KM = 149597870.7
                            distance_au = distance_km / AU_TO_KM
                            self.update_selected_body_property("semiMajorAxis", distance_au, "semiMajorAxis")
                            
                            body = self.get_selected_body()
                            if body:
                                # Visual orbit_radius and physics are recomputed automatically
                                self.recompute_orbit_parameters(body, force_recompute=True)
                                self.clear_orbit_points(body)
                                # Regenerate orbit grid with new radius
                                self.generate_orbit_grid(body)
                            self.orbital_distance_input_text = ""
                            self.show_custom_moon_orbital_distance_input = False
                    elif event.key == pygame.K_BACKSPACE:
                        self.orbital_distance_input_text = self.orbital_distance_input_text[:-1]
                    elif event.key == pygame.K_ESCAPE:
                        self.show_custom_moon_orbital_distance_input = False
                        self.orbital_distance_input_text = ""
                    elif event.unicode.isnumeric() or event.unicode == '.' or event.unicode.lower() == 'e' or event.unicode == '-' or event.unicode == '+':
                        # Allow numbers, decimal point, scientific notation (e, E), and signs
                        self.orbital_distance_input_text += event.unicode
                if self.show_custom_orbital_distance_input and self.selected_body and self.selected_body.get('type') == 'planet':
                    if event.key == pygame.K_RETURN:
                        dist = self._parse_input_value(self.orbital_distance_input_text)
                        if dist is not None and self.orbital_distance_min <= dist <= self.orbital_distance_max:
                            # CRITICAL: Use update_selected_body_property to ensure we update ONLY the selected body
                            self.update_selected_body_property("semiMajorAxis", dist, "semiMajorAxis")
                            # CRITICAL: Update position using centralized function
                            parent_star = next((b for b in self.placed_bodies if b["name"] == self.selected_body.get("parent")), None)
                            if parent_star is None and self.selected_body.get("parent_obj"):
                                parent_star = self.selected_body["parent_obj"]
                            if parent_star:
                                # Use centralized function to ensure position is derived from semiMajorAxis via visual scaling
                                self.compute_planet_position(self.selected_body, parent_star)
                                self.selected_body["position"][1] = parent_star["position"][1]
                            self.generate_orbit_grid(self.selected_body)
                            self.planet_orbital_distance_dropdown_selected = f"Custom ({dist:.2f} AU)"
                            self.orbital_distance_input_text = ""
                            self.orbital_distance_input_active = False
                            self.show_custom_orbital_distance_input = False
                    elif event.key == pygame.K_BACKSPACE:
                        self.orbital_distance_input_text = self.orbital_distance_input_text[:-1]
                    elif event.key == pygame.K_ESCAPE:
                        self.show_custom_orbital_distance_input = False
                        self.orbital_distance_input_text = ""
                    elif event.unicode.isnumeric() or event.unicode == '.':
                        self.orbital_distance_input_text += event.unicode
                if self.show_custom_orbital_eccentricity_input and self.selected_body and self.selected_body.get('type') == 'planet':
                    if event.key == pygame.K_RETURN:
                        try:
                            ecc = float(self.orbital_eccentricity_input_text)
                            if self.orbital_eccentricity_min <= ecc <= self.orbital_eccentricity_max:
                                # CRITICAL: Use centralized update to trigger recompute_orbit_parameters
                                self.update_selected_body_property("eccentricity", ecc, "eccentricity")
                                self.planet_orbital_eccentricity_dropdown_selected = f"Custom ({ecc:.3f})"
                            self.orbital_eccentricity_input_text = ""
                            self.orbital_eccentricity_input_active = False
                            self.show_custom_orbital_eccentricity_input = False
                        except ValueError:
                            pass
                    elif event.key == pygame.K_BACKSPACE:
                        self.orbital_eccentricity_input_text = self.orbital_eccentricity_input_text[:-1]
                    elif event.key == pygame.K_ESCAPE:
                        self.show_custom_orbital_eccentricity_input = False
                        self.orbital_eccentricity_input_text = ""
                    elif event.unicode.isnumeric() or event.unicode == '.':
                        self.orbital_eccentricity_input_text += event.unicode
                if self.show_custom_orbital_period_input and self.selected_body and self.selected_body.get('type') == 'planet':
                    if event.key == pygame.K_RETURN:
                        try:
                            period = float(self.orbital_period_input_text)
                            if self.orbital_period_min <= period <= self.orbital_period_max:
                                # CRITICAL: Use centralized update to trigger recompute_orbit_parameters
                                self.update_selected_body_property("orbital_period", period, "orbital_period")
                                self.planet_orbital_period_dropdown_selected = f"Custom ({period:.0f} days)"
                            self.orbital_period_input_text = ""
                            self.orbital_period_input_active = False
                            self.show_custom_orbital_period_input = False
                        except ValueError:
                            pass
                    elif event.key == pygame.K_BACKSPACE:
                        self.orbital_period_input_text = self.orbital_period_input_text[:-1]
                    elif event.key == pygame.K_ESCAPE:
                        self.show_custom_orbital_period_input = False
                        self.orbital_period_input_active = False
                        self.orbital_period_input_text = ""
                    elif event.unicode.isnumeric() or event.unicode == '.':
                        self.orbital_period_input_text += event.unicode
                
                if self.show_custom_moon_orbital_period_input and self.selected_body and self.selected_body.get('type') == 'moon':
                    if event.key == pygame.K_RETURN:
                        try:
                            period = float(self.orbital_period_input_text)
                            if self.orbital_period_min <= period <= self.orbital_period_max:
                                # CRITICAL: Use centralized update to trigger recompute_orbit_parameters
                                self.update_selected_body_property("orbital_period", period, "orbital_period")
                                self.moon_orbital_period_dropdown_selected = f"Custom ({period:.0f} days)"
                            self.orbital_period_input_text = ""
                            self.orbital_period_input_active = False
                            self.show_custom_moon_orbital_period_input = False
                        except ValueError:
                            pass
                    elif event.key == pygame.K_BACKSPACE:
                        self.orbital_period_input_text = self.orbital_period_input_text[:-1]
                    elif event.key == pygame.K_ESCAPE:
                        self.show_custom_moon_orbital_period_input = False
                        self.orbital_period_input_active = False
                        self.orbital_period_input_text = ""
                    elif event.unicode.isnumeric() or event.unicode == '.':
                        self.orbital_period_input_text += event.unicode
                if self.show_custom_stellar_flux_input and self.selected_body and self.selected_body.get('type') == 'planet':
                    if event.key == pygame.K_RETURN:
                        try:
                            flux = float(self.stellar_flux_input_text)
                            if self.stellar_flux_min <= flux <= self.stellar_flux_max:
                                # CRITICAL: Use update_selected_body_property for coupling/sync
                                self.update_selected_body_property("stellarFlux", flux, "stellarFlux")
                                self.planet_stellar_flux_dropdown_selected = f"Custom ({flux:.3f} EFU)"
                            self.stellar_flux_input_text = ""
                            self.stellar_flux_input_active = False
                            self.show_custom_stellar_flux_input = False
                        except ValueError:
                            pass
                    elif event.key == pygame.K_BACKSPACE:
                        self.stellar_flux_input_text = self.stellar_flux_input_text[:-1]
                    elif event.key == pygame.K_ESCAPE:
                        self.show_custom_stellar_flux_input = False
                        self.stellar_flux_input_text = ""
                    elif event.unicode.isnumeric() or event.unicode == '.':
                        self.stellar_flux_input_text += event.unicode
                if self.show_custom_planet_density_input and self.selected_body and self.selected_body.get('type') == 'planet':
                    if event.key == pygame.K_RETURN:
                        try:
                            density = float(self.planet_density_input_text)
                            if self.planet_density_min <= density <= self.planet_density_max:
                                # CRITICAL: Use update_selected_body_property to ensure we update ONLY the selected body
                                self.update_selected_body_property("density", density, "density")
                                self.planet_density_dropdown_selected = f"Custom ({density:.2f} g/cm³)"
                            self.planet_density_input_text = ""
                            self.show_custom_planet_density_input = False
                        except ValueError:
                            pass
                    elif event.key == pygame.K_BACKSPACE:
                        self.planet_density_input_text = self.planet_density_input_text[:-1]
                    elif event.key == pygame.K_ESCAPE:
                        self.show_custom_planet_density_input = False
                        self.planet_density_input_text = ""
                    elif event.unicode.isnumeric() or event.unicode == '.':
                        self.planet_density_input_text += event.unicode
            elif event.type == pygame.MOUSEMOTION:
                # DO NOT update preview_position here - it causes freezing!
                # Preview position is updated every frame in render loop using pygame.mouse.get_pos()
                # This ensures smooth, frame-accurate tracking regardless of event timing
                pass
        return True
    
    def clear_preview(self):
        """Clear placement preview - only call on explicit state changes"""
        self.preview_position = None
        self.preview_radius = None
        self.placement_mode_active = False
    
    def start_placement_mode(self, object_type):
        """Start placement mode for an object type"""
        self.placement_mode_active = True
        # Set preview radius once when placement mode begins
        if object_type == "planet":
            # For planets, radius is set when planet is selected from dropdown
            # This is handled separately in the dropdown selection code
            pass
        elif object_type == "star":
            # Default star size: 1.0 Solar radii = SUN_RADIUS_PX
            self.preview_radius = SUN_RADIUS_PX
        elif object_type == "moon":
            self.preview_radius = 4  # Default moon size (matches calculate_moon_visual_radius fallback)
    
    def update_ambient_colors(self):
        """Update the ambient colors for the title"""
        self.color_change_counter += 1
        if self.color_change_counter >= self.color_change_speed:
            self.color_change_counter = 0
            self.current_color_index = (self.current_color_index + 1) % len(self.ambient_colors)
    
    def clear_orbit_points(self, body):
        """Clear orbit points for a body when parameters change"""
        if "orbit_points" in body:
            body["orbit_points"].clear()
    
    def draw_orbit(self, body):
        """Draw orbit curve for a body using persistent orbit_points"""
        if not body.get("orbit_enabled", True):
            return
            
        if "orbit_points" not in body or len(body["orbit_points"]) < 2:
            return
        
        # For moons, orbit points are stored relative to planet
        # For planets, orbit points are stored in absolute world coordinates
        if body["type"] == "moon" and body.get("parent_obj") is not None:
            # Moon: convert relative points to world coordinates by adding planet position
            planet = body["parent_obj"]
            pts = [self.world_to_screen(planet["position"] + p) for p in body["orbit_points"]]
        else:
            # Planet: convert absolute world coordinates to screen coordinates
            pts = [self.world_to_screen(p) for p in body["orbit_points"]]
        
        # Choose color based on body type
        if body["type"] == "planet":
            color = self.GRAY
        else:  # moon
            color = (100, 100, 100)  # Slightly darker for moons
        
        # Draw the orbit line
        pygame.draw.lines(self.screen, color, False, pts, max(1, int(2 * self.camera_zoom)))
    
    def compute_planet_position(self, planet, parent_star):
        """
        MANDATORY: Centralized function to compute planet position from semiMajorAxis and eccentricity.
        Position is derived from:
        - parent_star.position (placed at one focus)
        - planet.orbit_angle (true anomaly θ)
        - semiMajorAxis (a)
        - eccentricity (e)
        
        Using the polar equation for an ellipse with focus at origin:
        r(θ) = a(1 - e²) / (1 + e*cos(θ))
        """
        if planet["type"] != "planet" or parent_star is None:
            return
        
        # Apply perceptual scaling to AU distance (a)
        semi_major_axis_au = planet.get("semiMajorAxis", 1.0)
        a = self.get_visual_orbit_radius(semi_major_axis_au)
        e = float(planet.get("eccentricity", 0.0))
        theta = float(planet.get("orbit_angle", 0.0))
        
        # Keplerian radial distance
        r = a * (1 - e**2) / (1 + e * math.cos(theta))
        
        # Compute position relative to parent star (focus)
        x = parent_star["position"][0] + r * math.cos(theta)
        y = parent_star["position"][1] + r * math.sin(theta)
        
        # Update position (derived, never stored independently)
        planet["position"] = np.array([x, y], dtype=float)
        
        # Update orbit_radius for backwards compatibility (this is the semi-major axis 'a' in our model)
        planet["orbit_radius"] = float(a)
    
    def compute_moon_position(self, moon, parent_planet):
        """
        MANDATORY: Centralized function to compute moon position from semiMajorAxis and eccentricity.
        Position is derived from:
        - parent_planet.position (placed at one focus)
        - moon.orbit_angle (true anomaly θ relative to planet)
        - semiMajorAxis (a)
        - eccentricity (e)
        
        r(θ) = a(1 - e²) / (1 + e*cos(θ))
        """
        if moon["type"] != "moon" or parent_planet is None:
            return
            
        # Get visual orbit radius (a)
        semi_major_axis_au = moon.get("semiMajorAxis", MOON_ORBIT_AU)
        a = self.get_visual_moon_orbit_radius(semi_major_axis_au)
        e = float(moon.get("eccentricity", 0.0))
        theta = float(moon.get("orbit_angle", 0.0))
        
        # Keplerian radial distance
        r = a * (1 - e**2) / (1 + e * math.cos(theta))
        
        # Compute position relative to parent planet
        x = parent_planet["position"][0] + r * math.cos(theta)
        y = parent_planet["position"][1] + r * math.sin(theta)
        
        # Update position
        moon["position"] = np.array([x, y], dtype=float)
        
        # Update orbit_radius for backwards compatibility
        moon["orbit_radius"] = float(a)

    def _apply_kepler_coupling(self, body, changed_field):
        """
        Enforce Kepler's Third Law: P² = a³ / M
        Couples orbital distance (AU) and orbital period (days) based on parent mass.
        """
        if body["type"] == "star":
            return

        # 1. Update driver status if the change came from UI
        if changed_field == "semiMajorAxis":
            body["orbital_driver"] = "semi_major_axis"
        elif changed_field == "orbital_period":
            body["orbital_driver"] = "orbital_period"
        
        driver = body.get("orbital_driver", "semi_major_axis")
        
        # 2. Get parent mass in Solar masses
        parent = body.get("parent_obj")
        if not parent and body.get("parent"):
             parent = next((b for b in self.placed_bodies if b["name"] == body["parent"]), None)
        
        if not parent:
            return

        # Internal star mass is in Solar masses; internal planet/moon mass is in Earth masses
        if parent["type"] == "star":
            M_total_solar = float(parent.get("mass", 1.0))
        else:
            # Internal mass is in Earth masses (Sun = 333,000)
            parent_mass_earth = float(parent.get("mass", 1.0))
            M_total_solar = parent_mass_earth / 333000.0
        
        if M_total_solar <= 0:
            return

        # 3. Apply coupling based on driver
        # If parent_mass changed, we update the non-driver field
        try:
            is_sun = body["type"] == "planet" and abs(M_total_solar - 1.0) < 1e-3
            is_parent_planet = parent["type"] == "planet"
            
            if driver == "semi_major_axis":
                if changed_field == "parent_mass" or changed_field == "semiMajorAxis":
                    # Update Period from AU: P (days) = 365.25 * sqrt(a³ / M_solar)
                    a = float(body.get("semiMajorAxis", 1.0))
                    
                    # DIRECT PRESET MATCHING: If AU matches a preset exactly, use that preset's period
                    # This ensures selection of "Mercury" AU results in exactly "Mercury" Period label
                    found_preset_p = None
                    if is_sun:
                        for p_name, p_data in SOLAR_SYSTEM_PLANET_PRESETS.items():
                            if abs(a - p_data["semiMajorAxis"]) < 1e-4:
                                found_preset_p = p_data["orbital_period"]
                                break
                    
                    if found_preset_p is not None:
                        new_period = float(found_preset_p)
                    else:
                        p_years = math.sqrt(max(0, a**3) / M_total_solar)
                        new_period = float(p_years * 365.25)
                        
                        # SNAP TO PRESET: If very close to a preset, snap anyway (handles small floating point drift)
                        if is_sun:
                            for name, preset_p in self.planet_orbital_period_dropdown_options:
                                if preset_p is not None and abs(new_period - preset_p) < 2.0:
                                    new_period = float(preset_p)
                                    break
                        elif is_parent_planet:
                            for name, preset_p in self.moon_orbital_period_dropdown_options:
                                if preset_p is not None and abs(new_period - preset_p) < 0.2:
                                    new_period = float(preset_p)
                                    break
                                
                    body["orbital_period"] = new_period
            else: # driver == "orbital_period"
                if changed_field == "parent_mass" or changed_field == "orbital_period":
                    # Update AU from Period: a = (M_solar * P_years²)^(1/3)
                    p_days = float(body.get("orbital_period", 365.25))
                    
                    # DIRECT PRESET MATCHING:
                    found_preset_au = None
                    if is_sun:
                        for p_name, p_data in SOLAR_SYSTEM_PLANET_PRESETS.items():
                            if abs(p_days - p_data["orbital_period"]) < 1e-3:
                                found_preset_au = p_data["semiMajorAxis"]
                                break
                    
                    if found_preset_au is not None:
                        new_au = float(found_preset_au)
                    else:
                        p_years = p_days / 365.25
                        new_au = (M_total_solar * (p_years**2))**(1/3)
                        
                        # SNAP TO PRESET:
                        if is_sun:
                            for name, preset_au in self.planet_orbital_distance_dropdown_options:
                                if preset_au is not None and abs(new_au - preset_au) < 0.02:
                                    new_au = float(preset_au)
                                    break
                        elif is_parent_planet:
                            KM_TO_AU = 6.68459e-9
                            for name, preset_km in self.moon_orbital_distance_dropdown_options:
                                if preset_km is not None:
                                    preset_au = preset_km * KM_TO_AU
                                    if abs(new_au - preset_au) < (10000 * KM_TO_AU): # 10000km tolerance
                                        new_au = float(preset_au)
                                        break
                                    
                    body["semiMajorAxis"] = float(new_au)
        except (ValueError, ZeroDivisionError, OverflowError):
            pass
            
        # Ensure stellar flux is updated whenever orbital parameters change
        self._update_stellar_flux(body)

    def _start_orbital_correction(self, body, new_au):
        """Helper to trigger the smooth orbital correction animation when AU changes."""
        parent_obj = body.get("parent_obj")
        if parent_obj is None and body.get("parent"):
            parent_obj = next((b for b in self.placed_bodies if b["name"] == body.get("parent")), None)
        
        if parent_obj:
            # Calculate new target position based on updated AU
            if body["type"] == "planet":
                new_orbit_radius_px = self.get_visual_orbit_radius(new_au)
            elif body["type"] == "moon":
                new_orbit_radius_px = self.get_visual_moon_orbit_radius(new_au)
            else:
                new_orbit_radius_px = new_au * AU_TO_PX
            
            current_angle = body.get("orbit_angle", 0.0)
            new_target_pos = np.array([
                parent_obj["position"][0] + new_orbit_radius_px * math.cos(current_angle),
                parent_obj["position"][1] + new_orbit_radius_px * math.sin(current_angle)
            ], dtype=float)
            
            # Start orbital correction animation
            body_id = body.get("id")
            if body_id:
                body["is_correcting_orbit"] = True
                body["target_position"] = new_target_pos.copy()
                body["orbit_radius_au"] = float(new_au)
                self.orbital_corrections[body_id] = {
                    "target_radius_px": new_orbit_radius_px,
                    "start_time": time.time(),
                    "duration": self.correction_animation_duration,
                    "start_pos": body["position"].copy(),
                    "target_pos": new_target_pos.copy(),
                    "parent_star_pos": parent_obj["position"].copy(),
                    "position_history": [body["position"].copy()]  # Track position history for trail
                }
            
            # Regenerate orbit grid for visualization
            if body["type"] != "star":
                self.generate_orbit_grid(body)
                self.clear_orbit_points(body)

    def recompute_orbit_parameters(self, body, force_recompute=False):
        """
        Recompute orbital parameters for a single body based on its current mass and parent.
        This ensures each body has independent physics calculations.
        
        Args:
            body: The body dict to recompute parameters for
            force_recompute: If True, always recalculate even if parameters exist
        """
        if body["type"] == "star":
            return
        
        # Find parent body
        parent = body.get("parent_obj")
        if parent is None:
            if body.get("parent"):
                parent = next((b for b in self.placed_bodies if b["name"] == body["parent"]), None)
            if parent is None:
                if body["type"] == "planet":
                    stars = [b for b in self.placed_bodies if b["type"] == "star"]
                    if stars:
                        parent = min(stars, key=lambda s: np.linalg.norm(s["position"] - body["position"]))
                elif body["type"] == "moon":
                    planets = [b for b in self.placed_bodies if b["type"] == "planet"]
                    if planets:
                        parent = min(planets, key=lambda p: np.linalg.norm(p["position"] - body["position"]))
        
        if not parent:
            return
        
        # Set permanent parent reference
        body["parent_obj"] = parent
        body["parent"] = parent["name"]
        
        # CRITICAL: For planets, orbit_radius MUST be derived using visual scaling
        # Position is derived, never stored. AU is the single source of truth.
        if body["type"] == "planet":
            # For planets: orbit_radius uses perceptual sqrt scaling
            semi_major_axis = body.get("semiMajorAxis", 1.0)
            orbit_radius = self.get_visual_orbit_radius(semi_major_axis)
            body["orbit_radius"] = float(orbit_radius)
            
            # Ensure eccentricity is set (default to Earth-like 0.0167 or 0.0)
            if "eccentricity" not in body:
                body["eccentricity"] = 0.0
            
            # Initialize orbit_angle if not set
            if body.get("orbit_angle", 0.0) == 0.0:
                body["orbit_angle"] = float(random.uniform(0, 2 * np.pi))
                print(f"ORBIT_ANGLE_INIT body_id={body.get('id', 'unknown')} name={body.get('name', 'unknown')} "
                      f"random_angle={body['orbit_angle']:.6f}")
            
            # CRITICAL: Update position immediately using centralized function
            self.compute_planet_position(body, parent)
        else:
            # For moons: calculate orbit radius from semiMajorAxis using visual scaling
            # Fallback to current distance if semiMajorAxis not set
            if body.get("semiMajorAxis"):
                orbit_radius = self.get_visual_moon_orbit_radius(body["semiMajorAxis"])
            else:
                if force_recompute or body.get("orbit_radius", 0.0) == 0.0:
                    orbit_radius = np.linalg.norm(parent["position"] - body["position"])
                else:
                    orbit_radius = body.get("orbit_radius", 0.0)
            
            body["orbit_radius"] = float(orbit_radius)
            
            # Ensure eccentricity is set for moons
            if "eccentricity" not in body:
                body["eccentricity"] = 0.0
            
            # CRITICAL: Initialize orbit_angle randomly to ensure independent orbital phase
            if body.get("orbit_angle", 0.0) == 0.0:
                body["orbit_angle"] = float(random.uniform(0, 2 * np.pi))
                print(f"ORBIT_ANGLE_INIT body_id={body.get('id', 'unknown')} name={body.get('name', 'unknown')} "
                      f"random_angle={body['orbit_angle']:.6f}")
            
            # CRITICAL: Update position immediately using centralized function
            self.compute_moon_position(body, parent)
        
        # CRITICAL: Recalculate orbital speed using BOTH parent mass AND body mass
        # Standard gravitational parameter: μ = G * (M_parent + M_body)
        # Angular speed: ω = sqrt(μ / r^3)
        if orbit_radius > 0:
            # Internal star mass is in Solar masses; internal planet/moon mass is in Earth masses
            if parent["type"] == "star":
                parent_mass_solar = float(parent.get("mass", 1.0))
                body_mass_solar = float(body.get("mass", 0.0)) / 333000.0
            else:
                parent_mass_solar = float(parent.get("mass", 1.0)) / 333000.0
                body_mass_solar = float(body.get("mass", 0.0)) / 333000.0
            
            # Combine masses in consistent units (Solar masses)
            mu_solar = self.G * (parent_mass_solar + body_mass_solar)
            mu = mu_solar # Keeping the variable name for dict storage
            
            # For speed calculation, use the ACTUAL physical radius (linear)
            # This ensures orbital periods remain scientifically correct for the given AU
            if body["type"] == "planet":
                physics_radius = body.get("semiMajorAxis", 1.0) * AU_TO_PX
            elif body["type"] == "moon" and body.get("semiMajorAxis"):
                # Use a specific physics scale for moons to keep their orbital speeds stable
                # Maps MOON_ORBIT_AU to MOON_ORBIT_PX (~40 units) for tuned kinematic physics
                MOON_PHYSICS_SCALE = MOON_ORBIT_PX / MOON_ORBIT_AU
                physics_radius = body["semiMajorAxis"] * MOON_PHYSICS_SCALE
            else:
                physics_radius = orbit_radius # Fallback for old bodies
                
            # CRITICAL: Calculate orbital speed based on user-defined orbital_period if available.
            # This allows users to change year length without affecting orbit size.
            # We use the natural physical speed (rad/year) derived from the period.
            
            # P is in days, convert to simulator years (1 year = 365.25 days)
            # Omega (rad/year) = 2*pi / (P / 365.25)
            default_period = 27.3 if body["type"] == "moon" else 365.25
            orbital_period_days = body.get("orbital_period", default_period)
            if orbital_period_days > 0:
                omega_real = (2 * np.pi * 365.25) / orbital_period_days
            else:
                omega_real = 0
            
            if body["type"] == "moon":
                # Moons already have short periods (e.g. 27.3 days) which result
                # in high natural omega. We no longer apply an extra speed factor
                # to avoid teleportation/aliasing at high simulation time steps.
                body["orbit_speed"] = float(omega_real)
            else:
                # Planets move at their natural Keplerian speed (2*pi/P)
                body["orbit_speed"] = float(omega_real)
            
            # Store mu per-body for reference (though no longer the primary driver of speed)
            body["mu"] = float(mu)
            
            # Recalculate velocity for circular orbit (relative to parent's frame)
            # This is used for other physics calculations if needed
            v = body["orbit_speed"] * orbit_radius
            body["velocity"] = np.array([-v * np.sin(body["orbit_angle"]), v * np.cos(body["orbit_angle"])], dtype=float).copy()
        else:
            body["orbit_speed"] = 0.0
            body["mu"] = 0.0
        
        # Verification print: show per-body orbit parameters
        print(f"PHYSICS_RECOMPUTE body_id={body.get('id', 'unknown')} name={body.get('name', 'unknown')} "
              f"mass={body.get('mass', 0.0):.6f} orbit_speed={body.get('orbit_speed', 0.0):.6f} "
              f"mu={body.get('mu', 0.0):.6f} parent_mass={parent.get('mass', 0.0):.6f}")
    
    def generate_orbit_grid(self, body):
        """Generate a circular grid for the orbit path"""
        if body["type"] == "star":
            return
        
        # Find parent body
        parent = None
        if body["type"] == "planet":
            # Find nearest star
            stars = [b for b in self.placed_bodies if b["type"] == "star"]
            if stars:
                parent = min(stars, key=lambda s: np.linalg.norm(s["position"] - body["position"]))
        elif body["type"] == "moon":
            # Find nearest planet
            planets = [b for b in self.placed_bodies if b["type"] == "planet"]
            if planets:
                parent = min(planets, key=lambda p: np.linalg.norm(p["position"] - body["position"]))
                
        if parent:
            # Set permanent parent reference
            body["parent_obj"] = parent
            
            # Check if orbital parameters are already set (e.g., for newly placed moons)
            # If orbit_radius and orbit_speed are already set, preserve them
            # For moons, we check if parameters are set regardless of parent match (parent might be set after)
            has_orbit_radius = body.get("orbit_radius", 0.0) > 0.0
            has_orbit_speed = body.get("orbit_speed", 0.0) > 0.0
            has_parent_match = body.get("parent") == parent["name"] if body.get("parent") else False
            
            # Preserve parameters if they're set AND (parent matches OR parent not set yet)
            preserve_params = has_orbit_radius and has_orbit_speed and (has_parent_match or not body.get("parent"))
            
            # Debug output for moons
            if body["type"] == "moon":
                print(f"DEBUG generate_orbit_grid for moon {body.get('name', 'unknown')}:")
                print(f"  has_orbit_radius={has_orbit_radius}, has_orbit_speed={has_orbit_speed}")
                print(f"  body parent={body.get('parent')}, parent name={parent['name']}, has_parent_match={has_parent_match}")
                print(f"  preserve_params={preserve_params}")
            
            if not preserve_params:
                # Use recompute_orbit_parameters to ensure per-body physics
                self.recompute_orbit_parameters(body, force_recompute=True)
                
                # Clear orbit points when orbital parameters change
                self.clear_orbit_points(body)
            else:
                # Parameters already set, just ensure parent is set
                body["parent"] = parent["name"]
            
            # CRITICAL: For planets and moons, orbit_radius is derived from semiMajorAxis via visual scaling
            if body["type"] == "planet":
                orbit_radius = self.get_visual_orbit_radius(body.get("semiMajorAxis", 1.0))
            elif body["type"] == "moon" and body.get("semiMajorAxis"):
                orbit_radius = self.get_visual_moon_orbit_radius(body["semiMajorAxis"])
            else:
                orbit_radius = body.get("orbit_radius", 0.0)
            
            # Always generate grid points for visualization
            grid_points = []
            e = float(body.get("eccentricity", 0.0))
            a = orbit_radius
            
            # Generate 200 points for a smooth ellipse
            for i in range(200):
                theta = i * 2 * np.pi / 200
                # Keplerian radial distance: r = a(1 - e²) / (1 + e*cosθ)
                r = a * (1 - e**2) / (1 + e * np.cos(theta))
                
                x = parent["position"][0] + r * np.cos(theta)
                y = parent["position"][1] + r * np.sin(theta)
                grid_points.append(np.array([x, y]))
            
            # Store grid points for visualization
            self.orbit_grid_points[body["name"]] = grid_points
            
            # CRITICAL: Clear screen cache to force redraw of the new elliptical grid
            if body["name"] in self.orbit_grid_screen_cache:
                del self.orbit_grid_screen_cache[body["name"]]
    
    def initialize_all_orbits(self):
        """Initialize orbital relationships and velocities for all bodies when simulation starts"""
        for body in self.placed_bodies:
            if body["type"] == "planet" and not body["parent"]:
                # Find nearest star
                stars = [b for b in self.placed_bodies if b["type"] == "star"]
                if stars:
                    nearest_star = min(stars, key=lambda s: np.linalg.norm(s["position"] - body["position"]))
                    body["parent"] = nearest_star["name"]
                    self.generate_orbit_grid(body)
            elif body["type"] == "moon":
                # For moons, check if they already have orbital parameters set
                if not body["parent"]:
                    # Find nearest planet
                    planets = [b for b in self.placed_bodies if b["type"] == "planet"]
                    if planets:
                        nearest_planet = min(planets, key=lambda p: np.linalg.norm(p["position"] - body["position"]))
                        # Only set up orbit if parameters aren't already set
                        if body.get("orbit_speed", 0.0) == 0.0 or body.get("orbit_radius", 0.0) == 0.0:
                            body["parent"] = nearest_planet["name"]
                            self.generate_orbit_grid(body)
                        else:
                            # Parameters already set, just set parent
                            body["parent"] = nearest_planet["name"]
                elif body["parent"]:
                    # Moon has parent, check if orbital parameters need initialization
                    parent = next((b for b in self.placed_bodies if b["name"] == body["parent"]), None)
                    if parent and (body.get("orbit_speed", 0.0) == 0.0 or body.get("orbit_radius", 0.0) == 0.0):
                        # Only regenerate if parameters aren't set
                        self.generate_orbit_grid(body)
            elif body["type"] != "star" and body["parent"]:
                # Ensure bodies with parents have orbital velocities initialized
                parent = next((b for b in self.placed_bodies if b["name"] == body["parent"]), None)
                if parent and (body.get("orbit_speed", 0.0) == 0.0 or body.get("orbit_radius", 0.0) == 0.0):
                    self.generate_orbit_grid(body)
    
    def update_physics(self):
        """Update positions and velocities of all bodies using global simulation clock"""
        # Update camera focus interpolation
        self.update_camera()
        
        trace("BEGIN_FRAME")
        
        # Determine elapsed real time since last frame
        dt_real_ms = self.clock.tick(60)
        dt_real = dt_real_ms / 1000.0
        
        if self.paused or self.time_scale <= 0.0:
            dt_sim = 0.0
        else:
            # dt_sim = dt_real * time_scale
            # Here time_scale is explicit simulated_seconds per real_second
            dt_sim = dt_real * self.time_scale
            
            # Advance global simulation clock
            self.simulation_time_seconds += dt_sim
            
        # effective_dt in simulation years for Keplerian motion
        # (orbit_speed is in rad/year internally for period calculations)
        effective_dt = dt_sim / self.SECONDS_PER_YEAR
        
        # Update orbital correction animations
        current_time = time.time()
        bodies_to_remove_from_correction = []
        
        for body_id, correction_data in list(self.orbital_corrections.items()):
            body = self.bodies_by_id.get(body_id)
            if not body:
                bodies_to_remove_from_correction.append(body_id)
                continue
            
            elapsed = current_time - correction_data["start_time"]
            progress = min(elapsed / correction_data["duration"], 1.0)
            
            # Smooth easing function (ease-out cubic)
            eased_progress = 1.0 - (1.0 - progress) ** 3
            
            if progress < 1.0:
                # Still animating - interpolate position
                start_pos = correction_data["start_pos"]
                target_pos = correction_data["target_pos"]
                
                # Interpolate position
                current_pos = start_pos + (target_pos - start_pos) * eased_progress
                body["position"] = np.array(current_pos, dtype=float)
                
                # Update visual position for rendering
                body["visual_position"] = np.array(current_pos, dtype=float)
                
                # Track position history for trail (keep last ~30 positions)
                if "position_history" not in correction_data:
                    correction_data["position_history"] = [start_pos.copy()]
                
                # Add current position to history
                position_history = correction_data["position_history"]
                position_history.append(current_pos.copy())
                # Keep only last 30 positions for smooth trail
                if len(position_history) > 30:
                    position_history.pop(0)
                
                # Update placement trajectory line (Step 1)
                if body.get("is_placing"):
                    dx = body["position"][0] - target_pos[0]
                    dy = body["position"][1] - target_pos[1]
                    distance = math.hypot(dx, dy)
                    
                    if distance < self.ARRIVAL_EPSILON:
                        body["is_placing"] = False
                        if body.get("placement_line"):
                            body["placement_line"]["fade_start_time"] = current_time
                
                # CRITICAL: Prevent persistent orbit point accumulation during placement animation
                # This stops a second "straight line" from appearing behind the planet
                if body.get("is_correcting_orbit") and "orbit_points" in body:
                    body["orbit_points"].clear()
            else:
                # Animation complete - snap to final position and compute from physics
                body["position"] = np.array(correction_data["target_pos"], dtype=float)
                body["visual_position"] = np.array(correction_data["target_pos"], dtype=float)
                body["is_correcting_orbit"] = False
                
                # Clear orbit points accumulated during animation to prevent a "straight line" trail
                if "orbit_points" in body:
                    body["orbit_points"].clear()
                
                # Clear position history when animation completes to prevent lingering trails
                if "position_history" in correction_data:
                    correction_data["position_history"].clear()
                
                # After animation, ensure position is computed from physics (AU)
                if body["type"] == "planet":
                    parent_star = body.get("parent_obj")
                    if parent_star:
                        self.compute_planet_position(body, parent_star)
                        body["visual_position"] = body["position"].copy()
                elif body["type"] == "moon":
                    parent_planet = body.get("parent_obj")
                    if parent_planet:
                        self.compute_moon_position(body, parent_planet)
                        body["visual_position"] = body["position"].copy()
                
                bodies_to_remove_from_correction.append(body_id)
        
        # Remove completed corrections
        for body_id in bodies_to_remove_from_correction:
            self.orbital_corrections.pop(body_id, None)
        
        # First, establish parent-child relationships if not already set
        for body in self.placed_bodies:
            if body["type"] == "planet" and (not body.get("parent") or body["parent"] is None):
                # Find nearest star
                stars = [b for b in self.placed_bodies if b["type"] == "star"]
                if stars:
                    nearest_star = min(stars, key=lambda s: np.linalg.norm(s["position"] - body["position"]))
                    body["parent"] = nearest_star["name"]
                    self.generate_orbit_grid(body)
            elif body["type"] == "moon" and (not body.get("parent") or body["parent"] is None):
                # Find nearest planet
                planets = [b for b in self.placed_bodies if b["type"] == "planet"]
                if planets:
                    nearest_planet = min(planets, key=lambda p: np.linalg.norm(p["position"] - body["position"]))
                    # Only set up orbit if parameters aren't already set
                    if body.get("orbit_speed", 0.0) == 0.0 or body.get("orbit_radius", 0.0) == 0.0:
                        body["parent"] = nearest_planet["name"]
                        self.generate_orbit_grid(body)
                    else:
                        # Parameters already set, just set parent
                        body["parent"] = nearest_planet["name"]
            
            # Ensure bodies with parents have orbital velocities initialized
            if body["type"] != "star" and body.get("parent") is not None:
                parent = next((b for b in self.placed_bodies if b["name"] == body["parent"]), None)
                if parent and (body.get("orbit_speed", 0.0) == 0.0 or body.get("orbit_radius", 0.0) == 0.0):
                    # Regenerate orbit grid if orbital parameters aren't set
                    self.generate_orbit_grid(body)

        # Update positions and velocities
        # IMPORTANT: Enforce deterministic update order: stars -> planets -> moons
        # Separate bodies by type to ensure correct update order
        stars = [b for b in self.placed_bodies if b["type"] == "star"]
        planets = [b for b in self.placed_bodies if b["type"] == "planet"]
        moons = [b for b in self.placed_bodies if b["type"] == "moon"]
        
        # Create ordered list for deterministic processing
        ordered_bodies = stars + planets + moons
        
        # Process planets first (they orbit stars)
        for body in planets:
            # Get parent using parent_obj reference (faster and more reliable)
            parent = body.get("parent_obj")
            if parent is None:
                # Fallback to name-based lookup if parent_obj not set
                if body.get("parent") is not None:
                    parent_candidate = next((b for b in self.placed_bodies if b["name"] == body["parent"]), None)
                    # Validate parent is a star
                    if parent_candidate and parent_candidate["type"] == "star":
                        parent = parent_candidate
                        body["parent_obj"] = parent
                    else:
                        # Invalid parent, clear it
                        body["parent"] = None
            
            # If no valid parent found, find nearest star
            if not parent:
                stars_list = [b for b in self.placed_bodies if b["type"] == "star"]
                if stars_list:
                    parent = min(stars_list, key=lambda s: np.linalg.norm(s["position"] - body["position"]))
                    body["parent"] = parent["name"]
                    body["parent_obj"] = parent
                    self.generate_orbit_grid(body)
            
            if parent and parent["type"] == "star":
                # CRITICAL: Ensure parent_obj is always set when we have a valid parent
                if body.get("parent_obj") is None or body["parent_obj"] != parent:
                    body["parent_obj"] = parent
                
                # CRITICAL: For planets and moons, orbit_radius is derived from semiMajorAxis via visual scaling
                if body["type"] == "planet":
                    orbit_radius = self.get_visual_orbit_radius(body.get("semiMajorAxis", 1.0))
                    body["orbit_radius"] = float(orbit_radius)
                elif body["type"] == "moon" and body.get("semiMajorAxis"):
                    orbit_radius = self.get_visual_moon_orbit_radius(body["semiMajorAxis"])
                    body["orbit_radius"] = float(orbit_radius)
                else:
                    orbit_radius = body.get("orbit_radius", 0.0)
                
                orbit_speed = body.get("orbit_speed", 0.0)
                
                if orbit_speed == 0.0 or orbit_radius == 0.0:
                    # Use recompute_orbit_parameters to ensure per-body physics calculation
                    self.recompute_orbit_parameters(body, force_recompute=True)
                    self.generate_orbit_grid(body)
                    # Re-get values after regeneration
                    if body["type"] == "planet":
                        orbit_radius = self.get_visual_orbit_radius(body.get("semiMajorAxis", 1.0))
                    else:
                        orbit_radius = body.get("orbit_radius", 0.0)
                    orbit_speed = body.get("orbit_speed", 0.0)
                
                # Get parent_obj reference (guaranteed to be set above)
                p = body["parent_obj"]
                if p is None:
                    p = parent  # Fallback to validated parent
                    body["parent_obj"] = p
                
                # PURE KINEMATIC KEPLERIAN ORBIT - follows Kepler's 2nd law
                # Update orbit angle (true anomaly) using variable angular speed
                if effective_dt > 0.0 and orbit_speed != 0.0 and not np.isnan(orbit_speed):
                    old_angle = body["orbit_angle"]
                    e = float(body.get("eccentricity", 0.0))
                    
                    # Kepler's 2nd law: dθ/dt = n * (1 + e*cosθ)² / (1 - e²)^(3/2)
                    # where n is the mean motion (orbit_speed)
                    if e > 0:
                        # Limit e to 0.999 to avoid division by zero/extreme speeds
                        e_safe = min(e, 0.999)
                        speed_factor = ((1 + e_safe * math.cos(old_angle))**2) / ((1 - e_safe**2)**1.5)
                        angular_delta = orbit_speed * speed_factor * effective_dt
                    else:
                        angular_delta = orbit_speed * effective_dt
                        
                    body["orbit_angle"] += angular_delta
                    trace(f"ORBIT_ANGLE_UPDATE {body['name']} old={old_angle:.6f} new={body['orbit_angle']:.6f} source=update_physics_planet")
                    # Keep orbit angle in [0, 2π) range for clean circular orbits
                    while body["orbit_angle"] >= 2 * np.pi:
                        body["orbit_angle"] -= 2 * np.pi
                    while body["orbit_angle"] < 0:
                        body["orbit_angle"] += 2 * np.pi
                    
                    # Verification print: show per-body physics parameters including orbit_angle
                    if DEBUG_ORBIT:
                        print(f"PHYSICS_UPDATE body_id={body.get('id', 'unknown')} name={body.get('name', 'unknown')} "
                              f"mass={body.get('mass', 0.0):.6f} orbit_speed={body.get('orbit_speed', 0.0):.6f} "
                              f"mu={body.get('mu', 0.0):.6f} orbit_radius={body.get('orbit_radius', 0.0):.6f} "
                              f"orbit_angle={body.get('orbit_angle', 0.0):.6f}")
                
                # CRITICAL: For planets, position MUST be computed from semiMajorAxis via visual scaling
                # Use centralized function to ensure AU is single source of truth
                # BUT: Skip position update if orbital correction animation is active
                if p is not None:
                    # Only update position from physics if not correcting orbit
                    if not body.get("is_correcting_orbit", False):
                        trace(f"PRE_WRITE {body['name']} pos={body['position'].copy()} source=update_physics_planet parent_pos={p['position'].copy()}")
                        self.compute_planet_position(body, p)
                        trace(f"POST_WRITE {body['name']} pos={body['position'].copy()} source=update_physics_planet")
                        # Update visual_position to match position when not animating
                        if "visual_position" in body:
                            body["visual_position"] = body["position"].copy()
                    orbit_radius = body.get("orbit_radius", 0.0)  # Update orbit_radius from computed value
                
                # Skip remaining updates if paused
                if effective_dt == 0.0:
                    continue
                
                # Update velocity for circular orbit (relative to parent's frame)
                v = body["orbit_speed"] * body["orbit_radius"]
                body["velocity"] = np.array([-v * np.sin(body["orbit_angle"]), v * np.cos(body["orbit_angle"])]).copy()  # Ensure independent copy
                
                # Update rotation angle
                body["rotation_angle"] += body["rotation_speed"] * effective_dt
                if body["rotation_angle"] >= 2 * np.pi:
                    body["rotation_angle"] -= 2 * np.pi
                
                # Store orbit points only when time progresses and orbit is enabled
                if effective_dt > 0.0 and body.get("orbit_enabled", True):
                    if "orbit_points" not in body:
                        body["orbit_points"] = []
                    if "max_orbit_points" not in body:
                        body["max_orbit_points"] = 2000
                    
                    body["orbit_points"].append(body["position"].copy())
                    
                    # Trim to max_orbit_points if exceeded
                    if len(body["orbit_points"]) > body["max_orbit_points"]:
                        body["orbit_points"].pop(0)
        
        # Process moons AFTER planets (so they use updated planet positions)
        for body in moons:
            # Get parent using parent_obj reference (faster and more reliable)
            parent = body.get("parent_obj")
            # Instrumentation: Parent lookup
            trace(f"PARENT_LOOKUP {body['name']} parent_name={body.get('parent')} parent_obj_exists={'parent_obj' in body}")
            if parent is None:
                # Fallback to name-based lookup if parent_obj not set
                if body.get("parent") is not None:
                    parent_candidate = next((b for b in self.placed_bodies if b["name"] == body["parent"]), None)
                    # Validate parent is a planet (moons cannot orbit stars directly)
                    if parent_candidate and parent_candidate["type"] == "planet":
                        parent = parent_candidate
                        body["parent_obj"] = parent
                    else:
                        # Invalid parent, clear it
                        body["parent"] = None
            
            # If no valid parent found, find nearest planet
            if not parent:
                planets_list = [b for b in self.placed_bodies if b["type"] == "planet"]
                if planets_list:
                    parent = min(planets_list, key=lambda p: np.linalg.norm(p["position"] - body["position"]))
                    # Only set up orbit if parameters aren't already set
                    if body.get("orbit_speed", 0.0) == 0.0 or body.get("orbit_radius", 0.0) == 0.0:
                        body["parent"] = parent["name"]
                        body["parent_obj"] = parent
                        self.generate_orbit_grid(body)
                    else:
                        # Parameters already set, just set parent
                        body["parent"] = parent["name"]
                        body["parent_obj"] = parent
            
            if parent and parent["type"] == "planet":
                # CRITICAL: Ensure parent_obj is always set when we have a valid parent
                if body.get("parent_obj") is None or body["parent_obj"] != parent:
                    body["parent_obj"] = parent
                
                # Ensure orbit_speed is set and non-zero
                orbit_radius = body.get("orbit_radius", 0.0)
                orbit_speed = body.get("orbit_speed", 0.0)
                
                if orbit_speed == 0.0 or orbit_radius == 0.0:
                    # Use recompute_orbit_parameters to ensure per-body physics calculation
                    self.recompute_orbit_parameters(body, force_recompute=True)
                    self.generate_orbit_grid(body)
                    # Re-get values after regeneration
                    orbit_radius = body.get("orbit_radius", 0.0)
                    orbit_speed = body.get("orbit_speed", 0.0)
                
                # Update orbit grid points for moons to follow their parent planet
                # Always update/create the orbit grid for moons so the orbit line is visible
                # Update EVERY FRAME (even when paused) so the orbit line follows the planet
                if body["type"] == "moon" and parent:
                    self.update_moon_orbit_grid(body, parent)
                    # Clear cache to force redraw with new planet position
                    if body["name"] in self.orbit_grid_screen_cache:
                        del self.orbit_grid_screen_cache[body["name"]]
                
                # Get parent_obj reference (guaranteed to be set above)
                p = body["parent_obj"]
                if p is None:
                    p = parent  # Fallback to validated parent
                    body["parent_obj"] = p
                
                # CRITICAL: Moon orbit update - MUST happen every frame
                # Step 1: Update orbit angle (true anomaly) using variable angular speed
                if effective_dt > 0.0 and orbit_speed != 0.0 and not np.isnan(orbit_speed):
                    old_angle = body["orbit_angle"]
                    e = float(body.get("eccentricity", 0.0))
                    
                    # Kepler's 2nd law: dθ/dt = n * (1 + e*cosθ)² / (1 - e²)^(3/2)
                    if e > 0:
                        e_safe = min(e, 0.999)
                        speed_factor = ((1 + e_safe * math.cos(old_angle))**2) / ((1 - e_safe**2)**1.5)
                        angular_delta = orbit_speed * speed_factor * effective_dt
                    else:
                        angular_delta = orbit_speed * effective_dt
                        
                    body["orbit_angle"] += angular_delta
                    trace(f"ORBIT_ANGLE_UPDATE {body['name']} old={old_angle:.6f} new={body['orbit_angle']:.6f} source=update_physics_moon")
                    # Keep orbit angle in [0, 2π) range for clean circular orbits
                    while body["orbit_angle"] >= 2 * np.pi:
                        body["orbit_angle"] -= 2 * np.pi
                    while body["orbit_angle"] < 0:
                        body["orbit_angle"] += 2 * np.pi
                    
                    # Verification print: show per-body physics parameters including orbit_angle
                    if DEBUG_ORBIT:
                        print(f"PHYSICS_UPDATE body_id={body.get('id', 'unknown')} name={body.get('name', 'unknown')} "
                              f"mass={body.get('mass', 0.0):.6f} orbit_speed={body.get('orbit_speed', 0.0):.6f} "
                              f"mu={body.get('mu', 0.0):.6f} orbit_radius={body.get('orbit_radius', 0.0):.6f} "
                              f"orbit_angle={body.get('orbit_angle', 0.0):.6f}")
                
                # Step 2: ALWAYS recalculate position from parent EVERY FRAME (even when paused)
                # Formula: moon.position = planet.position + [r * cos(angle), r * sin(angle)]
                # This ensures moon stays locked to planet's current position
                # CRITICAL: This MUST happen every frame, BEFORE any rendering
                # BUT: Skip position update if orbital correction animation is active
                if orbit_radius > 0.0 and not np.isnan(orbit_radius) and p is not None:
                    if not body.get("is_correcting_orbit", False):
                        trace(f"PRE_WRITE {body['name']} pos={body['position'].copy()} source=update_physics_moon parent_pos={p['position'].copy()}")
                        self.compute_moon_position(body, p)
                        trace(f"POST_WRITE {body['name']} pos={body['position'].copy()} source=update_physics_moon")
                        # Update visual_position to match position when not animating
                        if "visual_position" in body:
                            body["visual_position"] = body["position"].copy()
                    
                    # Verification: Log moon position relative to planet (every 60 frames to avoid spam)
                    actual_distance = np.linalg.norm(body["position"] - p["position"])
                    if not hasattr(self, '_moon_log_counter'):
                        self._moon_log_counter = 0
                    self._moon_log_counter += 1
                    
                    if abs(actual_distance - orbit_radius) > 0.1:  # Allow small floating point error
                        orbit_log(f"MOON {body['name']} pos={body['position']} PLANET {p['name']} pos={p['position']} distance={actual_distance:.2f} expected_radius={orbit_radius:.2f}")
                    elif self._moon_log_counter % 60 == 0:  # Log every 60 frames when correct
                        if DEBUG_ORBIT:
                            print(f"MOON {body['name']} pos={body['position']} PLANET {p['name']} pos={p['position']} distance={actual_distance:.2f} radius={orbit_radius:.2f}")
                    
                    # Verification: Log moon position relative to planet
                    actual_distance = np.linalg.norm(body["position"] - p["position"])
                    if abs(actual_distance - orbit_radius) > 0.1:  # Allow small floating point error
                        orbit_log(f"MOON {body['name']} pos={body['position']} PLANET {p['name']} pos={p['position']} distance={actual_distance:.2f} expected_radius={orbit_radius:.2f}")
                    else:
                        # Only log occasionally to avoid spam
                        if hasattr(self, '_moon_log_counter'):
                            self._moon_log_counter += 1
                        else:
                            self._moon_log_counter = 0
                        if self._moon_log_counter % 60 == 0:  # Log every 60 frames
                            orbit_log(f"MOON {body['name']} pos={body['position']} PLANET {p['name']} pos={p['position']} distance={actual_distance:.2f} radius={orbit_radius:.2f}")
                else:
                    # Debug output for moons with issues
                    if body["type"] == "moon":
                        if orbit_speed == 0.0:
                            print(f"WARNING: Moon {body.get('name', 'unknown')} has orbit_speed=0.0")
                            print(f"  parent={body.get('parent')}, orbit_radius={orbit_radius}")
                        if orbit_radius == 0.0:
                            print(f"WARNING: Moon {body.get('name', 'unknown')} has orbit_radius=0.0")
                        if p is None:
                            print(f"WARNING: Moon {body.get('name', 'unknown')} has parent_obj=None")
                
                # Skip remaining updates if paused (but position was already recalculated above)
                if effective_dt == 0.0:
                    continue
                
                # Update velocity for circular orbit (relative to parent's frame)
                v = body["orbit_speed"] * body["orbit_radius"]
                body["velocity"] = np.array([-v * np.sin(body["orbit_angle"]), v * np.cos(body["orbit_angle"])]).copy()  # Ensure independent copy
                
                # Update rotation angle
                body["rotation_angle"] += body["rotation_speed"] * effective_dt
                if body["rotation_angle"] >= 2 * np.pi:
                    body["rotation_angle"] -= 2 * np.pi
                
                # Store orbit points only when time progresses and orbit is enabled
                # For moons, store points RELATIVE to planet (not absolute world position)
                if effective_dt > 0.0 and body.get("orbit_enabled", True):
                    if "orbit_points" not in body:
                        body["orbit_points"] = []
                    if "max_orbit_points" not in body:
                        body["max_orbit_points"] = 2000
                    
                    # For moons, store offset from planet (relative position)
                    # For planets, store absolute world position
                    if body["type"] == "moon" and p is not None:
                        # Store relative offset from planet
                        relative_offset = body["position"] - p["position"]
                        body["orbit_points"].append(relative_offset.copy())
                    else:
                        # Planets store absolute position
                        body["orbit_points"].append(body["position"].copy())
                    
                    # Trim to max_orbit_points if exceeded
                    if len(body["orbit_points"]) > body["max_orbit_points"]:
                        body["orbit_points"].pop(0)
        
        # Automated assertions at end of frame
        trace("END_FRAME_ASSERTIONS")
        for body in moons:
            if body.get("parent_obj") is not None:
                p = body["parent_obj"]
                orbit_radius = body.get("orbit_radius", 0.0)
                orbit_angle = body.get("orbit_angle", 0.0)
                if orbit_radius > 0.0:
                    expected_pos = p["position"] + np.array([
                        orbit_radius * math.cos(orbit_angle),
                        orbit_radius * math.sin(orbit_angle)
                    ])
                    err = np.linalg.norm(body["position"] - expected_pos)
                    if err > 1e-3:
                        orbit_log(f"ASSERT FAIL: {body['name']} position mismatch err={err:.6e}")
                        orbit_log(f"  moon_pos={body['position']} expected={expected_pos}")
                        orbit_log(f"  parent_pos={p['position']} orbit_radius={orbit_radius} orbit_angle={orbit_angle}")
                        # Print last 50 trace lines
                        orbit_log("LAST 50 TRACE LINES:")
                        for line in frame_trace[-50:]:
                            print(line)
                        trace(f"ASSERT_FAIL {body['name']} err={err:.6e}")
        
        # Per-frame dump (first 300 lines)
        if len(frame_trace) > 0:
            orbit_log("FRAME TRACE START")
            for line in frame_trace[:300]:
                print(line)
            orbit_log("FRAME TRACE END")
            frame_trace.clear()

    def run_orbit_unit_test(self):
        """Deterministic unit test for moon orbit mechanics"""
        orbit_log("=" * 80)
        orbit_log("STARTING ORBIT UNIT TEST")
        orbit_log("=" * 80)
        
        # Create clean test objects (do NOT modify existing placed_bodies)
        star_test = {
            "position": np.array([0.0, 0.0], dtype=float),
            "type": "star",
            "name": "StarTest"
        }
        
        planet_test = {
            "orbit_radius": 100.0,  # px
            "orbit_angle": 0.0,
            "orbit_speed": 0.02,  # rad / frame
            "type": "planet",
            "name": "PlanetTest",
            "parent_obj": star_test,
            "position": np.array([100.0, 0.0], dtype=float)
        }
        
        moon_test = {
            "orbit_radius": 20.0,
            "orbit_angle": 0.0,
            "orbit_speed": 0.2,  # rad / frame
            "type": "moon",
            "name": "MoonTest",
            "parent_obj": planet_test,
            "position": planet_test["position"] + np.array([20.0, 0.0], dtype=float)
        }
        
        max_err = 0.0
        failures = []
        
        # Run deterministic loop for 200 frames
        for frame in range(1, 201):
            # Update planet first (simulate planet update code)
            planet_test["orbit_angle"] += planet_test["orbit_speed"]
            planet_test["position"][0] = star_test["position"][0] + planet_test["orbit_radius"] * math.cos(planet_test["orbit_angle"])
            planet_test["position"][1] = star_test["position"][1] + planet_test["orbit_radius"] * math.sin(planet_test["orbit_angle"])
            
            # Then update moon
            moon_test["orbit_angle"] += moon_test["orbit_speed"]
            moon_test["position"][0] = planet_test["position"][0] + moon_test["orbit_radius"] * math.cos(moon_test["orbit_angle"])
            moon_test["position"][1] = planet_test["position"][1] + moon_test["orbit_radius"] * math.sin(moon_test["orbit_angle"])
            
            # After update compute expected vector
            expected = planet_test["position"] + np.array([
                moon_test["orbit_radius"] * math.cos(moon_test["orbit_angle"]),
                moon_test["orbit_radius"] * math.sin(moon_test["orbit_angle"])
            ])
            
            # Record error between moon position and expected position
            err = np.linalg.norm(moon_test["position"] - expected)
            max_err = max(max_err, err)
            
            if frame <= 10 or frame % 20 == 0 or err > 1e-6:
                print(f"[UNIT_TEST] frame={frame} planet_pos={planet_test['position']} moon_pos={moon_test['position']} expected={expected} err={err:.6e}")
            
            if err > 1e-6:
                failures.append((frame, err, moon_test["position"].copy(), expected.copy()))
        
        # Print summary
        orbit_log("=" * 80)
        orbit_log(f"UNIT TEST SUMMARY: max_err={max_err:.6e}")
        if max_err > 1e-6:
            orbit_log(f"FAIL: {len(failures)} frames with error > 1e-6")
            for frame, err, moon_pos, exp_pos in failures[:10]:  # Show first 10 failures
                orbit_log(f"  frame={frame} err={err:.6e} moon={moon_pos} expected={exp_pos}")
        else:
            orbit_log("PASS: All frames within tolerance (err < 1e-6)")
        orbit_log("=" * 80)
        
        return max_err <= 1e-6
    
    def update_moon_orbit_grid(self, moon, planet):
        """Update the moon's orbit grid points to follow its parent planet with Keplerian ellipse"""
        a = moon.get("orbit_radius", MOON_ORBIT_PX)
        if a <= 0:
            a = MOON_ORBIT_PX
        
        e = float(moon.get("eccentricity", 0.0))
        grid_points = []
        
        # Generate 200 points for a smooth Keplerian ellipse
        for i in range(200):
            theta = i * 2 * np.pi / 200
            # r(θ) = a(1 - e²) / (1 + e*cosθ)
            r = a * (1 - e**2) / (1 + e * np.cos(theta))
            
            x = planet["position"][0] + r * math.cos(theta)
            y = planet["position"][1] + r * math.sin(theta)
            grid_points.append(np.array([x, y]))
        
        # Update special points: none needed for rendering
        self.orbit_grid_points[moon["name"]] = grid_points
    
    def draw_spacetime_grid(self):
        """Draw a static spacetime grid in the background"""
        # Calculate the space area boundaries
        space_top = self.tab_height + 2*self.tab_margin
        
        # Calculate the right boundary based on whether customization panel is visible
        right_boundary = self.width - self.customization_panel_width if self.show_customization_panel else self.width
        
        # Draw horizontal lines
        for y in range(space_top, self.height, self.grid_size):
            # Draw main lines
            pygame.draw.line(self.screen, self.GRID_COLOR, 
                            (0, y), 
                            (right_boundary, y), 1)
            
            # Draw secondary lines (darker)
            if y + self.grid_size//2 < self.height:
                pygame.draw.line(self.screen, (self.GRID_COLOR[0]//2, self.GRID_COLOR[1]//2, self.GRID_COLOR[2]//2), 
                                (0, y + self.grid_size//2), 
                                (right_boundary, y + self.grid_size//2), 1)
        
        # Draw vertical lines
        for x in range(0, right_boundary, self.grid_size):
            # Draw main lines
            pygame.draw.line(self.screen, self.GRID_COLOR, 
                            (x, space_top), 
                            (x, self.height), 1)
            
            # Draw secondary lines (darker)
            if x + self.grid_size//2 < right_boundary:
                pygame.draw.line(self.screen, (self.GRID_COLOR[0]//2, self.GRID_COLOR[1]//2, self.GRID_COLOR[2]//2), 
                                (x + self.grid_size//2, space_top), 
                                (x + self.grid_size//2, self.height), 1)
    
    # Home screen removed - start directly in sandbox
    # def render_home_screen(self):
    #     """Render the home screen with the title and create button"""
    #     # Fill background
    #     self.screen.fill(self.BLACK)
    #     
    #     # Update ambient colors
    #     self.update_ambient_colors()
    #     
    #     # Render title with current ambient color
    #     title_text = self.title_font.render("AIET", True, self.ambient_colors[self.current_color_index])
    #     title_rect = title_text.get_rect(center=(self.width//2, self.height//3))
    #     self.screen.blit(title_text, title_rect)
    #     
    #     # Draw instruction with current ambient color
    #     instruction_text = self.font.render("Create", True, self.ambient_colors[self.current_color_index])
    #     instruction_rect = instruction_text.get_rect(center=(self.width//2, self.height*3/4))
    #     
    #     # Draw a button around the text
    #     button_width = instruction_text.get_width() + 40
    #     button_height = instruction_text.get_height() + 20
    #     button_rect = pygame.Rect(
    #         instruction_rect.centerx - button_width//2,
    #         instruction_rect.centery - button_height//2,
    #         button_width,
    #         button_height
    #     )
    #     
    #     # Draw button with current ambient color
    #     pygame.draw.rect(self.screen, (50, 50, 50), button_rect, border_radius=10)
    #     pygame.draw.rect(self.screen, self.ambient_colors[self.current_color_index], button_rect, 2, border_radius=10)
    #     
    #     # Draw the text on top of the button
    #     self.screen.blit(instruction_text, instruction_rect)
    #     
    #     pygame.display.flip()
    
    def format_age_display(self, age: float) -> str:
        """Format age for display, converting to Myr if less than 0.5 Gyr"""
        # Ensure age is a Python float, not a numpy array/scalar
        if hasattr(age, 'item'):
            age = float(age.item())
        else:
            age = float(age)
        if age < 0.5:
            myr = age * 1000  # Convert Gyr to Myr
            return f"{myr:.1f} (Myr)"
        return f"{age:.1f} (Gyr)"

    def create_dropdown_surface(self):
        """Create a new surface for the dropdown menu that will float above everything"""
        if self.planet_orbital_distance_dropdown_visible:
            print('DEBUG: create_dropdown_surface called for orbital distance dropdown')
        if self.planet_stellar_flux_dropdown_visible:
            print('DEBUG: create_dropdown_surface called for stellar flux dropdown')
        if not (self.planet_dropdown_visible or self.moon_dropdown_visible or 
                self.star_mass_dropdown_visible or self.luminosity_dropdown_visible or
                self.planet_age_dropdown_visible or self.star_age_dropdown_visible or
                self.moon_age_dropdown_visible or self.moon_radius_dropdown_visible or self.moon_orbital_distance_dropdown_visible or self.moon_orbital_period_dropdown_visible or self.moon_temperature_dropdown_visible or self.moon_gravity_dropdown_visible or self.spectral_class_dropdown_visible or 
                self.radius_dropdown_visible or self.activity_dropdown_visible or 
                self.metallicity_dropdown_visible or self.planet_radius_dropdown_visible or 
                self.planet_temperature_dropdown_visible or self.planet_atmosphere_dropdown_visible or self.planet_gravity_dropdown_visible or 
                self.planet_orbital_distance_dropdown_visible or self.planet_orbital_eccentricity_dropdown_visible or
                self.planet_orbital_period_dropdown_visible or self.planet_stellar_flux_dropdown_visible or self.planet_density_dropdown_visible):
            return
        # Calculate dropdown dimensions
        option_height = self.dropdown_option_height
        if self.planet_dropdown_visible:
            options = self.planet_dropdown_options
            width = self.planet_dropdown_rect.width
        elif self.moon_dropdown_visible:
            options = self.moon_dropdown_options
            width = self.moon_dropdown_rect.width
        elif self.moon_age_dropdown_visible:
            options = self.moon_age_dropdown_options
            width = self.moon_age_dropdown_rect.width
        elif self.moon_radius_dropdown_visible:
            options = self.moon_radius_dropdown_options
            width = self.moon_radius_dropdown_rect.width
        elif self.moon_orbital_distance_dropdown_visible:
            options = self.moon_orbital_distance_dropdown_options
            width = self.moon_orbital_distance_dropdown_rect.width
        elif self.moon_orbital_period_dropdown_visible:
            options = self.moon_orbital_period_dropdown_options
            width = self.moon_orbital_period_dropdown_rect.width
        elif self.moon_temperature_dropdown_visible:
            options = self.moon_temperature_dropdown_options
            width = self.moon_temperature_dropdown_rect.width
        elif self.moon_gravity_dropdown_visible:
            options = self.moon_gravity_dropdown_options
            width = self.moon_gravity_dropdown_rect.width
        elif self.star_mass_dropdown_visible:
            options = self.star_mass_dropdown_options
            width = self.star_mass_dropdown_rect.width
        elif self.luminosity_dropdown_visible:
            options = self.luminosity_dropdown_options
            width = self.luminosity_dropdown_rect.width
        elif self.planet_age_dropdown_visible:
            options = self.planet_age_dropdown_options
            width = self.planet_age_dropdown_rect.width
        elif self.star_age_dropdown_visible:
            options = self.star_age_dropdown_options
            width = self.star_age_dropdown_rect.width
        elif self.spectral_class_dropdown_visible:
            options = self.spectral_class_dropdown_options
            width = self.spectral_class_dropdown_rect.width
        elif self.radius_dropdown_visible:
            options = self.radius_dropdown_options
            width = self.radius_dropdown_rect.width
        elif self.activity_dropdown_visible:
            options = self.activity_dropdown_options
            width = self.activity_dropdown_rect.width
        elif self.metallicity_dropdown_visible:
            options = self.metallicity_dropdown_options
            width = self.metallicity_dropdown_rect.width
        elif self.planet_radius_dropdown_visible:
            options = self.planet_radius_dropdown_options
            width = self.planet_radius_dropdown_rect.width
        elif self.planet_temperature_dropdown_visible:
            options = self.planet_temperature_dropdown_options
            width = self.planet_temperature_dropdown_rect.width
        elif self.planet_atmosphere_dropdown_visible:
            options = self.planet_atmosphere_dropdown_options
            width = self.planet_atmosphere_dropdown_rect.width
        elif self.planet_gravity_dropdown_visible:
            options = self.planet_gravity_dropdown_options
            width = self.planet_gravity_dropdown_rect.width
        elif self.planet_orbital_distance_dropdown_visible:
            options = self.planet_orbital_distance_dropdown_options
            width = self.planet_orbital_distance_dropdown_rect.width
        elif self.planet_orbital_eccentricity_dropdown_visible:
            options = self.planet_orbital_eccentricity_dropdown_options
            width = self.planet_orbital_eccentricity_dropdown_rect.width
        elif self.planet_orbital_period_dropdown_visible:
            options = self.planet_orbital_period_dropdown_options
            width = self.planet_orbital_period_dropdown_rect.width
        elif self.planet_stellar_flux_dropdown_visible:
            print(f'DEBUG: Creating stellar flux dropdown surface with {len(self.planet_stellar_flux_dropdown_options)} options')
            options = self.planet_stellar_flux_dropdown_options
            width = self.planet_stellar_flux_dropdown_rect.width
        elif self.planet_density_dropdown_visible:
            options = self.planet_density_dropdown_options
            width = self.planet_density_dropdown_rect.width
        else:  # luminosity dropdown
            options = self.luminosity_dropdown_options
            width = self.luminosity_dropdown_rect.width
        total_height = len(options) * option_height
        
        # Create a new surface for the dropdown
        self.dropdown_surface = pygame.Surface((width, total_height), pygame.SRCALPHA)
        self.dropdown_surface.fill((255, 255, 255))  # Solid white background
        
        # Create rects for each option
        self.dropdown_options_rects = []
        for i, option_data in enumerate(options):
            # Handle different option formats
            if len(option_data) == 3:  # moon_dropdown_options: (name, value, unit) or spectral_class_dropdown_options: (name, temp, color)
                name, value, third_param = option_data
                if self.moon_dropdown_visible:
                    unit = third_param
                else:
                    color = third_param
            else:  # other dropdowns: (name, value)
                name, value = option_data
                unit = None
                
            option_rect = pygame.Rect(
                0,  # x position relative to dropdown surface
                i * option_height,  # y position relative to dropdown surface
                width,
                option_height
            )
            self.dropdown_options_rects.append(option_rect)
            
            # Check if this option is selected (for planet dropdown in placement mode)
            is_selected = False
            if self.planet_dropdown_visible:
                # In placement mode, check if this option matches the selected planet
                if not self.selected_body or self.selected_body.get('type') != 'planet':
                    if self.planet_dropdown_selected == name:
                        is_selected = True
            
            # Draw option background with highlighting if selected
            if is_selected:
                pygame.draw.rect(self.dropdown_surface, (200, 220, 255), option_rect)  # Light blue highlight
            else:
                pygame.draw.rect(self.dropdown_surface, (255, 255, 255), option_rect)  # Solid white background
            pygame.draw.rect(self.dropdown_surface, self.dropdown_border_color, option_rect, self.dropdown_border_width)
            
            # Draw option text
            if value is not None:
                if self.planet_dropdown_visible:
                    text = f"{name} (" + self._format_value(value, 'M⊕') + ")"
                elif self.moon_dropdown_visible:
                    # Use specialized formatting for moon masses: "Name (Value M☾)"
                    formatted_val = self._format_value(value, 'M☾')
                    text = f"{name} ({formatted_val})"
                elif self.star_mass_dropdown_visible:
                    text = name  # Mass options already include the unit
                elif self.planet_age_dropdown_visible or self.star_age_dropdown_visible:
                    text = name  # Age options already include the unit
                elif self.moon_age_dropdown_visible:
                    if "Gyr" not in name:
                        text = f"{name} (" + self._format_value(value, 'Gyr') + ")"
                    else:
                        text = name
                elif self.moon_radius_dropdown_visible:
                    text = f"{name} (" + self._format_value(value, 'km') + ")"
                elif self.moon_orbital_distance_dropdown_visible:
                    text = f"{name} (" + self._format_value(value, 'km') + ")"
                elif self.moon_orbital_period_dropdown_visible:
                    text = f"{name} (" + self._format_value(value, 'days') + ")"
                elif self.moon_temperature_dropdown_visible:
                    text = f"{name} (" + self._format_value(value, 'K') + ")"
                elif self.moon_gravity_dropdown_visible:
                    text = f"{name} (" + self._format_value(value, 'm/s²') + ")"
                elif self.spectral_class_dropdown_visible:
                    text = name  # Spectral class options already include temperature
                elif self.radius_dropdown_visible:
                    if "(Sun)" in name:
                        text = f"Sun ({value:.2f} R☉)"
                    else:
                        text = f"{name} (" + self._format_value(value, 'R☉') + ")"
                elif self.activity_dropdown_visible:
                    if "(Sun)" in name:
                        text = f"Sun ({value:.2f})"
                    else:
                        text = f"{name} (" + self._format_value(value, '') + ")"
                elif self.metallicity_dropdown_visible:
                    text = name  # Metallicity options already include the unit (e.g., "0.0 (Sun)")
                elif self.planet_radius_dropdown_visible:
                    text = f"{name} (" + self._format_value(value, 'R🜨') + ")"
                elif self.planet_temperature_dropdown_visible:
                    text = f"{name} (" + self._format_value(value, 'K') + ")"
                elif self.planet_atmosphere_dropdown_visible:
                    text = name
                elif self.planet_gravity_dropdown_visible:
                    text = f"{name} (" + self._format_value(value, 'm/s²') + ")"
                elif self.planet_orbital_distance_dropdown_visible:
                    text = f"{name} (" + self._format_value(value, 'AU') + ")"
                elif self.planet_orbital_eccentricity_dropdown_visible:
                    text = f"{name} (" + self._format_value(value, '') + ")"
                elif self.planet_orbital_period_dropdown_visible:
                    text = f"{name} (" + self._format_value(value, 'days') + ")"
                elif self.planet_stellar_flux_dropdown_visible:
                    text = f"{name} (" + self._format_value(value, 'EFU') + ")"
                elif self.planet_density_dropdown_visible:
                    text = f"{name} (" + self._format_value(value, 'g/cm³') + ")"
                else:  # luminosity dropdown
                    text = f"{name} (" + self._format_value(value, 'L☉') + ")"
            else:
                # Handle "Custom" options without redundancy
                if name.strip() == "Custom":
                    if self.planet_dropdown_visible: text = "Custom (M⊕)"
                    elif self.moon_dropdown_visible: text = "Custom (M☾)"
                    elif self.star_mass_dropdown_visible: text = "Custom (M☉)"
                    elif self.planet_radius_dropdown_visible: text = "Custom (R🜨)"
                    elif self.moon_radius_dropdown_visible: text = "Custom (km)"
                    elif self.planet_temperature_dropdown_visible or self.moon_temperature_dropdown_visible: text = "Custom (K)"
                    elif self.planet_atmosphere_dropdown_visible: text = "Custom (K ΔT)"
                    elif self.planet_gravity_dropdown_visible or self.moon_gravity_dropdown_visible: text = "Custom (m/s²)"
                    elif self.planet_orbital_distance_dropdown_visible: text = "Custom (AU)"
                    elif self.moon_orbital_distance_dropdown_visible: text = "Custom (km)"
                    elif self.planet_orbital_period_dropdown_visible or self.moon_orbital_period_dropdown_visible: text = "Custom (days)"
                    elif self.planet_age_dropdown_visible or self.star_age_dropdown_visible or self.moon_age_dropdown_visible: text = "Custom (Gyr)"
                    elif self.spectral_class_dropdown_visible: text = "Custom (K)"
                    elif self.luminosity_dropdown_visible: text = "Custom (L☉)"
                    elif self.radius_dropdown_visible: text = "Custom (R☉)"
                    elif self.planet_stellar_flux_dropdown_visible: text = "Custom (EFU)"
                    elif self.planet_density_dropdown_visible: text = "Custom (g/cm³)"
                    else: text = name
                else:
                    # If name is not "Custom" but value is None, only append if it doesn't already say it
                    if "(Custom)" in name:
                        text = name
                    else:
                        text = f"{name} (Custom)"
            if "hydrogen-burning limit" in text:
                # Special split rendering for Red Dwarf label
                parts = text.split("hydrogen-burning limit")
                first_part = parts[0]
                limit_text = "hydrogen-burning limit"
                last_part = parts[1] if len(parts) > 1 else ""
                
                # Render first part
                surf1 = self.subtitle_font.render(first_part, True, self.dropdown_text_color)
                rect1 = surf1.get_rect(midleft=(self.dropdown_padding, option_rect.centery))
                self.dropdown_surface.blit(surf1, rect1)
                
                # Render limit text smaller
                surf2 = self.tiny_font.render(limit_text, True, self.dropdown_text_color)
                rect2 = surf2.get_rect(midleft=(rect1.right, option_rect.centery))
                self.dropdown_surface.blit(surf2, rect2)
                
                # Render last part (the closing parenthesis)
                if last_part:
                    surf3 = self.subtitle_font.render(last_part, True, self.dropdown_text_color)
                    rect3 = surf3.get_rect(midleft=(rect2.right, option_rect.centery))
                    self.dropdown_surface.blit(surf3, rect3)
            else:
                text_surface = self.subtitle_font.render(text, True, self.dropdown_text_color)
                text_rect = text_surface.get_rect(midleft=(self.dropdown_padding, option_rect.centery))
                self.dropdown_surface.blit(text_surface, text_rect)
        
        # Create the dropdown rect for positioning
        if self.planet_dropdown_visible:
            self.dropdown_rect = pygame.Rect(
                self.planet_dropdown_rect.left,
                self.planet_dropdown_rect.bottom,
                width,
                total_height
            )
        elif self.moon_dropdown_visible:
            self.dropdown_rect = pygame.Rect(
                self.moon_dropdown_rect.left,
                self.moon_dropdown_rect.bottom,
                width,
                total_height
            )
        elif self.star_mass_dropdown_visible:
            self.dropdown_rect = pygame.Rect(
                self.star_mass_dropdown_rect.left,
                self.star_mass_dropdown_rect.bottom,
                width,
                total_height
            )
        elif self.planet_age_dropdown_visible:
            self.dropdown_rect = pygame.Rect(
                self.planet_age_dropdown_rect.left,
                self.planet_age_dropdown_rect.bottom,
                width,
                total_height
            )
        elif self.star_age_dropdown_visible:
            self.dropdown_rect = pygame.Rect(
                self.star_age_dropdown_rect.left,
                self.star_age_dropdown_rect.bottom,
                width,
                total_height
            )
        elif self.spectral_class_dropdown_visible:
            self.dropdown_rect = pygame.Rect(
                self.spectral_class_dropdown_rect.left,
                self.spectral_class_dropdown_rect.bottom,
                width,
                total_height
            )
        elif self.moon_age_dropdown_visible:
            self.dropdown_rect = pygame.Rect(
                self.moon_age_dropdown_rect.left,
                self.moon_age_dropdown_rect.bottom,
                width,
                total_height
            )
        elif self.moon_radius_dropdown_visible:
            self.dropdown_rect = pygame.Rect(
                self.moon_radius_dropdown_rect.left,
                self.moon_radius_dropdown_rect.bottom,
                width,
                total_height
            )
        elif self.moon_orbital_distance_dropdown_visible:
            self.dropdown_rect = pygame.Rect(
                self.moon_orbital_distance_dropdown_rect.left,
                self.moon_orbital_distance_dropdown_rect.bottom,
                width,
                total_height
            )
        elif self.moon_orbital_period_dropdown_visible:
            self.dropdown_rect = pygame.Rect(
                self.moon_orbital_period_dropdown_rect.left,
                self.moon_orbital_period_dropdown_rect.bottom,
                width,
                total_height
            )
        elif self.moon_temperature_dropdown_visible:
            self.dropdown_rect = pygame.Rect(
                self.moon_temperature_dropdown_rect.left,
                self.moon_temperature_dropdown_rect.bottom,
                width,
                total_height
            )
        elif self.moon_gravity_dropdown_visible:
            self.dropdown_rect = pygame.Rect(
                self.moon_gravity_dropdown_rect.left,
                self.moon_gravity_dropdown_rect.bottom,
                width,
                total_height
            )
        elif self.radius_dropdown_visible:
            self.dropdown_rect = pygame.Rect(
                self.radius_dropdown_rect.left,
                self.radius_dropdown_rect.bottom,
                width,
                total_height
            )
        elif self.activity_dropdown_visible:
            self.dropdown_rect = pygame.Rect(
                self.activity_dropdown_rect.left,
                self.activity_dropdown_rect.bottom,
                width,
                total_height
            )
        elif self.metallicity_dropdown_visible:
            self.dropdown_rect = pygame.Rect(
                self.metallicity_dropdown_rect.left,
                self.metallicity_dropdown_rect.bottom,
                width,
                total_height
            )
        elif self.planet_radius_dropdown_visible:
            self.dropdown_rect = pygame.Rect(
                self.planet_radius_dropdown_rect.left,
                self.planet_radius_dropdown_rect.bottom,
                width,
                total_height
            )
        elif self.planet_temperature_dropdown_visible:
            self.dropdown_rect = pygame.Rect(
                self.planet_temperature_dropdown_rect.left,
                self.planet_temperature_dropdown_rect.bottom,
                width,
                total_height
            )
        elif self.planet_atmosphere_dropdown_visible:
            self.dropdown_rect = pygame.Rect(
                self.planet_atmosphere_dropdown_rect.left,
                self.planet_atmosphere_dropdown_rect.bottom,
                width,
                total_height
            )
        elif self.planet_gravity_dropdown_visible:
            self.dropdown_rect = pygame.Rect(
                self.planet_gravity_dropdown_rect.left,
                self.planet_gravity_dropdown_rect.bottom,
                width,
                total_height
            )
        elif self.planet_orbital_distance_dropdown_visible:
            self.dropdown_rect = pygame.Rect(
                self.planet_orbital_distance_dropdown_rect.left,
                self.planet_orbital_distance_dropdown_rect.bottom,
                width,
                total_height
            )
        elif self.planet_orbital_eccentricity_dropdown_visible:
            self.dropdown_rect = pygame.Rect(
                self.planet_orbital_eccentricity_dropdown_rect.left,
                self.planet_orbital_eccentricity_dropdown_rect.bottom,
                width,
                total_height
            )
        elif self.planet_orbital_period_dropdown_visible:
            self.dropdown_rect = pygame.Rect(
                self.planet_orbital_period_dropdown_rect.left,
                self.planet_orbital_period_dropdown_rect.bottom,
                width,
                total_height
            )
        elif self.planet_stellar_flux_dropdown_visible:
            self.dropdown_rect = pygame.Rect(
                self.planet_stellar_flux_dropdown_rect.left,
                self.planet_stellar_flux_dropdown_rect.bottom,
                width,
                total_height
            )
        elif self.planet_density_dropdown_visible:
            self.dropdown_rect = pygame.Rect(
                self.planet_density_dropdown_rect.left,
                self.planet_density_dropdown_rect.bottom,
                width,
                total_height
            )
        else:  # luminosity dropdown
            self.dropdown_rect = pygame.Rect(
                self.luminosity_dropdown_rect.left,
                self.luminosity_dropdown_rect.bottom,
                width,
                total_height
            )

    def close_all_dropdowns(self):
        """Close all visible dropdown menus and set their active states to False."""
        self.planet_preset_dropdown_visible = False
        self.planet_dropdown_visible = False
        self.planet_age_dropdown_visible = False
        self.planet_radius_dropdown_visible = False
        self.planet_temperature_dropdown_visible = False
        self.planet_atmosphere_dropdown_visible = False
        self.planet_gravity_dropdown_visible = False
        self.moon_dropdown_visible = False
        self.moon_age_dropdown_visible = False
        self.moon_radius_dropdown_visible = False
        self.moon_orbital_distance_dropdown_visible = False
        self.moon_orbital_period_dropdown_visible = False
        self.moon_temperature_dropdown_visible = False
        self.moon_gravity_dropdown_visible = False
        self.luminosity_dropdown_visible = False
        self.spectral_class_dropdown_visible = False
        self.star_mass_dropdown_visible = False
        self.star_age_dropdown_visible = False
        self.radius_dropdown_visible = False
        self.activity_dropdown_visible = False
        self.metallicity_dropdown_visible = False
        self.planet_orbital_distance_dropdown_visible = False
        self.planet_orbital_eccentricity_dropdown_visible = False
        self.planet_orbital_period_dropdown_visible = False
        self.planet_stellar_flux_dropdown_visible = False
        self.planet_density_dropdown_visible = False
        
        # Reset active states
        self.planet_dropdown_active = False
        self.moon_dropdown_active = False
        self.star_mass_dropdown_active = False
        self.luminosity_dropdown_active = False
        self.planet_age_dropdown_active = False
        self.star_age_dropdown_active = False
        self.moon_age_dropdown_active = False
        self.moon_radius_dropdown_active = False
        self.moon_orbital_distance_dropdown_active = False
        self.moon_orbital_period_dropdown_active = False
        self.moon_temperature_dropdown_active = False
        self.moon_gravity_dropdown_active = False
        self.spectral_class_dropdown_active = False
        self.radius_dropdown_active = False
        self.activity_dropdown_active = False
        self.metallicity_dropdown_active = False
        self.planet_orbital_distance_dropdown_active = False
        self.planet_orbital_eccentricity_dropdown_active = False
        self.planet_orbital_period_dropdown_active = False
        self.planet_stellar_flux_dropdown_active = False
        self.planet_density_dropdown_active = False

        # CRITICAL: Clear dropdown rects to avoid indexing errors when switching between 
        # dropdowns of different sizes
        self.dropdown_options_rects = []
        self.dropdown_surface = None
        self.dropdown_rect = None

    def _handle_dropdown_selection(self, pos):
        """
        Handle clicks on dropdown menu options.
        Returns True if a click was handled, False otherwise.
        """
        # 1. Handle Planet Preset Dropdown (special case)
        if self.planet_preset_dropdown_visible and self.planet_preset_dropdown_rect:
            if self.planet_preset_dropdown_rect.collidepoint(pos):
                relative_y = pos[1] - self.planet_preset_dropdown_rect.top
                option_index = relative_y // 28
                if 0 <= option_index < len(self.planet_preset_options):
                    preset_name = self.planet_preset_options[option_index]
                    if self.selected_body and self.selected_body.get('type') == 'planet':
                        self.apply_planet_preset(preset_name)
                        self.planet_preset_dropdown_visible = False
                    else:
                        if self.planet_dropdown_selected == preset_name:
                            self.planet_dropdown_selected = None
                            self.clear_preview()
                        else:
                            self.planet_dropdown_selected = preset_name
                            if preset_name in SOLAR_SYSTEM_PLANET_PRESETS:
                                preset = SOLAR_SYSTEM_PLANET_PRESETS[preset_name]
                                planet_radius_re = preset["radius"]
                                visual_radius = EARTH_RADIUS_PX * math.pow(planet_radius_re, 0.4)
                                self.preview_radius = int(max(MIN_PLANET_PX, min(visual_radius, SUN_RADIUS_PX * 0.85)))
                            else:
                                self.preview_radius = int(MIN_PLANET_PX)
                            self.preview_position = pygame.mouse.get_pos()
                            self.placement_mode_active = True
                    self.create_dropdown_surface()
                    return True

        # 2. Handle Specialized Planet Dropdowns (Flux, Density)
        if self.selected_body and self.selected_body.get('type') == 'planet':
            # Stellar Flux
            if self.planet_stellar_flux_dropdown_visible and self.planet_stellar_flux_dropdown_rect:
                dropdown_y = self.planet_stellar_flux_dropdown_rect.bottom
                for i, (flux_name, flux) in enumerate(self.planet_stellar_flux_dropdown_options):
                    option_rect = pygame.Rect(self.planet_stellar_flux_dropdown_rect.left, dropdown_y + i * 30, self.planet_stellar_flux_dropdown_rect.width, 30)
                    if option_rect.collidepoint(pos):
                        if flux_name == "Custom":
                            self.show_custom_stellar_flux_input = True
                            self.stellar_flux_input_active = True
                            self.stellar_flux_input_text = f"{self.selected_body.get('stellarFlux', 1.0):.3f}"
                        else:
                            self.update_selected_body_property("stellarFlux", flux, "stellarFlux")
                            self.show_custom_stellar_flux_input = False
                        self.planet_stellar_flux_dropdown_selected = flux_name
                        self.planet_stellar_flux_dropdown_visible = False
                        self.planet_stellar_flux_dropdown_active = False
                        return True
            
            # Density
            if self.planet_density_dropdown_visible and self.planet_density_dropdown_rect:
                dropdown_y = self.planet_density_dropdown_rect.bottom
                for i, (density_name, density) in enumerate(self.planet_density_dropdown_options):
                    option_rect = pygame.Rect(self.planet_density_dropdown_rect.left, dropdown_y + i * 30, self.planet_density_dropdown_rect.width, 30)
                    if option_rect.collidepoint(pos):
                        if density_name == "Custom":
                            body = self.get_selected_body()
                            self.previous_dropdown_selection = body.get("planet_density_dropdown_selected") if body else None
                            self.show_custom_modal = True
                            self.custom_modal_title = "Set Custom Planet Density"
                            self.custom_modal_helper = "Enter density in g/cm³. Earth = 5.51. Range: 0.5 – 25.0"
                            self.custom_modal_min = 0.5
                            self.custom_modal_max = 25.0
                            self.custom_modal_unit = "g/cm³"
                            self.pending_custom_field = "density"
                            self.pending_custom_body_id = self.selected_body_id
                            self.custom_modal_text = ""
                            self.validate_custom_modal_input()
                        else:
                            self.update_selected_body_property("density", density, "density")
                        self.planet_density_dropdown_selected = density_name
                        self.planet_density_dropdown_visible = False
                        self.planet_density_dropdown_active = False
                        return True

        # 3. Handle Generalized Dropdowns (using self.dropdown_rect)
        if self._any_dropdown_active() and self.dropdown_rect:
            if self.dropdown_rect.collidepoint(pos):
                for i, option_rect in enumerate(self.dropdown_options_rects):
                    screen_option_rect = pygame.Rect(
                        self.dropdown_rect.left,
                        self.dropdown_rect.top + i * self.dropdown_option_height,
                        option_rect.width,
                        option_rect.height
                    )
                    
                    if screen_option_rect.collidepoint(pos):
                        # Handle selection for all standard dropdowns (moved logic from loop)
                        if self.planet_dropdown_visible:
                            if i < len(self.planet_dropdown_options):
                                name, value = self.planet_dropdown_options[i]
                                if not self.selected_body_id or not self.selected_body or self.selected_body.get('type') != 'planet':
                                    if name == "Custom":
                                        self.planet_dropdown_selected = None
                                        self.clear_preview()
                                    else:
                                        self.planet_dropdown_selected = name
                                        if name in SOLAR_SYSTEM_PLANET_PRESETS:
                                            preset = SOLAR_SYSTEM_PLANET_PRESETS[name]
                                            planet_radius_re = preset["radius"]
                                            visual_radius = EARTH_RADIUS_PX * math.pow(planet_radius_re, 0.4)
                                            self.preview_radius = int(max(MIN_PLANET_PX, min(visual_radius, SUN_RADIUS_PX * 0.85)))
                                        else:
                                            self.preview_radius = int(MIN_PLANET_PX)
                                        self.placement_mode_active = True
                                    self.create_dropdown_surface()
                                else:
                                    if value is not None:
                                        mass_value = float(value) if not hasattr(value, 'item') else float(value.item())
                                        if self.update_selected_body_property("mass", mass_value, "mass"):
                                            body = self.get_selected_body()
                                            if body: body["planet_dropdown_selected"] = name
                                    else:
                                        self.show_custom_planet_mass_input = True
                                        self.planet_mass_input_active = True
                                    self.planet_age_dropdown_selected = "Solar System Planets (4.6 Gyr)"
                                    self.planet_gravity_dropdown_selected = "Earth"
                                    self.planet_dropdown_visible = False
                                    self.planet_dropdown_active = False
                            return True

                        elif self.moon_dropdown_visible:
                            if i < len(self.moon_dropdown_options):
                                option_data = self.moon_dropdown_options[i]
                                if len(option_data) == 3: name, value, unit = option_data
                                else: name, value = option_data; unit = None
                                if value is not None:
                                    if self.selected_body:
                                        mass_value = value / 7.35e22 if unit == "kg" else value
                                        mass_float = float(mass_value.item()) if hasattr(mass_value, 'item') else float(mass_value)
                                        self.update_selected_body_property("mass", mass_float, "mass")
                                else:
                                    self.show_custom_moon_mass_input = True
                                self.moon_dropdown_selected = name
                                self.moon_dropdown_visible = False
                                self.moon_dropdown_active = False
                            return True
                            
                        elif self.star_mass_dropdown_visible:
                            if i < len(self.star_mass_dropdown_options):
                                name, value = self.star_mass_dropdown_options[i]
                                if value is not None:
                                    if self.selected_body:
                                        self.update_selected_body_property("mass", value, "mass")
                                else:
                                    self.show_custom_star_mass_input = True
                                    self.star_mass_input_active = True
                                self.star_mass_dropdown_selected = name
                                self.star_mass_dropdown_visible = False
                                self.star_mass_dropdown_active = False
                            return True

                        elif self.luminosity_dropdown_visible:
                            if i < len(self.luminosity_dropdown_options):
                                name, value = self.luminosity_dropdown_options[i]
                                if value is not None:
                                    self.update_selected_body_property("luminosity", value, "luminosity")
                                else:
                                    self.show_custom_luminosity_input = True
                                    self.luminosity_input_active = True
                                self.luminosity_dropdown_selected = name
                                self.luminosity_dropdown_visible = False
                                self.luminosity_dropdown_active = False
                            return True

                        elif self.planet_age_dropdown_visible:
                            if i < len(self.planet_age_dropdown_options):
                                name, value = self.planet_age_dropdown_options[i]
                                if value is not None:
                                    if self.selected_body:
                                        self.update_selected_body_property("age", value, "age")
                                else:
                                    self.show_custom_planet_age_input = True
                                    self.planet_age_input_active = True
                                self.planet_age_dropdown_selected = name
                                self.planet_age_dropdown_visible = False
                                self.planet_age_dropdown_active = False
                            return True

                        elif self.planet_radius_dropdown_visible:
                            if i < len(self.planet_radius_dropdown_options):
                                name, value = self.planet_radius_dropdown_options[i]
                                if name == "Custom":
                                    body = self.get_selected_body()
                                    self.previous_dropdown_selection = body.get("planet_radius_dropdown_selected") if body else None
                                    self.show_custom_modal = True
                                    self.custom_modal_title = "Set Custom Planet Radius"
                                    self.custom_modal_helper = "Enter radius in Earth radii (R⊕). Allowed: 0.1 – 20"
                                    self.custom_modal_min = 0.1
                                    self.custom_modal_max = 20.0
                                    self.custom_modal_unit = "R⊕"
                                    self.pending_custom_field = "radius"
                                    self.pending_custom_body_id = self.selected_body_id
                                    self.custom_modal_text = ""
                                    self.validate_custom_modal_input()
                                else:
                                    self.update_selected_body_property("radius", value, "radius")
                                self.planet_radius_dropdown_selected = name
                                self.planet_radius_dropdown_visible = False
                                self.planet_radius_dropdown_active = False
                            return True

                        elif self.planet_temperature_dropdown_visible:
                            if i < len(self.planet_temperature_dropdown_options):
                                name, value = self.planet_temperature_dropdown_options[i]
                                if name == "Custom":
                                    body = self.get_selected_body()
                                    self.previous_dropdown_selection = body.get("planet_temperature_dropdown_selected") if body else None
                                    self.show_custom_modal = True
                                    self.custom_modal_title = "Set Planet Temperature"
                                    self.custom_modal_helper = "Enter equilibrium temperature in Kelvin (K). Allowed: 1 – 5000"
                                    self.custom_modal_min = 1.0
                                    self.custom_modal_max = 5000.0
                                    self.custom_modal_unit = "K"
                                    self.pending_custom_field = "temperature"
                                    self.pending_custom_body_id = self.selected_body_id
                                    self.custom_modal_text = ""
                                    self.validate_custom_modal_input()
                                else:
                                    self.update_selected_body_property("temperature", value, "temperature")
                                self.planet_temperature_dropdown_selected = name
                                self.planet_temperature_dropdown_visible = False
                                self.planet_temperature_dropdown_active = False
                            return True

                        elif self.planet_atmosphere_dropdown_visible:
                            if i < len(self.planet_atmosphere_dropdown_options):
                                name, value = self.planet_atmosphere_dropdown_options[i]
                                self.update_selected_body_property("greenhouse_offset", value, "greenhouse_offset")
                                self.planet_atmosphere_dropdown_selected = name
                                self.planet_atmosphere_dropdown_visible = False
                                self.planet_atmosphere_dropdown_active = False
                            return True

                        elif self.planet_gravity_dropdown_visible:
                            if i < len(self.planet_gravity_dropdown_options):
                                name, value = self.planet_gravity_dropdown_options[i]
                                if name == "Custom":
                                    body = self.get_selected_body()
                                    self.previous_dropdown_selection = body.get("planet_gravity_dropdown_selected") if body else None
                                    self.show_custom_modal = True
                                    self.custom_modal_title = "Set Surface Gravity"
                                    self.custom_modal_helper = "Enter surface gravity in g-units (Earth = 1.0). Allowed: 0.01 – 50"
                                    self.custom_modal_min = 0.01
                                    self.custom_modal_max = 50.0
                                    self.custom_modal_unit = "g"
                                    self.pending_custom_field = "gravity"
                                    self.pending_custom_body_id = self.selected_body_id
                                    self.custom_modal_text = ""
                                    self.validate_custom_modal_input()
                                else:
                                    self.update_selected_body_property("gravity", value, "gravity")
                                self.planet_gravity_dropdown_selected = name
                                self.planet_gravity_dropdown_visible = False
                                self.planet_gravity_dropdown_active = False
                            return True

                        elif self.star_age_dropdown_visible:
                            if i < len(self.star_age_dropdown_options):
                                name, value = self.star_age_dropdown_options[i]
                                if value is not None:
                                    self.update_selected_body_property("age", value, "age")
                                else:
                                    self.show_custom_star_age_input = True
                                    self.star_age_input_active = True
                                self.star_age_dropdown_selected = name
                                self.star_age_dropdown_visible = False
                                self.star_age_dropdown_active = False
                            return True

                        elif self.moon_age_dropdown_visible:
                            if i < len(self.moon_age_dropdown_options):
                                name, value = self.moon_age_dropdown_options[i]
                                if value is not None:
                                    if self.selected_body:
                                        self.update_selected_body_property("age", value, "age")
                                else:
                                    self.show_custom_moon_age_input = True
                                    self.moon_age_input_active = True
                                self.moon_age_dropdown_selected = name
                                self.moon_age_dropdown_visible = False
                                self.moon_age_dropdown_active = False
                            return True

                        elif self.moon_radius_dropdown_visible:
                            if i < len(self.moon_radius_dropdown_options):
                                name, value = self.moon_radius_dropdown_options[i]
                                if value is not None:
                                    if self.selected_body:
                                        self.update_selected_body_property("radius", value, "radius")
                                else:
                                    self.show_custom_moon_radius_input = True
                                self.moon_radius_dropdown_selected = name
                                self.moon_radius_dropdown_visible = False
                                self.moon_radius_dropdown_active = False
                            return True

                        elif self.moon_orbital_distance_dropdown_visible:
                            if i < len(self.moon_orbital_distance_dropdown_options):
                                name, value = self.moon_orbital_distance_dropdown_options[i]
                                if value is not None:
                                    body = self.get_selected_body()
                                    if body:
                                        KM_TO_AU = 6.68459e-9
                                        distance_au = float(value) * KM_TO_AU
                                        self.update_selected_body_property("semiMajorAxis", distance_au, "semiMajorAxis")
                                        self.clear_orbit_points(body)
                                    self.generate_orbit_grid(self.selected_body)
                                else:
                                    self.show_custom_moon_orbital_distance_input = True
                                self.moon_orbital_distance_dropdown_selected = name
                                self.moon_orbital_distance_dropdown_visible = False
                                self.moon_orbital_distance_dropdown_active = False
                            return True

                        elif self.moon_orbital_period_dropdown_visible:
                            if i < len(self.moon_orbital_period_dropdown_options):
                                name, value = self.moon_orbital_period_dropdown_options[i]
                                if value is not None:
                                    self.update_selected_body_property("orbital_period", value, "orbital_period")
                                else:
                                    self.show_custom_moon_orbital_period_input = True
                                self.moon_orbital_period_dropdown_selected = name
                                self.moon_orbital_period_dropdown_visible = False
                                self.moon_orbital_period_dropdown_active = False
                            return True

                        elif self.moon_temperature_dropdown_visible:
                            if i < len(self.moon_temperature_dropdown_options):
                                name, value = self.moon_temperature_dropdown_options[i]
                                if value is not None:
                                    self.update_selected_body_property("temperature", value, "temperature")
                                else:
                                    self.show_custom_moon_temperature_input = True
                                self.moon_temperature_dropdown_selected = name
                                self.moon_temperature_dropdown_visible = False
                                self.moon_temperature_dropdown_active = False
                            return True

                        elif self.moon_gravity_dropdown_visible:
                            if i < len(self.moon_gravity_dropdown_options):
                                name, value = self.moon_gravity_dropdown_options[i]
                                if value is not None:
                                    self.update_selected_body_property("gravity", value, "gravity")
                                else:
                                    self.show_custom_moon_gravity_input = True
                                self.moon_gravity_dropdown_selected = name
                                self.moon_gravity_dropdown_visible = False
                                self.moon_gravity_dropdown_active = False
                            return True

                        elif self.spectral_class_dropdown_visible:
                            if i < len(self.spectral_class_dropdown_options):
                                option_data = self.spectral_class_dropdown_options[i]
                                name = option_data[0]
                                temp = option_data[1]
                                if temp is not None:
                                    self.update_selected_body_property("temperature", temp, "temperature")
                                    self.spectral_class_dropdown_selected = name
                                else:
                                    # Handle "Custom" selection
                                    self.show_custom_temperature_input = True
                                    self.temperature_input_active = True
                                    body = self.get_selected_body()
                                    if body:
                                        self.temperature_input_text = f"{body.get('temperature', 5800):.0f}"
                                
                                self.spectral_class_dropdown_visible = False
                                self.spectral_class_dropdown_active = False
                            return True

                        elif self.radius_dropdown_visible:
                            if i < len(self.radius_dropdown_options):
                                name, value = self.radius_dropdown_options[i]
                                if value is not None:
                                    body = self.get_selected_body()
                                    if body: self.update_selected_body_property("radius", value, "radius")
                                else:
                                    self.show_custom_radius_input = True
                                    self.radius_input_active = True
                                self.radius_dropdown_selected = name
                                self.radius_dropdown_visible = False
                                self.radius_dropdown_active = False
                            return True

                        elif self.activity_dropdown_visible:
                            if i < len(self.activity_dropdown_options):
                                name, value = self.activity_dropdown_options[i]
                                if value is not None:
                                    self.update_selected_body_property("activity", value, "activity")
                                else:
                                    self.show_custom_activity_input = True
                                    self.activity_input_active = True
                                self.activity_dropdown_selected = name
                                self.activity_dropdown_visible = False
                                self.activity_dropdown_active = False
                            return True

                        elif self.metallicity_dropdown_visible:
                            if i < len(self.metallicity_dropdown_options):
                                name, value = self.metallicity_dropdown_options[i]
                                if value is not None:
                                    self.update_selected_body_property("metallicity", value, "metallicity")
                                else:
                                    self.show_custom_metallicity_input = True
                                    self.metallicity_input_active = True
                                self.metallicity_dropdown_selected = name
                                self.metallicity_dropdown_visible = False
                                self.metallicity_dropdown_active = False
                            return True

                        elif self.planet_orbital_distance_dropdown_visible:
                            if i < len(self.planet_orbital_distance_dropdown_options):
                                name, value = self.planet_orbital_distance_dropdown_options[i]
                                if value is not None:
                                    self.update_selected_body_property("semiMajorAxis", value, "semiMajorAxis")
                                    body = self.get_selected_body()
                                    if body: self.generate_orbit_grid(body)
                                    self.show_custom_orbital_distance_input = False
                                else:
                                    self.show_custom_orbital_distance_input = True
                                    self.orbital_distance_input_active = True
                                    self.orbital_distance_input_text = f"{self.selected_body.get('semiMajorAxis', 1.0):.2f}"
                                self.planet_orbital_distance_dropdown_selected = name
                                self.planet_orbital_distance_dropdown_visible = False
                                self.planet_orbital_distance_dropdown_active = False
                            return True

                        elif self.planet_orbital_eccentricity_dropdown_visible:
                            if i < len(self.planet_orbital_eccentricity_dropdown_options):
                                name, value = self.planet_orbital_eccentricity_dropdown_options[i]
                                if value is not None:
                                    self.update_selected_body_property("eccentricity", value, "eccentricity")
                                    self.show_custom_orbital_eccentricity_input = False
                                else:
                                    self.show_custom_orbital_eccentricity_input = True
                                    self.orbital_eccentricity_input_active = True
                                    self.orbital_eccentricity_input_text = f"{self.selected_body.get('eccentricity', 0.0167):.3f}"
                                self.planet_orbital_eccentricity_dropdown_selected = name
                                self.planet_orbital_eccentricity_dropdown_visible = False
                                self.planet_orbital_eccentricity_dropdown_active = False
                            return True

                        elif self.planet_orbital_period_dropdown_visible:
                            if i < len(self.planet_orbital_period_dropdown_options):
                                name, value = self.planet_orbital_period_dropdown_options[i]
                                if value is not None:
                                    self.update_selected_body_property("orbital_period", value, "orbital_period")
                                    self.show_custom_orbital_period_input = False
                                else:
                                    self.show_custom_orbital_period_input = True
                                    self.orbital_period_input_active = True
                                    self.orbital_period_input_text = f"{self.selected_body.get('orbital_period', 365.25):.0f}"
                                self.planet_orbital_period_dropdown_selected = name
                                self.planet_orbital_period_dropdown_visible = False
                                self.planet_orbital_period_dropdown_active = False
                            return True

        return False

    def render_dropdown(self):
        """Render the dropdown menu on top of everything else"""
        if self.planet_orbital_distance_dropdown_visible:
            print('DEBUG: render_dropdown called for orbital distance dropdown')
        if self.planet_stellar_flux_dropdown_visible:
            print('DEBUG: render_dropdown called for stellar flux dropdown')
        if (self.planet_dropdown_visible or self.moon_dropdown_visible or self.star_mass_dropdown_visible or 
            self.luminosity_dropdown_visible or self.planet_age_dropdown_visible or self.star_age_dropdown_visible or 
            self.moon_age_dropdown_visible or self.moon_radius_dropdown_visible or self.moon_orbital_distance_dropdown_visible or self.moon_orbital_period_dropdown_visible or self.moon_temperature_dropdown_visible or self.moon_gravity_dropdown_visible or self.spectral_class_dropdown_visible or 
            self.radius_dropdown_visible or self.activity_dropdown_visible or self.metallicity_dropdown_visible or 
            self.planet_radius_dropdown_visible or self.planet_temperature_dropdown_visible or 
            self.planet_atmosphere_dropdown_visible or self.planet_gravity_dropdown_visible or self.planet_orbital_distance_dropdown_visible or
            self.planet_orbital_eccentricity_dropdown_visible or self.planet_orbital_period_dropdown_visible or
            self.planet_stellar_flux_dropdown_visible or self.planet_density_dropdown_visible) and self.dropdown_surface:
            # Create a new surface that covers the entire screen
            overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))  # Semi-transparent dark background
            
            # Draw the dropdown surface on the overlay
            if self.dropdown_surface:
                overlay.blit(self.dropdown_surface, self.dropdown_rect)
            
            # Draw the overlay on the screen
            self.screen.blit(overlay, (0, 0))
    
    def render_planet_preset_dropdown(self):
        """Render the planet preset dropdown menu on top of everything else"""
        if self.planet_preset_dropdown_visible and self.active_tab == "planet":
            # Get the planet tab rect
            planet_tab_rect = self.tabs.get("planet")
            if planet_tab_rect:
                dropdown_width = 140
                dropdown_height = len(self.planet_preset_options) * 28
                self.planet_preset_dropdown_rect = pygame.Rect(
                    planet_tab_rect.right - dropdown_width,
                    planet_tab_rect.bottom + 2,
                    dropdown_width,
                    dropdown_height
                )
                
                # Create an overlay surface to ensure dropdown is on top
                overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
                overlay.fill((0, 0, 0, 0))  # Transparent background
                
                # Draw dropdown background with white fill on overlay
                pygame.draw.rect(overlay, self.WHITE, self.planet_preset_dropdown_rect)
                pygame.draw.rect(overlay, self.BLACK, self.planet_preset_dropdown_rect, 2)
                
                # Draw preset options on overlay
                for i, preset_name in enumerate(self.planet_preset_options):
                    option_rect = pygame.Rect(
                        self.planet_preset_dropdown_rect.left,
                        self.planet_preset_dropdown_rect.top + i * 28,
                        self.planet_preset_dropdown_rect.width,
                        28
                    )
                    # Check if this option is selected (for placement mode)
                    is_selected = False
                    if not self.selected_body or self.selected_body.get('type') != 'planet':
                        if self.planet_dropdown_selected == preset_name:
                            is_selected = True
                    
                    # Draw option background with highlighting if selected
                    if is_selected:
                        pygame.draw.rect(overlay, (200, 220, 255), option_rect)  # Light blue highlight
                    # Border is already drawn by the outer rect, but draw separator lines between options
                    if i > 0:
                        pygame.draw.line(overlay, self.BLACK, 
                                       (option_rect.left, option_rect.top), 
                                       (option_rect.right, option_rect.top), 1)
                    
                    text_surface = self.tab_font.render(preset_name, True, self.BLACK)
                    text_rect = text_surface.get_rect(midleft=(option_rect.left + 8, option_rect.centery))
                    overlay.blit(text_surface, text_rect)
                
                # Blit overlay onto screen (ensures it's on top)
                self.screen.blit(overlay, (0, 0))
                
                # Debug output (only print once per frame to avoid spam)
                if not hasattr(self, '_last_dropdown_debug') or self._last_dropdown_debug != self.planet_preset_dropdown_visible:
                    print(f"DEBUG: Rendering preset dropdown at {self.planet_preset_dropdown_rect}, visible={self.planet_preset_dropdown_visible}, active_tab={self.active_tab}, tab_rect={planet_tab_rect}")
                    self._last_dropdown_debug = self.planet_preset_dropdown_visible

    def render_simulation(self, engine: SimulationEngine):
        """Render the solar system simulation"""
        # Fill background with dark blue
        self.screen.fill(self.DARK_BLUE)
        
        # Draw white bar at the top
        top_bar_height = self.tab_height + 2*self.tab_margin
        top_bar_rect = pygame.Rect(0, 0, self.width, top_bar_height)
        pygame.draw.rect(self.screen, self.WHITE, top_bar_rect)
        
        # Draw tabs
        for tab_name, tab_rect in self.tabs.items():
            # Draw tab background
            if tab_name == self.active_tab:
                # Draw active tab with bright color and border
                pygame.draw.rect(self.screen, self.ACTIVE_TAB_COLOR, tab_rect, border_radius=5)
                pygame.draw.rect(self.screen, self.WHITE, tab_rect, 2, border_radius=5)
            else:
                pygame.draw.rect(self.screen, self.GRAY, tab_rect, border_radius=5)

            # Draw tab text
            tab_text = self.tab_font.render(tab_name.capitalize(), True, self.WHITE)
            tab_text_rect = tab_text.get_rect(center=tab_rect.center)
            self.screen.blit(tab_text, tab_text_rect)
            
            # Draw preset selector arrow for Planet tab (bottom-right corner)
            if tab_name == "planet":
                # Calculate arrow position in bottom-right of tab
                arrow_x = tab_rect.right - self.planet_preset_arrow_size - 3
                arrow_y = tab_rect.bottom - self.planet_preset_arrow_size - 3
                self.planet_preset_arrow_rect = pygame.Rect(
                    arrow_x - 2, arrow_y - 2,
                    self.planet_preset_arrow_size + 4, self.planet_preset_arrow_size + 4
                )
                
                # Draw small chevron/down arrow
                arrow_points = [
                    (arrow_x, arrow_y),
                    (arrow_x + self.planet_preset_arrow_size, arrow_y),
                    (arrow_x + self.planet_preset_arrow_size // 2, arrow_y + self.planet_preset_arrow_size)
                ]
                pygame.draw.polygon(self.screen, self.WHITE, arrow_points)

        # Draw customization panel only if a body is selected
        if self.show_customization_panel and self.selected_body:
            # Draw plain white panel
            pygame.draw.rect(self.screen, self.WHITE, self.customization_panel)
            
            # Draw close button (X)
            pygame.draw.rect(self.screen, self.BLACK, self.close_button, 2)
            # Draw X lines
            pygame.draw.line(self.screen, self.BLACK, 
                            (self.close_button.left + 5, self.close_button.top + 5),
                            (self.close_button.right - 5, self.close_button.bottom - 5), 2)
            pygame.draw.line(self.screen, self.BLACK, 
                            (self.close_button.left + 5, self.close_button.bottom - 5),
                            (self.close_button.right - 5, self.close_button.top + 5), 2)
            
            # Draw customization panel title
            title_text = self.font.render(f"Customize {self.selected_body['type'].capitalize()}", True, self.BLACK)
            title_rect = title_text.get_rect(center=(self.width - self.customization_panel_width//2, 50))
            self.screen.blit(title_text, title_rect)
            
            # Draw habitability potential at the top center (only for planets, not moons or stars)
            if self.selected_body and self.selected_body.get('type') == 'planet':
                # Shift right by 60 pixels
                habitability_text = self.font.render(f"Habitability Potential: {self.selected_body.get('habit_score', 0.0):.2f}%", True, (0, 255, 0))  # Green color
                habitability_rect = habitability_text.get_rect(center=(self.width//2 + 60, 30))
                self.screen.blit(habitability_text, habitability_rect)
            
            # --- STATUS BADGES & WARNINGS (CONSEQUENCES FIRST) ---
            badge_y = 75
            badge_x = self.width - self.customization_panel_width + 50
            
            if self.selected_body.get('type') == 'planet' or self.selected_body.get('type') == 'moon':
                # Info icon for climate/axial information (single shared icon above dropdowns)
                info_icon_size = 16
                self.climate_info_icon_rect = pygame.Rect(badge_x, badge_y, info_icon_size, info_icon_size)
                
                # Draw info icon (circle with 'i')
                pygame.draw.circle(self.screen, (100, 150, 255), self.climate_info_icon_rect.center, info_icon_size // 2, 1)
                info_text = self.tiny_font.render("i", True, (100, 150, 255))
                info_text_rect = info_text.get_rect(center=self.climate_info_icon_rect.center)
                self.screen.blit(info_text, info_text_rect)
                
                # Build tooltip content from current body state
                body = self.get_selected_body()
                if body:
                    stability = body.get("axial_stability", "Stable")
                    regime = body.get("climate_regime", "Stable")
                    var_pct = body.get("seasonal_flux_variation", 0.0)
                    temp_swing = body.get("seasonal_temp_swing", 0.0)
                    
                    tooltip_lines = [
                        f"Axial: {stability}",
                        f"Climate: {regime}",
                        f"Seasonal Flux: ±{var_pct:.1f}%",
                        f"Temp Swing: ±{temp_swing:.1f} K",
                        "Eccentric orbits amplify seasonal extremes."
                    ]
                    
                    # Show tooltip on hover or if pinned
                    mouse_pos = pygame.mouse.get_pos()
                    hovering_icon = self.climate_info_icon_rect.collidepoint(mouse_pos)
                    if hovering_icon or getattr(self, "climate_info_pinned", False):
                        self.active_tooltip = tooltip_lines
                
                # (Magnetosphere and internal heating warnings removed from UI per design:
                #  only dropdowns and sandbox visuals should respond to parameter changes.)
            
            # MASS SECTION (always show mass label/input, dropdown only for planets)
            if self.selected_body.get('type') == 'moon':
                mass_label = self.subtitle_font.render("Mass (M☾)", True, self.BLACK)
            elif self.selected_body.get('type') == 'star':
                mass_label = self.subtitle_font.render("Mass (M☉)", True, self.BLACK)
            else:
                mass_label = self.subtitle_font.render("Mass (M⊕)", True, self.BLACK)
            mass_label_rect = mass_label.get_rect(midleft=(self.width - self.customization_panel_width + 50, 105))
            self.screen.blit(mass_label, mass_label_rect)
            
            if self.selected_body.get('type') == 'planet':
                # For planets, show the planet dropdown
                pygame.draw.rect(self.screen, self.WHITE, self.planet_dropdown_rect, 2)
                pygame.draw.rect(self.screen, self.BLUE if self.planet_dropdown_active else self.GRAY, 
                               self.planet_dropdown_rect, 1)
                dropdown_text = "Select Reference Planet"
                
                body = self.get_selected_body()
                body_dropdown_selected = body.get("planet_dropdown_selected", self.planet_dropdown_selected) if body else self.planet_dropdown_selected

                if body_dropdown_selected:
                    selected = next(((name, value) for name, value in self.planet_dropdown_options if name == body_dropdown_selected), None)
                    if selected:
                        name, value = selected
                        if value is not None:
                            dropdown_text = f"{name} (" + self._format_value(value, 'M⊕') + ")"
                        else:
                            dropdown_text = name  # Custom
                    else:
                        dropdown_text = body_dropdown_selected # SHOW CUSTOM VALUE LABEL
                text_surface = self.subtitle_font.render(dropdown_text, True, self.BLACK)
                text_rect = text_surface.get_rect(midleft=(self.planet_dropdown_rect.left + 5, 
                                                         self.planet_dropdown_rect.centery))
                self.screen.blit(text_surface, text_rect)
            elif self.selected_body.get('type') == 'moon':
                # For moons, show the moon dropdown
                pygame.draw.rect(self.screen, self.WHITE, self.moon_dropdown_rect, 2)
                pygame.draw.rect(self.screen, self.BLUE if self.moon_dropdown_active else self.GRAY, 
                               self.moon_dropdown_rect, 1)
                dropdown_text = "Select Reference Moon"
                body = self.get_selected_body()
                body_dropdown_selected = body.get("moon_dropdown_selected", self.moon_dropdown_selected) if body else self.moon_dropdown_selected
                
                if body_dropdown_selected:
                    # Find the selected option's value
                    selected = next((option_data for option_data in self.moon_dropdown_options if option_data[0] == body_dropdown_selected), None)
                    if selected:
                        if len(selected) == 3:
                            name, value, unit = selected
                        else:
                            name, value = selected
                            unit = None
                        if value is not None:
                            dropdown_text = f"{name} ({self._format_value(value, 'M☾')})"
                        else:
                            dropdown_text = name  # Custom
                    else:
                        # Fallback: display the custom label directly
                        dropdown_text = body_dropdown_selected
                text_surface = self.subtitle_font.render(dropdown_text, True, self.BLACK)
                text_rect = text_surface.get_rect(midleft=(self.moon_dropdown_rect.left + 5, 
                                                         self.moon_dropdown_rect.centery))
                self.screen.blit(text_surface, text_rect)
            elif self.selected_body.get('type') == 'star':
                # For stars, show the star mass dropdown
                pygame.draw.rect(self.screen, self.WHITE, self.star_mass_dropdown_rect, 2)
                pygame.draw.rect(self.screen, self.BLUE if self.star_mass_dropdown_active else self.GRAY, 
                               self.star_mass_dropdown_rect, 1)
                dropdown_text = "Select Star Mass"
                
                body = self.get_selected_body()
                body_dropdown_selected = body.get("star_mass_dropdown_selected", self.star_mass_dropdown_selected) if body else self.star_mass_dropdown_selected
                
                if body_dropdown_selected:
                    selected = next(((name, value) for name, value in self.star_mass_dropdown_options if name == body_dropdown_selected), None)
                    if selected:
                        name, value = selected
                        dropdown_text = name
                    else:
                        dropdown_text = body_dropdown_selected

                if "hydrogen-burning limit" in dropdown_text:
                    # Special split rendering for Red Dwarf label
                    parts = dropdown_text.split("hydrogen-burning limit")
                    first_part = parts[0]
                    limit_text = "hydrogen-burning limit"
                    last_part = parts[1] if len(parts) > 1 else ""
                    
                    # Render first part
                    surf1 = self.subtitle_font.render(first_part, True, self.BLACK)
                    rect1 = surf1.get_rect(midleft=(self.star_mass_dropdown_rect.left + 5, self.star_mass_dropdown_rect.centery))
                    self.screen.blit(surf1, rect1)
                    
                    # Render limit text smaller
                    surf2 = self.tiny_font.render(limit_text, True, self.BLACK)
                    rect2 = surf2.get_rect(midleft=(rect1.right, self.star_mass_dropdown_rect.centery))
                    self.screen.blit(surf2, rect2)
                    
                    # Render last part (the closing parenthesis)
                    if last_part:
                        surf3 = self.subtitle_font.render(last_part, True, self.BLACK)
                        rect3 = surf3.get_rect(midleft=(rect2.right, self.star_mass_dropdown_rect.centery))
                        self.screen.blit(surf3, rect3)
                else:
                    text_surface = self.subtitle_font.render(dropdown_text, True, self.BLACK)
                    text_rect = text_surface.get_rect(midleft=(self.star_mass_dropdown_rect.left + 5, self.star_mass_dropdown_rect.centery))
                    self.screen.blit(text_surface, text_rect)
                
                # REMOVED: Inline custom mass input (now handled by modal)
            else:
                # For non-planets (moons/stars), show the mass input box
                # UNIFIED: Use modal for custom input everywhere
                pygame.draw.rect(self.screen, self.WHITE, self.mass_input_rect, 2)
                pygame.draw.rect(self.screen, self.BLUE if self.mass_input_active else self.GRAY, 
                               self.mass_input_rect, 1)
                
                body = self.get_selected_body()
                if body:
                    # Show value or label
                    label = body.get("mass_dropdown_selected" if body["type"] == "moon" else "star_mass_dropdown_selected")
                    if label:
                        text_surface = self.subtitle_font.render(label, True, self.BLACK)
                    else:
                        text_surface = self.subtitle_font.render(self._format_value(body.get('mass', 1.0), '', for_dropdown=False), True, self.BLACK)
                else:
                    text_surface = self.subtitle_font.render("N/A", True, self.BLACK)
                
                text_rect = text_surface.get_rect(midleft=(self.mass_input_rect.left + 5, self.mass_input_rect.centery))
                self.screen.blit(text_surface, text_rect)
            
            # AGE SECTION (moved up closer to mass section)
            age_label = self.subtitle_font.render("Age (Gyr)", True, self.BLACK)
            age_label_rect = age_label.get_rect(midleft=(self.width - self.customization_panel_width + 50, 165))
            self.screen.blit(age_label, age_label_rect)

            if self.selected_body.get('type') == 'planet':
                # For planets, show the age dropdown
                pygame.draw.rect(self.screen, self.WHITE, self.planet_age_dropdown_rect, 2)
                pygame.draw.rect(self.screen, self.BLUE if self.planet_age_dropdown_active else self.GRAY, 
                               self.planet_age_dropdown_rect, 1)
                dropdown_text = "Select Age"
                
                body = self.get_selected_body()
                body_dropdown_selected = body.get("planet_age_dropdown_selected", self.planet_age_dropdown_selected) if body else self.planet_age_dropdown_selected
                
                if body_dropdown_selected:
                    dropdown_text = body_dropdown_selected
                text_surface = self.subtitle_font.render(dropdown_text, True, self.BLACK)
                text_rect = text_surface.get_rect(midleft=(self.planet_age_dropdown_rect.left + 5, 
                                                         self.planet_age_dropdown_rect.centery))
                self.screen.blit(text_surface, text_rect)

                # Show custom age input if "Custom Age" is selected
                if self.show_custom_age_input:
                    custom_age_label = self.subtitle_font.render("Enter Custom Age (Gyr):", True, self.BLACK)
                    custom_age_label_rect = custom_age_label.get_rect(midleft=(self.width - self.customization_panel_width + 50, 225))
                    self.screen.blit(custom_age_label, custom_age_label_rect)
                    
                    pygame.draw.rect(self.screen, self.WHITE, self.age_input_rect, 2)
                    pygame.draw.rect(self.screen, self.BLUE if self.age_input_active else self.GRAY, 
                                   self.age_input_rect, 1)
                    if self.age_input_active:
                        text_surface = self.subtitle_font.render(self.age_input_text, True, self.BLACK)
                    else:
                        text_surface = self.subtitle_font.render(self._format_value(self.selected_body.get('age', 0.0), '', for_dropdown=False), True, self.BLACK)
                    text_rect = text_surface.get_rect(midleft=(self.age_input_rect.left + 5, 
                                                             self.age_input_rect.centery))
                    self.screen.blit(text_surface, text_rect)
            elif self.selected_body.get('type') == 'moon':
                # Info icon and climate tooltip are handled in the shared planet/moon section above

                # For moons, show the age dropdown
                pygame.draw.rect(self.screen, self.WHITE, self.moon_age_dropdown_rect, 2)
                pygame.draw.rect(self.screen, self.BLUE if self.moon_age_dropdown_active else self.GRAY, 
                               self.moon_age_dropdown_rect, 1)
                dropdown_text = "Select Age"
                if self.moon_age_dropdown_selected:
                    selected = next(((name, value) for name, value in self.moon_age_dropdown_options if name == self.moon_age_dropdown_selected), None)
                    if selected:
                        name, value = selected
                        if value is not None and "Gyr" not in name:
                            dropdown_text = f"{name} (" + self._format_value(value, 'Gyr') + ")"
                        else:
                            dropdown_text = name
                text_surface = self.subtitle_font.render(dropdown_text, True, self.BLACK)
                text_rect = text_surface.get_rect(midleft=(self.moon_age_dropdown_rect.left + 5, 
                                                         self.moon_age_dropdown_rect.centery))
                self.screen.blit(text_surface, text_rect)

                # REMOVED: Inline custom age input (now handled by modal)
                
                # RADIUS SECTION (only for moons)
                radius_label = self.subtitle_font.render("Radius (km)", True, self.BLACK)
                radius_label_rect = radius_label.get_rect(midleft=(self.width - self.customization_panel_width + 50, 225))
                self.screen.blit(radius_label, radius_label_rect)
                
                # Draw the moon radius dropdown
                pygame.draw.rect(self.screen, self.WHITE, self.moon_radius_dropdown_rect, 2)
                pygame.draw.rect(self.screen, self.BLUE if self.moon_radius_dropdown_active else self.GRAY, 
                               self.moon_radius_dropdown_rect, 1)
                dropdown_text = "Select Radius"
                
                body = self.get_selected_body()
                body_dropdown_selected = body.get("moon_radius_dropdown_selected", self.moon_radius_dropdown_selected) if body else self.moon_radius_dropdown_selected

                if body_dropdown_selected:
                    selected = next(((name, value) for name, value in self.moon_radius_dropdown_options if name == body_dropdown_selected), None)
                    if selected:
                        name, value = selected
                        if value is not None:
                            dropdown_text = f"{name} (" + self._format_value(value, 'km') + ")"
                        else:
                            dropdown_text = name
                    else:
                        dropdown_text = body_dropdown_selected
                text_surface = self.subtitle_font.render(dropdown_text, True, self.BLACK)
                text_rect = text_surface.get_rect(midleft=(self.moon_radius_dropdown_rect.left + 5, 
                                                         self.moon_radius_dropdown_rect.centery))
                self.screen.blit(text_surface, text_rect)

                # REMOVED: Inline custom radius input (now handled by modal)
                
                # ORBITAL DISTANCE SECTION (only for moons)
                orbital_distance_label = self.subtitle_font.render("Orbital Distance (km)", True, self.BLACK)
                orbital_distance_label_rect = orbital_distance_label.get_rect(midleft=(self.width - self.customization_panel_width + 50, 285))
                self.screen.blit(orbital_distance_label, orbital_distance_label_rect)
                
                # Draw the moon orbital distance dropdown
                pygame.draw.rect(self.screen, self.WHITE, self.moon_orbital_distance_dropdown_rect, 2)
                pygame.draw.rect(self.screen, self.BLUE if self.moon_orbital_distance_dropdown_active else self.GRAY, 
                               self.moon_orbital_distance_dropdown_rect, 1)
                dropdown_text = "Select Orbital Distance"
                
                body = self.get_selected_body()
                body_dropdown_selected = body.get("moon_orbital_distance_dropdown_selected", self.moon_orbital_distance_dropdown_selected) if body else self.moon_orbital_distance_dropdown_selected

                if body_dropdown_selected:
                    selected = next(((name, value) for name, value in self.moon_orbital_distance_dropdown_options if name == body_dropdown_selected), None)
                    if selected:
                        name, value = selected
                        if value is not None:
                            dropdown_text = f"{name} (" + self._format_value(value, 'km') + ")"
                        else:
                            dropdown_text = name
                    else:
                        dropdown_text = body_dropdown_selected
                text_surface = self.subtitle_font.render(dropdown_text, True, self.BLACK)
                text_rect = text_surface.get_rect(midleft=(self.moon_orbital_distance_dropdown_rect.left + 5, 
                                                         self.moon_orbital_distance_dropdown_rect.centery))
                self.screen.blit(text_surface, text_rect)
                
                # REMOVED: Inline custom orbital distance input (now handled by modal)
                
                # ORBITAL PERIOD SECTION (only for moons)
                orbital_period_label = self.subtitle_font.render("Orbital Period (days)", True, self.BLACK)
                orbital_period_label_rect = orbital_period_label.get_rect(midleft=(self.width - self.customization_panel_width + 50, 345))
                self.screen.blit(orbital_period_label, orbital_period_label_rect)
                
                # Draw the moon orbital period dropdown
                pygame.draw.rect(self.screen, self.WHITE, self.moon_orbital_period_dropdown_rect, 2)
                pygame.draw.rect(self.screen, self.BLUE if self.moon_orbital_period_dropdown_active else self.GRAY, 
                               self.moon_orbital_period_dropdown_rect, 1)
                dropdown_text = "Select Orbital Period"
                
                body = self.get_selected_body()
                body_dropdown_selected = body.get("moon_orbital_period_dropdown_selected", self.moon_orbital_period_dropdown_selected) if body else self.moon_orbital_period_dropdown_selected

                if body_dropdown_selected:
                    selected = next(((name, value) for name, value in self.moon_orbital_period_dropdown_options if name == body_dropdown_selected), None)
                    if selected:
                        name, value = selected
                        if value is not None:
                            dropdown_text = f"{name} (" + self._format_value(value, 'days') + ")"
                        else:
                            dropdown_text = name
                    else:
                        dropdown_text = body_dropdown_selected
                text_surface = self.subtitle_font.render(dropdown_text, True, self.BLACK)
                text_rect = text_surface.get_rect(midleft=(self.moon_orbital_period_dropdown_rect.left + 5, 
                                                         self.moon_orbital_period_dropdown_rect.centery))
                self.screen.blit(text_surface, text_rect)
                
                # Show custom orbital period input if "Custom" is selected
                if self.show_custom_moon_orbital_period_input:
                    custom_period_label = self.subtitle_font.render("Enter Custom Period (days):", True, self.BLACK)
                    custom_period_label_rect = custom_period_label.get_rect(midleft=(self.width - self.customization_panel_width + 50, 405))
                    self.screen.blit(custom_period_label, custom_period_label_rect)
                    
                    custom_period_input_rect = pygame.Rect(self.width - self.customization_panel_width + 50, 435, self.customization_panel_width - 100, 30)
                    pygame.draw.rect(self.screen, self.WHITE, custom_period_input_rect, 2)
                    pygame.draw.rect(self.screen, self.BLUE, custom_period_input_rect, 1)
                    text_surface = self.subtitle_font.render(self.orbital_period_input_text, True, self.BLACK)
                    text_rect = text_surface.get_rect(midleft=(custom_period_input_rect.left + 5, custom_period_input_rect.centery))
                    self.screen.blit(text_surface, text_rect)
                
                # SURFACE TEMPERATURE SECTION (only for moons)
                temperature_label = self.subtitle_font.render("Surface Temperature (K)", True, self.BLACK)
                temperature_label_rect = temperature_label.get_rect(midleft=(self.width - self.customization_panel_width + 50, 405))
                self.screen.blit(temperature_label, temperature_label_rect)
                
                # Draw the moon temperature dropdown
                pygame.draw.rect(self.screen, self.WHITE, self.moon_temperature_dropdown_rect, 2)
                pygame.draw.rect(self.screen, self.BLUE if self.moon_temperature_dropdown_active else self.GRAY, 
                               self.moon_temperature_dropdown_rect, 1)
                dropdown_text = "Select Temperature"
                
                body = self.get_selected_body()
                body_dropdown_selected = body.get("moon_temperature_dropdown_selected", self.moon_temperature_dropdown_selected) if body else self.moon_temperature_dropdown_selected

                if body_dropdown_selected:
                    selected = next(((name, value) for name, value in self.moon_temperature_dropdown_options if name == body_dropdown_selected), None)
                    if selected:
                        name, value = selected
                        if value is not None:
                            dropdown_text = f"{name} (" + self._format_value(value, 'K') + ")"
                        else:
                            dropdown_text = name
                    else:
                        dropdown_text = body_dropdown_selected
                text_surface = self.subtitle_font.render(dropdown_text, True, self.BLACK)
                text_rect = text_surface.get_rect(midleft=(self.moon_temperature_dropdown_rect.left + 5, 
                                                         self.moon_temperature_dropdown_rect.centery))
                self.screen.blit(text_surface, text_rect)

                # --- DERIVED BREAKDOWNS REMOVED (Moved to Info Panel) ---
                moon_temp_push_y = 0
                # ---------------------------------------------------------

                # Show custom temperature input if "Custom" is selected
                if self.show_custom_moon_temperature_input:
                    custom_temp_label = self.subtitle_font.render("Enter Custom Temperature (K):", True, self.BLACK)
                    custom_temp_label_rect = custom_temp_label.get_rect(midleft=(self.width - self.customization_panel_width + 50, 465))
                    self.screen.blit(custom_temp_label, custom_temp_label_rect)
                    
                    custom_temp_input_rect = pygame.Rect(self.width - self.customization_panel_width + 50, 495, self.customization_panel_width - 100, 30)
                    pygame.draw.rect(self.screen, self.WHITE, custom_temp_input_rect, 2)
                    pygame.draw.rect(self.screen, self.BLUE, custom_temp_input_rect, 1)
                    text_surface = self.subtitle_font.render(self.planet_temperature_input_text, True, self.BLACK)
                    text_rect = text_surface.get_rect(midleft=(custom_temp_input_rect.left + 5, custom_temp_input_rect.centery))
                    self.screen.blit(text_surface, text_rect)
                
                # SURFACE GRAVITY SECTION (only for moons)
                gravity_label = self.subtitle_font.render("Surface Gravity (m/s²)", True, self.BLACK)
                gravity_label_rect = gravity_label.get_rect(midleft=(self.width - self.customization_panel_width + 50, 465 + moon_temp_push_y))
                self.screen.blit(gravity_label, gravity_label_rect)
                
                # Draw the moon gravity dropdown
                self.moon_gravity_dropdown_rect.y = 480 + moon_temp_push_y
                pygame.draw.rect(self.screen, self.WHITE, self.moon_gravity_dropdown_rect, 2)
                pygame.draw.rect(self.screen, self.BLUE if self.moon_gravity_dropdown_active else self.GRAY, 
                               self.moon_gravity_dropdown_rect, 1)
                dropdown_text = "Select Gravity"
                
                body = self.get_selected_body()
                body_dropdown_selected = body.get("moon_gravity_dropdown_selected", self.moon_gravity_dropdown_selected) if body else self.moon_gravity_dropdown_selected

                if body_dropdown_selected:
                    selected = next(((name, value) for name, value in self.moon_gravity_dropdown_options if name == body_dropdown_selected), None)
                    if selected:
                        name, value = selected
                        if value is not None:
                            dropdown_text = f"{name} (" + self._format_value(value, 'm/s²') + ")"
                        else:
                            dropdown_text = name
                    else:
                        dropdown_text = body_dropdown_selected
                text_surface = self.subtitle_font.render(dropdown_text, True, self.BLACK)
                text_rect = text_surface.get_rect(midleft=(self.moon_gravity_dropdown_rect.left + 5, 
                                                         self.moon_gravity_dropdown_rect.centery))
                self.screen.blit(text_surface, text_rect)

                # Show custom gravity input if "Custom" is selected
                if self.show_custom_moon_gravity_input:
                    custom_gravity_label = self.subtitle_font.render("Enter Custom Gravity (m/s²):", True, self.BLACK)
                    custom_gravity_label_rect = custom_gravity_label.get_rect(midleft=(self.width - self.customization_panel_width + 50, 525 + moon_temp_push_y))
                    self.screen.blit(custom_gravity_label, custom_gravity_label_rect)
                    
                    custom_gravity_input_rect = pygame.Rect(self.width - self.customization_panel_width + 50, 555 + moon_temp_push_y, self.customization_panel_width - 100, 30)
                    pygame.draw.rect(self.screen, self.WHITE, custom_gravity_input_rect, 2)
                    pygame.draw.rect(self.screen, self.BLUE, custom_gravity_input_rect, 1)
                    text_surface = self.subtitle_font.render(self.moon_gravity_input_text, True, self.BLACK)
                    text_rect = text_surface.get_rect(midleft=(custom_gravity_input_rect.left + 5, custom_gravity_input_rect.centery))
                    self.screen.blit(text_surface, text_rect)
            else:
                # Stars handled in star section below
                pass
            
            # RADIUS SECTION (only for planets)
            if self.selected_body.get('type') == 'planet':
                radius_label = self.subtitle_font.render("Radius (R🜨)", True, self.BLACK)
                radius_label_rect = radius_label.get_rect(midleft=(self.width - self.customization_panel_width + 50, 225))
                self.screen.blit(radius_label, radius_label_rect)
                
                # Draw the planet radius dropdown
                pygame.draw.rect(self.screen, self.WHITE, self.planet_radius_dropdown_rect, 2)
                pygame.draw.rect(self.screen, self.BLUE if self.planet_radius_dropdown_active else self.GRAY, 
                               self.planet_radius_dropdown_rect, 1)
                dropdown_text = "Select Reference Planet"
                
                body = self.get_selected_body()
                body_dropdown_selected = body.get("planet_radius_dropdown_selected", self.planet_radius_dropdown_selected) if body else self.planet_radius_dropdown_selected

                if body_dropdown_selected:
                    selected = next(((name, value) for name, value in self.planet_radius_dropdown_options if name == body_dropdown_selected), None)
                    if selected:
                        name, value = selected
                        if value is not None:
                            dropdown_text = f"{name} (" + self._format_value(value, 'R🜨') + ")"
                        else:
                            dropdown_text = name
                    else:
                        dropdown_text = body_dropdown_selected
                text_surface = self.subtitle_font.render(dropdown_text, True, self.BLACK)
                text_rect = text_surface.get_rect(midleft=(self.planet_radius_dropdown_rect.left + 5, 
                                                         self.planet_radius_dropdown_rect.centery))
                self.screen.blit(text_surface, text_rect)
                
                # Show custom radius input if "Custom" is selected, just below dropdown
                if self.show_custom_radius_input and self.selected_body.get('type') == 'planet':
                    custom_radius_label = self.subtitle_font.render("Enter Custom Radius (R⊕):", True, self.BLACK)
                    custom_radius_label_rect = custom_radius_label.get_rect(midleft=(self.width - self.customization_panel_width + 50, 275))
                    self.screen.blit(custom_radius_label, custom_radius_label_rect)
                    
                    custom_radius_input_rect = pygame.Rect(self.width - self.customization_panel_width + 50, 305, self.customization_panel_width - 100, 30)
                    pygame.draw.rect(self.screen, self.WHITE, custom_radius_input_rect, 2)
                    pygame.draw.rect(self.screen, self.BLUE if self.radius_input_active else self.GRAY,
                                   custom_radius_input_rect, 1)
                    text_surface = self.subtitle_font.render(self.radius_input_text, True, self.BLACK)
                    text_rect = text_surface.get_rect(midleft=(custom_radius_input_rect.left + 5,
                                                             custom_radius_input_rect.centery))
                    self.screen.blit(text_surface, text_rect)
            
            # TEMPERATURE SECTION (only for planets)
            if self.selected_body.get('type') == 'planet':
                temperature_label = self.subtitle_font.render("Temperature (K)", True, self.BLACK)
                temperature_label_rect = temperature_label.get_rect(midleft=(self.width - self.customization_panel_width + 50, 285))
                self.screen.blit(temperature_label, temperature_label_rect)
                
                # Draw the planet temperature dropdown
                pygame.draw.rect(self.screen, self.WHITE, self.planet_temperature_dropdown_rect, 2)
                pygame.draw.rect(self.screen, self.BLUE if self.planet_temperature_dropdown_active else self.GRAY, 
                               self.planet_temperature_dropdown_rect, 1)
                dropdown_text = "Select Reference Planet"
                
                body = self.get_selected_body()
                body_dropdown_selected = body.get("planet_temperature_dropdown_selected", self.planet_temperature_dropdown_selected) if body else self.planet_temperature_dropdown_selected

                if body_dropdown_selected:
                    selected = next(((name, value) for name, value in self.planet_temperature_dropdown_options if name == body_dropdown_selected), None)
                    if selected:
                        name, value = selected
                        if value is not None:
                            dropdown_text = f"{name} (" + self._format_value(value, 'K') + ")"
                        else:
                            dropdown_text = name
                    else:
                        dropdown_text = body_dropdown_selected
                text_surface = self.subtitle_font.render(dropdown_text, True, self.BLACK)
                text_rect = text_surface.get_rect(midleft=(self.planet_temperature_dropdown_rect.left + 5, 
                                                         self.planet_temperature_dropdown_rect.centery))
                self.screen.blit(text_surface, text_rect)

                # --- DERIVED BREAKDOWNS REMOVED (Moved to Info Panel) ---
                temp_push_y = 0
                # --------------------------------------------------------

                # ATMOSPHERIC COMPOSITION / GREENHOUSE TYPE SECTION (only for planets)
                atmosphere_label = self.subtitle_font.render("Atmospheric Composition / Greenhouse Type", True, self.BLACK)
                atmosphere_label_rect = atmosphere_label.get_rect(midleft=(self.width - self.customization_panel_width + 50, 345 + temp_push_y))
                self.screen.blit(atmosphere_label, atmosphere_label_rect)
                
                # Draw the planet atmospheric composition dropdown
                self.planet_atmosphere_dropdown_rect.y = 360 + temp_push_y
                pygame.draw.rect(self.screen, self.WHITE, self.planet_atmosphere_dropdown_rect, 2)
                pygame.draw.rect(self.screen, self.BLUE if self.planet_atmosphere_dropdown_active else self.GRAY, 
                               self.planet_atmosphere_dropdown_rect, 1)
                dropdown_text = "Select Atmospheric Composition"
                
                body = self.get_selected_body()
                body_dropdown_selected = body.get("planet_atmosphere_dropdown_selected", self.planet_atmosphere_dropdown_selected) if body else self.planet_atmosphere_dropdown_selected

                if body_dropdown_selected:
                    selected = next(((name, value) for name, value in self.planet_atmosphere_dropdown_options if name == body_dropdown_selected), None)
                    if selected:
                        name, value = selected
                        dropdown_text = name
                    else:
                        dropdown_text = body_dropdown_selected
                text_surface = self.subtitle_font.render(dropdown_text, True, self.BLACK)
                text_rect = text_surface.get_rect(midleft=(self.planet_atmosphere_dropdown_rect.left + 5, 
                                                         self.planet_atmosphere_dropdown_rect.centery))
                self.screen.blit(text_surface, text_rect)

                # --- MAGNETIC FIELD & SHIELDING REMOVED (Moved to Info Panel) ---
                # Warnings are handled in the status badges section at the top
                temp_push_y = 0
                
                # Show custom atmosphere input if "Custom" is selected, just below dropdown
                if self.show_custom_atmosphere_input:
                    custom_atmosphere_label = self.subtitle_font.render("Enter Custom ΔT (K):", True, self.BLACK)
                    custom_atmosphere_label_rect = custom_atmosphere_label.get_rect(midleft=(self.width - self.customization_panel_width + 50, 415))
                    self.screen.blit(custom_atmosphere_label, custom_atmosphere_label_rect)
                    
                    custom_atmosphere_input_rect = pygame.Rect(self.width - self.customization_panel_width + 50, 445, self.customization_panel_width - 100, 30)
                    pygame.draw.rect(self.screen, self.WHITE, custom_atmosphere_input_rect, 2)
                    pygame.draw.rect(self.screen, self.BLUE, custom_atmosphere_input_rect, 1)
                    text_surface = self.subtitle_font.render(self.planet_atmosphere_input_text, True, self.BLACK)
                    text_rect = text_surface.get_rect(midleft=(custom_atmosphere_input_rect.left + 5, custom_atmosphere_input_rect.centery))
                    self.screen.blit(text_surface, text_rect)

                # GRAVITY SECTION (only for planets)
                gravity_label = self.subtitle_font.render("Surface Gravity (m/s²)", True, self.BLACK)
                gravity_label_rect = gravity_label.get_rect(midleft=(self.width - self.customization_panel_width + 50, 405 + temp_push_y))
                self.screen.blit(gravity_label, gravity_label_rect)
                self.planet_gravity_dropdown_rect.y = 420 + temp_push_y
                pygame.draw.rect(self.screen, self.WHITE, self.planet_gravity_dropdown_rect, 2)
                pygame.draw.rect(self.screen, self.BLUE if self.planet_gravity_dropdown_active else self.GRAY, self.planet_gravity_dropdown_rect, 1)
                dropdown_text = "Select Reference Planet"
                
                body = self.get_selected_body()
                body_dropdown_selected = body.get("planet_gravity_dropdown_selected", self.planet_gravity_dropdown_selected) if body else self.planet_gravity_dropdown_selected

                if body_dropdown_selected:
                    selected = next(((name, value) for name, value in self.planet_gravity_dropdown_options if name == body_dropdown_selected), None)
                    if selected:
                        name, value = selected
                        if value is not None:
                            dropdown_text = f"{name} (" + self._format_value(value, 'm/s²') + ")"
                        else:
                            dropdown_text = name
                    else:
                        dropdown_text = body_dropdown_selected
                text_surface = self.subtitle_font.render(dropdown_text, True, self.BLACK)
                text_rect = text_surface.get_rect(midleft=(self.planet_gravity_dropdown_rect.left + 5, self.planet_gravity_dropdown_rect.centery))
                self.screen.blit(text_surface, text_rect)
                
                # Show custom gravity input if "Custom" is selected, just below dropdown
                if self.show_custom_planet_gravity_input:
                    custom_gravity_label = self.subtitle_font.render("Enter Custom Gravity (m/s²):", True, self.BLACK)
                    custom_gravity_label_rect = custom_gravity_label.get_rect(midleft=(self.width - self.customization_panel_width + 50, 415 + temp_push_y))
                    self.screen.blit(custom_gravity_label, custom_gravity_label_rect)
                    
                    custom_gravity_input_rect = pygame.Rect(self.width - self.customization_panel_width + 50, 445 + temp_push_y, self.customization_panel_width - 100, 30)
                    pygame.draw.rect(self.screen, self.WHITE, custom_gravity_input_rect, 2)
                    pygame.draw.rect(self.screen, self.BLUE, custom_gravity_input_rect, 1)
                    text_surface = self.subtitle_font.render(self.planet_gravity_input_text, True, self.BLACK)
                    text_rect = text_surface.get_rect(midleft=(custom_gravity_input_rect.left + 5, custom_gravity_input_rect.centery))
                    self.screen.blit(text_surface, text_rect)
                # Orbital Distance (AU) dropdown
                orbital_distance_label = self.subtitle_font.render("Orbital Distance (AU)", True, self.BLACK)
                orbital_distance_label_rect = orbital_distance_label.get_rect(midleft=(self.width - self.customization_panel_width + 50, 465 + temp_push_y))
                self.screen.blit(orbital_distance_label, orbital_distance_label_rect)
                self.planet_orbital_distance_dropdown_rect.y = 480 + temp_push_y
                pygame.draw.rect(self.screen, self.WHITE, self.planet_orbital_distance_dropdown_rect, 2)
                pygame.draw.rect(self.screen, self.BLUE if self.planet_orbital_distance_dropdown_active else self.GRAY, self.planet_orbital_distance_dropdown_rect, 1)
                dropdown_text = "Select Orbital Distance"
                
                body = self.get_selected_body()
                body_dropdown_selected = body.get("planet_orbital_distance_dropdown_selected", self.planet_orbital_distance_dropdown_selected) if body else self.planet_orbital_distance_dropdown_selected

                if body_dropdown_selected:
                    selected = next(((name, value) for name, value in self.planet_orbital_distance_dropdown_options if name == body_dropdown_selected), None)
                    if selected:
                        name, value = selected
                        if value is not None:
                            dropdown_text = f"{name} (" + self._format_value(value, 'AU') + ")"
                        else:
                            dropdown_text = name
                    else:
                        dropdown_text = body_dropdown_selected
                text_surface = self.subtitle_font.render(dropdown_text, True, self.BLACK)
                text_rect = text_surface.get_rect(midleft=(self.planet_orbital_distance_dropdown_rect.left + 5, self.planet_orbital_distance_dropdown_rect.centery))
                self.screen.blit(text_surface, text_rect)
                
                # Show custom orbital distance input if "Custom" is selected, just below dropdown
                if self.show_custom_orbital_distance_input:
                    custom_orbital_label = self.subtitle_font.render("Enter Custom Distance (AU):", True, self.BLACK)
                    custom_orbital_label_rect = custom_orbital_label.get_rect(midleft=(self.width - self.customization_panel_width + 50, 455 + temp_push_y))
                    self.screen.blit(custom_orbital_label, custom_orbital_label_rect)
                    custom_orbital_input_rect = pygame.Rect(self.width - self.customization_panel_width + 50, 485 + temp_push_y, self.customization_panel_width - 100, 30)
                    pygame.draw.rect(self.screen, self.WHITE, custom_orbital_input_rect, 2)
                    pygame.draw.rect(self.screen, self.BLUE, custom_orbital_input_rect, 1)
                    text_surface = self.subtitle_font.render(self.orbital_distance_input_text, True, self.BLACK)
                    text_rect = text_surface.get_rect(midleft=(custom_orbital_input_rect.left + 5, custom_orbital_input_rect.centery))
                    self.screen.blit(text_surface, text_rect)
                # Orbital Eccentricity dropdown
                orbital_eccentricity_label = self.subtitle_font.render("Orbital Eccentricity", True, self.BLACK)
                orbital_eccentricity_label_rect = orbital_eccentricity_label.get_rect(midleft=(self.width - self.customization_panel_width + 50, 525 + temp_push_y))
                self.screen.blit(orbital_eccentricity_label, orbital_eccentricity_label_rect)
                self.planet_orbital_eccentricity_dropdown_rect.y = 540 + temp_push_y
                pygame.draw.rect(self.screen, self.WHITE, self.planet_orbital_eccentricity_dropdown_rect, 2)
                pygame.draw.rect(self.screen, self.BLUE if self.planet_orbital_eccentricity_dropdown_active else self.GRAY, self.planet_orbital_eccentricity_dropdown_rect, 1)

                dropdown_text = "Select Orbital Eccentricity"
                
                body = self.get_selected_body()
                body_dropdown_selected = body.get("planet_orbital_eccentricity_dropdown_selected", self.planet_orbital_eccentricity_dropdown_selected) if body else self.planet_orbital_eccentricity_dropdown_selected

                if body_dropdown_selected:
                    selected = next(((name, value) for name, value in self.planet_orbital_eccentricity_dropdown_options if name == body_dropdown_selected), None)
                    if selected:
                        name, value = selected
                        if value is not None:
                            dropdown_text = f"{name} (" + self._format_value(value, '') + ")"
                        else:
                            dropdown_text = name
                    else:
                        dropdown_text = body_dropdown_selected
                text_surface = self.subtitle_font.render(dropdown_text, True, self.BLACK)
                text_rect = text_surface.get_rect(midleft=(self.planet_orbital_eccentricity_dropdown_rect.left + 5, self.planet_orbital_eccentricity_dropdown_rect.centery))
                self.screen.blit(text_surface, text_rect)

                # Show custom orbital eccentricity input if "Custom" is selected, just below dropdown
                if self.show_custom_orbital_eccentricity_input:
                    custom_eccentricity_label = self.subtitle_font.render("Enter Custom Eccentricity:", True, self.BLACK)
                    custom_eccentricity_label_rect = custom_eccentricity_label.get_rect(midleft=(self.width - self.customization_panel_width + 50, 515))
                    self.screen.blit(custom_eccentricity_label, custom_eccentricity_label_rect)
                    custom_eccentricity_input_rect = pygame.Rect(self.width - self.customization_panel_width + 50, 545, self.customization_panel_width - 100, 30)
                    pygame.draw.rect(self.screen, self.WHITE, custom_eccentricity_input_rect, 2)
                    pygame.draw.rect(self.screen, self.BLUE, custom_eccentricity_input_rect, 1)
                    text_surface = self.subtitle_font.render(self.orbital_eccentricity_input_text, True, self.BLACK)
                    text_rect = text_surface.get_rect(midleft=(custom_eccentricity_input_rect.left + 5, custom_eccentricity_input_rect.centery))
                    self.screen.blit(text_surface, text_rect)
                
                # Orbital Period dropdown
                orbital_period_label = self.subtitle_font.render("Orbital Period (days)", True, self.BLACK)
                orbital_period_label_rect = orbital_period_label.get_rect(midleft=(self.width - self.customization_panel_width + 50, 585 + temp_push_y))
                self.screen.blit(orbital_period_label, orbital_period_label_rect)
                self.planet_orbital_period_dropdown_rect.y = 600 + temp_push_y
                pygame.draw.rect(self.screen, self.WHITE, self.planet_orbital_period_dropdown_rect, 2)
                pygame.draw.rect(self.screen, self.BLUE if self.planet_orbital_period_dropdown_active else self.GRAY, self.planet_orbital_period_dropdown_rect, 1)
                dropdown_text = "Select Orbital Period"
                
                body = self.get_selected_body()
                body_dropdown_selected = body.get("planet_orbital_period_dropdown_selected", self.planet_orbital_period_dropdown_selected) if body else self.planet_orbital_period_dropdown_selected

                if body_dropdown_selected:
                    selected = next(((name, value) for name, value in self.planet_orbital_period_dropdown_options if name == body_dropdown_selected), None)
                    if selected:
                        name, value = selected
                        if value is not None:
                            dropdown_text = f"{name} (" + self._format_value(value, 'days', allow_scientific=False) + ")"
                        else:
                            dropdown_text = name
                    else:
                        dropdown_text = body_dropdown_selected
                text_surface = self.subtitle_font.render(dropdown_text, True, self.BLACK)
                text_rect = text_surface.get_rect(midleft=(self.planet_orbital_period_dropdown_rect.left + 5, self.planet_orbital_period_dropdown_rect.centery))
                self.screen.blit(text_surface, text_rect)
                
                # REMOVED: Inline custom orbital period input (now handled by modal)
                # Stellar Flux dropdown
                stellar_flux_label = self.subtitle_font.render("Stellar Flux (Earth Units)", True, self.BLACK)
                stellar_flux_label_rect = stellar_flux_label.get_rect(midleft=(self.width - self.customization_panel_width + 50, 645 + temp_push_y))
                self.screen.blit(stellar_flux_label, stellar_flux_label_rect)
                self.planet_stellar_flux_dropdown_rect.y = 660 + temp_push_y
                pygame.draw.rect(self.screen, self.WHITE, self.planet_stellar_flux_dropdown_rect, 2)
                pygame.draw.rect(self.screen, self.BLUE if self.planet_stellar_flux_dropdown_active else self.GRAY, self.planet_stellar_flux_dropdown_rect, 1)
                dropdown_text = "Select Stellar Flux"
                
                body = self.get_selected_body()
                body_dropdown_selected = body.get("planet_stellar_flux_dropdown_selected", self.planet_stellar_flux_dropdown_selected) if body else self.planet_stellar_flux_dropdown_selected

                if body_dropdown_selected:
                    selected = next(((name, value) for name, value in self.planet_stellar_flux_dropdown_options if name == body_dropdown_selected), None)
                    if selected:
                        name, value = selected
                        if value is not None:
                            dropdown_text = f"{name} (" + self._format_value(value, 'EFU') + ")"
                        else:
                            dropdown_text = name
                    else:
                        dropdown_text = body_dropdown_selected
                text_surface = self.subtitle_font.render(dropdown_text, True, self.BLACK)
                text_rect = text_surface.get_rect(midleft=(self.planet_stellar_flux_dropdown_rect.left + 5, self.planet_stellar_flux_dropdown_rect.centery))
                self.screen.blit(text_surface, text_rect)
                # Show custom stellar flux input if "Custom" is selected, just below dropdown
                if self.show_custom_stellar_flux_input:
                    custom_flux_label = self.subtitle_font.render("Enter Custom Flux (EFU):", True, self.BLACK)
                    custom_flux_label_rect = custom_flux_label.get_rect(midleft=(self.width - self.customization_panel_width + 50, 685 + temp_push_y))
                    self.screen.blit(custom_flux_label, custom_flux_label_rect)
                    custom_flux_input_rect = pygame.Rect(self.width - self.customization_panel_width + 50, 715 + temp_push_y, self.customization_panel_width - 100, 30)
                    pygame.draw.rect(self.screen, self.WHITE, custom_flux_input_rect, 2)
                    pygame.draw.rect(self.screen, self.BLUE, custom_flux_input_rect, 1)
                    text_surface = self.subtitle_font.render(self.stellar_flux_input_text, True, self.BLACK)
                    text_rect = text_surface.get_rect(midleft=(custom_flux_input_rect.left + 5, custom_flux_input_rect.centery))
                    self.screen.blit(text_surface, text_rect)
                # Density dropdown (NEW)
                density_label = self.subtitle_font.render("Density (g/cm³)", True, self.BLACK)
                density_label_rect = density_label.get_rect(midleft=(self.width - self.customization_panel_width + 50, 705 + temp_push_y))
                self.screen.blit(density_label, density_label_rect)
                self.planet_density_dropdown_rect.y = 720 + temp_push_y
                pygame.draw.rect(self.screen, self.WHITE, self.planet_density_dropdown_rect, 2)
                pygame.draw.rect(self.screen, self.BLUE if self.planet_density_dropdown_active else self.GRAY, self.planet_density_dropdown_rect, 1)
                dropdown_text = "Select Density"
                
                # CRITICAL: Use per-body selection if available
                body = self.get_selected_body()
                body_dropdown_selected = body.get("planet_density_dropdown_selected") if body else None
                if not body_dropdown_selected:
                    body_dropdown_selected = self.planet_density_dropdown_selected

                if body_dropdown_selected:
                    selected = next(((name, value) for name, value in self.planet_density_dropdown_options if name == body_dropdown_selected), None)
                    if selected:
                        name, value = selected
                        if value is not None:
                            dropdown_text = f"{name} (" + self._format_value(value, 'g/cm³') + ")"
                        else:
                            dropdown_text = name
                    else:
                        dropdown_text = body_dropdown_selected
                text_surface = self.subtitle_font.render(dropdown_text, True, self.BLACK)
                text_rect = text_surface.get_rect(midleft=(self.planet_density_dropdown_rect.left + 5, self.planet_density_dropdown_rect.centery))
                self.screen.blit(text_surface, text_rect)
                # Show custom density input if "Custom" is selected, just below dropdown
                if self.show_custom_planet_density_input:
                    custom_density_label = self.subtitle_font.render("Enter Custom Density (g/cm³):", True, self.BLACK)
                    custom_density_label_rect = custom_density_label.get_rect(midleft=(self.width - self.customization_panel_width + 50, 745 + temp_push_y))
                    self.screen.blit(custom_density_label, custom_density_label_rect)
                    custom_density_input_rect = pygame.Rect(self.width - self.customization_panel_width + 50, 775 + temp_push_y, self.customization_panel_width - 100, 30)
                    pygame.draw.rect(self.screen, self.WHITE, custom_density_input_rect, 2)
                    pygame.draw.rect(self.screen, self.BLUE, custom_density_input_rect, 1)
                    text_surface = self.subtitle_font.render(self.planet_density_input_text, True, self.BLACK)
                    text_rect = text_surface.get_rect(midleft=(custom_density_input_rect.left + 5, custom_density_input_rect.centery))
                    self.screen.blit(text_surface, text_rect)
            # Draw spectral class dropdown for stars (merged from spectral and temperature dropdowns)
            if self.selected_body and self.selected_body.get('type') == 'star':
                spectral_class_label = self.subtitle_font.render("Spectral Class (Temperature)", True, self.BLACK)
                spectral_class_label_rect = spectral_class_label.get_rect(midleft=(self.width - self.customization_panel_width + 50, 225))
                self.screen.blit(spectral_class_label, spectral_class_label_rect)
                pygame.draw.rect(self.screen, self.WHITE, self.spectral_class_dropdown_rect, 2)
                pygame.draw.rect(self.screen, self.BLUE if self.spectral_class_dropdown_active else self.GRAY, self.spectral_class_dropdown_rect, 1)
                dropdown_text = "Select Spectral Class"
                if self.spectral_class_dropdown_selected:
                    dropdown_text = self.spectral_class_dropdown_selected
                text_surface = self.subtitle_font.render(dropdown_text, True, self.BLACK)
                text_rect = text_surface.get_rect(midleft=(self.spectral_class_dropdown_rect.left + 5, self.spectral_class_dropdown_rect.centery))
                self.screen.blit(text_surface, text_rect)

                # Draw custom temperature input if "Custom" is selected
                if self.show_custom_temperature_input:
                    custom_temp_label = self.subtitle_font.render("Enter Custom Temperature (K):", True, self.BLACK)
                    custom_temp_label_rect = custom_temp_label.get_rect(midleft=(self.width - self.customization_panel_width + 50, 265))
                    self.screen.blit(custom_temp_label, custom_temp_label_rect)
                    custom_temp_input_rect = pygame.Rect(self.width - self.customization_panel_width + 50, 285, self.customization_panel_width - 100, 30)
                    pygame.draw.rect(self.screen, self.WHITE, custom_temp_input_rect, 2)
                    pygame.draw.rect(self.screen, self.BLUE if self.temperature_input_active else self.GRAY, custom_temp_input_rect, 1)
                    if self.temperature_input_active:
                        text_surface = self.subtitle_font.render(self.temperature_input_text, True, self.BLACK)
                    else:
                        text_surface = self.subtitle_font.render(f"{self.selected_body.get('temperature', 5800):.0f} K", True, self.BLACK)
                    text_rect = text_surface.get_rect(midleft=(custom_temp_input_rect.left + 5, custom_temp_input_rect.centery))
                    self.screen.blit(text_surface, text_rect)

                # Draw star age dropdown
                age_label = self.subtitle_font.render("Age (Gyr)", True, self.BLACK)
                age_label_rect = age_label.get_rect(midleft=(self.width - self.customization_panel_width + 50, 165))
                self.screen.blit(age_label, age_label_rect)
                if not self.show_custom_star_age_input:
                    pygame.draw.rect(self.screen, self.WHITE, self.star_age_dropdown_rect, 2)
                    pygame.draw.rect(self.screen, self.BLUE if self.star_age_dropdown_active else self.GRAY, 
                                   self.star_age_dropdown_rect, 1)
                    dropdown_text = "Select Age"
                    if self.star_age_dropdown_selected:
                        selected = next(((name, value) for name, value in self.star_age_dropdown_options if name == self.star_age_dropdown_selected), None)
                        if selected:
                            name, value = selected
                            dropdown_text = name
                    text_surface = self.subtitle_font.render(dropdown_text, True, self.BLACK)
                    text_rect = text_surface.get_rect(midleft=(self.star_age_dropdown_rect.left + 5, self.star_age_dropdown_rect.centery))
                    self.screen.blit(text_surface, text_rect)
                else:
                    # Draw custom age input in place of the dropdown when "Custom" is selected
                    pygame.draw.rect(self.screen, self.WHITE, self.star_age_dropdown_rect, 2)
                    pygame.draw.rect(self.screen, self.BLUE if self.age_input_active else self.GRAY, self.star_age_dropdown_rect, 1)
                    if self.age_input_active:
                        text_surface = self.subtitle_font.render(self.age_input_text, True, self.BLACK)
                    else:
                        text_surface = self.subtitle_font.render(
                            self.format_age_display(self.selected_body.get("age", 0.0)), 
                            True, self.BLACK
                        )
                    text_rect = text_surface.get_rect(midleft=(self.star_age_dropdown_rect.left + 5, self.star_age_dropdown_rect.centery))
                    self.screen.blit(text_surface, text_rect)

                # Draw luminosity input for stars
                luminosity_label = self.subtitle_font.render("Luminosity (L☉)", True, self.BLACK)
                luminosity_label_rect = luminosity_label.get_rect(midleft=(self.width - self.customization_panel_width + 50, 285))
                self.screen.blit(luminosity_label, luminosity_label_rect)
                pygame.draw.rect(self.screen, self.WHITE, self.luminosity_dropdown_rect, 2)
                pygame.draw.rect(self.screen, self.BLUE if self.luminosity_dropdown_active else self.GRAY, 
                               self.luminosity_dropdown_rect, 1)
                
                # Use synchronized label from body or fallback to generic "Select"
                dropdown_text = self.luminosity_dropdown_selected if self.luminosity_dropdown_selected else "Select Star Luminosity"
                
                # Check if current selection matches a preset name to add value in parentheses
                if self.luminosity_dropdown_selected:
                    selected = next(((name, value) for name, value in self.luminosity_dropdown_options if name == self.luminosity_dropdown_selected), None)
                    if selected and selected[1] is not None:
                        name, value = selected
                        dropdown_text = f"{name} (" + self._format_value(value, 'L☉') + ")"
                
                text_surface = self.subtitle_font.render(dropdown_text, True, self.BLACK)
                text_rect = text_surface.get_rect(midleft=(self.luminosity_dropdown_rect.left + 5, self.luminosity_dropdown_rect.centery))
                self.screen.blit(text_surface, text_rect)

                # Draw radius dropdown
                radius_label = self.subtitle_font.render("Radius (R☉)", True, self.BLACK)
                radius_label_rect = radius_label.get_rect(midleft=(self.width - self.customization_panel_width + 50, 345))
                self.screen.blit(radius_label, radius_label_rect)
                pygame.draw.rect(self.screen, self.WHITE, self.radius_dropdown_rect, 2)
                pygame.draw.rect(self.screen, self.BLUE if self.radius_dropdown_active else self.GRAY, 
                               self.radius_dropdown_rect, 1)
                
                # Use synchronized label from body or fallback
                dropdown_text = self.radius_dropdown_selected if self.radius_dropdown_selected else "Select Radius"
                
                # Check if current selection matches a preset name
                if self.radius_dropdown_selected:
                    selected = next(((name, value) for name, value in self.radius_dropdown_options if name == self.radius_dropdown_selected), None)
                    if selected and selected[1] is not None:
                        name, value = selected
                        if "(Sun)" in name:
                            dropdown_text = f"Sun ({value:.2f} R☉)"
                        else:
                            dropdown_text = f"{name} (" + self._format_value(value, 'R☉') + ")"
                
                text_surface = self.subtitle_font.render(dropdown_text, True, self.BLACK)
                text_rect = text_surface.get_rect(midleft=(self.radius_dropdown_rect.left + 5, self.radius_dropdown_rect.centery))
                self.screen.blit(text_surface, text_rect)

                # Draw activity level dropdown
                activity_label = self.subtitle_font.render("Activity Level", True, self.BLACK)
                activity_label_rect = activity_label.get_rect(midleft=(self.width - self.customization_panel_width + 50, 405))
                self.screen.blit(activity_label, activity_label_rect)
                pygame.draw.rect(self.screen, self.WHITE, self.activity_dropdown_rect, 2)
                pygame.draw.rect(self.screen, self.BLUE if self.activity_dropdown_active else self.GRAY, 
                               self.activity_dropdown_rect, 1)
                dropdown_text = "Select Activity Level"
                if self.activity_dropdown_selected:
                    selected = next(((name, value) for name, value in self.activity_dropdown_options if name == self.activity_dropdown_selected), None)
                    if selected:
                        name, value = selected
                        if value is not None and "(Sun)" in name:
                            dropdown_text = f"Sun ({value:.2f})"
                        elif value is not None:
                            dropdown_text = f"{name} (" + self._format_value(value, '') + ")"
                        else:
                            dropdown_text = name
                text_surface = self.subtitle_font.render(dropdown_text, True, self.BLACK)
                text_rect = text_surface.get_rect(midleft=(self.activity_dropdown_rect.left + 5, self.activity_dropdown_rect.centery))
                self.screen.blit(text_surface, text_rect)

                # Draw metallicity dropdown
                metallicity_label = self.subtitle_font.render("Metallicity [Fe/H]", True, self.BLACK)
                metallicity_label_rect = metallicity_label.get_rect(midleft=(self.width - self.customization_panel_width + 50, 465))
                self.screen.blit(metallicity_label, metallicity_label_rect)
                pygame.draw.rect(self.screen, self.WHITE, self.metallicity_dropdown_rect, 2)
                pygame.draw.rect(self.screen, self.BLUE if self.metallicity_dropdown_active else self.GRAY, 
                               self.metallicity_dropdown_rect, 1)
                dropdown_text = "Select Metallicity"
                if self.metallicity_dropdown_selected:
                    selected = next(((name, value) for name, value in self.metallicity_dropdown_options if name == self.metallicity_dropdown_selected), None)
                    if selected:
                        name, value = selected
                        if value is not None and "(Sun)" in name:
                            dropdown_text = f"Sun ({value:+.2f} [Fe/H])"
                        elif value is not None:
                            dropdown_text = f"{name} (" + self._format_value(value, '[Fe/H]') + ")"
                        else:
                            dropdown_text = name
                text_surface = self.subtitle_font.render(dropdown_text, True, self.BLACK)
                text_rect = text_surface.get_rect(midleft=(self.metallicity_dropdown_rect.left + 5, self.metallicity_dropdown_rect.centery))
                self.screen.blit(text_surface, text_rect)

                # Draw spacetime grid
                self.draw_spacetime_grid()
        
        # Draw spacetime grid
        self.draw_spacetime_grid()
        
        # Update physics
        self.update_physics()
        
        # Update and Draw placement trajectory lines (Step 2 & 3)
        current_time = time.time()
        for body in self.placed_bodies:
            line = body.get("placement_line")
            if line:
                # Step 2: Fade the line over time
                if line["fade_start_time"] is not None:
                    t = current_time - line["fade_start_time"]
                    fade_progress = min(t / line["fade_duration"], 1.0)
                    line["alpha"] = int(255 * (1.0 - fade_progress))
                    
                    if line["alpha"] <= 0:
                        body["placement_line"] = None
                        continue
                
                # Step 3: Render with alpha
                # Only draw if the line exists and alpha > 0
                if body.get("placement_line"):
                    alpha = line["alpha"]
                    color = (255, 255, 255, alpha)
                    
                    # Shrink from the back: start follows the planet, end is fixed at target
                    # This makes the trajectory ahead of the planet shrink as it moves
                    current_body_pos = body.get("visual_position", body["position"])
                    start_screen = self.world_to_screen(current_body_pos)
                    end_screen = self.world_to_screen(line["end"])
                    
                    # Only draw if there's actual length
                    if np.linalg.norm(np.array(start_screen) - np.array(end_screen)) > 1.0:
                        pygame.draw.line(self.screen, color, start_screen, end_screen, 1)

        # Draw orbit grid lines first (so they appear behind the bodies)
        for body in self.placed_bodies:
            if body["type"] != "star" and body["name"] in self.orbit_grid_points:
                grid_points = self.orbit_grid_points[body["name"]]
                if len(grid_points) > 1:
                    # For moons, don't cache since grid moves with planet every frame
                    # For planets, use cache since grid is static relative to star
                    if body["type"] == "moon":
                        # Convert directly without caching (planet position changes every frame)
                        screen_points = [np.array(self.world_to_screen(p)) for p in grid_points]
                    else:
                        # Planets can use cache (static relative to star)
                        screen_points = self._cached_screen_points(body["name"], grid_points, self.orbit_grid_screen_cache)
                    if body["type"] == "planet":
                        color = self.LIGHT_GRAY
                    else:  # moon
                        color = (150, 150, 150)  # Slightly darker for moons
                    
                    # Optional: Fade orbits with high eccentricity (> 0.6) to flag instability
                    e = float(body.get("eccentricity", 0.0))
                    if e > 0.6:
                        fade_factor = max(0.3, 1.0 - (e - 0.6) / 0.4)
                        color = tuple(int(c * fade_factor) for c in color)
                        
                    pygame.draw.lines(self.screen, color, True, screen_points, max(1, int(2 * self.camera_zoom)))
        
        # Draw habitable zones for all stars (before drawing bodies, after orbit grids)
        # DISABLED: Habitable zone visualization removed per user request
        # for body in self.placed_bodies:
        #     if body.get("type") == "star":
        #         self.draw_habitable_zone(self.screen, body)
        
        # Draw motion trails FIRST (behind everything) for planets during orbital correction
        for body in self.placed_bodies:
            if body.get("is_correcting_orbit", False) and body["type"] == "planet":
                body_id = body.get("id")
                if body_id and body_id in self.orbital_corrections:
                    correction = self.orbital_corrections[body_id]
                    current_time = time.time()
                    elapsed = current_time - correction["start_time"]
                    progress = min(elapsed / correction["duration"], 1.0)
                    
                    # Check if planet has reached target and is orbiting
                    target_pos = correction.get("target_pos")
                    current_pos = body.get("position", body.get("visual_position"))
                    distance_to_target = np.linalg.norm(current_pos - target_pos) if target_pos is not None else float('inf')
                    position_threshold = 5.0
                    is_at_target = distance_to_target < position_threshold
                    orbit_speed = body.get("orbit_speed", 0.0)
                    is_orbiting = orbit_speed > 0.0
                    is_still_moving = (not is_at_target) or (not is_orbiting)
                    
                    # Draw trail if animation is in progress and planet is moving
                    if progress < 1.0 and is_still_moving and "position_history" in correction:
                        position_history = correction["position_history"]
                        if len(position_history) > 1:
                            # Calculate total distance traveled
                            total_distance = 0.0
                            for i in range(1, len(position_history)):
                                total_distance += np.linalg.norm(position_history[i] - position_history[i-1])
                            
                            if total_distance > 1.0:
                                # Get planet color
                                base_color_hex = body.get("base_color")
                                if base_color_hex:
                                    planet_color = hex_to_rgb(base_color_hex)
                                else:
                                    planet_color = hex_to_rgb(CELESTIAL_BODY_COLORS.get(body.get("name", "Earth"), "#2E7FFF"))
                                
                                # Trail length: 15% of distance traveled
                                trail_length_ratio = 0.15
                                target_trail_distance = total_distance * trail_length_ratio
                                
                                # Find positions that make up the trail
                                trail_positions = []
                                accumulated_distance = 0.0
                                current_pos = body.get("position", body.get("visual_position", position_history[-1]))
                                trail_positions.append(current_pos)
                                
                                # Work backwards through position history
                                for i in range(len(position_history) - 1, 0, -1):
                                    segment_distance = np.linalg.norm(position_history[i] - position_history[i-1])
                                    if accumulated_distance + segment_distance <= target_trail_distance:
                                        trail_positions.insert(0, position_history[i-1])
                                        accumulated_distance += segment_distance
                                    else:
                                        remaining = target_trail_distance - accumulated_distance
                                        if remaining > 0 and segment_distance > 0:
                                            t = remaining / segment_distance
                                            interpolated_pos = position_history[i-1] + (position_history[i] - position_history[i-1]) * t
                                            trail_positions.insert(0, interpolated_pos)
                                        break
                                    if accumulated_distance >= target_trail_distance:
                                        break
                                
                                # Draw trail if we have at least 2 points
                                if len(trail_positions) >= 2:
                                    # Calculate fade
                                    fade_progress = progress
                                    eased_fade = 1.0 - ((1.0 - fade_progress) ** 3)
                                    trail_alpha = int(255 * (1.0 - eased_fade))
                                    
                                    if trail_alpha > 10:
                                        # Desaturate planet color for trail
                                        trail_color = desaturate_color(planet_color, saturation=0.6)
                                        
                                        # Convert positions to screen coordinates
                                        screen_positions = [self.world_to_screen(pos) for pos in trail_positions]
                                        
                                        # Draw soft trail line with alpha blending
                                        trail_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
                                        for i in range(3):
                                            line_width = max(1, 3 - i)
                                            line_alpha = max(0, trail_alpha - (i * 60))
                                            if line_alpha > 0:
                                                trail_color_alpha = (*trail_color, line_alpha)
                                                pygame.draw.lines(trail_surface, trail_color_alpha, False, screen_positions, line_width)
                                        
                                        self.screen.blit(trail_surface, (0, 0))
        
        # Draw orbit lines and bodies
        for body in self.placed_bodies:
            # Draw orbit line using persistent orbit_points
            if body["type"] != "star":
                self.draw_orbit(body)
            
            # Draw body using base_color (per-object, stored as hex string)
            base_color_hex = body.get("base_color")
            if base_color_hex:
                color = hex_to_rgb(base_color_hex)
            else:
                # Fallback to default colors if base_color not set
                if body["type"] == "star":
                    color = hex_to_rgb(CELESTIAL_BODY_COLORS.get("Sun", "#FDB813"))
                elif body["type"] == "planet":
                    color = hex_to_rgb(CELESTIAL_BODY_COLORS.get(body.get("name", "Earth"), "#2E7FFF"))
                else:  # moon
                    color = hex_to_rgb(CELESTIAL_BODY_COLORS.get("Moon", "#B0B0B0"))
            
            if body["type"] == "star":
                pos = self.world_to_screen(body["position"])
                # Calculate visual radius using authoritative formula
                visual_radius = self.calculate_star_visual_radius(body)
                pygame.draw.circle(self.screen, color, (int(pos[0]), int(pos[1])), max(1, int(visual_radius * self.camera_zoom)))
            else:
                self.draw_rotating_body(body, color)
            
            # Highlight selected body (compare object identity, not name, to ensure only the clicked object is highlighted)
            if self.selected_body is body:
                pos = self.world_to_screen(body["position"])
                # CRITICAL: For planets, use NASA-style normalized visual radius
                # For moons, scale relative to parent planet
                if body["type"] == "planet":
                    visual_radius = self.calculate_planet_visual_radius(body, self.placed_bodies)
                elif body["type"] == "moon":
                    visual_radius = self.calculate_moon_visual_radius(body, self.placed_bodies)
                else:
                    # Stars: calculate visual radius using authoritative formula
                    visual_radius = self.calculate_star_visual_radius(body)
                pygame.draw.circle(self.screen, self.RED, (int(pos[0]), int(pos[1])), max(1, int((visual_radius + 5) * self.camera_zoom)), 2)
        
        # Update placement preview position EVERY FRAME (frame-driven, not event-driven)
        # This ensures smooth cursor following regardless of event timing
        mouse_pos = pygame.mouse.get_pos()
        
        # Update preview position every frame if in placement mode
        if self.placement_mode_active or self.planet_dropdown_selected:
            # For planets, always follow cursor everywhere
            if self.planet_dropdown_selected:
                self.preview_position = mouse_pos
            # For stars and moons, only show when mouse is over space area
            elif self.active_tab:
                space_area = pygame.Rect(0, self.tab_height + 2*self.tab_margin, 
                                          self.width, 
                                          self.height - (self.tab_height + 2*self.tab_margin))
                if space_area.collidepoint(mouse_pos):
                    self.preview_position = mouse_pos
                else:
                    # Clear preview for stars/moons when mouse leaves space area
                    self.preview_position = None
        else:
                self.preview_position = None
        
        # Draw time control box on top of scene elements
        self.draw_time_controls(self.screen)
        
        # Draw Reset View button (UI layer, screen space)
        self.draw_reset_button()
        
        # Draw Zoom-Aware Distance Scale Indicator
        self.draw_scale_indicator()
        
        # Draw preview BEFORE dropdowns so it appears behind them if they overlap
        # BUT hide it entirely when any modal dropdown is active for better UX
        if self.preview_position and self.preview_radius and self.preview_radius > 0 and not self._any_dropdown_active():
            self.draw_placement_preview()

        # Render dropdown menu last, so it appears on top of everything
        self.render_dropdown()
        
        # Render planet preset dropdown on top of everything
        self.render_planet_preset_dropdown()

        # Draw the custom modal last if active
        self.draw_custom_modal()

        pygame.display.flip()
    
    def draw_placement_preview(self):
        """Draw a semi-transparent preview of the object being placed"""
        if not self.preview_position or not self.preview_radius:
            # Debug: Print why preview isn't drawing
            if not hasattr(self, '_last_no_draw_debug') or time.time() - self._last_no_draw_debug > 1.0:
                print(f"DEBUG: draw_placement_preview skipped. preview_position={self.preview_position}, preview_radius={self.preview_radius}")
                self._last_no_draw_debug = time.time()
            return
        
        # Ensure preview_radius is valid (greater than 0)
        if self.preview_radius <= 0:
            print(f"WARNING: Invalid preview_radius: {self.preview_radius}")
            return
        
        scaled_preview_radius = int(self.preview_radius * self.camera_zoom)
        if scaled_preview_radius <= 0:
            scaled_preview_radius = 1
        
        # Debug: Print once per second when drawing preview
        if not hasattr(self, '_last_draw_debug') or time.time() - self._last_draw_debug > 1.0:
            print(f"DEBUG: Drawing preview at {self.preview_position}, radius={self.preview_radius}, scaled={scaled_preview_radius}, planet={self.planet_dropdown_selected}")
            self._last_draw_debug = time.time()
        
        # Determine preview color and glow color based on object type
        # Use base_color from presets when available
        if self.planet_dropdown_selected:
            # Get planet's base_color from preset
            planet_name = self.planet_dropdown_selected
            if planet_name in SOLAR_SYSTEM_PLANET_PRESETS:
                base_color_hex = SOLAR_SYSTEM_PLANET_PRESETS[planet_name].get("base_color")
                if base_color_hex:
                    preview_color = hex_to_rgb(base_color_hex)
                else:
                    # Fallback to Earth color if not found
                    preview_color = hex_to_rgb(CELESTIAL_BODY_COLORS.get("Earth", "#2E7FFF"))
            else:
                # Fallback for custom planets
                preview_color = hex_to_rgb(CELESTIAL_BODY_COLORS.get("Earth", "#2E7FFF"))
            # Create lighter version for glow (brighten by ~30%)
            glow_color = tuple(min(255, int(c * 1.3)) for c in preview_color)
        elif self.active_tab == "star":
            # Use Sun color
            preview_color = hex_to_rgb(CELESTIAL_BODY_COLORS.get("Sun", "#FDB813"))
            glow_color = tuple(min(255, int(c * 1.2)) for c in preview_color)  # Slightly brighter
        elif self.active_tab == "planet":
            # Generic planet color (shouldn't happen, but fallback)
            preview_color = hex_to_rgb(CELESTIAL_BODY_COLORS.get("Earth", "#2E7FFF"))
            glow_color = tuple(min(255, int(c * 1.3)) for c in preview_color)
        else:  # moon
            # Use Moon color
            preview_color = hex_to_rgb(CELESTIAL_BODY_COLORS.get("Moon", "#B0B0B0"))
            glow_color = tuple(min(255, int(c * 1.2)) for c in preview_color)  # Slightly brighter
        
        center_x, center_y = int(self.preview_position[0]), int(self.preview_position[1])
        
        # Draw glow outline (outer ring) - multiple circles for smoother glow effect
        glow_radius = scaled_preview_radius + 3
        for i in range(3):
            alpha = 64 - i * 15
            if alpha > 0:
                glow_surface = pygame.Surface((glow_radius * 2 + 6, glow_radius * 2 + 6), pygame.SRCALPHA)
                pygame.draw.circle(glow_surface, (*glow_color, alpha), 
                                  (glow_surface.get_width() // 2, glow_surface.get_height() // 2), 
                                  max(1, glow_radius - i), 2)
                self.screen.blit(glow_surface, (center_x - glow_surface.get_width() // 2, 
                                               center_y - glow_surface.get_height() // 2))
        
        # Draw main preview circle (semi-transparent)
        preview_surface = pygame.Surface((scaled_preview_radius * 2 + 4, scaled_preview_radius * 2 + 4), pygame.SRCALPHA)
        preview_center = (preview_surface.get_width() // 2, preview_surface.get_height() // 2)
        # Use alpha 180 for better visibility (was 128)
        pygame.draw.circle(preview_surface, (*preview_color, 180), preview_center, scaled_preview_radius)
        self.screen.blit(preview_surface, (center_x - preview_surface.get_width() // 2,
                                          center_y - preview_surface.get_height() // 2))
        
        # Draw outer outline for better visibility
        outline_surface = pygame.Surface((scaled_preview_radius * 2 + 4, scaled_preview_radius * 2 + 4), pygame.SRCALPHA)
        pygame.draw.circle(outline_surface, (*glow_color, 255), preview_center, scaled_preview_radius, 2)
        self.screen.blit(outline_surface, (center_x - outline_surface.get_width() // 2,
                                          center_y - outline_surface.get_height() // 2))
        
        # Draw tooltip below the preview
        # Show planet name if a planet is selected for placement (regardless of active_tab)
        if self.planet_dropdown_selected:
            tooltip_text = f"Click to place {self.planet_dropdown_selected}"
        else:
            tooltip_text = "Click to confirm placement."
        tooltip_surface = self.subtitle_font.render(tooltip_text, True, self.WHITE)
        tooltip_rect = tooltip_surface.get_rect(center=(center_x, center_y + scaled_preview_radius + 25))
        
        # Draw tooltip background for better visibility
        tooltip_bg_rect = tooltip_rect.inflate(10, 5)
        tooltip_bg_surface = pygame.Surface(tooltip_bg_rect.size, pygame.SRCALPHA)
        tooltip_bg_surface.fill((0, 0, 0, 180))  # Semi-transparent black background
        self.screen.blit(tooltip_bg_surface, tooltip_bg_rect)
        self.screen.blit(tooltip_surface, tooltip_rect)
    
    def create_habitable_zone(self, star):
        """
        Create a habitable zone (HZ) annulus surface for a star.
        Uses Kopparapu-style approximation:
        - inner_HZ_AU = sqrt(L / 1.1)
        - outer_HZ_AU = sqrt(L / 0.53)
        
        Args:
            star: Star object with 'luminosity' attribute
            
        Returns:
            pygame.Surface with SRCALPHA containing the HZ annulus, or None if invalid
        """
        if star.get("type") != "star":
            return None
        
        luminosity = star.get("luminosity", 1.0)
        if luminosity <= 0:
            return None
        
        # Calculate HZ boundaries in AU
        inner_HZ_AU = np.sqrt(luminosity / 1.1)
        outer_HZ_AU = np.sqrt(luminosity / 0.53)
        
        # Convert to pixels using perceptual scaling
        inner_HZ_px = int(self.get_visual_orbit_radius(inner_HZ_AU))
        outer_HZ_px = int(self.get_visual_orbit_radius(outer_HZ_AU))
        
        if outer_HZ_px <= 0:
            return None
        
        # SAFETY LIMIT: Cap the HZ surface size to prevent memory errors and crashes
        # 4000x4000 is ~64MB in RGBA, which is safe for most systems.
        if outer_HZ_px > 2000:
            outer_HZ_px = 2000
            if inner_HZ_px > 1990: inner_HZ_px = 1990

        # Create surface large enough for the outer radius
        surface_size = (outer_HZ_px * 2 + 10, outer_HZ_px * 2 + 10)
        try:
            hz_surface = pygame.Surface(surface_size, pygame.SRCALPHA)
        except (pygame.error, MemoryError):
            print(f"WARNING: Could not create HZ surface of size {surface_size}")
            return None
            
        center = (surface_size[0] // 2, surface_size[1] // 2)
        
        # OPTIMIZED DRAWING: Draw the annulus using circle width
        # This is much faster than the nested pixel loop which caused freezes/crashes
        width = max(1, outer_HZ_px - inner_HZ_px)
        mid_radius = inner_HZ_px + width // 2
        pygame.draw.circle(hz_surface, (0, 255, 0, 60), center, mid_radius, width)
        
        # Store HZ parameters in star for later use
        star["hz_inner_px"] = inner_HZ_px
        star["hz_outer_px"] = outer_HZ_px
        star["hz_center_offset"] = (surface_size[0] // 2, surface_size[1] // 2)
        
        return hz_surface
    
    def draw_habitable_zone(self, screen, star):
        """
        Draw the habitable zone ring for a star on the screen.
        Caches scaled surface to improve performance.
        """
        if star.get("type") != "star":
            return
        
        # Get or create HZ surface
        hz_surface = star.get("hz_surface")
        if hz_surface is None:
            hz_surface = self.create_habitable_zone(star)
            if hz_surface is None:
                return
            star["hz_surface"] = hz_surface
        
        # Get star position (screen space)
        star_pos = self.world_to_screen(star["position"])
        
        # Use cached scaled surface if zoom hasn't changed significantly
        zoom_threshold = 0.05
        last_zoom = star.get("hz_last_zoom", 0)
        cached_scaled = star.get("hz_surface_scaled")
        
        if abs(self.camera_zoom - last_zoom) > zoom_threshold or cached_scaled is None:
            # Scale surface based on camera zoom
            scaled_size = (
                max(1, int(hz_surface.get_width() * self.camera_zoom)),
                max(1, int(hz_surface.get_height() * self.camera_zoom)),
            )
            try:
                if self.camera_zoom != 1.0:
                    hz_surface_scaled = pygame.transform.smoothscale(hz_surface, scaled_size)
                else:
                    hz_surface_scaled = hz_surface
                
                star["hz_surface_scaled"] = hz_surface_scaled
                star["hz_last_zoom"] = self.camera_zoom
            except (pygame.error, MemoryError):
                return
        else:
            hz_surface_scaled = cached_scaled
            
        center_offset = (hz_surface_scaled.get_width() // 2, hz_surface_scaled.get_height() // 2)
        blit_pos = (star_pos[0] - center_offset[0], star_pos[1] - center_offset[1])
        screen.blit(hz_surface_scaled, blit_pos)
    
    def draw_rotating_body(self, body, color):
        """Draw a celestial body with rotation"""
        # Apply density-based color tinting for planets
        if body["type"] == "planet" and "density_tint" in body:
            tint = body["density_tint"]
            # Blend 40% tint, 60% original color for a subtle but noticeable effect
            color = (
                int(color[0] * 0.6 + tint[0] * 0.4),
                int(color[1] * 0.6 + tint[1] * 0.4),
                int(color[2] * 0.6 + tint[2] * 0.4)
            )

        # CRITICAL: For planets, radius is in Earth radii (R⊕), compute visual radius
        # For stars, radius is in pixels (legacy)
        # For moons, scale relative to parent planet
        if body["type"] == "planet":
            # Use NASA-style normalized visual radius with perceptual scaling
            visual_radius = self.calculate_planet_visual_radius(body, self.placed_bodies)
        elif body["type"] == "moon":
            # Scale moon relative to parent planet
            visual_radius = self.calculate_moon_visual_radius(body, self.placed_bodies)
        else:
            # Stars: radius is already in pixels
            visual_radius = body["radius"]
        
        radius = max(1, int(visual_radius * self.camera_zoom))
        
        # Use visual_position if correcting orbit, otherwise use position
        render_pos = body.get("visual_position", body["position"])
        
        surface_size = radius * 2 + 4  # Add some padding
        surface = pygame.Surface((surface_size, surface_size), pygame.SRCALPHA)
        
        # Draw the main body
        pygame.draw.circle(surface, color, (radius + 2, radius + 2), radius)
        
        # Draw gravity-based atmosphere glow for planets
        if body["type"] == "planet":
            gravity_strength = body.get("gravity_visual_strength", 1.0)
            # Subtle outer glow that gets stronger with gravity
            glow_radius = radius + 2
            glow_color = (min(255, color[0] + 50), min(255, color[1] + 50), min(255, color[2] + 50), int(100 * min(1.0, gravity_strength)))
            pygame.draw.circle(surface, glow_color, (radius + 2, radius + 2), glow_radius, 2)
        
        # Draw a line to indicate rotation
        rotation_angle = body["rotation_angle"]
        end_x = radius + 2 + radius * 0.8 * np.cos(rotation_angle)
        end_y = radius + 2 + radius * 0.8 * np.sin(rotation_angle)
        pygame.draw.line(surface, self.WHITE, (radius + 2, radius + 2), (end_x, end_y), 2)
        
        # render_pos is already defined above for trail calculation
        pos = self.world_to_screen(render_pos)
        self.screen.blit(surface, (pos[0] - radius - 2, pos[1] - radius - 2))
        
    
    def render_simulation_builder(self):
        """Render the simulation builder screen with tabs and space area"""
        # Fill background
        self.screen.fill(self.DARK_BLUE)
        
        # 1. Background Layer: Spacetime Grid
        self.draw_spacetime_grid()
        
        # 2. Physics & Space Elements Layer (Drawn behind UI)
        self.update_physics()
        
        # Draw placement trajectory lines
        current_time = time.time()
        for body in self.placed_bodies:
            line = body.get("placement_line")
            if line:
                if line["fade_start_time"] is not None:
                    t = current_time - line["fade_start_time"]
                    fade_progress = min(t / line["fade_duration"], 1.0)
                    line["alpha"] = int(255 * (1.0 - fade_progress))
                    if line["alpha"] <= 0:
                        body["placement_line"] = None
                        continue
                if body.get("placement_line"):
                    alpha = line["alpha"]
                    color = (255, 255, 255, alpha)
                    current_body_pos = body.get("visual_position", body["position"])
                    start_screen = self.world_to_screen(current_body_pos)
                    end_screen = self.world_to_screen(line["end"])
                    if np.linalg.norm(np.array(start_screen) - np.array(end_screen)) > 1.0:
                        pygame.draw.line(self.screen, color, start_screen, end_screen, 1)

        # Draw orbit grid lines (behind bodies)
        for body in self.placed_bodies:
            if body["type"] != "star" and body["name"] in self.orbit_grid_points:
                grid_points = self.orbit_grid_points[body["name"]]
                if len(grid_points) > 1:
                    if body["type"] == "moon":
                        screen_points = [np.array(self.world_to_screen(p)) for p in grid_points]
                    else:
                        screen_points = self._cached_screen_points(body["name"], grid_points, self.orbit_grid_screen_cache)
                    color = self.LIGHT_GRAY if body["type"] == "planet" else (150, 150, 150)
                    e = float(body.get("eccentricity", 0.0))
                    if e > 0.6:
                        fade_factor = max(0.3, 1.0 - (e - 0.6) / 0.4)
                        color = tuple(int(c * fade_factor) for c in color)
                    pygame.draw.lines(self.screen, color, True, screen_points, max(1, int(2 * self.camera_zoom)))
        
        # Draw orbit lines and bodies
        for body in self.placed_bodies:
            if body["type"] != "star":
                self.draw_orbit(body)
            base_color_hex = body.get("base_color")
            color = hex_to_rgb(base_color_hex) if base_color_hex else \
                    (hex_to_rgb(CELESTIAL_BODY_COLORS.get("Sun", "#FDB813")) if body["type"] == "star" else \
                     hex_to_rgb(CELESTIAL_BODY_COLORS.get(body.get("name", "Earth"), "#2E7FFF")) if body["type"] == "planet" else \
                     hex_to_rgb(CELESTIAL_BODY_COLORS.get("Moon", "#B0B0B0")))
            
            if body["type"] == "star":
                pos = self.world_to_screen(body["position"])
                visual_radius = self.calculate_star_visual_radius(body)
                pygame.draw.circle(self.screen, color, (int(pos[0]), int(pos[1])), max(1, int(visual_radius * self.camera_zoom)))
            else:
                self.draw_rotating_body(body, color)
            
            if self.selected_body is body:
                pos = self.world_to_screen(body["position"])
                visual_radius = self.calculate_planet_visual_radius(body, self.placed_bodies) if body["type"] == "planet" else \
                                self.calculate_moon_visual_radius(body, self.placed_bodies) if body["type"] == "moon" else \
                                self.calculate_star_visual_radius(body)
                pygame.draw.circle(self.screen, self.RED, (int(pos[0]), int(pos[1])), max(1, int((visual_radius + 5) * self.camera_zoom)), 2)

        # 3. UI Layer (Always on top of space area elements)
        
        # Draw white bar at the top
        top_bar_height = self.tab_height + 2*self.tab_margin
        top_bar_rect = pygame.Rect(0, 0, self.width, top_bar_height)
        pygame.draw.rect(self.screen, self.WHITE, top_bar_rect)
        
        # Draw tabs
        for tab_name, tab_rect in self.tabs.items():
            if tab_name == self.active_tab:
                pygame.draw.rect(self.screen, self.ACTIVE_TAB_COLOR, tab_rect, border_radius=5)
                pygame.draw.rect(self.screen, self.WHITE, tab_rect, 2, border_radius=5)
            else:
                pygame.draw.rect(self.screen, self.GRAY, tab_rect, border_radius=5)
            tab_text = self.tab_font.render(tab_name.capitalize(), True, self.WHITE)
            tab_text_rect = tab_text.get_rect(center=tab_rect.center)
            self.screen.blit(tab_text, tab_text_rect)
            if tab_name == "planet":
                arrow_x = tab_rect.right - self.planet_preset_arrow_size - 3
                arrow_y = tab_rect.bottom - self.planet_preset_arrow_size - 3
                self.planet_preset_arrow_rect = pygame.Rect(arrow_x - 2, arrow_y - 2, self.planet_preset_arrow_size + 4, self.planet_preset_arrow_size + 4)
                arrow_points = [(arrow_x, arrow_y), (arrow_x + self.planet_preset_arrow_size, arrow_y), (arrow_x + self.planet_preset_arrow_size // 2, arrow_y + self.planet_preset_arrow_size)]
                pygame.draw.polygon(self.screen, self.WHITE, arrow_points)

        # Draw customization panel
        if self.show_customization_panel and self.selected_body:
            pygame.draw.rect(self.screen, self.WHITE, self.customization_panel)
            pygame.draw.rect(self.screen, self.BLACK, self.close_button, 2)
            pygame.draw.line(self.screen, self.BLACK, (self.close_button.left + 5, self.close_button.top + 5), (self.close_button.right - 5, self.close_button.bottom - 5), 2)
            pygame.draw.line(self.screen, self.BLACK, (self.close_button.left + 5, self.close_button.bottom - 5), (self.close_button.right - 5, self.close_button.top + 5), 2)
            title_text = self.font.render(f"Customize {self.selected_body['type'].capitalize()}", True, self.BLACK)
            title_rect = title_text.get_rect(center=(self.width - self.customization_panel_width//2, 50))
            self.screen.blit(title_text, title_rect)
            
            # --- INTERNAL PANEL CONTENT START ---
            body = self.get_selected_body()
            if not body: body = self.selected_body
            
            if body.get('type') == 'star':
                # Star-specific UI logic (re-implemented simply here)
                self._render_star_ui_content()
            elif body.get('type') == 'planet':
                # Planet-specific UI logic
                self._render_planet_ui_content()
            elif body.get('type') == 'moon':
                # Moon-specific UI logic
                self._render_moon_ui_content()
            
            # Common elements: Orbit Toggles
            if body.get('type') in ['planet', 'moon']:
                self._render_orbit_toggles(body)
            # --- INTERNAL PANEL CONTENT END ---

        # 4. Global UI Elements (Top-most)
        
        # Update preview position based on mouse
        mouse_pos = pygame.mouse.get_pos()
        if self.placement_mode_active or self.planet_dropdown_selected:
            if self.planet_dropdown_selected:
                self.preview_position = mouse_pos
            elif self.active_tab:
                space_area = pygame.Rect(0, self.tab_height + 2*self.tab_margin, self.width, self.height - (self.tab_height + 2*self.tab_margin))
                if space_area.collidepoint(mouse_pos):
                    self.preview_position = mouse_pos
                else:
                    self.preview_position = None
        
        # Instructions
        if len(self.placed_bodies) > 0:
            stars = [b for b in self.placed_bodies if b["type"] == "star"]
            planets = [b for b in self.placed_bodies if b["type"] == "planet"]
            instruction = "Add at least one star" if len(stars) == 0 else "Add at least one planet" if len(planets) == 0 else "Simulation active"
            instruction_text = self.subtitle_font.render(instruction, True, self.WHITE)
            instruction_rect = instruction_text.get_rect(center=(self.width//2, self.height - 30))
            self.screen.blit(instruction_text, instruction_rect)
        
        self.draw_reset_button()
        self.draw_scale_indicator()
        self.render_dropdown()
        self.render_planet_preset_dropdown()
        
        should_hide_preview = (self.planet_preset_dropdown_visible and self.planet_preset_dropdown_rect and self.planet_preset_dropdown_rect.collidepoint(mouse_pos))
        if self.preview_position and self.preview_radius and self.preview_radius > 0 and not should_hide_preview:
            self.draw_placement_preview()

        self.draw_custom_modal()
        pygame.display.flip()

    def _render_star_ui_content(self):
        """Helper to render star customization options inside the panel"""
        body = self.get_selected_body()
        # Draw Star Mass
        label = self.subtitle_font.render("Mass", True, self.BLACK)
        self.screen.blit(label, (self.width - self.customization_panel_width + 50, 105))
        pygame.draw.rect(self.screen, self.WHITE, self.star_mass_dropdown_rect, 2)
        pygame.draw.rect(self.screen, self.BLUE if self.star_mass_dropdown_active else self.GRAY, self.star_mass_dropdown_rect, 1)
        text = body.get("star_mass_dropdown_selected", "Select Mass")
        self.screen.blit(self.subtitle_font.render(text, True, self.BLACK), (self.star_mass_dropdown_rect.left + 5, self.star_mass_dropdown_rect.centery - 10))
        
        # Draw Star Age
        label = self.subtitle_font.render("Age (Gyr)", True, self.BLACK)
        self.screen.blit(label, (self.width - self.customization_panel_width + 50, 165))
        pygame.draw.rect(self.screen, self.WHITE, self.star_age_dropdown_rect, 2)
        pygame.draw.rect(self.screen, self.BLUE if self.star_age_dropdown_active else self.GRAY, self.star_age_dropdown_rect, 1)
        text = body.get("star_age_dropdown_selected", "Select Age")
        self.screen.blit(self.subtitle_font.render(text, True, self.BLACK), (self.star_age_dropdown_rect.left + 5, self.star_age_dropdown_rect.centery - 10))

        # Draw Spectral Class (Temp)
        label = self.subtitle_font.render("Spectral Class (Temp)", True, self.BLACK)
        self.screen.blit(label, (self.width - self.customization_panel_width + 50, 225))
        pygame.draw.rect(self.screen, self.WHITE, self.spectral_class_dropdown_rect, 2)
        pygame.draw.rect(self.screen, self.BLUE if self.spectral_class_dropdown_active else self.GRAY, self.spectral_class_dropdown_rect, 1)
        text = body.get("spectral_class_dropdown_selected", "Select Spectral Class")
        self.screen.blit(self.subtitle_font.render(text, True, self.BLACK), (self.spectral_class_dropdown_rect.left + 5, self.spectral_class_dropdown_rect.centery - 10))

        # Draw Luminosity
        label = self.subtitle_font.render("Luminosity (L☉)", True, self.BLACK)
        self.screen.blit(label, (self.width - self.customization_panel_width + 50, 285))
        pygame.draw.rect(self.screen, self.WHITE, self.luminosity_dropdown_rect, 2)
        pygame.draw.rect(self.screen, self.BLUE if self.luminosity_dropdown_active else self.GRAY, self.luminosity_dropdown_rect, 1)
        text = body.get("luminosity_dropdown_selected", "Select Luminosity")
        self.screen.blit(self.subtitle_font.render(text, True, self.BLACK), (self.luminosity_dropdown_rect.left + 5, self.luminosity_dropdown_rect.centery - 10))

        # Draw Radius
        label = self.subtitle_font.render("Radius (R☉)", True, self.BLACK)
        self.screen.blit(label, (self.width - self.customization_panel_width + 50, 345))
        pygame.draw.rect(self.screen, self.WHITE, self.radius_dropdown_rect, 2)
        pygame.draw.rect(self.screen, self.BLUE if self.radius_dropdown_active else self.GRAY, self.radius_dropdown_rect, 1)
        text = body.get("radius_dropdown_selected", "Select Radius")
        self.screen.blit(self.subtitle_font.render(text, True, self.BLACK), (self.radius_dropdown_rect.left + 5, self.radius_dropdown_rect.centery - 10))

    def _render_planet_ui_content(self):
        """Helper to render planet customization options"""
        body = self.get_selected_body()
        # Simplified for brevity, following the star pattern
        label = self.subtitle_font.render("Reference Planet", True, self.BLACK)
        self.screen.blit(label, (self.width - self.customization_panel_width + 50, 105))
        pygame.draw.rect(self.screen, self.WHITE, self.planet_dropdown_rect, 2)
        pygame.draw.rect(self.screen, self.BLUE if self.planet_dropdown_active else self.GRAY, self.planet_dropdown_rect, 1)
        text = body.get("planet_dropdown_selected", "Select Planet")
        self.screen.blit(self.subtitle_font.render(text, True, self.BLACK), (self.planet_dropdown_rect.left + 5, self.planet_dropdown_rect.centery - 10))
        # (Other planet dropdowns follow similarly)

    def _render_moon_ui_content(self):
        """Helper to render moon customization options"""
        body = self.get_selected_body()
        label = self.subtitle_font.render("Reference Moon", True, self.BLACK)
        self.screen.blit(label, (self.width - self.customization_panel_width + 50, 105))
        pygame.draw.rect(self.screen, self.WHITE, self.moon_dropdown_rect, 2)
        pygame.draw.rect(self.screen, self.BLUE if self.moon_dropdown_active else self.GRAY, self.moon_dropdown_rect, 1)
        text = body.get("moon_dropdown_selected", "Select Moon")
        self.screen.blit(self.subtitle_font.render(text, True, self.BLACK), (self.moon_dropdown_rect.left + 5, self.moon_dropdown_rect.centery - 10))

    def _render_orbit_toggles(self, body):
        """Render the show orbit / last rev toggles"""
        pygame.draw.rect(self.screen, self.BLACK, self.orbit_enabled_checkbox, 2)
        if body.get("orbit_enabled", True):
            pygame.draw.line(self.screen, self.BLACK, (self.orbit_enabled_checkbox.left + 4, self.orbit_enabled_checkbox.centery), (self.orbit_enabled_checkbox.left + 8, self.orbit_enabled_checkbox.bottom - 4), 2)
            pygame.draw.line(self.screen, self.BLACK, (self.orbit_enabled_checkbox.left + 8, self.orbit_enabled_checkbox.bottom - 4), (self.orbit_enabled_checkbox.right - 4, self.orbit_enabled_checkbox.top + 4), 2)
        label = self.subtitle_font.render("Show Orbit", True, self.BLACK)
        self.screen.blit(label, (self.orbit_enabled_checkbox.right + 10, self.orbit_enabled_checkbox.top))

    def render(self, engine: SimulationEngine):
        """Main render function"""
        self.active_tooltip = None  # Reset tooltip for this frame
        
        # Home screen removed - start directly in sandbox
        # if self.show_home_screen:
        #     self.render_home_screen()
        if self.show_simulation_builder:
            self.render_simulation_builder()
        elif self.show_simulation:
            self.render_simulation(engine)
            self.render_info_panel() # Render popover last
        
        # Draw active tooltip if any
        if self.active_tooltip:
            self._render_generic_tooltip(self.active_tooltip, self.screen)
        
        pygame.display.flip() 

    def validate_custom_modal_input(self):
        """Validate the input text against physical bounds."""
        if not self.custom_modal_text:
            self.custom_modal_error = None
            return

        try:
            value = float(self.custom_modal_text)
            if value < self.custom_modal_min or value > self.custom_modal_max:
                self.custom_modal_error = f"Value outside physical limits: {self.custom_modal_min} - {self.custom_modal_max}"
                self.pending_custom_value = None
            else:
                self.custom_modal_error = None
                self.pending_custom_value = value
        except ValueError:
            self.custom_modal_error = "Invalid numeric value"
            self.pending_custom_value = None

    def apply_custom_value(self):
        """Apply the validated custom value to the target body and update visuals."""
        if self.pending_custom_value is None:
            return

        body = self.bodies_by_id.get(self.pending_custom_body_id) if self.pending_custom_body_id else self.get_selected_body()
        if not body:
            self.show_custom_modal = False
            return

        field = self.pending_custom_field
        value = self.pending_custom_value
        
        # Determine internal field name and apply conversions
        internal_field = field
        if field == "mass":
            if body["type"] == "star":
                # Star mass is stored in Solar masses internally, no conversion needed from modal input
                pass
            elif body["type"] == "moon" and self.custom_modal_unit == "kg":
                value /= 7.35e22  # Convert kg to Lunar masses (M☾)
        elif field == "semi_major_axis":
            internal_field = "semiMajorAxis"
            if body["type"] == "moon":
                value /= 149597870.7  # km to AU
        elif field == "radius":
            if body["type"] == "moon":
                internal_field = "actual_radius"
        elif field == "atmosphere":
            internal_field = "greenhouse_offset"
        elif field == "temperature" and body["type"] == "planet":
            internal_field = "equilibrium_temperature"
        elif field == "density":
            internal_field = "density"
        elif field == "gravity":
            internal_field = "gravity"
        
        # Update the property using the safe registry update
        if self.update_selected_body_property(internal_field, value, internal_field):
            # Track custom overrides
            if "custom_overrides" not in body:
                body["custom_overrides"] = {}
            body["custom_overrides"][field] = self.pending_custom_value # Original custom value
            
            # Clear preset type to indicate custom state
            body["preset_type"] = None
            
            # Update the dropdown selection label to show the value
            unit = self.custom_modal_unit
            field_display = field.replace("_", " ").capitalize()
            if field == "semi_major_axis":
                field_display = "AU" if body["type"] == "planet" else "Dist"
            elif field == "atmosphere":
                field_display = "ΔT"
            
            # Use appropriate unit label for the display
            display_unit = unit
            if field == "mass":
                if body["type"] == "planet": display_unit = "M⊕"
                elif body["type"] == "star": display_unit = "M☉"
                elif body["type"] == "moon": display_unit = "M☾"
            
            label = f"{field_display}: {self.pending_custom_value} {display_unit}"
            
            if field == "mass":
                if body["type"] == "planet":
                    body["planet_dropdown_selected"] = label
                elif body["type"] == "star":
                    body["star_mass_dropdown_selected"] = label
                elif body["type"] == "moon":
                    body["moon_dropdown_selected"] = label
            
            # Recalculate habitability score immediately
            # NOTE: update_selected_body_property already triggers _update_planet_scores,
            # but we call it again here to ensure ML gets all updated values after
            # any additional processing (e.g., temperature/atmosphere updates below)
            if body["type"] == "planet":
                # Temporary set selected body to this body to reuse _update_planet_scores
                old_selected = self.selected_body
                self.selected_body = body
                # _update_planet_scores reads ALL current parameters fresh from body dict
                self._update_planet_scores()
                self.selected_body = old_selected
            
            elif field == "semi_major_axis":
                if body["type"] == "planet":
                    body["planet_orbital_distance_dropdown_selected"] = label
                elif body["type"] == "moon":
                    body["moon_orbital_distance_dropdown_selected"] = label
            elif field == "radius":
                if body["type"] == "planet":
                    body["planet_radius_dropdown_selected"] = label
                elif body["type"] == "moon":
                    body["moon_radius_dropdown_selected"] = label
            elif field == "age":
                if body["type"] == "planet":
                    body["planet_age_dropdown_selected"] = label
                elif body["type"] == "star":
                    body["star_age_dropdown_selected"] = label
                elif body["type"] == "moon":
                    body["moon_age_dropdown_selected"] = label
            elif field == "temperature":
                if body["type"] == "planet":
                    body["planet_temperature_dropdown_selected"] = label
                elif body["type"] == "moon":
                    body["moon_temperature_dropdown_selected"] = label
            elif field == "density":
                if body["type"] == "planet":
                    body["planet_density_dropdown_selected"] = label
            elif field == "gravity":
                if body["type"] == "planet":
                    body["planet_gravity_dropdown_selected"] = label
                elif body["type"] == "moon":
                    body["moon_gravity_dropdown_selected"] = label
            elif field == "atmosphere":
                body["planet_atmosphere_dropdown_selected"] = label
            elif field == "gravity":
                if body["type"] == "planet":
                    body["planet_gravity_dropdown_selected"] = label
                elif body["type"] == "moon":
                    body["moon_gravity_dropdown_selected"] = label
            
            # For semi-major axis, also update position and orbit grid
            if internal_field == "semiMajorAxis":
                parent_star = next((b for b in self.placed_bodies if b["name"] == body.get("parent")), None)
                if parent_star is None and body.get("parent_obj"):
                    parent_star = body["parent_obj"]
                if parent_star:
                    self.compute_planet_position(body, parent_star)
                self.generate_orbit_grid(body)
            
            # For temperature/atmosphere, recalculate greenhouse effects if applicable
            if field in ["temperature", "atmosphere"] and body["type"] == "planet":
                eq_temp = body.get("equilibrium_temperature", 255.0)
                offset = body.get("greenhouse_offset", 0.0)
                self.update_selected_body_property("temperature", eq_temp + offset, "temperature")
                self._update_planet_scores()

            print(f"Applied custom {field}={value} (internal {internal_field}) to body ID={body.get('id')}")

        self.show_custom_modal = False
        self.pending_custom_field = None
        self.pending_custom_body_id = None
        self.pending_custom_value = None

    def cancel_custom_modal(self):
        """Revert dropdown selection and close modal."""
        self.show_custom_modal = False
        
        # Revert dropdown to previous selection
        if self.previous_dropdown_selection and self.pending_custom_field:
            body = self.bodies_by_id.get(self.pending_custom_body_id) if self.pending_custom_body_id else self.get_selected_body()
            if body:
                field = self.pending_custom_field
                if field == "mass":
                    if body["type"] == "planet": body["planet_dropdown_selected"] = self.previous_dropdown_selection
                    elif body["type"] == "star": body["star_mass_dropdown_selected"] = self.previous_dropdown_selection
                    elif body["type"] == "moon": body["moon_dropdown_selected"] = self.previous_dropdown_selection
                elif field == "semi_major_axis":
                    if body["type"] == "planet": body["planet_orbital_distance_dropdown_selected"] = self.previous_dropdown_selection
                    elif body["type"] == "moon": body["moon_orbital_distance_dropdown_selected"] = self.previous_dropdown_selection
                elif field == "radius":
                    if body["type"] == "planet": body["planet_radius_dropdown_selected"] = self.previous_dropdown_selection
                    elif body["type"] == "moon": body["moon_radius_dropdown_selected"] = self.previous_dropdown_selection
                elif field == "age":
                    if body["type"] == "planet": body["planet_age_dropdown_selected"] = self.previous_dropdown_selection
                    elif body["type"] == "star": body["star_age_dropdown_selected"] = self.previous_dropdown_selection
                    elif body["type"] == "moon": body["moon_age_dropdown_selected"] = self.previous_dropdown_selection

        self.pending_custom_field = None
        self.pending_custom_body_id = None
        self.pending_custom_value = None
        self.previous_dropdown_selection = None

    def draw_custom_modal(self):
        """Draw the centered custom value input modal."""
        if not self.show_custom_modal:
            return

        # Modal dimensions
        modal_width = 400
        modal_height = 300
        modal_rect = pygame.Rect((self.width - modal_width) // 2, (self.height - modal_height) // 2, modal_width, modal_height)

        # Draw semi-transparent overlay
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        self.screen.blit(overlay, (0, 0))

        # Draw modal background
        pygame.draw.rect(self.screen, self.WHITE, modal_rect, border_radius=10)
        pygame.draw.rect(self.screen, self.BLUE, modal_rect, 2, border_radius=10)

        # Title
        title_surf = self.font.render(self.custom_modal_title, True, self.BLACK)
        title_rect = title_surf.get_rect(midtop=(modal_rect.centerx, modal_rect.top + 20))
        self.screen.blit(title_surf, title_rect)

        # Helper text (wrapped if too long)
        helper_y = self._render_wrapped_text(self.custom_modal_helper, self.subtitle_font, self.GRAY, modal_width - 40, modal_rect.centerx, title_rect.bottom + 10)

        # Input field (position based on wrapped text bottom)
        input_rect = pygame.Rect(modal_rect.left + 50, helper_y + 15, modal_width - 100, 40)
        pygame.draw.rect(self.screen, self.LIGHT_GRAY, input_rect, border_radius=5)
        pygame.draw.rect(self.screen, self.BLUE, input_rect, 1, border_radius=5)
        
        # Input text
        display_text = self.custom_modal_text + ("|" if (pygame.time.get_ticks() // 500) % 2 == 0 else "")
        text_surf = self.font.render(display_text, True, self.BLACK)
        text_rect = text_surf.get_rect(midleft=(input_rect.left + 10, input_rect.centery))
        self.screen.blit(text_surf, text_rect)

        # Unit label
        if self.custom_modal_unit:
            unit_surf = self.subtitle_font.render(self.custom_modal_unit, True, self.BLACK)
            unit_rect = unit_surf.get_rect(midleft=(input_rect.right + 5, input_rect.centery))
            self.screen.blit(unit_surf, unit_rect)

        # Error message (wrapped if too long)
        if self.custom_modal_error:
            self._render_wrapped_text(self.custom_modal_error, self.subtitle_font, self.RED, modal_width - 40, modal_rect.centerx, input_rect.bottom + 10)

        # Buttons
        button_width = 100
        button_height = 40
        
        # Apply button
        apply_btn_rect = pygame.Rect(modal_rect.centerx - 110, modal_rect.bottom - 60, button_width, button_height)
        apply_color = self.BLUE if not self.custom_modal_error and self.custom_modal_text else self.GRAY
        pygame.draw.rect(self.screen, apply_color, apply_btn_rect, border_radius=5)
        apply_text = self.subtitle_font.render("Apply", True, self.WHITE)
        self.screen.blit(apply_text, apply_text.get_rect(center=apply_btn_rect.center))
        self.apply_btn_rect = apply_btn_rect # Store for click detection

        # Cancel button
        cancel_btn_rect = pygame.Rect(modal_rect.centerx + 10, modal_rect.bottom - 60, button_width, button_height)
        pygame.draw.rect(self.screen, self.RED, cancel_btn_rect, border_radius=5)
        cancel_text = self.subtitle_font.render("Cancel", True, self.WHITE)
        self.screen.blit(cancel_text, cancel_text.get_rect(center=cancel_btn_rect.center))
        self.cancel_btn_rect = cancel_btn_rect # Store for click detection

    def _parse_input_value(self, input_text):
        """Parse input text that can be in decimal or scientific notation, with optional units."""
        if not input_text or input_text.strip() == "":
            return None
        
        # Remove unit characters for parsing
        text = input_text.strip().lower()
        # Remove common unit suffixes
        for unit in ['kg', 'm☾', 'm⊕', 'm☉', 'k', 'g', '☾', '⊕', '☉']:
            if text.endswith(unit):
                text = text[:-len(unit)].strip()
                break
        
        try:
            # Try to parse as float (handles both decimal and scientific notation)
            value = float(text)
            return value
        except ValueError:
            return None
    
    def _render_wrapped_text(self, text, font, color, max_width, center_x, top_y):
        """Render multi-line wrapped text centered at center_x."""
        words = text.split(' ')
        lines = []
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            if font.size(test_line)[0] <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
        if current_line:
            lines.append(' '.join(current_line))
        
        y = top_y
        for line in lines:
            surf = font.render(line, True, color)
            rect = surf.get_rect(midtop=(center_x, y))
            self.screen.blit(surf, rect)
            y += font.get_linesize()
        return y # Return new bottom y

    def _render_tooltip(self, rect, text, font, color=(128, 128, 128)):
        """Render tooltip text in an input field."""
        if not text:
            return
        tooltip_surface = font.render(text, True, color)
        tooltip_rect = tooltip_surface.get_rect(midleft=(rect.left + 5, rect.centery))
        self.screen.blit(tooltip_surface, tooltip_rect)
    
    def _render_generic_tooltip(self, text, surface=None):
        """Render a generic floating tooltip at the mouse position. Supports multi-line text."""
        if not text:
            return
        
        if surface is None:
            surface = self.screen
            
        mouse_pos = pygame.mouse.get_pos()
        padding = 6
        
        # Handle multi-line text
        if isinstance(text, list):
            lines = text
        else:
            lines = [text]
        
        # Render each line
        rendered_lines = []
        max_width = 0
        total_height = 0
        for line in lines:
            if line:
                line_surface = self.subtitle_font.render(str(line), True, self.WHITE)
                rendered_lines.append(line_surface)
                max_width = max(max_width, line_surface.get_width())
                total_height += line_surface.get_height() + 2  # 2px spacing between lines
            else:
                rendered_lines.append(None)
                total_height += self.subtitle_font.get_height() + 2
        
        if not rendered_lines:
            return
        
        # Create background rect
        bg_rect = pygame.Rect(0, 0, max_width + padding * 2, total_height + padding * 2)
        
        # Position slightly offset from mouse
        bg_rect.topleft = (mouse_pos[0] + 12, mouse_pos[1] - bg_rect.height // 2)
        
        # Ensure tooltip stays on screen
        if bg_rect.right > self.width:
            bg_rect.right = mouse_pos[0] - 12
        if bg_rect.bottom > self.height:
            bg_rect.bottom = self.height - 5
        if bg_rect.top < 0:
            bg_rect.top = 5
            
        tooltip_bg = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
        tooltip_bg.fill((0, 0, 0, 180))
        surface.blit(tooltip_bg, bg_rect.topleft)
        
        # Blit each line
        y_offset = padding
        for line_surface in rendered_lines:
            if line_surface:
                surface.blit(line_surface, (bg_rect.left + padding, bg_rect.top + y_offset))
                y_offset += line_surface.get_height() + 2
            else:
                y_offset += self.subtitle_font.get_height() + 2

    def _update_planet_scores(self):
        """
        Update habitability score using ML model.
        
        CRITICAL: This function reads ALL current parameters directly from the body dict.
        It is called AFTER all cascading parameter updates complete (e.g., mass → radius → density,
        AU → flux → temperature). The ML model receives the complete, current state of all parameters.
        """
        if not self.selected_body or self.selected_body.get('type') != 'planet':
            return
        
        # Use ML calculator if available
        if hasattr(self, 'ml_calculator') and self.ml_calculator:
            # CRITICAL: Always read from registry to ensure we have the latest body dict
            # This prevents reading stale values if body reference changed during updates
            body_id = self.selected_body.get('id')
            if body_id and body_id in self.bodies_by_id:
                body = self.bodies_by_id[body_id]
            else:
                body = self.selected_body
            
            # Find host star (simplified: take the first star found)
            star = next((b for b in self.placed_bodies if b.get("type") == "star"), None)
            if star:
                # CRITICAL: Read ALL parameters directly from body dict to ensure we get
                # the latest values after all cascading updates (mass→radius, AU→flux→temp, etc.)
                
                # Planet parameters - read fresh from body dict
                pl_rade = float(body.get('radius', 1.0))
                pl_masse = float(body.get('mass', 1.0))
                pl_orbper = float(body.get('orbital_period', 365.25))
                pl_orbeccen = float(body.get('eccentricity', 0.0))
                
                # Stellar parameters - read fresh from star dict
                st_teff = float(star.get('temperature', 5778.0))
                st_mass = float(star.get('mass', 1.0)) # in solar masses
                st_rad = float(star.get('radius', 1.0)) # already in Solar Radii (R☉)
                
                # Calculate stellar luminosity from current stellar parameters
                # Stefan-Boltzmann: L = R^2 * (T/T_sun)^4
                st_lum = (st_rad**2) * ((st_teff / 5778.0)**4)
                
                # Calculate insolation flux from current orbital distance and stellar luminosity
                # F = L / d^2 (in Earth flux units)
                dist_au = float(body.get('semiMajorAxis', 1.0))
                pl_insol = st_lum / (dist_au**2)
                
                # Build feature dict with ALL current parameters for ML model
                # This ensures ML receives complete, up-to-date state after all physics updates
                features = {
                    "pl_rade": float(pl_rade),      # Current radius (may have changed via mass coupling)
                    "pl_masse": float(pl_masse),    # Current mass
                    "pl_orbper": float(pl_orbper),  # Current orbital period (may have changed via AU coupling)
                    "pl_orbeccen": float(pl_orbeccen), # Current eccentricity
                    "pl_insol": float(pl_insol),     # Current insolation (calculated from updated AU and stellar L)
                    "st_teff": float(st_teff),       # Current stellar temperature
                    "st_mass": float(st_mass),       # Current stellar mass
                    "st_rad": float(st_rad),         # Current stellar radius
                    "st_lum": float(st_lum)          # Current stellar luminosity (calculated from T and R)
                }
                
                ml_score = self.ml_calculator.predict(features)
                
                # UNCONDITIONAL Debug logging to help identify unit issues
                print(f"[ML_DIAGNOSTICS] Planet: {self.selected_body.get('name')} | Score: {ml_score:.2f}%")
                print(f"  Inputs: R={features['pl_rade']:.3f} M={features['pl_masse']:.3f} P={features['pl_orbper']:.1f} E={features['pl_orbeccen']:.3f}")
                print(f"  Stellar: T={features['st_teff']:.0f} M={features['st_mass']:.2f} R={features['st_rad']:.2f} L={features['st_lum']:.3f} Insol={features['pl_insol']:.4f}")
                
                self.selected_body['habit_score'] = ml_score
                # Also update H for compatibility
                self.selected_body['H'] = ml_score / 100.0
                return

        # Fallback to basic calculation if ML is not available
        temp = self.selected_body.get('temperature', 288)
        ideal_temp = 288.0  # Earth's average temperature
        temp_range = 100.0  # Temperature range for habitability
        
        f_T = max(0.0, 1.0 - abs(temp - ideal_temp) / temp_range)
        self.selected_body['f_T'] = f_T
        
        # Calculate composite habitability H (simple weighted average)
        f_M = 1.0  # Mass factor (placeholder)
        f_R = 1.0  # Radius factor (placeholder)
        f_G = 1.0  # Gravity factor (placeholder)
        
        # Simple composite score
        H = (f_T * 0.4 + f_M * 0.2 + f_R * 0.2 + f_G * 0.2)
        self.selected_body['H'] = H
        self.selected_body['habit_score'] = H * 100.0  # Convert to percentage
        
    def _format_value(self, value, unit, allow_scientific=True, for_dropdown=True):
        if value is None:
            return "Custom"
        
        # For dropdown menus, always show human-readable values
        if for_dropdown:
            # Special handling for specific units to show human-readable values
            if unit == 'days':
                if abs(value) < 1:
                    return f"{value:.4f} {unit}"
                elif abs(value) < 10:
                    return f"{value:.3f} {unit}"
                elif abs(value) < 100:
                    return f"{value:.2f} {unit}"
                else:
                    return f"{value:.0f} {unit}"
            elif unit == 'Gyr':
                if abs(value) < 1:
                    return f"{value:.1f} ({unit})"
                else:
                    return f"{value:.1f} ({unit})"
            elif unit == 'M⊕' or unit == 'M☉' or unit == 'M☾':
                # Special adaptive formatting for moon masses (M☾)
                if unit == 'M☾':
                    if abs(value) < 1e-3:
                        return f"{value:.1e} {unit}"
                    elif abs(value) < 0.1:
                        return f"{value:.4f} {unit}"
                    else:
                        return f"{value:.2f} {unit}"
                # Standard formatting for other planetary masses
                if abs(value) < 0.01:
                    return f"{value:.4f} {unit}"
                elif abs(value) < 1:
                    return f"{value:.3f} {unit}"
                elif abs(value) < 10:
                    return f"{value:.2f} {unit}"
                else:
                    return f"{value:.1f} {unit}"
            elif unit == 'kg':
                # For kg units, use scientific notation for large values
                if abs(value) >= 1e12:
                    return f"{value:.2e} {unit}"
                elif abs(value) >= 1e9:
                    return f"{value/1e9:.1f}×10⁹ {unit}"
                elif abs(value) >= 1e6:
                    return f"{value/1e6:.1f}×10⁶ {unit}"
                elif abs(value) >= 1e3:
                    return f"{value/1e3:.1f}×10³ {unit}"
                else:
                    return f"{value:.0f} {unit}"
            elif unit == 'R🜨' or unit == 'R☉':
                if abs(value) < 1:
                    return f"{value:.3f} {unit}"
                else:
                    return f"{value:.2f} {unit}"
            elif unit == 'km':
                if abs(value) < 1000:
                    return f"{value:.0f} {unit}"
                else:
                    return f"{value:.0f} {unit}"
            elif unit == 'AU':
                if abs(value) < 1:
                    return f"{value:.3f} {unit}"
                else:
                    return f"{value:.2f} {unit}"
            elif unit == 'K':
                return f"{value:.0f} {unit}"
            elif unit == 'm/s²':
                if abs(value) < 1:
                    return f"{value:.3f} {unit}"
                else:
                    return f"{value:.2f} {unit}"
            elif unit == 'L☉':
                if abs(value) < 1:
                    return f"{value:.3f} {unit}"
                else:
                    return f"{value:.2f} {unit}"
            elif unit == 'g/cm³':
                return f"{value:.2f} {unit}"
            elif unit == 'EFU':
                if abs(value) < 1:
                    return f"{value:.3f} {unit}"
                else:
                    return f"{value:.2f} {unit}"
            elif unit == '[Fe/H]':
                return f"{value:+.2f} {unit}"
            else:
                # Generic formatting for other units
                if abs(value) < 1:
                    return f"{value:.3f} {unit}"
                elif abs(value) < 10:
                    return f"{value:.2f} {unit}"
                else:
                    return f"{value:.1f} {unit}"
        
        # For custom input display, use scientific notation rule
        else:
            # Scientific notation rule: < 0.001 or ≥ 10,000
            if abs(value) < 0.001 or abs(value) >= 10000:
                return f"{value:.2e}"
            else:
                # Decimal format with 2-3 significant digits
                if abs(value) < 1:
                    return f"{value:.3f}"
                elif abs(value) < 10:
                    return f"{value:.2f}"
                else:
                    return f"{value:.1f}"

    # --- In all dropdown rendering code blocks (moons, planets, stars) ---
    # Replace formatting like f"{name} ({value:.3f} M☾)" with:
    # f"{name} (" + self._format_value(value, 'M☾') + ")"
    # And similarly for all other units (Earth masses, R🜨, km, AU, L☉, etc.)
    # For example:
    # dropdown_text = f"{name} (" + self._format_value(value, 'M☾') + ")"
    # or
    # dropdown_text = f"{name} (" + self._format_value(value, 'Earth masses') + ")"
    # etc.