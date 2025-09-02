import pygame
import numpy as np
from typing import List, Tuple
from simulation_engine import CelestialBody, SimulationEngine

class SolarSystemVisualizer:
    def __init__(self, width: int = 1200, height: int = 800):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("AIET - Solar System Simulator")
        self.clock = pygame.time.Clock()
        self.scale = 100  # pixels per AU
        self.center = np.array([width/2, height/2])
        
        # Initialize fonts consistently using pygame.font.Font
        self.font = pygame.font.Font(None, 36)
        self.title_font = pygame.font.Font(None, 72)
        self.subtitle_font = pygame.font.Font(None, 24)
        self.tab_font = pygame.font.Font(None, 20)
        self.button_font = pygame.font.Font(None, 36)
        
        # Screen states
        self.show_home_screen = True
        self.show_simulation_builder = False
        self.show_simulation = False
        self.show_customization_panel = False
        
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
        
        # Track placed celestial bodies
        self.placed_bodies = []
        self.body_counter = {"moon": 0, "planet": 0, "star": 0}
        
        # Physics parameters
        self.G = 0.5  # Gravitational constant (reduced for more stable orbits)
        self.time_step = 0.05  # Simulation time step (reduced for smoother motion)
        self.orbit_points = {}  # Store orbit points for visualization
        self.orbit_history = {}  # Store orbit history for trail effect
        self.orbit_grid_points = {}  # Store grid points for orbit visualization
        
        # Spacetime grid parameters
        self.grid_size = 50  # Size of grid cells
        
        # Rotation parameters
        self.rotation_speed = 0.1  # Base rotation speed
        
        # Customization panel
        self.selected_body = None
        self.customization_panel_width = 400  # Increased from 300 to 400 for more space
        self.customization_panel = pygame.Rect(self.width - self.customization_panel_width, 0, 
                                             self.customization_panel_width, self.height)
        self.mass_input_rect = pygame.Rect(self.width - self.customization_panel_width + 50, 120, 
                                         self.customization_panel_width - 100, 30)
        self.mass_input_active = False
        self.mass_input_text = ""
        self.mass_min = 0.1  # Minimum mass (0.1 Earth masses)
        self.mass_max = 1000.0  # Maximum mass (1000 Earth masses)
        
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
        self.planet_dropdown_selected = "Earth"
        self.planet_dropdown_visible = False
        self.show_custom_mass_input = False
        
        # Planet age dropdown properties
        self.planet_age_dropdown_rect = pygame.Rect(self.width - self.customization_panel_width + 50, 180,
                                                  self.customization_panel_width - 100, 30)
        self.planet_age_dropdown_active = False
        self.planet_age_dropdown_options = [
            ("0.1 Gyr", 0.1),
            ("1.0 Gyr (1 billion years)", 1.0),
            ("4.6 Gyr (Earth’s age)", 4.6),
            ("6 Gyr", 6.0),
            ("10.0 Gyr", 10.0),
            ("Custom", None)
        ]
        self.planet_age_dropdown_selected = "4.6 Gyr (Earth’s age)"
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
        
        # Planet gravity dropdown properties
        self.planet_gravity_dropdown_rect = pygame.Rect(self.width - self.customization_panel_width + 50, 540,
                                                      self.customization_panel_width - 100, 30)
        self.planet_gravity_dropdown_active = False
        self.planet_gravity_dropdown_options = [
            ("Mercury", 3.7),
            ("Mars", 3.7),
            ("Venus", 8.87),
            ("Earth", 9.81),
            ("Uranus", 8.69),
            ("Neptune", 11.15),
            ("Saturn", 10.44),
            ("Jupiter", 24.79),
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
            ("Deimos", 0.000003),
            ("Phobos", 0.0000001),
            ("Europa", 0.0073),
            ("Enceladus", 0.0001),
            ("Titan", 0.0135),
            ("Ganymede", 0.0148),
            ("Callisto", 0.0107),
            ("Moon", 0.0123),
            ("Custom", None)
        ]
        self.moon_dropdown_selected = "Moon"
        self.moon_dropdown_visible = False
        self.show_custom_moon_mass_input = False
        
        # Moon age dropdown properties
        self.moon_age_dropdown_rect = pygame.Rect(self.width - self.customization_panel_width + 50, 180,
                                                self.customization_panel_width - 100, 30)
        self.moon_age_dropdown_active = False
        self.moon_age_dropdown_options = [
            ("Deimos", 1.0),
            ("Phobos", 1.0),
            ("Europa", 4.6),
            ("Enceladus", 4.6),
            ("Titan", 10.0),
            ("Ganymede", 10.0),
            ("Callisto", 10.0),
            ("Moon", 4.5),
            ("Custom", None)
        ]
        self.moon_age_dropdown_selected = "Moon"
        self.moon_age_dropdown_visible = False
        self.show_custom_moon_age_input = False
        
        # Moon radius dropdown properties
        self.moon_radius_dropdown_rect = pygame.Rect(self.width - self.customization_panel_width + 50, 240,
                                                   self.customization_panel_width - 100, 30)
        self.moon_radius_dropdown_active = False
        self.moon_radius_dropdown_options = [
            ("Deimos", 6.2),
            ("Phobos", 11.3),
            ("Europa", 1560.8),
            ("Enceladus", 252.1),
            ("Titan", 2574.7),
            ("Ganymede", 2634.1),
            ("Callisto", 2410.3),
            ("Moon", 1737.4),
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
            ("Europa", 670900),
            ("Enceladus", 237950),
            ("Titan", 1221870),
            ("Ganymede", 1070400),
            ("Callisto", 1882700),
            ("Moon", 384400),
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
            ("Europa", 3.55),
            ("Enceladus", 1.37),
            ("Titan", 15.95),
            ("Ganymede", 7.15),
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
            ("Europa", 102),
            ("Enceladus", 75),
            ("Ganymede", 110),
            ("Callisto", 134),
            ("Titan", 94),
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
            ("Europa", 1.314),
            ("Titan", 1.352),
            ("Ganymede", 1.428),
            ("Callisto", 1.235),
            ("Moon", 1.62),
            ("Custom", None)
        ]
        self.moon_gravity_dropdown_selected = "Moon"
        self.moon_gravity_dropdown_visible = False
        self.show_custom_moon_gravity_input = False
        
        # Close button for customization panel
        self.close_button_size = 20
        self.close_button = pygame.Rect(self.width - self.close_button_size - 10, 10, 
                                      self.close_button_size, self.close_button_size)
        
        # Age input properties
        self.age_input_rect = pygame.Rect(self.width - self.customization_panel_width + 50, 180, 
                                        self.customization_panel_width - 100, 30)
        self.age_input_active = False
        self.age_input_text = ""
        self.age_min = 0.0  # 0 Gyr
        self.age_max = 10.0  # 10 Gyr
        
        # Luminosity input properties
        self.luminosity_dropdown_rect = pygame.Rect(self.width - self.customization_panel_width + 50, 360,
                                                  self.customization_panel_width - 100, 30)
        self.luminosity_dropdown_active = False
        self.luminosity_dropdown_options = [
            ("Red Dwarf", 0.04),
            ("Orange Dwarf", 0.15),
            ("White Dwarf (Sun)", 1.00),
            ("Bright F-type Star", 2.00),
            ("Blue A-type Star", 25.00),
            ("B-type Giant", 10000.00),
            ("O-type Supergiant", 100000.00),
            ("Custom", None)
        ]
        self.luminosity_dropdown_selected = "White Dwarf (Sun)"  # Default to Sun
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
            ("O-type (Blue, ~40,000 K)", 40000, (0, 0, 255)),
            ("B-type (Blue-white, ~20,000 K)", 20000, (173, 216, 230)),
            ("A-type (White, ~10,000 K)", 10000, (255, 255, 255)),
            ("F-type (Yellow-white, ~7,500 K)", 7500, (255, 255, 224)),
            ("G-type (Yellow, Sun, ~5,800 K)", 5800, (255, 255, 0)),
            ("K-type (Orange, ~4,500 K)", 4500, (255, 165, 0)),
            ("M-type (Red, ~3,000 K)", 3000, (255, 0, 0)),
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
            ("0.08 M☉ (Hydrogen-burning limit)", 0.08),
            ("0.5 M☉", 0.5),
            ("1.0 M☉ (Sun)", 1.0),
            ("1.5 M☉", 1.5),
            ("3.0 M☉", 3.0),
            ("5.0 M☉", 5.0),
            ("10.0 M☉", 10.0),
            ("20.0 M☉", 20.0),
            ("50.0 M☉", 50.0),
            ("100.0 M☉", 100.0),
            ("Custom", None)
        ]
        self.star_mass_dropdown_selected = "1.0 M☉ (Sun)"  # Default to Sun
        self.star_mass_dropdown_visible = False
        self.show_custom_star_mass_input = False
    
        # Star age dropdown properties
        self.star_age_dropdown_rect = pygame.Rect(self.width - self.customization_panel_width + 50, 180,
                                                self.customization_panel_width - 100, 30)
        self.star_age_dropdown_active = False
        self.star_age_dropdown_options = [
            ("1.0 Gyr", 1.0),
            ("Sun (4.6 Gyr)", 4.6),
            ("7.0 Gyr", 7.0),
            ("10.0 Gyr", 10.0),
            ("Custom", None)
        ]
        self.star_age_dropdown_selected = "Sun (4.6 Gyr)"  # Default to Sun
        self.star_age_dropdown_visible = False
        self.show_custom_star_age_input = False
    
        # Star radius dropdown properties
        self.radius_dropdown_rect = pygame.Rect(self.width - self.customization_panel_width + 50, 420,
                                              self.customization_panel_width - 100, 30)
        self.radius_dropdown_active = False
        self.radius_dropdown_options = [
            ("O-type", 10.0),
            ("B-type", 5.0),
            ("A-type", 2.0),
            ("F-type", 1.3),
            ("G-type (Sun)", 1.0),
            ("K-type", 0.8),
            ("M-type", 0.3),
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
        self.activity_dropdown_rect = pygame.Rect(self.width - self.customization_panel_width + 50, 480,
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
        self.metallicity_dropdown_rect = pygame.Rect(self.width - self.customization_panel_width + 50, 540,
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
        self.planet_orbital_distance_dropdown_rect = pygame.Rect(self.width - self.customization_panel_width + 50, 420, self.customization_panel_width - 100, 30)
        self.planet_orbital_distance_dropdown_active = False
        self.planet_orbital_distance_dropdown_options = [
            ("Mercury", 0.39),
            ("Venus", 0.72),
            ("Earth", 1.0),
            ("Mars", 1.52),
            ("Jupiter", 5.2),
            ("Saturn", 9.58),
            ("Uranus", 19.2),
            ("Neptune", 30.1),
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
        self.planet_orbital_eccentricity_dropdown_rect = pygame.Rect(self.width - self.customization_panel_width + 50, 480, self.customization_panel_width - 100, 30)
        self.planet_orbital_eccentricity_dropdown_active = False
        self.planet_orbital_eccentricity_dropdown_options = [
            ("Circular Orbit", 0.0),
            ("Earth", 0.0167),
            ("Mars", 0.093),
            ("Mercury", 0.205),
            ("Pluto", 0.248),
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
        self.planet_orbital_period_dropdown_rect = pygame.Rect(self.width - self.customization_panel_width + 50, 540, self.customization_panel_width - 100, 30)
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
        self.planet_stellar_flux_dropdown_rect = pygame.Rect(self.width - self.customization_panel_width + 50, 600, self.customization_panel_width - 100, 30)
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
        self.planet_density_dropdown_rect = pygame.Rect(self.width - self.customization_panel_width + 50, 660, self.customization_panel_width - 100, 30)
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
    
    def handle_events(self) -> bool:
        """Handle pygame events. Returns False if the window should close."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if self.show_home_screen:
                    # Check if click is within the create button area
                    button_center_x = self.width//2
                    button_center_y = self.height*3/4
                    button_width = 200
                    button_height = 50
                    button_rect = pygame.Rect(
                        button_center_x - button_width//2,
                        button_center_y - button_height//2,
                        button_width,
                        button_height
                    )
                    if button_rect.collidepoint(event.pos):
                        self.show_home_screen = False
                        self.show_simulation_builder = True
                elif self.show_simulation_builder or self.show_simulation:
                    # Check if click is in the customization panel
                    if self.show_customization_panel and self.customization_panel.collidepoint(event.pos):
                        print(f'DEBUG: Mouse click at {event.pos}')
                        print(f'DEBUG: Orbital distance rect: {self.planet_orbital_distance_dropdown_rect}')
                        # Check if close button was clicked
                        if self.close_button.collidepoint(event.pos):
                            self.show_customization_panel = False
                            self.selected_body = None
                            self.mass_input_active = False
                            self.planet_dropdown_active = False
                            self.planet_dropdown_visible = False
                            self.luminosity_input_active = False
                        # Handle planet dropdown (only for planets)
                        elif (self.selected_body and self.selected_body.get('type') == 'planet' and 
                              self.planet_dropdown_rect.collidepoint(event.pos)):
                            self.planet_dropdown_active = True
                            self.mass_input_active = False
                            self.planet_dropdown_visible = True
                            self.create_dropdown_surface()
                        # Check if clicked on a planet option (only for planets)
                        elif (self.selected_body and self.selected_body.get('type') == 'planet' and 
                              self.planet_dropdown_visible):
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
                                        self.show_custom_mass_input = True
                                        self.mass_input_active = True
                                        self.mass_input_text = f"{self.selected_body.get('mass', 1.0):.3f}"
                                    else:
                                        self.selected_body["mass"] = mass
                                        self.show_custom_mass_input = False
                                        if self.selected_body["type"] != "star":
                                            self.generate_orbit_grid(self.selected_body)
                                    self.planet_dropdown_selected = planet_name
                                    self.planet_age_dropdown_selected = "4.5 Gyr (Earth’s age)"
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
                                        self.show_custom_age_input = True
                                        self.age_input_active = True
                                        self.age_input_text = f"{self.selected_body.get('age', 0.0):.1f}"
                                    else:
                                        self.selected_body["age"] = age
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
                                        self.show_custom_radius_input = True
                                    else:
                                        self.selected_body["radius"] = radius
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
                                        self.show_custom_planet_temperature_input = True
                                    else:
                                        self.selected_body["temperature"] = temp
                                        self.show_custom_planet_temperature_input = False
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
                        # Handle planet gravity dropdown (only for planets)
                        elif (self.selected_body and self.selected_body.get('type') == 'planet' and 
                              self.planet_gravity_dropdown_rect.collidepoint(event.pos)):
                            self.planet_gravity_dropdown_active = True
                            self.planet_gravity_dropdown_visible = True
                            self.create_dropdown_surface()
                        # Handle planet orbital distance dropdown (only for planets)
                        elif (self.selected_body and self.selected_body.get('type') == 'planet' and 
                              self.planet_orbital_distance_dropdown_rect.collidepoint(event.pos)):
                            print('DEBUG: Orbital distance dropdown clicked')
                            self.planet_orbital_distance_dropdown_active = True
                            self.planet_orbital_distance_dropdown_visible = True
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
                            self.planet_orbital_eccentricity_dropdown_active = False
                            self.planet_orbital_eccentricity_dropdown_visible = False
                            self.planet_stellar_flux_dropdown_active = False
                            self.planet_stellar_flux_dropdown_visible = False
                            self.create_dropdown_surface()
                        # Handle planet orbital eccentricity dropdown (only for planets)
                        elif (self.selected_body and self.selected_body.get('type') == 'planet' and 
                              self.planet_orbital_eccentricity_dropdown_rect.collidepoint(event.pos)):
                            print('DEBUG: Orbital eccentricity dropdown clicked')
                            self.planet_orbital_eccentricity_dropdown_active = True
                            self.planet_orbital_eccentricity_dropdown_visible = True
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
                            self.planet_stellar_flux_dropdown_active = False
                            self.planet_stellar_flux_dropdown_visible = False
                            self.create_dropdown_surface()
                        # Handle planet orbital period dropdown (only for planets)
                        elif (self.selected_body and self.selected_body.get('type') == 'planet' and 
                              self.planet_orbital_period_dropdown_rect.collidepoint(event.pos)):
                            print('DEBUG: Orbital period dropdown clicked')
                            self.planet_orbital_period_dropdown_active = True
                            self.planet_orbital_period_dropdown_visible = True
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
                            self.planet_stellar_flux_dropdown_active = False
                            self.planet_stellar_flux_dropdown_visible = False
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
                                        self.show_custom_planet_gravity_input = True
                                    else:
                                        self.selected_body["gravity"] = gravity
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
                                        self.show_custom_orbital_distance_input = True
                                        self.orbital_distance_input_active = True
                                        self.orbital_distance_input_text = f"{self.selected_body.get('semiMajorAxis', 1.0):.2f}"
                                    else:
                                        self.selected_body["semiMajorAxis"] = dist
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
                                        self.orbital_eccentricity_input_text = f"{self.selected_body.get('eccentricity', 0.017):.3f}"
                                    else:
                                        self.selected_body["eccentricity"] = ecc
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
                                        self.orbital_period_input_text = f"{self.selected_body.get('orbital_period', 365):.0f}"
                                    else:
                                        self.selected_body["orbital_period"] = period
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
                                        self.selected_body["stellarFlux"] = flux
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
                                        self.show_custom_planet_density_input = True
                                        self.planet_density_input_text = f"{self.selected_body.get('density', 5.51):.2f}"
                                    else:
                                        self.selected_body["density"] = density
                                        self.show_custom_planet_density_input = False
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
                            self.moon_dropdown_active = True
                            self.mass_input_active = False
                            self.moon_dropdown_visible = True
                            self.create_dropdown_surface()
                        # Handle moon age dropdown (only for moons)
                        elif (self.selected_body and self.selected_body.get('type') == 'moon' and 
                              self.moon_age_dropdown_rect.collidepoint(event.pos)):
                            self.moon_age_dropdown_active = True
                            self.age_input_active = False
                            self.moon_age_dropdown_visible = True
                            self.create_dropdown_surface()
                        # Handle moon radius dropdown (only for moons)
                        elif (self.selected_body and self.selected_body.get('type') == 'moon' and 
                              self.moon_radius_dropdown_rect.collidepoint(event.pos)):
                            self.moon_radius_dropdown_active = True
                            self.moon_radius_dropdown_visible = True
                            self.create_dropdown_surface()
                        # Handle moon orbital distance dropdown (only for moons)
                        elif (self.selected_body and self.selected_body.get('type') == 'moon' and 
                              self.moon_orbital_distance_dropdown_rect.collidepoint(event.pos)):
                            self.moon_orbital_distance_dropdown_active = True
                            self.moon_orbital_distance_dropdown_visible = True
                            self.create_dropdown_surface()
                        # Handle moon orbital period dropdown (only for moons)
                        elif (self.selected_body and self.selected_body.get('type') == 'moon' and 
                              self.moon_orbital_period_dropdown_rect.collidepoint(event.pos)):
                            self.moon_orbital_period_dropdown_active = True
                            self.moon_orbital_period_dropdown_visible = True
                            self.create_dropdown_surface()
                        # Handle moon temperature dropdown (only for moons)
                        elif (self.selected_body and self.selected_body.get('type') == 'moon' and 
                              self.moon_temperature_dropdown_rect.collidepoint(event.pos)):
                            self.moon_temperature_dropdown_active = True
                            self.moon_temperature_dropdown_visible = True
                            self.create_dropdown_surface()
                        # Handle moon gravity dropdown (only for moons)
                        elif (self.selected_body and self.selected_body.get('type') == 'moon' and 
                              self.moon_gravity_dropdown_rect.collidepoint(event.pos)):
                            self.moon_gravity_dropdown_active = True
                            self.moon_gravity_dropdown_visible = True
                            self.create_dropdown_surface()
                        # Check if clicked on a moon option (only for moons)
                        elif (self.selected_body and self.selected_body.get('type') == 'moon' and 
                              self.moon_dropdown_visible):
                            dropdown_y = self.moon_dropdown_rect.bottom
                            for i, (moon_name, mass) in enumerate(self.moon_dropdown_options):
                                option_rect = pygame.Rect(
                                    self.moon_dropdown_rect.left,
                                    dropdown_y + i * 30,
                                    self.moon_dropdown_rect.width,
                                    30
                                )
                                if option_rect.collidepoint(event.pos):
                                    if moon_name == "Custom":
                                        self.show_custom_moon_mass_input = True
                                        self.mass_input_active = True
                                        self.mass_input_text = f"{self.selected_body.get('mass', 1.0):.3f}"
                                    else:
                                        self.selected_body["mass"] = mass
                                        self.show_custom_moon_mass_input = False
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
                                        self.show_custom_moon_age_input = True
                                        self.age_input_active = True
                                        self.age_input_text = f"{self.selected_body.get('age', 0.0):.1f}"
                                    else:
                                        self.selected_body["age"] = age
                                        self.show_custom_moon_age_input = False
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
                                        self.show_custom_moon_radius_input = True
                                    else:
                                        # Scale the radius for visual display (convert km to appropriate pixel size)
                                        # Use a reasonable scale factor for moons
                                        self.selected_body["radius"] = max(5, min(20, radius / 100))  # Scale down and clamp
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
                                        self.show_custom_moon_orbital_distance_input = True
                                    else:
                                        # Scale the orbital distance for visual display
                                        self.selected_body["orbit_radius"] = max(50, min(200, distance / 1000))  # Scale down and clamp
                                        self.show_custom_moon_orbital_distance_input = False
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
                                    else:
                                        self.selected_body["orbitalPeriod"] = period
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
                                        self.show_custom_moon_temperature_input = True
                                    else:
                                        self.selected_body["temperature"] = temp
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
                                        self.show_custom_moon_gravity_input = True
                                    else:
                                        self.selected_body["surfaceGravity"] = gravity
                                        self.show_custom_moon_gravity_input = False
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
                                        self.luminosity_input_text = f"{self.selected_body.get('luminosity', 1.0):.3f}"
                                    else:
                                        self.selected_body["luminosity"] = luminosity
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
                            self.spectral_class_dropdown_active = True
                            self.spectral_class_dropdown_visible = True
                            self.create_dropdown_surface()
                        # Handle star mass dropdown (only for stars)
                        elif (self.selected_body and self.selected_body.get('type') == 'star' and 
                              self.star_mass_dropdown_rect.collidepoint(event.pos)):
                            self.star_mass_dropdown_active = True
                            self.mass_input_active = False
                            self.star_mass_dropdown_visible = True
                            self.create_dropdown_surface()
                        # Handle star age dropdown (only for stars)
                        elif (self.selected_body and self.selected_body.get('type') == 'star' and 
                              self.star_age_dropdown_rect.collidepoint(event.pos)):
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
                                        self.age_input_text = f"{self.selected_body.get('age', 0.0):.1f}"
                                    else:
                                        self.selected_body["age"] = age
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
                                        self.show_custom_star_mass_input = True
                                        self.mass_input_active = True
                                        self.mass_input_text = f"{self.selected_body.get('mass', 1.0):.3f}"
                                    else:
                                        self.selected_body["mass"] = mass * 1000.0  # Convert to Earth masses
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
                                            self.selected_body["temperature"] = temp
                                            self.selected_body["color"] = color
                                        self.spectral_class_dropdown_selected = name
                                    else:  # Custom option
                                        self.show_custom_temperature_input = True
                                        self.temperature_input_active = True
                                        self.temperature_input_text = f"{self.selected_body.get('temperature', 5800):.0f}"
                                    self.spectral_class_dropdown_visible = False
                                    self.spectral_class_dropdown_active = False
                            else:
                                # Click outside the spectral class dropdown area
                                self.spectral_class_dropdown_visible = False
                                self.spectral_class_dropdown_active = False
                        # Handle radius dropdown (only for stars)
                        elif (self.selected_body and self.selected_body.get('type') == 'star' and 
                              self.radius_dropdown_rect.collidepoint(event.pos)):
                            self.radius_dropdown_active = True
                            self.radius_input_active = False
                            self.radius_dropdown_visible = True
                            self.create_dropdown_surface()
                        # Handle activity level dropdown (only for stars)
                        elif (self.selected_body and self.selected_body.get('type') == 'star' and 
                              self.activity_dropdown_rect.collidepoint(event.pos)):
                            self.activity_dropdown_active = True
                            self.activity_input_active = False
                            self.activity_dropdown_visible = True
                            self.create_dropdown_surface()
                        elif (self.selected_body and self.selected_body.get('type') == 'star' and 
                              self.metallicity_dropdown_rect.collidepoint(event.pos)):
                            self.metallicity_dropdown_active = True
                            self.metallicity_input_active = False
                            self.metallicity_dropdown_visible = True
                            self.create_dropdown_surface()
                        # Handle dropdown option selection
                        if (self.planet_dropdown_visible or self.moon_dropdown_visible or 
                            self.star_mass_dropdown_visible or self.luminosity_dropdown_visible or
                            self.planet_age_dropdown_visible or self.star_age_dropdown_visible or
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
                                        name, value = self.planet_dropdown_options[i]
                                        if value is not None:
                                            self.selected_body["mass"] = value
                                        else:
                                            self.show_custom_planet_mass_input = True
                                            self.planet_mass_input_active = True
                                        self.planet_dropdown_selected = name
                                        self.planet_age_dropdown_selected = "4.5 Gyr (Earth’s age)"
                                        self.planet_gravity_dropdown_selected = "Earth"
                                        self.planet_dropdown_visible = False
                                        self.planet_dropdown_active = False
                                    elif self.moon_dropdown_visible:
                                        name, value = self.moon_dropdown_options[i]
                                        if value is not None:
                                            self.selected_body["mass"] = value
                                        else:
                                            self.show_custom_moon_mass_input = True
                                            self.moon_mass_input_active = True
                                        self.moon_dropdown_selected = name
                                        self.moon_dropdown_visible = False
                                        self.moon_dropdown_active = False
                                    elif self.moon_age_dropdown_visible:
                                        name, value = self.moon_age_dropdown_options[i]
                                        if value is not None:
                                            self.selected_body["age"] = value
                                        else:
                                            self.show_custom_moon_age_input = True
                                            self.age_input_active = True
                                        self.moon_age_dropdown_selected = name
                                        self.moon_age_dropdown_visible = False
                                        self.moon_age_dropdown_active = False
                                    elif self.moon_radius_dropdown_visible:
                                        name, value = self.moon_radius_dropdown_options[i]
                                        if value is not None:
                                            # Scale the radius for visual display (convert km to appropriate pixel size)
                                            # Use a reasonable scale factor for moons
                                            self.selected_body["radius"] = max(5, min(20, value / 100))  # Scale down and clamp
                                        else:
                                            self.show_custom_moon_radius_input = True
                                        self.moon_radius_dropdown_selected = name
                                        self.moon_radius_dropdown_visible = False
                                        self.moon_radius_dropdown_active = False
                                    elif self.moon_orbital_distance_dropdown_visible:
                                        name, value = self.moon_orbital_distance_dropdown_options[i]
                                        if value is not None:
                                            # Scale the orbital distance for visual display
                                            self.selected_body["orbit_radius"] = max(50, min(200, value / 1000))  # Scale down and clamp
                                        else:
                                            self.show_custom_moon_orbital_distance_input = True
                                        self.moon_orbital_distance_dropdown_selected = name
                                        self.moon_orbital_distance_dropdown_visible = False
                                        self.moon_orbital_distance_dropdown_active = False
                                    elif self.star_mass_dropdown_visible:
                                        name, value = self.star_mass_dropdown_options[i]
                                        if value is not None:
                                            self.selected_body["mass"] = value
                                        else:
                                            self.show_custom_star_mass_input = True
                                            self.star_mass_input_active = True
                                        self.star_mass_dropdown_selected = name
                                        self.star_mass_dropdown_visible = False
                                        self.star_mass_dropdown_active = False
                                    elif self.luminosity_dropdown_visible:
                                        name, value = self.luminosity_dropdown_options[i]
                                        if value is not None:
                                            self.selected_body["luminosity"] = value
                                        else:
                                            self.show_custom_luminosity_input = True
                                            self.luminosity_input_active = True
                                        self.luminosity_dropdown_selected = name
                                        self.luminosity_dropdown_visible = False
                                        self.luminosity_dropdown_active = False
                                    elif self.planet_age_dropdown_visible:
                                        name, value = self.planet_age_dropdown_options[i]
                                        if value is not None:
                                            self.selected_body["age"] = value
                                        else:
                                            self.show_custom_planet_age_input = True
                                            self.planet_age_input_active = True
                                        self.planet_age_dropdown_selected = name
                                        self.planet_age_dropdown_visible = False
                                        self.planet_age_dropdown_active = False
                                    elif self.star_age_dropdown_visible:
                                        name, value = self.star_age_dropdown_options[i]
                                        if value is not None:
                                            self.selected_body["age"] = value
                                        else:
                                            self.show_custom_star_age_input = True
                                            self.star_age_input_active = True
                                        self.star_age_dropdown_selected = name
                                        self.star_age_dropdown_visible = False
                                        self.star_age_dropdown_active = False
                                    elif self.radius_dropdown_visible:
                                        name, value = self.radius_dropdown_options[i]
                                        if value is not None:
                                            self.selected_body["radius"] = value
                                        else:
                                            self.show_custom_radius_input = True
                                            self.radius_input_active = True
                                        self.radius_dropdown_selected = name
                                        self.radius_dropdown_visible = False
                                        self.radius_dropdown_active = False
                                    elif self.activity_dropdown_visible:
                                        name, value = self.activity_dropdown_options[i]
                                        if value is not None:
                                            self.selected_body["activity"] = value
                                        else:
                                            self.show_custom_activity_input = True
                                            self.activity_input_active = True
                                        self.activity_dropdown_selected = name
                                        self.activity_dropdown_visible = False
                                        self.activity_dropdown_active = False
                                    elif self.metallicity_dropdown_visible:
                                        name, value = self.metallicity_dropdown_options[i]
                                        if value is not None:
                                            self.selected_body["metallicity"] = value
                                        else:
                                            self.show_custom_metallicity_input = True
                                            self.metallicity_input_active = True
                                        self.metallicity_dropdown_selected = name
                                        self.metallicity_dropdown_visible = False
                                        self.metallicity_dropdown_active = False
                                    elif self.planet_orbital_distance_dropdown_visible:
                                        name, value = self.planet_orbital_distance_dropdown_options[i]
                                        if value is not None:
                                            self.selected_body["semiMajorAxis"] = value
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
                                        name, value = self.planet_orbital_eccentricity_dropdown_options[i]
                                        if value is not None:
                                            self.selected_body["eccentricity"] = value
                                            self.show_custom_orbital_eccentricity_input = False
                                        else:
                                            self.show_custom_orbital_eccentricity_input = True
                                            self.orbital_eccentricity_input_active = True
                                            self.orbital_eccentricity_input_text = f"{self.selected_body.get('eccentricity', 0.0167):.3f}"
                                        self.planet_orbital_eccentricity_dropdown_selected = ecc_name
                                        self.planet_orbital_eccentricity_dropdown_visible = False
                                        self.planet_orbital_eccentricity_dropdown_active = False
                                        break
                                    elif self.planet_orbital_period_dropdown_visible:
                                        name, value = self.planet_orbital_period_dropdown_options[i]
                                        if value is not None:
                                            self.selected_body["orbital_period"] = value
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
                        # Handle tab clicks
                        for tab_name, tab_rect in self.tabs.items():
                            if tab_rect.collidepoint(event.pos):
                                # Toggle active tab - if clicking the same tab, deactivate it
                                if self.active_tab == tab_name:
                                    self.active_tab = None
                                else:
                                    self.active_tab = tab_name
                                # Clear selected body when changing tabs
                                self.selected_body = None
                                self.show_customization_panel = False
                                self.mass_input_active = False
                                self.planet_dropdown_active = False
                                self.planet_dropdown_visible = False
                                break
                        
                        # Handle space area clicks
                        space_area = pygame.Rect(0, self.tab_height + 2*self.tab_margin, 
                                              self.width, 
                                              self.height - (self.tab_height + 2*self.tab_margin))
                        
                        # Check if click is on a celestial body
                        clicked_body = None
                        for body in self.placed_bodies:
                            body_pos = body["position"].astype(int)
                            body_radius = body["radius"]
                            if (event.pos[0] - body_pos[0])**2 + (event.pos[1] - body_pos[1])**2 <= body_radius**2:
                                clicked_body = body
                                break
                        
                        if clicked_body:
                            # Select the clicked body
                            self.selected_body = clicked_body
                            self.show_customization_panel = True
                            self.mass_input_active = False
                            self.planet_dropdown_active = False
                            self.planet_dropdown_visible = False
                            # --- Moon age dropdown selection logic ---
                            if self.selected_body.get('type') == 'moon':
                                age = self.selected_body.get('age', 4.5)
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
                                orbit_radius = self.selected_body.get('orbit_radius', 384400)
                                found = False
                                for name, preset_distance in self.moon_orbital_distance_dropdown_options:
                                    if preset_distance is not None and abs(preset_distance - orbit_radius) < 1:
                                        self.moon_orbital_distance_dropdown_selected = name
                                        found = True
                                        break
                                if not found:
                                    self.moon_orbital_distance_dropdown_selected = "Moon"  # Default to Moon instead of Custom
                                
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
                        elif self.active_tab and space_area.collidepoint(event.pos):
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
                                default_radius = 20
                            elif self.active_tab == "planet":
                                # Earth-like defaults
                                default_mass = 1.0  # Earth masses
                                default_age = 4.5  # Gyr
                                default_name = "Earth"
                                default_radius = 15
                            else:  # moon
                                # Luna-like defaults
                                default_mass = 1.0  # Earth's Moon mass (1 lunar mass)
                                default_age = 4.5  # Gyr
                                default_name = "Moon"
                                default_radius = 10  # Visual radius for display
                            
                            body = {
                                "type": self.active_tab,
                                "position": np.array([event.pos[0], event.pos[1]], dtype=float),
                                "velocity": np.array([0.0, 0.0]),
                                "radius": default_radius,
                                "name": default_name,
                                "mass": default_mass * (1000.0 if self.active_tab == "star" else 1.0),  # Convert to Earth masses for stars
                                "parent": None,
                                "orbit_radius": 0.0,  # Distance from parent
                                "orbit_angle": 0.0,   # Current angle in orbit
                                "orbit_speed": 0.0,   # Angular speed
                                "rotation_angle": 0.0, # Current rotation angle
                                "rotation_speed": self.rotation_speed * (1.0 if self.active_tab == "planet" else 2.0 if self.active_tab == "moon" else 0.0), # Rotation speed
                                "age": default_age,  # Set default age
                                "habit_score": 0.0,  # Added habitability score attribute
                            }
                            
                            # Add planet-specific attributes
                            if self.active_tab == "planet":
                                body.update({
                                    "gravity": 9.81,  # Earth's gravity in m/s²
                                    "semiMajorAxis": 1.0,  # Default orbital distance (AU)
                                    "eccentricity": 0.017,  # Default orbital eccentricity (Earth-like)
                                    "orbital_period": 365,  # Default orbital period (days)
                                    "stellarFlux": 1.0,  # Default stellar flux (Earth units)
                                })
                            
                            # Add star-specific attributes
                            if self.active_tab == "star":
                                body.update({
                                    "luminosity": default_luminosity,
                                    "star_temperature": 5778,  # Sun's temperature in Kelvin
                                    "star_color": (255, 255, 0),  # Yellow color for G-type star
                                })
                            
                            # Add moon-specific attributes
                            if self.active_tab == "moon":
                                body.update({
                                    "actual_radius": 1737.4,  # Actual radius in km (The Moon) - for dropdown logic
                                    "radius": 10,  # Visual radius in pixels for display
                                    "orbit_radius": 384400,  # Actual orbital distance in km (Earth's Moon)
                                    "temperature": 220,  # Surface temperature in K (Earth's Moon)
                                    "gravity": 1.62,  # Surface gravity in m/s² (Earth's Moon)
                                    "orbital_period": 27.3,  # Orbital period in days (Earth's Moon)
                                })
                            
                            self.placed_bodies.append(body)
                            self.orbit_points[body["name"]] = []
                            self.orbit_history[body["name"]] = []
                            
                            # Set dropdown selections to match defaults
                            if self.active_tab == "star":
                                self.star_mass_dropdown_selected = "1.0 M☉ (Sun)"
                                self.star_age_dropdown_selected = "Sun (4.6 Gyr)"
                                self.spectral_dropdown_selected = "G-type (Yellow, Sun) (5,778 K)"
                                self.luminosity_dropdown_selected = "White Dwarf (Sun)"
                                self.temperature_dropdown_selected = "G-type (Sun) (5,800 K)"
                                self.radius_dropdown_selected = "G-type (Sun)"
                                self.activity_dropdown_selected = "Moderate (Sun)"
                                self.metallicity_dropdown_selected = "0.0 (Sun)"
                            elif self.active_tab == "planet":
                                self.planet_dropdown_selected = "Earth"
                                self.planet_age_dropdown_selected = "4.6 Gyr (Earth’s age)"
                                self.planet_gravity_dropdown_selected = "Earth"
                                self.planet_orbital_distance_dropdown_selected = "Earth"
                                self.planet_orbital_eccentricity_dropdown_selected = "Earth"
                                self.planet_orbital_period_dropdown_selected = "Earth"
                                self.planet_stellar_flux_dropdown_selected = "Earth"
                                if self.selected_body:
                                    self.selected_body["semiMajorAxis"] = 1.0
                                    self.selected_body["eccentricity"] = 0.0167
                                    self.selected_body["orbital_period"] = 365.25
                                    self.selected_body["stellarFlux"] = 1.0
                            else:  # moon
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
                                print(f"DEBUG: show_customization_panel after: {self.show_customization_panel}")
                        else:
                            # Clicked empty space, deselect body
                            self.selected_body = None
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
                        try:
                            new_mass = float(self.mass_input_text)
                            if self.selected_body.get('type') == 'moon':
                                # For moons, mass is already in Lunar masses
                                if self.mass_min <= new_mass <= self.mass_max:
                                    self.selected_body["mass"] = new_mass
                                    self.generate_orbit_grid(self.selected_body)
                            else:
                                if self.mass_min <= new_mass <= self.mass_max:
                                    self.selected_body["mass"] = new_mass
                                    if self.selected_body["type"] != "star":
                                        self.generate_orbit_grid(self.selected_body)
                                self.mass_input_active = False
                        except ValueError:
                            # Invalid input, keep current value
                            if self.selected_body.get('type') == 'moon':
                                # For moons, mass is already in Lunar masses
                                lunar_mass = self.selected_body.get('mass', 1.0)
                                self.mass_input_text = f"{lunar_mass:.3f}"
                            else:
                                self.mass_input_text = f"{self.selected_body.get('mass', 1.0):.3f}"
                    elif event.key == pygame.K_BACKSPACE:
                        self.mass_input_text = self.mass_input_text[:-1]
                    elif event.key == pygame.K_ESCAPE:
                        self.mass_input_active = False
                        if self.selected_body.get('type') == 'moon':
                            # For moons, mass is already in Lunar masses
                            lunar_mass = self.selected_body.get('mass', 1.0)
                            self.mass_input_text = f"{lunar_mass:.3f}"
                        else:
                            self.mass_input_text = f"{self.selected_body.get('mass', 1.0):.3f}"
                    elif event.unicode.isnumeric() or event.unicode == '.':
                        # Only allow numbers and a single decimal point, not as the first character
                        if event.unicode == '.':
                            if '.' in self.mass_input_text or len(self.mass_input_text) == 0:
                                pass  # Don't allow multiple decimals or starting with a decimal
                            else:
                                self.mass_input_text += event.unicode
                        else:
                            # Prevent leading zeros unless followed by a decimal
                            if self.mass_input_text == "0" and event.unicode != '.':
                                pass
                            else:
                                self.mass_input_text += event.unicode
                if self.luminosity_input_active and self.selected_body and self.selected_body.get('type') == 'star':
                    if event.key == pygame.K_RETURN:
                        # Try to convert input to float and validate
                        try:
                            new_luminosity = float(self.luminosity_input_text)
                            if self.luminosity_min <= new_luminosity <= self.luminosity_max:
                                self.selected_body["luminosity"] = new_luminosity
                            selfFluminosity_input_active = False
                        except ValueError:
                            # Invalid input, keep current value
                            self.luminosity_input_text = f"{self.selected_body.get('luminosity', 1.0):.3f}"
                    elif event.key == pygame.K_BACKSPACE:
                        self.luminosity_input_text = self.luminosity_input_text[:-1]
                    elif event.key == pygame.K_ESCAPE:
                        self.luminosity_input_active = False
                        self.luminosity_input_text = f"{self.selected_body.get('luminosity', 1.0):.3f}"
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
                        try:
                            temp = float(self.temperature_input_text)
                            if self.temperature_min <= temp <= self.temperature_max:
                                if self.selected_body:
                                    self.selected_body["temperature"] = temp
                                self.temperature_dropdown_selected = f"Custom ({temp:.0f} K)"
                            self.temperature_input_text = ""
                            self.temperature_input_active = False
                            self.show_custom_temperature_input = False
                        except ValueError:
                            pass
                    elif event.key == pygame.K_BACKSPACE:
                        self.temperature_input_text = self.temperature_input_text[:-1]
                    elif event.unicode.isnumeric() or event.unicode == '.':
                        self.temperature_input_text += event.unicode
                if self.metallicity_input_active and self.selected_body and self.selected_body.get('type') == 'star':
                    if event.key == pygame.K_RETURN:
                        try:
                            metallicity = float(self.metallicity_input_text)
                            if self.metallicity_min <= metallicity <= self.metallicity_max:
                                self.selected_body["metallicity"] = metallicity
                            self.metallicity_input_text = ""
                            self.metallicity_input_active = False
                            self.show_custom_metallicity_input = False
                        except ValueError:
                            pass
                    elif event.key == pygame.K_BACKSPACE:
                        self.metallicity_input_text = self.metallicity_input_text[:-1]
                    elif event.unicode.isnumeric() or event.unicode == '.':
                        self.metallicity_input_text += event.unicode
                if self.show_custom_planet_gravity_input and self.selected_body and self.selected_body.get('type') == 'planet':
                    if event.key == pygame.K_RETURN:
                        try:
                            gravity = float(self.planet_gravity_input_text)
                            if 0.1 <= gravity <= 100.0:  # Reasonable gravity range
                                self.selected_body["gravity"] = gravity
                            self.planet_gravity_input_text = ""
                            self.show_custom_planet_gravity_input = False
                        except ValueError:
                            pass
                    elif event.key == pygame.K_BACKSPACE:
                        self.planet_gravity_input_text = self.planet_gravity_input_text[:-1]
                    elif event.key == pygame.K_ESCAPE:
                        self.show_custom_planet_gravity_input = False
                        self.planet_gravity_input_text = ""
                    elif event.unicode.isnumeric() or event.unicode == '.':
                        self.planet_gravity_input_text += event.unicode
                if self.show_custom_orbital_distance_input and self.selected_body and self.selected_body.get('type') == 'planet':
                    if event.key == pygame.K_RETURN:
                        try:
                            dist = float(self.orbital_distance_input_text)
                            if self.orbital_distance_min <= dist <= self.orbital_distance_max:
                                self.selected_body["semiMajorAxis"] = dist
                                self.planet_orbital_distance_dropdown_selected = f"Custom ({dist:.2f} AU)"
                            self.orbital_distance_input_text = ""
                            self.orbital_distance_input_active = False
                            self.show_custom_orbital_distance_input = False
                        except ValueError:
                            pass
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
                                self.selected_body["eccentricity"] = ecc
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
                                self.selected_body["orbital_period"] = period
                                self.planet_orbital_period_dropdown_selected = f"Custom ({period:.0f} days)"
                            self.orbital_period_input_text = ""
                            self.orbital_period_input_active = False
                            self.show_custom_orbital_period_input = False
                        except ValueError:
                            pass
                    elif event.key == pygame.K_BACKSPACE:
                        self.orbital_period_input_text = self.orbital_period_input_text[:-1]
                    elif event.unicode.isnumeric() or event.unicode == '.':
                        self.orbital_period_input_text += event.unicode
                if self.show_custom_stellar_flux_input and self.selected_body and self.selected_body.get('type') == 'planet':
                    if event.key == pygame.K_RETURN:
                        try:
                            flux = float(self.stellar_flux_input_text)
                            if self.stellar_flux_min <= flux <= self.stellar_flux_max:
                                self.selected_body["stellarFlux"] = flux
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
                                self.selected_body["density"] = density
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
        return True
    
    def update_ambient_colors(self):
        """Update the ambient colors for the title"""
        self.color_change_counter += 1
        if self.color_change_counter >= self.color_change_speed:
            self.color_change_counter = 0
            self.current_color_index = (self.current_color_index + 1) % len(self.ambient_colors)
    
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
            # Calculate orbit radius
            orbit_radius = np.linalg.norm(parent["position"] - body["position"])
            body["orbit_radius"] = orbit_radius
            body["parent"] = parent["name"]
            
            # Generate grid points for a perfect circle
            grid_points = []
            for i in range(100):  # 100 points for a smooth circle
                angle = i * 2 * np.pi / 100
                x = parent["position"][0] + orbit_radius * np.cos(angle)
                y = parent["position"][1] + orbit_radius * np.sin(angle)
                grid_points.append(np.array([x, y]))
            
            self.orbit_grid_points[body["name"]] = grid_points
            
            # Set initial orbit angle
            dx = body["position"][0] - parent["position"][0]
            dy = body["position"][1] - parent["position"][1]
            body["orbit_angle"] = np.arctan2(dy, dx)
            
            # Calculate orbital speed for circular orbit
            # v = sqrt(G * M / r) for circular orbit
            # For moons, use a faster orbital speed relative to their planet
            if body["type"] == "moon":
                # Moons orbit faster around planets
                body["orbit_speed"] = np.sqrt(self.G * parent["mass"] / orbit_radius) / orbit_radius * 3.0
            else:
                body["orbit_speed"] = np.sqrt(self.G * parent["mass"] / orbit_radius) / orbit_radius
            
            # Set initial velocity for circular orbit
            # v_x = -v * sin(angle), v_y = v * cos(angle)
            v = body["orbit_speed"] * orbit_radius
            body["velocity"] = np.array([-v * np.sin(body["orbit_angle"]), v * np.cos(body["orbit_angle"])])
    
    def update_physics(self):
        """Update positions and velocities of all bodies"""
        # First, establish parent-child relationships if not already set
        for body in self.placed_bodies:
            if body["type"] == "planet" and not body["parent"]:
                # Find nearest star
                stars = [b for b in self.placed_bodies if b["type"] == "star"]
                if stars:
                    nearest_star = min(stars, key=lambda s: np.linalg.norm(s["position"] - body["position"]))
                    body["parent"] = nearest_star["name"]
                    self.generate_orbit_grid(body)
            elif body["type"] == "moon" and not body["parent"]:
                # Find nearest planet
                planets = [b for b in self.placed_bodies if b["type"] == "planet"]
                if planets:
                    nearest_planet = min(planets, key=lambda p: np.linalg.norm(p["position"] - body["position"]))
                    body["parent"] = nearest_planet["name"]
                    self.generate_orbit_grid(body)

        # Update positions and velocities
        for body in self.placed_bodies:
            if body["type"] == "star":
                # Stars remain stationary
                continue
            
            # Find parent body
            parent = next((b for b in self.placed_bodies if b["name"] == body["parent"]), None)
            if parent:
                # Update orbit angle
                body["orbit_angle"] += body["orbit_speed"] * self.time_step
                
                # Calculate new position based on orbit angle
                body["position"][0] = parent["position"][0] + body["orbit_radius"] * np.cos(body["orbit_angle"])
                body["position"][1] = parent["position"][1] + body["orbit_radius"] * np.sin(body["orbit_angle"])
                
                # Update velocity for circular orbit
                v = body["orbit_speed"] * body["orbit_radius"]
                body["velocity"] = np.array([-v * np.sin(body["orbit_angle"]), v * np.cos(body["orbit_angle"])])
                
                # Update rotation angle
                body["rotation_angle"] += body["rotation_speed"] * self.time_step
                if body["rotation_angle"] >= 2 * np.pi:
                    body["rotation_angle"] -= 2 * np.pi
                
                # Store orbit points
                self.orbit_points[body["name"]].append(body["position"].copy())
                if len(self.orbit_points[body["name"]]) > 100:  # Limit number of points
                    self.orbit_points[body["name"]].pop(0)
                
                # Update orbit grid points for moons to follow their parent planet
                if body["type"] == "moon" and body["name"] in self.orbit_grid_points:
                    self.update_moon_orbit_grid(body, parent)

    def update_moon_orbit_grid(self, moon, planet):
        """Update the moon's orbit grid points to follow its parent planet"""
        orbit_radius = moon["orbit_radius"]
        grid_points = []
        
        # Generate grid points for a perfect circle around the planet's current position
        for i in range(100):  # 100 points for a smooth circle
            angle = i * 2 * np.pi / 100
            x = planet["position"][0] + orbit_radius * np.cos(angle)
            y = planet["position"][1] + orbit_radius * np.sin(angle)
            grid_points.append(np.array([x, y]))
        
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
    
    def render_home_screen(self):
        """Render the home screen with the title and create button"""
        # Fill background
        self.screen.fill(self.BLACK)
        
        # Update ambient colors
        self.update_ambient_colors()
        
        # Render title with current ambient color
        title_text = self.title_font.render("AIET", True, self.ambient_colors[self.current_color_index])
        title_rect = title_text.get_rect(center=(self.width//2, self.height//3))
        self.screen.blit(title_text, title_rect)
        
        # Draw instruction with current ambient color
        instruction_text = self.font.render("Create", True, self.ambient_colors[self.current_color_index])
        instruction_rect = instruction_text.get_rect(center=(self.width//2, self.height*3/4))
        
        # Draw a button around the text
        button_width = instruction_text.get_width() + 40
        button_height = instruction_text.get_height() + 20
        button_rect = pygame.Rect(
            instruction_rect.centerx - button_width//2,
            instruction_rect.centery - button_height//2,
            button_width,
            button_height
        )
        
        # Draw button with current ambient color
        pygame.draw.rect(self.screen, (50, 50, 50), button_rect, border_radius=10)
        pygame.draw.rect(self.screen, self.ambient_colors[self.current_color_index], button_rect, 2, border_radius=10)
        
        # Draw the text on top of the button
        self.screen.blit(instruction_text, instruction_rect)
        
        pygame.display.flip()
    
    def format_age_display(self, age: float) -> str:
        """Format age for display, converting to Myr if less than 0.5 Gyr"""
        if age < 0.5:
            myr = age * 1000  # Convert Gyr to Myr
            return f"{myr:.1f} Myr"
        return f"{age:.1f} Gyr"

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
                self.planet_temperature_dropdown_visible or self.planet_gravity_dropdown_visible or 
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
            if len(option_data) == 3:  # spectral_class_dropdown_options: (name, temp, color)
                name, value, color = option_data
            else:  # other dropdowns: (name, value)
                name, value = option_data
                
            option_rect = pygame.Rect(
                0,  # x position relative to dropdown surface
                i * option_height,  # y position relative to dropdown surface
                width,
                option_height
            )
            self.dropdown_options_rects.append(option_rect)
            
            # Draw option background
            pygame.draw.rect(self.dropdown_surface, (255, 255, 255), option_rect)  # Solid white background
            pygame.draw.rect(self.dropdown_surface, self.dropdown_border_color, option_rect, self.dropdown_border_width)
            
            # Draw option text
            if value is not None:
                if self.planet_dropdown_visible:
                    text = f"{name} (" + self._format_value(value, 'M⊕') + ")"
                elif self.moon_dropdown_visible:
                    text = f"{name} (" + self._format_value(value, 'M🌕') + ")"
                elif self.star_mass_dropdown_visible:
                    text = name  # Mass options already include the unit
                elif self.planet_age_dropdown_visible or self.star_age_dropdown_visible:
                    text = name  # Age options already include the unit
                elif self.moon_age_dropdown_visible:
                    text = f"{name} (" + self._format_value(value, 'Gyr') + ")"
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
                    text = name  # Radius options already include the unit
                elif self.activity_dropdown_visible:
                    text = name  # Activity options already include the unit
                elif self.metallicity_dropdown_visible:
                    text = name  # Metallicity options already include the unit
                elif self.planet_radius_dropdown_visible:
                    text = f"{name} (" + self._format_value(value, 'R🜨') + ")"
                elif self.planet_temperature_dropdown_visible:
                    text = f"{name} (" + self._format_value(value, 'K') + ")"
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
                text = f"{name} (Custom)"
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
            self.planet_gravity_dropdown_visible or self.planet_orbital_distance_dropdown_visible or
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
            
            # Draw habitability probability at the top center
            if self.selected_body and self.selected_body.get('type') != 'star':
                # Shift right by 60 pixels
                habitability_text = self.font.render(f"Habitability Probability: {self.selected_body.get('habit_score', 0.0):.2f}%", True, (0, 255, 0))  # Green color
                habitability_rect = habitability_text.get_rect(center=(self.width//2 + 60, 30))
                self.screen.blit(habitability_text, habitability_rect)
            
            # MASS SECTION (always show mass label/input, dropdown only for planets)
            if self.selected_body.get('type') == 'moon':
                mass_label = self.subtitle_font.render("Mass (M🌕)", True, self.BLACK)
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
                if self.planet_dropdown_selected:
                    # Find the selected option's value
                    selected = next(((name, value) for name, value in self.planet_dropdown_options if name == self.planet_dropdown_selected), None)
                    if selected:
                        name, value = selected
                        if value is not None:
                            dropdown_text = f"{name} ({value:.3f} Earth masses)"
                        else:
                            dropdown_text = name  # Custom
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
                if self.moon_dropdown_selected:
                    # Find the selected option's value
                    selected = next(((name, value) for name, value in self.moon_dropdown_options if name == self.moon_dropdown_selected), None)
                    if selected:
                        name, value = selected
                        if value is not None:
                            dropdown_text = f"{name} ({value:.3f} M☽)"
                        else:
                            dropdown_text = name  # Custom
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
                if self.star_mass_dropdown_selected:
                    selected = next(((name, value) for name, value in self.star_mass_dropdown_options if name == self.star_mass_dropdown_selected), None)
                    if selected:
                        name, value = selected
                        if value is not None and "(Sun)" in name:
                            dropdown_text = f"Sun ({value:.2f} M☉)"
                        else:
                            dropdown_text = name
                text_surface = self.subtitle_font.render(dropdown_text, True, self.BLACK)
                text_rect = text_surface.get_rect(midleft=(self.star_mass_dropdown_rect.left + 5, self.star_mass_dropdown_rect.centery))
                self.screen.blit(text_surface, text_rect)

                # Show custom mass input if "Custom Mass" is selected
                if self.show_custom_star_mass_input:
                    custom_mass_label = self.subtitle_font.render("Enter Custom Mass (M☉):", True, self.BLACK)
                    custom_mass_label_rect = custom_mass_label.get_rect(midleft=(self.width - self.customization_panel_width + 50, 165))
                    self.screen.blit(custom_mass_label, custom_mass_label_rect)
                    
                    pygame.draw.rect(self.screen, self.WHITE, self.mass_input_rect, 2)
                    pygame.draw.rect(self.screen, self.BLUE if self.mass_input_active else self.GRAY, 
                                   self.mass_input_rect, 1)
                    if self.selected_body.get('type') == 'moon':
                        # For moons, mass is already in Lunar masses
                        lunar_mass = self.selected_body.get('mass', 1.0)
                        if self.mass_input_active:
                            text_surface = self.subtitle_font.render(self.mass_input_text, True, self.BLACK)
                        else:
                            text_surface = self.subtitle_font.render(f"{lunar_mass:.3f}", True, self.BLACK)
                    else:
                        if self.mass_input_active:
                            text_surface = self.subtitle_font.render(self.mass_input_text, True, self.BLACK)
                        else:
                            text_surface = self.subtitle_font.render(f"{self.selected_body.get('mass', 1.0):.3f}", True, self.BLACK)
                    text_rect = text_surface.get_rect(midleft=(self.mass_input_rect.left + 5, 
                                                             self.mass_input_rect.centery))
                    self.screen.blit(text_surface, text_rect)
            else:
                # For non-planets, show the mass input box
                pygame.draw.rect(self.screen, self.WHITE, self.mass_input_rect, 2)
                pygame.draw.rect(self.screen, self.BLUE if self.mass_input_active else self.GRAY, 
                               self.mass_input_rect, 1)
                if self.selected_body.get('type') == 'moon':
                    # For moons, mass is already in Lunar masses
                    lunar_mass = self.selected_body.get('mass', 1.0)
                    if self.mass_input_active:
                        text_surface = self.subtitle_font.render(self.mass_input_text, True, self.BLACK)
                    else:
                        text_surface = self.subtitle_font.render(f"{lunar_mass:.3f}", True, self.BLACK)
                else:
                    if self.mass_input_active:
                        text_surface = self.subtitle_font.render(self.mass_input_text, True, self.BLACK)
                    else:
                        text_surface = self.subtitle_font.render(f"{self.selected_body.get('mass', 1.0):.3f}", True, self.BLACK)
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
                if self.planet_age_dropdown_selected:
                    dropdown_text = self.planet_age_dropdown_selected
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
                        text_surface = self.subtitle_font.render(f"{self.selected_body.get('age', 0.0):.1f}", True, self.BLACK)
                    text_rect = text_surface.get_rect(midleft=(self.age_input_rect.left + 5, 
                                                             self.age_input_rect.centery))
                    self.screen.blit(text_surface, text_rect)
            elif self.selected_body.get('type') == 'moon':
                # For moons, show the age dropdown
                pygame.draw.rect(self.screen, self.WHITE, self.moon_age_dropdown_rect, 2)
                pygame.draw.rect(self.screen, self.BLUE if self.moon_age_dropdown_active else self.GRAY, 
                               self.moon_age_dropdown_rect, 1)
                dropdown_text = "Select Age"
                if self.moon_age_dropdown_selected:
                    selected = next(((name, value) for name, value in self.moon_age_dropdown_options if name == self.moon_age_dropdown_selected), None)
                    if selected:
                        name, value = selected
                        if value is not None:
                            dropdown_text = f"{name} ({value:.1f} Gyr)"
                        else:
                            dropdown_text = name
                text_surface = self.subtitle_font.render(dropdown_text, True, self.BLACK)
                text_rect = text_surface.get_rect(midleft=(self.moon_age_dropdown_rect.left + 5, 
                                                         self.moon_age_dropdown_rect.centery))
                self.screen.blit(text_surface, text_rect)

                # Show custom age input if "Custom Age" is selected
                if self.show_custom_moon_age_input:
                    custom_age_label = self.subtitle_font.render("Enter Custom Age (Gyr):", True, self.BLACK)
                    custom_age_label_rect = custom_age_label.get_rect(midleft=(self.width - self.customization_panel_width + 50, 225))
                    self.screen.blit(custom_age_label, custom_age_label_rect)
                    
                    pygame.draw.rect(self.screen, self.WHITE, self.age_input_rect, 2)
                    pygame.draw.rect(self.screen, self.BLUE if self.age_input_active else self.GRAY, 
                                   self.age_input_rect, 1)
                    if self.age_input_active:
                        text_surface = self.subtitle_font.render(self.age_input_text, True, self.BLACK)
                    else:
                        text_surface = self.subtitle_font.render(f"{self.selected_body.get('age', 0.0):.1f}", True, self.BLACK)
                    text_rect = text_surface.get_rect(midleft=(self.age_input_rect.left + 5, 
                                                             self.age_input_rect.centery))
                    self.screen.blit(text_surface, text_rect)
                
                # RADIUS SECTION (only for moons)
                radius_label = self.subtitle_font.render("Radius (km)", True, self.BLACK)
                radius_label_rect = radius_label.get_rect(midleft=(self.width - self.customization_panel_width + 50, 225))
                self.screen.blit(radius_label, radius_label_rect)
                
                # Draw the moon radius dropdown
                pygame.draw.rect(self.screen, self.WHITE, self.moon_radius_dropdown_rect, 2)
                pygame.draw.rect(self.screen, self.BLUE if self.moon_radius_dropdown_active else self.GRAY, 
                               self.moon_radius_dropdown_rect, 1)
                dropdown_text = "Select Radius"
                if self.moon_radius_dropdown_selected:
                    selected = next(((name, value) for name, value in self.moon_radius_dropdown_options if name == self.moon_radius_dropdown_selected), None)
                    if selected:
                        name, value = selected
                        if value is not None:
                            dropdown_text = f"{name} ({value:.0f} km)"
                        else:
                            dropdown_text = name
                text_surface = self.subtitle_font.render(dropdown_text, True, self.BLACK)
                text_rect = text_surface.get_rect(midleft=(self.moon_radius_dropdown_rect.left + 5, 
                                                         self.moon_radius_dropdown_rect.centery))
                self.screen.blit(text_surface, text_rect)

                # Show custom radius input if "Custom" is selected
                if self.show_custom_moon_radius_input:
                    custom_radius_label = self.subtitle_font.render("Enter Custom Radius (km):", True, self.BLACK)
                    custom_radius_label_rect = custom_radius_label.get_rect(midleft=(self.width - self.customization_panel_width + 50, 285))
                    self.screen.blit(custom_radius_label, custom_radius_label_rect)
                    
                    custom_radius_input_rect = pygame.Rect(self.width - self.customization_panel_width + 50, 315, self.customization_panel_width - 100, 30)
                    pygame.draw.rect(self.screen, self.WHITE, custom_radius_input_rect, 2)
                    pygame.draw.rect(self.screen, self.BLUE, custom_radius_input_rect, 1)
                    text_surface = self.subtitle_font.render("", True, self.BLACK)  # Placeholder for input text
                    text_rect = text_surface.get_rect(midleft=(custom_radius_input_rect.left + 5, custom_radius_input_rect.centery))
                    self.screen.blit(text_surface, text_rect)
                
                # ORBITAL DISTANCE SECTION (only for moons)
                orbital_distance_label = self.subtitle_font.render("Orbital Distance (km)", True, self.BLACK)
                orbital_distance_label_rect = orbital_distance_label.get_rect(midleft=(self.width - self.customization_panel_width + 50, 285))
                self.screen.blit(orbital_distance_label, orbital_distance_label_rect)
                
                # Draw the moon orbital distance dropdown
                pygame.draw.rect(self.screen, self.WHITE, self.moon_orbital_distance_dropdown_rect, 2)
                pygame.draw.rect(self.screen, self.BLUE if self.moon_orbital_distance_dropdown_active else self.GRAY, 
                               self.moon_orbital_distance_dropdown_rect, 1)
                dropdown_text = "Select Orbital Distance"
                if self.moon_orbital_distance_dropdown_selected:
                    selected = next(((name, value) for name, value in self.moon_orbital_distance_dropdown_options if name == self.moon_orbital_distance_dropdown_selected), None)
                    if selected:
                        name, value = selected
                        if value is not None:
                            dropdown_text = f"{name} ({value:.0f} km)"
                        else:
                            dropdown_text = name
                text_surface = self.subtitle_font.render(dropdown_text, True, self.BLACK)
                text_rect = text_surface.get_rect(midleft=(self.moon_orbital_distance_dropdown_rect.left + 5, 
                                                         self.moon_orbital_distance_dropdown_rect.centery))
                self.screen.blit(text_surface, text_rect)

                # Show custom orbital distance input if "Custom" is selected
                if self.show_custom_moon_orbital_distance_input:
                    custom_distance_label = self.subtitle_font.render("Enter Custom Distance (km):", True, self.BLACK)
                    custom_distance_label_rect = custom_distance_label.get_rect(midleft=(self.width - self.customization_panel_width + 50, 345))
                    self.screen.blit(custom_distance_label, custom_distance_label_rect)
                    
                    custom_distance_input_rect = pygame.Rect(self.width - self.customization_panel_width + 50, 375, self.customization_panel_width - 100, 30)
                    pygame.draw.rect(self.screen, self.WHITE, custom_distance_input_rect, 2)
                    pygame.draw.rect(self.screen, self.BLUE, custom_distance_input_rect, 1)
                    text_surface = self.subtitle_font.render("", True, self.BLACK)  # Placeholder for input text
                    text_rect = text_surface.get_rect(midleft=(custom_distance_input_rect.left + 5, custom_distance_input_rect.centery))
                    self.screen.blit(text_surface, text_rect)
                
                # ORBITAL PERIOD SECTION (only for moons)
                orbital_period_label = self.subtitle_font.render("Orbital Period (days)", True, self.BLACK)
                orbital_period_label_rect = orbital_period_label.get_rect(midleft=(self.width - self.customization_panel_width + 50, 345))
                self.screen.blit(orbital_period_label, orbital_period_label_rect)
                
                # Draw the moon orbital period dropdown
                pygame.draw.rect(self.screen, self.WHITE, self.moon_orbital_period_dropdown_rect, 2)
                pygame.draw.rect(self.screen, self.BLUE if self.moon_orbital_period_dropdown_active else self.GRAY, 
                               self.moon_orbital_period_dropdown_rect, 1)
                dropdown_text = "Select Orbital Period"
                if self.moon_orbital_period_dropdown_selected:
                    selected = next(((name, value) for name, value in self.moon_orbital_period_dropdown_options if name == self.moon_orbital_period_dropdown_selected), None)
                    if selected:
                        name, value = selected
                        if value is not None:
                            dropdown_text = f"{name} ({value:.1f} days)"
                        else:
                            dropdown_text = name
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
                    text_surface = self.subtitle_font.render("", True, self.BLACK)  # Placeholder for input text
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
                if self.moon_temperature_dropdown_selected:
                    selected = next(((name, value) for name, value in self.moon_temperature_dropdown_options if name == self.moon_temperature_dropdown_selected), None)
                    if selected:
                        name, value = selected
                        if value is not None:
                            dropdown_text = f"{name} ({value:.0f} K)"
                        else:
                            dropdown_text = name
                text_surface = self.subtitle_font.render(dropdown_text, True, self.BLACK)
                text_rect = text_surface.get_rect(midleft=(self.moon_temperature_dropdown_rect.left + 5, 
                                                         self.moon_temperature_dropdown_rect.centery))
                self.screen.blit(text_surface, text_rect)

                # Show custom temperature input if "Custom" is selected
                if self.show_custom_moon_temperature_input:
                    custom_temp_label = self.subtitle_font.render("Enter Custom Temperature (K):", True, self.BLACK)
                    custom_temp_label_rect = custom_temp_label.get_rect(midleft=(self.width - self.customization_panel_width + 50, 465))
                    self.screen.blit(custom_temp_label, custom_temp_label_rect)
                    
                    custom_temp_input_rect = pygame.Rect(self.width - self.customization_panel_width + 50, 495, self.customization_panel_width - 100, 30)
                    pygame.draw.rect(self.screen, self.WHITE, custom_temp_input_rect, 2)
                    pygame.draw.rect(self.screen, self.BLUE, custom_temp_input_rect, 1)
                    text_surface = self.subtitle_font.render("", True, self.BLACK)  # Placeholder for input text
                    text_rect = text_surface.get_rect(midleft=(custom_temp_input_rect.left + 5, custom_temp_input_rect.centery))
                    self.screen.blit(text_surface, text_rect)
                
                # SURFACE GRAVITY SECTION (only for moons)
                gravity_label = self.subtitle_font.render("Surface Gravity (m/s²)", True, self.BLACK)
                gravity_label_rect = gravity_label.get_rect(midleft=(self.width - self.customization_panel_width + 50, 465))
                self.screen.blit(gravity_label, gravity_label_rect)
                
                # Draw the moon gravity dropdown
                pygame.draw.rect(self.screen, self.WHITE, self.moon_gravity_dropdown_rect, 2)
                pygame.draw.rect(self.screen, self.BLUE if self.moon_gravity_dropdown_active else self.GRAY, 
                               self.moon_gravity_dropdown_rect, 1)
                dropdown_text = "Select Gravity"
                if self.moon_gravity_dropdown_selected:
                    selected = next(((name, value) for name, value in self.moon_gravity_dropdown_options if name == self.moon_gravity_dropdown_selected), None)
                    if selected:
                        name, value = selected
                        if value is not None:
                            dropdown_text = f"{name} ({value:.2f} m/s²)"
                        else:
                            dropdown_text = name
                text_surface = self.subtitle_font.render(dropdown_text, True, self.BLACK)
                text_rect = text_surface.get_rect(midleft=(self.moon_gravity_dropdown_rect.left + 5, 
                                                         self.moon_gravity_dropdown_rect.centery))
                self.screen.blit(text_surface, text_rect)

                # Show custom gravity input if "Custom" is selected
                if self.show_custom_moon_gravity_input:
                    custom_gravity_label = self.subtitle_font.render("Enter Custom Gravity (m/s²):", True, self.BLACK)
                    custom_gravity_label_rect = custom_gravity_label.get_rect(midleft=(self.width - self.customization_panel_width + 50, 525))
                    self.screen.blit(custom_gravity_label, custom_gravity_label_rect)
                    
                    custom_gravity_input_rect = pygame.Rect(self.width - self.customization_panel_width + 50, 555, self.customization_panel_width - 100, 30)
                    pygame.draw.rect(self.screen, self.WHITE, custom_gravity_input_rect, 2)
                    pygame.draw.rect(self.screen, self.BLUE, custom_gravity_input_rect, 1)
                    text_surface = self.subtitle_font.render("", True, self.BLACK)  # Placeholder for input text
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
                if self.planet_radius_dropdown_selected:
                    selected = next(((name, value) for name, value in self.planet_radius_dropdown_options if name == self.planet_radius_dropdown_selected), None)
                    if selected:
                        name, value = selected
                        if value is not None:
                            dropdown_text = f"{name} ({value:.2f} R🜨)"
                        else:
                            dropdown_text = name
                text_surface = self.subtitle_font.render(dropdown_text, True, self.BLACK)
                text_rect = text_surface.get_rect(midleft=(self.planet_radius_dropdown_rect.left + 5, 
                                                         self.planet_radius_dropdown_rect.centery))
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
                if self.planet_temperature_dropdown_selected:
                    selected = next(((name, value) for name, value in self.planet_temperature_dropdown_options if name == self.planet_temperature_dropdown_selected), None)
                    if selected:
                        name, value = selected
                        if value is not None:
                            dropdown_text = f"{name} ({value:.0f} K)"
                        else:
                            dropdown_text = name
                text_surface = self.subtitle_font.render(dropdown_text, True, self.BLACK)
                text_rect = text_surface.get_rect(midleft=(self.planet_temperature_dropdown_rect.left + 5, 
                                                         self.planet_temperature_dropdown_rect.centery))
                self.screen.blit(text_surface, text_rect)

                # GRAVITY SECTION (only for planets)
                gravity_label = self.subtitle_font.render("Surface Gravity (m/s²)", True, self.BLACK)
                gravity_label_rect = gravity_label.get_rect(midleft=(self.width - self.customization_panel_width + 50, 345))
                self.screen.blit(gravity_label, gravity_label_rect)
                self.planet_gravity_dropdown_rect.y = 360
                pygame.draw.rect(self.screen, self.WHITE, self.planet_gravity_dropdown_rect, 2)
                pygame.draw.rect(self.screen, self.BLUE if self.planet_gravity_dropdown_active else self.GRAY, self.planet_gravity_dropdown_rect, 1)
                dropdown_text = "Select Reference Planet"
                if self.planet_gravity_dropdown_selected:
                    selected = next(((name, value) for name, value in self.planet_gravity_dropdown_options if name == self.planet_gravity_dropdown_selected), None)
                    if selected:
                        name, value = selected
                        if value is not None:
                            dropdown_text = f"{name} ({value:.2f} m/s²)"
                        else:
                            dropdown_text = name
                text_surface = self.subtitle_font.render(dropdown_text, True, self.BLACK)
                text_rect = text_surface.get_rect(midleft=(self.planet_gravity_dropdown_rect.left + 5, self.planet_gravity_dropdown_rect.centery))
                self.screen.blit(text_surface, text_rect)
                
                # Show custom gravity input if "Custom" is selected, just below dropdown
                if self.show_custom_planet_gravity_input:
                    custom_gravity_label = self.subtitle_font.render("Enter Custom Gravity (m/s²):", True, self.BLACK)
                    custom_gravity_label_rect = custom_gravity_label.get_rect(midleft=(self.width - self.customization_panel_width + 50, 415))
                    self.screen.blit(custom_gravity_label, custom_gravity_label_rect)
                    
                    custom_gravity_input_rect = pygame.Rect(self.width - self.customization_panel_width + 50, 445, self.customization_panel_width - 100, 30)
                    pygame.draw.rect(self.screen, self.WHITE, custom_gravity_input_rect, 2)
                    pygame.draw.rect(self.screen, self.BLUE, custom_gravity_input_rect, 1)
                    text_surface = self.subtitle_font.render(self.planet_gravity_input_text, True, self.BLACK)
                    text_rect = text_surface.get_rect(midleft=(custom_gravity_input_rect.left + 5, custom_gravity_input_rect.centery))
                    self.screen.blit(text_surface, text_rect)
                # Orbital Distance (AU) dropdown
                orbital_distance_label = self.subtitle_font.render("Orbital Distance (AU)", True, self.BLACK)
                orbital_distance_label_rect = orbital_distance_label.get_rect(midleft=(self.width - self.customization_panel_width + 50, 405))
                self.screen.blit(orbital_distance_label, orbital_distance_label_rect)
                pygame.draw.rect(self.screen, self.WHITE, self.planet_orbital_distance_dropdown_rect, 2)
                pygame.draw.rect(self.screen, self.BLUE if self.planet_orbital_distance_dropdown_active else self.GRAY, self.planet_orbital_distance_dropdown_rect, 1)
                dropdown_text = "Select Orbital Distance"
                if self.planet_orbital_distance_dropdown_selected:
                    selected = next(((name, value) for name, value in self.planet_orbital_distance_dropdown_options if name == self.planet_orbital_distance_dropdown_selected), None)
                    if selected:
                        name, value = selected
                        if value is not None:
                            dropdown_text = f"{name} ({value:.2f} AU)"
                        else:
                            dropdown_text = name
                text_surface = self.subtitle_font.render(dropdown_text, True, self.BLACK)
                text_rect = text_surface.get_rect(midleft=(self.planet_orbital_distance_dropdown_rect.left + 5, self.planet_orbital_distance_dropdown_rect.centery))
                self.screen.blit(text_surface, text_rect)
                # Show custom orbital distance input if "Custom" is selected, just below dropdown
                if self.show_custom_orbital_distance_input:
                    custom_orbital_label = self.subtitle_font.render("Enter Custom Distance (AU):", True, self.BLACK)
                    custom_orbital_label_rect = custom_orbital_label.get_rect(midleft=(self.width - self.customization_panel_width + 50, 455))
                    self.screen.blit(custom_orbital_label, custom_orbital_label_rect)
                    custom_orbital_input_rect = pygame.Rect(self.width - self.customization_panel_width + 50, 485, self.customization_panel_width - 100, 30)
                    pygame.draw.rect(self.screen, self.WHITE, custom_orbital_input_rect, 2)
                    pygame.draw.rect(self.screen, self.BLUE, custom_orbital_input_rect, 1)
                    text_surface = self.subtitle_font.render(self.orbital_distance_input_text, True, self.BLACK)
                    text_rect = text_surface.get_rect(midleft=(custom_orbital_input_rect.left + 5, custom_orbital_input_rect.centery))
                    self.screen.blit(text_surface, text_rect)
                # Orbital Eccentricity dropdown
                orbital_eccentricity_label = self.subtitle_font.render("Orbital Eccentricity", True, self.BLACK)
                orbital_eccentricity_label_rect = orbital_eccentricity_label.get_rect(midleft=(self.width - self.customization_panel_width + 50, 465))
                self.screen.blit(orbital_eccentricity_label, orbital_eccentricity_label_rect)
                pygame.draw.rect(self.screen, self.WHITE, self.planet_orbital_eccentricity_dropdown_rect, 2)
                pygame.draw.rect(self.screen, self.BLUE if self.planet_orbital_eccentricity_dropdown_active else self.GRAY, self.planet_orbital_eccentricity_dropdown_rect, 1)
                dropdown_text = "Select Orbital Eccentricity"
                if self.planet_orbital_eccentricity_dropdown_selected:
                    selected = next(((name, value) for name, value in self.planet_orbital_eccentricity_dropdown_options if name == self.planet_orbital_eccentricity_dropdown_selected), None)
                    if selected:
                        name, value = selected
                        if value is not None:
                            dropdown_text = f"{name} ({value:.3f})"
                        else:
                            dropdown_text = name
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
                orbital_period_label_rect = orbital_period_label.get_rect(midleft=(self.width - self.customization_panel_width + 50, 525))
                self.screen.blit(orbital_period_label, orbital_period_label_rect)
                pygame.draw.rect(self.screen, self.WHITE, self.planet_orbital_period_dropdown_rect, 2)
                pygame.draw.rect(self.screen, self.BLUE if self.planet_orbital_period_dropdown_active else self.GRAY, self.planet_orbital_period_dropdown_rect, 1)
                dropdown_text = "Select Orbital Period"
                if self.planet_orbital_period_dropdown_selected:
                    selected = next(((name, value) for name, value in self.planet_orbital_period_dropdown_options if name == self.planet_orbital_period_dropdown_selected), None)
                    if selected:
                        name, value = selected
                        if value is not None:
                            dropdown_text = f"{name} (" + self._format_value(value, 'days', allow_scientific=False) + ")"
                        else:
                            dropdown_text = name
                text_surface = self.subtitle_font.render(dropdown_text, True, self.BLACK)
                text_rect = text_surface.get_rect(midleft=(self.planet_orbital_period_dropdown_rect.left + 5, self.planet_orbital_period_dropdown_rect.centery))
                self.screen.blit(text_surface, text_rect)
                # Show custom orbital period input if "Custom" is selected, just below dropdown
                if self.show_custom_orbital_period_input:
                    custom_period_label = self.subtitle_font.render("Enter Custom Period (days):", True, self.BLACK)
                    custom_period_label_rect = custom_period_label.get_rect(midleft=(self.width - self.customization_panel_width + 50, 565))
                    self.screen.blit(custom_period_label, custom_period_label_rect)
                    custom_period_input_rect = pygame.Rect(self.width - self.customization_panel_width + 50, 595, self.customization_panel_width - 100, 30)
                    pygame.draw.rect(self.screen, self.WHITE, custom_period_input_rect, 2)
                    pygame.draw.rect(self.screen, self.BLUE, custom_period_input_rect, 1)
                    text_surface = self.subtitle_font.render(self.orbital_period_input_text, True, self.BLACK)
                    text_rect = text_surface.get_rect(midleft=(custom_period_input_rect.left + 5, custom_period_input_rect.centery))
                    self.screen.blit(text_surface, text_rect)
                # Stellar Flux dropdown
                stellar_flux_label = self.subtitle_font.render("Stellar Flux (Earth Units)", True, self.BLACK)
                stellar_flux_label_rect = stellar_flux_label.get_rect(midleft=(self.width - self.customization_panel_width + 50, 585))
                self.screen.blit(stellar_flux_label, stellar_flux_label_rect)
                pygame.draw.rect(self.screen, self.WHITE, self.planet_stellar_flux_dropdown_rect, 2)
                pygame.draw.rect(self.screen, self.BLUE if self.planet_stellar_flux_dropdown_active else self.GRAY, self.planet_stellar_flux_dropdown_rect, 1)
                dropdown_text = "Select Stellar Flux"
                if self.planet_stellar_flux_dropdown_selected:
                    selected = next(((name, value) for name, value in self.planet_stellar_flux_dropdown_options if name == self.planet_stellar_flux_dropdown_selected), None)
                    if selected:
                        name, value = selected
                        if value is not None:
                            dropdown_text = f"{name} ({value:.3f} EFU)"
                        else:
                            dropdown_text = name
                text_surface = self.subtitle_font.render(dropdown_text, True, self.BLACK)
                text_rect = text_surface.get_rect(midleft=(self.planet_stellar_flux_dropdown_rect.left + 5, self.planet_stellar_flux_dropdown_rect.centery))
                self.screen.blit(text_surface, text_rect)
                # Show custom stellar flux input if "Custom" is selected, just below dropdown
                if self.show_custom_stellar_flux_input:
                    custom_flux_label = self.subtitle_font.render("Enter Custom Flux (EFU):", True, self.BLACK)
                    custom_flux_label_rect = custom_flux_label.get_rect(midleft=(self.width - self.customization_panel_width + 50, 625))
                    self.screen.blit(custom_flux_label, custom_flux_label_rect)
                    custom_flux_input_rect = pygame.Rect(self.width - self.customization_panel_width + 50, 655, self.customization_panel_width - 100, 30)
                    pygame.draw.rect(self.screen, self.WHITE, custom_flux_input_rect, 2)
                    pygame.draw.rect(self.screen, self.BLUE, custom_flux_input_rect, 1)
                    text_surface = self.subtitle_font.render(self.stellar_flux_input_text, True, self.BLACK)
                    text_rect = text_surface.get_rect(midleft=(custom_flux_input_rect.left + 5, custom_flux_input_rect.centery))
                    self.screen.blit(text_surface, text_rect)
                # Density dropdown (NEW)
                density_label = self.subtitle_font.render("Density (g/cm³)", True, self.BLACK)
                density_label_rect = density_label.get_rect(midleft=(self.width - self.customization_panel_width + 50, 645))
                self.screen.blit(density_label, density_label_rect)
                pygame.draw.rect(self.screen, self.WHITE, self.planet_density_dropdown_rect, 2)
                pygame.draw.rect(self.screen, self.BLUE if self.planet_density_dropdown_active else self.GRAY, self.planet_density_dropdown_rect, 1)
                dropdown_text = "Select Density"
                if self.planet_density_dropdown_selected:
                    selected = next(((name, value) for name, value in self.planet_density_dropdown_options if name == self.planet_density_dropdown_selected), None)
                    if selected:
                        name, value = selected
                        if value is not None:
                            dropdown_text = f"{name} ({value:.2f} g/cm³)"
                        else:
                            dropdown_text = name
                text_surface = self.subtitle_font.render(dropdown_text, True, self.BLACK)
                text_rect = text_surface.get_rect(midleft=(self.planet_density_dropdown_rect.left + 5, self.planet_density_dropdown_rect.centery))
                self.screen.blit(text_surface, text_rect)
                # Show custom density input if "Custom" is selected, just below dropdown
                if self.show_custom_planet_density_input:
                    custom_density_label = self.subtitle_font.render("Enter Custom Density (g/cm³):", True, self.BLACK)
                    custom_density_label_rect = custom_density_label.get_rect(midleft=(self.width - self.customization_panel_width + 50, 745))
                    self.screen.blit(custom_density_label, custom_density_label_rect)
                    custom_density_input_rect = pygame.Rect(self.width - self.customization_panel_width + 50, 775, self.customization_panel_width - 100, 30)
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
                luminosity_label_rect = luminosity_label.get_rect(midleft=(self.width - self.customization_panel_width + 50, 345))
                self.screen.blit(luminosity_label, luminosity_label_rect)
                pygame.draw.rect(self.screen, self.WHITE, self.luminosity_dropdown_rect, 2)
                pygame.draw.rect(self.screen, self.BLUE if self.luminosity_dropdown_active else self.GRAY, 
                               self.luminosity_dropdown_rect, 1)
                dropdown_text = "Select Star Luminosity"
                if self.luminosity_dropdown_selected:
                    selected = next(((name, value) for name, value in self.luminosity_dropdown_options if name == self.luminosity_dropdown_selected), None)
                    if selected:
                        name, value = selected
                        if value is not None and "(Sun)" in name:
                            dropdown_text = f"Sun ({value:.4f} L☉)"
                        elif value is not None:
                            dropdown_text = f"{name} ({value:.4f} L☉)"
                        else:
                            dropdown_text = name
                text_surface = self.subtitle_font.render(dropdown_text, True, self.BLACK)
                text_rect = text_surface.get_rect(midleft=(self.luminosity_dropdown_rect.left + 5, self.luminosity_dropdown_rect.centery))
                self.screen.blit(text_surface, text_rect)

                # Draw radius dropdown
                radius_label = self.subtitle_font.render("Radius (R☉)", True, self.BLACK)
                radius_label_rect = radius_label.get_rect(midleft=(self.width - self.customization_panel_width + 50, 405))
                self.screen.blit(radius_label, radius_label_rect)
                pygame.draw.rect(self.screen, self.WHITE, self.radius_dropdown_rect, 2)
                pygame.draw.rect(self.screen, self.BLUE if self.radius_dropdown_active else self.GRAY, 
                               self.radius_dropdown_rect, 1)
                dropdown_text = "Select Radius"
                if self.radius_dropdown_selected:
                    selected = next(((name, value) for name, value in self.radius_dropdown_options if name == self.radius_dropdown_selected), None)
                    if selected:
                        name, value = selected
                        if value is not None and "(Sun)" in name:
                            dropdown_text = f"Sun ({value:.2f} R☉)"
                        elif value is not None:
                            dropdown_text = f"{name} ({value:.2f} R☉)"
                        else:
                            dropdown_text = name
                text_surface = self.subtitle_font.render(dropdown_text, True, self.BLACK)
                text_rect = text_surface.get_rect(midleft=(self.radius_dropdown_rect.left + 5, self.radius_dropdown_rect.centery))
                self.screen.blit(text_surface, text_rect)

                # Draw activity level dropdown
                activity_label = self.subtitle_font.render("Activity Level", True, self.BLACK)
                activity_label_rect = activity_label.get_rect(midleft=(self.width - self.customization_panel_width + 50, 465))
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
                            dropdown_text = f"{name} ({value:.2f})"
                        else:
                            dropdown_text = name
                text_surface = self.subtitle_font.render(dropdown_text, True, self.BLACK)
                text_rect = text_surface.get_rect(midleft=(self.activity_dropdown_rect.left + 5, self.activity_dropdown_rect.centery))
                self.screen.blit(text_surface, text_rect)

                # Draw metallicity dropdown
                metallicity_label = self.subtitle_font.render("Metallicity [Fe/H]", True, self.BLACK)
                metallicity_label_rect = metallicity_label.get_rect(midleft=(self.width - self.customization_panel_width + 50, 525))
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
                            dropdown_text = f"{name} ({value:+.2f} [Fe/H])"
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
        
        # Draw orbit grid lines first (so they appear behind the bodies)
        for body in self.placed_bodies:
            if body["type"] != "star" and body["name"] in self.orbit_grid_points:
                grid_points = self.orbit_grid_points[body["name"]]
                if len(grid_points) > 1:
                    # Use different colors for different body types
                    if body["type"] == "planet":
                        color = self.LIGHT_GRAY
                    else:  # moon
                        color = (150, 150, 150)  # Slightly darker for moons
                    pygame.draw.lines(self.screen, color, True, grid_points, 1)
        
        # Draw orbit lines and bodies
        for body in self.placed_bodies:
            # Draw orbit line
            if body["type"] != "star" and body["name"] in self.orbit_points:
                points = self.orbit_points[body["name"]]
                if len(points) > 1:
                    # Use different colors for different body types
                    if body["type"] == "planet":
                        color = self.GRAY
                    else:  # moon
                        color = (100, 100, 100)  # Slightly darker for moons
                    pygame.draw.lines(self.screen, color, False, points, 1)
            
            # Draw body
            if body["type"] == "star":
                # Stars don't rotate
                color = self.YELLOW
                pygame.draw.circle(self.screen, color, body["position"].astype(int), body["radius"])
            else:
                # Planets and moons rotate
                color = self.BLUE if body["type"] == "planet" else self.WHITE
                
                # Draw the rotating body
                self.draw_rotating_body(body, color)
            
            # Highlight selected body
            if self.selected_body and body["name"] == self.selected_body["name"]:
                # Draw a circle around the selected body
                pygame.draw.circle(self.screen, self.RED, body["position"].astype(int), body["radius"] + 5, 2)
        
        # Draw instructions
        if self.active_tab:
            instruction_text = self.subtitle_font.render(f"Click in the space to place a {self.active_tab}", True, self.WHITE)
        else:
            instruction_text = self.subtitle_font.render("Select a tab to place celestial bodies", True, self.WHITE)
        instruction_rect = instruction_text.get_rect(center=(self.width//2, self.height - 30))
        self.screen.blit(instruction_text, instruction_rect)
        
        # Render dropdown menu last, so it appears on top of everything
        self.render_dropdown()
        
        pygame.display.flip()
    
    def draw_rotating_body(self, body, color):
        """Draw a celestial body with rotation"""
        # Create a surface for the body
        radius = body["radius"]
        surface_size = radius * 2 + 4  # Add some padding
        surface = pygame.Surface((surface_size, surface_size), pygame.SRCALPHA)
        
        # Draw the main body
        pygame.draw.circle(surface, color, (radius + 2, radius + 2), radius)
        
        # Draw a line to indicate rotation
        rotation_angle = body["rotation_angle"]
        end_x = radius + 2 + radius * 0.8 * np.cos(rotation_angle)
        end_y = radius + 2 + radius * 0.8 * np.sin(rotation_angle)
        pygame.draw.line(surface, self.WHITE, (radius + 2, radius + 2), (end_x, end_y), 2)
        
        # Blit the surface onto the screen
        self.screen.blit(surface, (body["position"][0] - radius - 2, body["position"][1] - radius - 2))
    
    def render_simulation_builder(self):
        """Render the simulation builder screen with tabs and space area"""
        # Fill background
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
            
            # MASS SECTION (always show mass label/input, dropdown only for planets)
            mass_label = self.subtitle_font.render("Mass", True, self.BLACK)
            mass_label_rect = mass_label.get_rect(midleft=(self.width - self.customization_panel_width + 50, 105))
            self.screen.blit(mass_label, mass_label_rect)
            
            if self.selected_body.get('type') == 'planet':
                # For planets, only show the dropdown
                pygame.draw.rect(self.screen, self.WHITE, self.planet_dropdown_rect, 2)
                pygame.draw.rect(self.screen, self.BLUE if self.planet_dropdown_active else self.GRAY, 
                               self.planet_dropdown_rect, 1)
                dropdown_text = "Select Reference Planet"
                if self.planet_dropdown_selected:
                    dropdown_text = self.planet_dropdown_selected
                text_surface = self.subtitle_font.render(dropdown_text, True, self.BLACK)
                text_rect = text_surface.get_rect(midleft=(self.planet_dropdown_rect.left + 5, 
                                                         self.planet_dropdown_rect.centery))
                self.screen.blit(text_surface, text_rect)

                # Show custom mass input if "Custom Mass" is selected
                if self.show_custom_mass_input:
                    custom_mass_label = self.subtitle_font.render("Enter Custom Mass:", True, self.BLACK)
                    custom_mass_label_rect = custom_mass_label.get_rect(midleft=(self.width - self.customization_panel_width + 50, 165))
                    self.screen.blit(custom_mass_label, custom_mass_label_rect)
                    
                    pygame.draw.rect(self.screen, self.WHITE, self.mass_input_rect, 2)
                    pygame.draw.rect(self.screen, self.BLUE if self.mass_input_active else self.GRAY, 
                                   self.mass_input_rect, 1)
                    if self.selected_body.get('type') == 'moon':
                        # For moons, mass is already in Lunar masses
                        lunar_mass = self.selected_body.get('mass', 1.0)
                        if self.mass_input_active:
                            text_surface = self.subtitle_font.render(self.mass_input_text, True, self.BLACK)
                        else:
                            text_surface = self.subtitle_font.render(f"{lunar_mass:.3f}", True, self.BLACK)
                    else:
                        if self.mass_input_active:
                            text_surface = self.subtitle_font.render(self.mass_input_text, True, self.BLACK)
                        else:
                            text_surface = self.subtitle_font.render(f"{self.selected_body.get('mass', 1.0):.3f}", True, self.BLACK)
                    text_rect = text_surface.get_rect(midleft=(self.mass_input_rect.left + 5, 
                                                             self.mass_input_rect.centery))
                    self.screen.blit(text_surface, text_rect)
            else:
                # For non-planets, show the mass input box
                pygame.draw.rect(self.screen, self.WHITE, self.mass_input_rect, 2)
                pygame.draw.rect(self.screen, self.BLUE if self.mass_input_active else self.GRAY, 
                               self.mass_input_rect, 1)
                if self.selected_body.get('type') == 'moon':
                    # For moons, mass is already in Lunar masses
                    lunar_mass = self.selected_body.get('mass', 1.0)
                    if self.mass_input_active:
                        text_surface = self.subtitle_font.render(self.mass_input_text, True, self.BLACK)
                    else:
                        text_surface = self.subtitle_font.render(f"{lunar_mass:.3f}", True, self.BLACK)
                else:
                    if self.mass_input_active:
                        text_surface = self.subtitle_font.render(self.mass_input_text, True, self.BLACK)
                    else:
                        text_surface = self.subtitle_font.render(f"{self.selected_body.get('mass', 1.0):.3f}", True, self.BLACK)
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
                if self.planet_age_dropdown_selected:
                    dropdown_text = self.planet_age_dropdown_selected
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
                        text_surface = self.subtitle_font.render(f"{self.selected_body.get('age', 0.0):.1f}", True, self.BLACK)
                    text_rect = text_surface.get_rect(midleft=(self.age_input_rect.left + 5, 
                                                             self.age_input_rect.centery))
                    self.screen.blit(text_surface, text_rect)
            elif self.selected_body.get('type') == 'moon':
                # For moons, show the age dropdown
                pygame.draw.rect(self.screen, self.WHITE, self.moon_age_dropdown_rect, 2)
                pygame.draw.rect(self.screen, self.BLUE if self.moon_age_dropdown_active else self.GRAY, 
                               self.moon_age_dropdown_rect, 1)
                dropdown_text = "Select Age"
                if self.moon_age_dropdown_selected:
                    dropdown_text = self.moon_age_dropdown_selected
                text_surface = self.subtitle_font.render(dropdown_text, True, self.BLACK)
                text_rect = text_surface.get_rect(midleft=(self.moon_age_dropdown_rect.left + 5, 
                                                         self.moon_age_dropdown_rect.centery))
                self.screen.blit(text_surface, text_rect)

                # Show custom age input if "Custom Age" is selected
                if self.show_custom_moon_age_input:
                    custom_age_label = self.subtitle_font.render("Enter Custom Age (Gyr):", True, self.BLACK)
                    custom_age_label_rect = custom_age_label.get_rect(midleft=(self.width - self.customization_panel_width + 50, 225))
                    self.screen.blit(custom_age_label, custom_age_label_rect)
                    
                    pygame.draw.rect(self.screen, self.WHITE, self.age_input_rect, 2)
                    pygame.draw.rect(self.screen, self.BLUE if self.age_input_active else self.GRAY, 
                                   self.age_input_rect, 1)
                    if self.age_input_active:
                        text_surface = self.subtitle_font.render(self.age_input_text, True, self.BLACK)
                    else:
                        text_surface = self.subtitle_font.render(f"{self.selected_body.get('age', 0.0):.1f}", True, self.BLACK)
                    text_rect = text_surface.get_rect(midleft=(self.age_input_rect.left + 5, 
                                                             self.age_input_rect.centery))
                    self.screen.blit(text_surface, text_rect)
                
                # RADIUS SECTION (only for moons)
                radius_label = self.subtitle_font.render("Radius (km)", True, self.BLACK)
                radius_label_rect = radius_label.get_rect(midleft=(self.width - self.customization_panel_width + 50, 225))
                self.screen.blit(radius_label, radius_label_rect)
                
                # Draw the moon radius dropdown
                pygame.draw.rect(self.screen, self.WHITE, self.moon_radius_dropdown_rect, 2)
                pygame.draw.rect(self.screen, self.BLUE if self.moon_radius_dropdown_active else self.GRAY, 
                               self.moon_radius_dropdown_rect, 1)
                dropdown_text = "Select Radius"
                if self.moon_radius_dropdown_selected:
                    dropdown_text = self.moon_radius_dropdown_selected
                text_surface = self.subtitle_font.render(dropdown_text, True, self.BLACK)
                text_rect = text_surface.get_rect(midleft=(self.moon_radius_dropdown_rect.left + 5, 
                                                         self.moon_radius_dropdown_rect.centery))
                self.screen.blit(text_surface, text_rect)

                # Show custom radius input if "Custom" is selected
                if self.show_custom_moon_radius_input:
                    custom_radius_label = self.subtitle_font.render("Enter Custom Radius (km):", True, self.BLACK)
                    custom_radius_label_rect = custom_radius_label.get_rect(midleft=(self.width - self.customization_panel_width + 50, 285))
                    self.screen.blit(custom_radius_label, custom_radius_label_rect)
                    
                    custom_radius_input_rect = pygame.Rect(self.width - self.customization_panel_width + 50, 315, self.customization_panel_width - 100, 30)
                    pygame.draw.rect(self.screen, self.WHITE, custom_radius_input_rect, 2)
                    pygame.draw.rect(self.screen, self.BLUE, custom_radius_input_rect, 1)
                    text_surface = self.subtitle_font.render("", True, self.BLACK)  # Placeholder for input text
                    text_rect = text_surface.get_rect(midleft=(custom_radius_input_rect.left + 5, custom_radius_input_rect.centery))
                    self.screen.blit(text_surface, text_rect)
                
                # ORBITAL DISTANCE SECTION (only for moons)
                orbital_distance_label = self.subtitle_font.render("Orbital Distance (km)", True, self.BLACK)
                orbital_distance_label_rect = orbital_distance_label.get_rect(midleft=(self.width - self.customization_panel_width + 50, 285))
                self.screen.blit(orbital_distance_label, orbital_distance_label_rect)
                
                # Draw the moon orbital distance dropdown
                pygame.draw.rect(self.screen, self.WHITE, self.moon_orbital_distance_dropdown_rect, 2)
                pygame.draw.rect(self.screen, self.BLUE if self.moon_orbital_distance_dropdown_active else self.GRAY, 
                               self.moon_orbital_distance_dropdown_rect, 1)
                dropdown_text = "Select Orbital Distance"
                if self.moon_orbital_distance_dropdown_selected:
                    selected = next(((name, value) for name, value in self.moon_orbital_distance_dropdown_options if name == self.moon_orbital_distance_dropdown_selected), None)
                    if selected:
                        name, value = selected
                        if value is not None:
                            dropdown_text = f"{name} ({value:.0f} km)"
                        else:
                            dropdown_text = name
                text_surface = self.subtitle_font.render(dropdown_text, True, self.BLACK)
                text_rect = text_surface.get_rect(midleft=(self.moon_orbital_distance_dropdown_rect.left + 5, 
                                                         self.moon_orbital_distance_dropdown_rect.centery))
                self.screen.blit(text_surface, text_rect)

                # Show custom orbital distance input if "Custom" is selected
                if self.show_custom_moon_orbital_distance_input:
                    custom_distance_label = self.subtitle_font.render("Enter Custom Distance (km):", True, self.BLACK)
                    custom_distance_label_rect = custom_distance_label.get_rect(midleft=(self.width - self.customization_panel_width + 50, 345))
                    self.screen.blit(custom_distance_label, custom_distance_label_rect)
                    
                    custom_distance_input_rect = pygame.Rect(self.width - self.customization_panel_width + 50, 375, self.customization_panel_width - 100, 30)
                    pygame.draw.rect(self.screen, self.WHITE, custom_distance_input_rect, 2)
                    pygame.draw.rect(self.screen, self.BLUE, custom_distance_input_rect, 1)
                    text_surface = self.subtitle_font.render("", True, self.BLACK)  # Placeholder for input text
                    text_rect = text_surface.get_rect(midleft=(custom_distance_input_rect.left + 5, custom_distance_input_rect.centery))
                    self.screen.blit(text_surface, text_rect)
                
                # ORBITAL PERIOD SECTION (only for moons)
                orbital_period_label = self.subtitle_font.render("Orbital Period (days)", True, self.BLACK)
                orbital_period_label_rect = orbital_period_label.get_rect(midleft=(self.width - self.customization_panel_width + 50, 345))
                self.screen.blit(orbital_period_label, orbital_period_label_rect)
                
                # Draw the moon orbital period dropdown
                pygame.draw.rect(self.screen, self.WHITE, self.moon_orbital_period_dropdown_rect, 2)
                pygame.draw.rect(self.screen, self.BLUE if self.moon_orbital_period_dropdown_active else self.GRAY, 
                               self.moon_orbital_period_dropdown_rect, 1)
                dropdown_text = "Select Orbital Period"
                if self.moon_orbital_period_dropdown_selected:
                    selected = next(((name, value) for name, value in self.moon_orbital_period_dropdown_options if name == self.moon_orbital_period_dropdown_selected), None)
                    if selected:
                        name, value = selected
                        if value is not None:
                            dropdown_text = f"{name} ({value:.1f} days)"
                        else:
                            dropdown_text = name
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
                    text_surface = self.subtitle_font.render("", True, self.BLACK)  # Placeholder for input text
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
                if self.moon_temperature_dropdown_selected:
                    selected = next(((name, value) for name, value in self.moon_temperature_dropdown_options if name == self.moon_temperature_dropdown_selected), None)
                    if selected:
                        name, value = selected
                        if value is not None:
                            dropdown_text = f"{name} ({value:.0f} K)"
                        else:
                            dropdown_text = name
                text_surface = self.subtitle_font.render(dropdown_text, True, self.BLACK)
                text_rect = text_surface.get_rect(midleft=(self.moon_temperature_dropdown_rect.left + 5, 
                                                         self.moon_temperature_dropdown_rect.centery))
                self.screen.blit(text_surface, text_rect)

                # Show custom temperature input if "Custom" is selected
                if self.show_custom_moon_temperature_input:
                    custom_temp_label = self.subtitle_font.render("Enter Custom Temperature (K):", True, self.BLACK)
                    custom_temp_label_rect = custom_temp_label.get_rect(midleft=(self.width - self.customization_panel_width + 50, 465))
                    self.screen.blit(custom_temp_label, custom_temp_label_rect)
                    
                    custom_temp_input_rect = pygame.Rect(self.width - self.customization_panel_width + 50, 495, self.customization_panel_width - 100, 30)
                    pygame.draw.rect(self.screen, self.WHITE, custom_temp_input_rect, 2)
                    pygame.draw.rect(self.screen, self.BLUE, custom_temp_input_rect, 1)
                    text_surface = self.subtitle_font.render("", True, self.BLACK)  # Placeholder for input text
                    text_rect = text_surface.get_rect(midleft=(custom_temp_input_rect.left + 5, custom_temp_input_rect.centery))
                    self.screen.blit(text_surface, text_rect)
                
                # SURFACE GRAVITY SECTION (only for moons)
                gravity_label = self.subtitle_font.render("Surface Gravity (m/s²)", True, self.BLACK)
                gravity_label_rect = gravity_label.get_rect(midleft=(self.width - self.customization_panel_width + 50, 465))
                self.screen.blit(gravity_label, gravity_label_rect)
                
                # Draw the moon gravity dropdown
                pygame.draw.rect(self.screen, self.WHITE, self.moon_gravity_dropdown_rect, 2)
                pygame.draw.rect(self.screen, self.BLUE if self.moon_gravity_dropdown_active else self.GRAY, 
                               self.moon_gravity_dropdown_rect, 1)
                dropdown_text = "Select Gravity"
                if self.moon_gravity_dropdown_selected:
                    selected = next(((name, value) for name, value in self.moon_gravity_dropdown_options if name == self.moon_gravity_dropdown_selected), None)
                    if selected:
                        name, value = selected
                        if value is not None:
                            dropdown_text = f"{name} ({value:.2f} m/s²)"
                        else:
                            dropdown_text = name
                text_surface = self.subtitle_font.render(dropdown_text, True, self.BLACK)
                text_rect = text_surface.get_rect(midleft=(self.moon_gravity_dropdown_rect.left + 5, 
                                                         self.moon_gravity_dropdown_rect.centery))
                self.screen.blit(text_surface, text_rect)

                # Show custom gravity input if "Custom" is selected
                if self.show_custom_moon_gravity_input:
                    custom_gravity_label = self.subtitle_font.render("Enter Custom Gravity (m/s²):", True, self.BLACK)
                    custom_gravity_label_rect = custom_gravity_label.get_rect(midleft=(self.width - self.customization_panel_width + 50, 525))
                    self.screen.blit(custom_gravity_label, custom_gravity_label_rect)
                    
                    custom_gravity_input_rect = pygame.Rect(self.width - self.customization_panel_width + 50, 555, self.customization_panel_width - 100, 30)
                    pygame.draw.rect(self.screen, self.WHITE, custom_gravity_input_rect, 2)
                    pygame.draw.rect(self.screen, self.BLUE, custom_gravity_input_rect, 1)
                    text_surface = self.subtitle_font.render("", True, self.BLACK)  # Placeholder for input text
                    text_rect = text_surface.get_rect(midleft=(custom_gravity_input_rect.left + 5, custom_gravity_input_rect.centery))
                    self.screen.blit(text_surface, text_rect)
            else:
                # For stars, show the age input box
                pygame.draw.rect(self.screen, self.WHITE, self.age_input_rect, 2)
                pygame.draw.rect(self.screen, self.BLUE if self.age_input_active else self.GRAY, self.age_input_rect, 1)
                if self.age_input_active:
                    text_surface = self.subtitle_font.render(self.age_input_text, True, self.BLACK)
                else:
                    text_surface = self.subtitle_font.render(
                        self.format_age_display(self.selected_body.get("age", 0.0)), 
                        True, self.BLACK
                    )
                text_rect = text_surface.get_rect(midleft=(self.age_input_rect.left + 5, self.age_input_rect.centery))
                self.screen.blit(text_surface, text_rect)
            
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
                if self.planet_radius_dropdown_selected:
                    selected = next(((name, value) for name, value in self.planet_radius_dropdown_options if name == self.planet_radius_dropdown_selected), None)
                    if selected:
                        name, value = selected
                        if value is not None:
                            dropdown_text = f"{name} ({value:.2f} R🜨)"
                        else:
                            dropdown_text = name
                text_surface = self.subtitle_font.render(dropdown_text, True, self.BLACK)
                text_rect = text_surface.get_rect(midleft=(self.planet_radius_dropdown_rect.left + 5, 
                                                         self.planet_radius_dropdown_rect.centery))
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
                if self.planet_temperature_dropdown_selected:
                    selected = next(((name, value) for name, value in self.planet_temperature_dropdown_options if name == self.planet_temperature_dropdown_selected), None)
                    if selected:
                        name, value = selected
                        if value is not None:
                            dropdown_text = f"{name} ({value:.0f} K)"
                        else:
                            dropdown_text = name
                text_surface = self.subtitle_font.render(dropdown_text, True, self.BLACK)
                text_rect = text_surface.get_rect(midleft=(self.planet_temperature_dropdown_rect.left + 5, 
                                                         self.planet_temperature_dropdown_rect.centery))
                self.screen.blit(text_surface, text_rect)

                # GRAVITY SECTION (only for planets)
                gravity_label = self.subtitle_font.render("Surface Gravity (m/s²)", True, self.BLACK)
                gravity_label_rect = gravity_label.get_rect(midleft=(self.width - self.customization_panel_width + 50, 345))
                self.screen.blit(gravity_label, gravity_label_rect)
                self.planet_gravity_dropdown_rect.y = 360
                pygame.draw.rect(self.screen, self.WHITE, self.planet_gravity_dropdown_rect, 2)
                pygame.draw.rect(self.screen, self.BLUE if self.planet_gravity_dropdown_active else self.GRAY, self.planet_gravity_dropdown_rect, 1)
                dropdown_text = "Select Reference Planet"
                if self.planet_gravity_dropdown_selected:
                    selected = next(((name, value) for name, value in self.planet_gravity_dropdown_options if name == self.planet_gravity_dropdown_selected), None)
                    if selected:
                        name, value = selected
                        if value is not None:
                            dropdown_text = f"{name} ({value:.2f} m/s²)"
                        else:
                            dropdown_text = name
                text_surface = self.subtitle_font.render(dropdown_text, True, self.BLACK)
                text_rect = text_surface.get_rect(midleft=(self.planet_gravity_dropdown_rect.left + 5, self.planet_gravity_dropdown_rect.centery))
                self.screen.blit(text_surface, text_rect)
                
                # Show custom gravity input if "Custom" is selected, just below dropdown
                if self.show_custom_planet_gravity_input:
                    custom_gravity_label = self.subtitle_font.render("Enter Custom Gravity (m/s²):", True, self.BLACK)
                    custom_gravity_label_rect = custom_gravity_label.get_rect(midleft=(self.width - self.customization_panel_width + 50, 415))
                    self.screen.blit(custom_gravity_label, custom_gravity_label_rect)
                    
                    custom_gravity_input_rect = pygame.Rect(self.width - self.customization_panel_width + 50, 445, self.customization_panel_width - 100, 30)
                    pygame.draw.rect(self.screen, self.WHITE, custom_gravity_input_rect, 2)
                    pygame.draw.rect(self.screen, self.BLUE, custom_gravity_input_rect, 1)
                    text_surface = self.subtitle_font.render(self.planet_gravity_input_text, True, self.BLACK)
                    text_rect = text_surface.get_rect(midleft=(custom_gravity_input_rect.left + 5, custom_gravity_input_rect.centery))
                    self.screen.blit(text_surface, text_rect)
                # Orbital Distance (AU) dropdown
                orbital_distance_label = self.subtitle_font.render("Orbital Distance (AU)", True, self.BLACK)
                orbital_distance_label_rect = orbital_distance_label.get_rect(midleft=(self.width - self.customization_panel_width + 50, 405))
                self.screen.blit(orbital_distance_label, orbital_distance_label_rect)
                pygame.draw.rect(self.screen, self.WHITE, self.planet_orbital_distance_dropdown_rect, 2)
                pygame.draw.rect(self.screen, self.BLUE if self.planet_orbital_distance_dropdown_active else self.GRAY, self.planet_orbital_distance_dropdown_rect, 1)
                dropdown_text = "Select Orbital Distance"
                if self.planet_orbital_distance_dropdown_selected:
                    selected = next(((name, value) for name, value in self.planet_orbital_distance_dropdown_options if name == self.planet_orbital_distance_dropdown_selected), None)
                    if selected:
                        name, value = selected
                        if value is not None:
                            dropdown_text = f"{name} ({value:.2f} AU)"
                        else:
                            dropdown_text = name
                text_surface = self.subtitle_font.render(dropdown_text, True, self.BLACK)
                text_rect = text_surface.get_rect(midleft=(self.planet_orbital_distance_dropdown_rect.left + 5, self.planet_orbital_distance_dropdown_rect.centery))
                self.screen.blit(text_surface, text_rect)
                # Show custom orbital distance input if "Custom" is selected, just below dropdown
                if self.show_custom_orbital_distance_input:
                    custom_orbital_label = self.subtitle_font.render("Enter Custom Distance (AU):", True, self.BLACK)
                    custom_orbital_label_rect = custom_orbital_label.get_rect(midleft=(self.width - self.customization_panel_width + 50, 455))
                    self.screen.blit(custom_orbital_label, custom_orbital_label_rect)
                    custom_orbital_input_rect = pygame.Rect(self.width - self.customization_panel_width + 50, 485, self.customization_panel_width - 100, 30)
                    pygame.draw.rect(self.screen, self.WHITE, custom_orbital_input_rect, 2)
                    pygame.draw.rect(self.screen, self.BLUE, custom_orbital_input_rect, 1)
                    text_surface = self.subtitle_font.render(self.orbital_distance_input_text, True, self.BLACK)
                    text_rect = text_surface.get_rect(midleft=(custom_orbital_input_rect.left + 5, custom_orbital_input_rect.centery))
                    self.screen.blit(text_surface, text_rect)
                # Orbital Eccentricity dropdown
                orbital_eccentricity_label = self.subtitle_font.render("Orbital Eccentricity", True, self.BLACK)
                orbital_eccentricity_label_rect = orbital_eccentricity_label.get_rect(midleft=(self.width - self.customization_panel_width + 50, 465))
                self.screen.blit(orbital_eccentricity_label, orbital_eccentricity_label_rect)
                pygame.draw.rect(self.screen, self.WHITE, self.planet_orbital_eccentricity_dropdown_rect, 2)
                pygame.draw.rect(self.screen, self.BLUE if self.planet_orbital_eccentricity_dropdown_active else self.GRAY, self.planet_orbital_eccentricity_dropdown_rect, 1)
                dropdown_text = "Select Orbital Eccentricity"
                if self.planet_orbital_eccentricity_dropdown_selected:
                    selected = next(((name, value) for name, value in self.planet_orbital_eccentricity_dropdown_options if name == self.planet_orbital_eccentricity_dropdown_selected), None)
                    if selected:
                        name, value = selected
                        if value is not None:
                            dropdown_text = f"{name} ({value:.3f})"
                        else:
                            dropdown_text = name
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
                orbital_period_label_rect = orbital_period_label.get_rect(midleft=(self.width - self.customization_panel_width + 50, 525))
                self.screen.blit(orbital_period_label, orbital_period_label_rect)
                pygame.draw.rect(self.screen, self.WHITE, self.planet_orbital_period_dropdown_rect, 2)
                pygame.draw.rect(self.screen, self.BLUE if self.planet_orbital_period_dropdown_active else self.GRAY, self.planet_orbital_period_dropdown_rect, 1)
                dropdown_text = "Select Orbital Period"
                if self.planet_orbital_period_dropdown_selected:
                    selected = next(((name, value) for name, value in self.planet_orbital_period_dropdown_options if name == self.planet_orbital_period_dropdown_selected), None)
                    if selected:
                        name, value = selected
                        if value is not None:
                            dropdown_text = f"{name} (" + self._format_value(value, 'days', allow_scientific=False) + ")"
                        else:
                            dropdown_text = name
                text_surface = self.subtitle_font.render(dropdown_text, True, self.BLACK)
                text_rect = text_surface.get_rect(midleft=(self.planet_orbital_period_dropdown_rect.left + 5, self.planet_orbital_period_dropdown_rect.centery))
                self.screen.blit(text_surface, text_rect)
                # Show custom orbital period input if "Custom" is selected, just below dropdown
                if self.show_custom_orbital_period_input:
                    custom_period_label = self.subtitle_font.render("Enter Custom Period (days):", True, self.BLACK)
                    custom_period_label_rect = custom_period_label.get_rect(midleft=(self.width - self.customization_panel_width + 50, 565))
                    self.screen.blit(custom_period_label, custom_period_label_rect)
                    custom_period_input_rect = pygame.Rect(self.width - self.customization_panel_width + 50, 595, self.customization_panel_width - 100, 30)
                    pygame.draw.rect(self.screen, self.WHITE, custom_period_input_rect, 2)
                    pygame.draw.rect(self.screen, self.BLUE, custom_period_input_rect, 1)
                    text_surface = self.subtitle_font.render(self.orbital_period_input_text, True, self.BLACK)
                    text_rect = text_surface.get_rect(midleft=(custom_period_input_rect.left + 5, custom_period_input_rect.centery))
                    self.screen.blit(text_surface, text_rect)
                # Stellar Flux dropdown
                stellar_flux_label = self.subtitle_font.render("Stellar Flux (Earth Units)", True, self.BLACK)
                stellar_flux_label_rect = stellar_flux_label.get_rect(midleft=(self.width - self.customization_panel_width + 50, 585))
                self.screen.blit(stellar_flux_label, stellar_flux_label_rect)
                pygame.draw.rect(self.screen, self.WHITE, self.planet_stellar_flux_dropdown_rect, 2)
                pygame.draw.rect(self.screen, self.BLUE if self.planet_stellar_flux_dropdown_active else self.GRAY, self.planet_stellar_flux_dropdown_rect, 1)
                dropdown_text = "Select Stellar Flux"
                if self.planet_stellar_flux_dropdown_selected:
                    selected = next(((name, value) for name, value in self.planet_stellar_flux_dropdown_options if name == self.planet_stellar_flux_dropdown_selected), None)
                    if selected:
                        name, value = selected
                        if value is not None:
                            dropdown_text = f"{name} ({value:.3f} EFU)"
                        else:
                            dropdown_text = name
                text_surface = self.subtitle_font.render(dropdown_text, True, self.BLACK)
                text_rect = text_surface.get_rect(midleft=(self.planet_stellar_flux_dropdown_rect.left + 5, self.planet_stellar_flux_dropdown_rect.centery))
                self.screen.blit(text_surface, text_rect)
                # Show custom stellar flux input if "Custom" is selected, just below dropdown
                if self.show_custom_stellar_flux_input:
                    custom_flux_label = self.subtitle_font.render("Enter Custom Flux (EFU):", True, self.BLACK)
                    custom_flux_label_rect = custom_flux_label.get_rect(midleft=(self.width - self.customization_panel_width + 50, 625))
                    self.screen.blit(custom_flux_label, custom_flux_label_rect)
                    custom_flux_input_rect = pygame.Rect(self.width - self.customization_panel_width + 50, 655, self.customization_panel_width - 100, 30)
                    pygame.draw.rect(self.screen, self.WHITE, custom_flux_input_rect, 2)
                    pygame.draw.rect(self.screen, self.BLUE, custom_flux_input_rect, 1)
                    text_surface = self.subtitle_font.render(self.stellar_flux_input_text, True, self.BLACK)
                    text_rect = text_surface.get_rect(midleft=(custom_flux_input_rect.left + 5, custom_flux_input_rect.centery))
                    self.screen.blit(text_surface, text_rect)
                # Density dropdown (NEW)
                density_label = self.subtitle_font.render("Density (g/cm³)", True, self.BLACK)
                density_label_rect = density_label.get_rect(midleft=(self.width - self.customization_panel_width + 50, 645))
                self.screen.blit(density_label, density_label_rect)
                pygame.draw.rect(self.screen, self.WHITE, self.planet_density_dropdown_rect, 2)
                pygame.draw.rect(self.screen, self.BLUE if self.planet_density_dropdown_active else self.GRAY, self.planet_density_dropdown_rect, 1)
                dropdown_text = "Select Density"
                if self.planet_density_dropdown_selected:
                    selected = next(((name, value) for name, value in self.planet_density_dropdown_options if name == self.planet_density_dropdown_selected), None)
                    if selected:
                        name, value = selected
                        if value is not None:
                            dropdown_text = f"{name} ({value:.2f} g/cm³)"
                        else:
                            dropdown_text = name
                text_surface = self.subtitle_font.render(dropdown_text, True, self.BLACK)
                text_rect = text_surface.get_rect(midleft=(self.planet_density_dropdown_rect.left + 5, self.planet_density_dropdown_rect.centery))
                self.screen.blit(text_surface, text_rect)
                # Show custom density input if "Custom" is selected, just below dropdown
                if self.show_custom_planet_density_input:
                    custom_density_label = self.subtitle_font.render("Enter Custom Density (g/cm³):", True, self.BLACK)
                    custom_density_label_rect = custom_density_label.get_rect(midleft=(self.width - self.customization_panel_width + 50, 745))
                    self.screen.blit(custom_density_label, custom_density_label_rect)
                    custom_density_input_rect = pygame.Rect(self.width - self.customization_panel_width + 50, 775, self.customization_panel_width - 100, 30)
                    pygame.draw.rect(self.screen, self.WHITE, custom_density_input_rect, 2)
                    pygame.draw.rect(self.screen, self.BLUE, custom_density_input_rect, 1)
                    text_surface = self.subtitle_font.render(self.planet_density_input_text, True, self.BLACK)
                    text_rect = text_surface.get_rect(midleft=(custom_density_input_rect.left + 5, custom_density_input_rect.centery))
                    self.screen.blit(text_surface, text_rect)
            # Draw spacetime grid
                spectral_label = self.subtitle_font.render("Spectral Type", True, self.BLACK)
                spectral_label_rect = spectral_label.get_rect(midleft=(self.width - self.customization_panel_width + 50, 225))
                self.screen.blit(spectral_label, spectral_label_rect)
                pygame.draw.rect(self.screen, self.WHITE, self.spectral_dropdown_rect, 2)
                pygame.draw.rect(self.screen, self.BLUE if self.spectral_dropdown_active else self.GRAY, self.spectral_dropdown_rect, 1)
                dropdown_text = "Select Spectral Type"
                if self.spectral_dropdown_selected:
                    # Extract the base spectral type name from the selected text
                    base_name = self.spectral_dropdown_selected.split(" (")[0] + " (" + self.spectral_dropdown_selected.split(" (")[1].split(")")[0] + ")"
                    selected = next(((name, value, color) for name, value, color in self.spectral_dropdown_options if name == base_name), None)
                    if selected:
                        name, value, _ = selected
                        if value is not None and "(Sun)" in name:
                            dropdown_text = f"Sun ({value:.0f} K)"
                        elif value is not None:
                            dropdown_text = f"{name} ({value:.0f} K)"
                        else:
                            dropdown_text = name
                    else:
                        # If lookup fails, just display the selected text directly
                        dropdown_text = self.spectral_dropdown_selected
                text_surface = self.subtitle_font.render(dropdown_text, True, self.BLACK)
                text_rect = text_surface.get_rect(midleft=(self.spectral_dropdown_rect.left + 5, self.spectral_dropdown_rect.centery))
                self.screen.blit(text_surface, text_rect)

                # Draw star age dropdown
                age_label = self.subtitle_font.render("Age (Gyr)", True, self.BLACK)
                age_label_rect = age_label.get_rect(midleft=(self.width - self.customization_panel_width + 50, 165))
                self.screen.blit(age_label, age_label_rect)
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

                # Draw luminosity input for stars
                luminosity_label = self.subtitle_font.render("Luminosity (L☉)", True, self.BLACK)
                luminosity_label_rect = luminosity_label.get_rect(midleft=(self.width - self.customization_panel_width + 50, 345))
                self.screen.blit(luminosity_label, luminosity_label_rect)
                pygame.draw.rect(self.screen, self.WHITE, self.luminosity_dropdown_rect, 2)
                pygame.draw.rect(self.screen, self.BLUE if self.luminosity_dropdown_active else self.GRAY, 
                               self.luminosity_dropdown_rect, 1)
                dropdown_text = "Select Star Luminosity"
                if self.luminosity_dropdown_selected:
                    selected = next(((name, value) for name, value in self.luminosity_dropdown_options if name == self.luminosity_dropdown_selected), None)
                    if selected:
                        name, value = selected
                        if value is not None:
                            dropdown_text = f"{name} ({value:.4f} L☉)"
                        else:
                            dropdown_text = name
                text_surface = self.subtitle_font.render(dropdown_text, True, self.BLACK)
                text_rect = text_surface.get_rect(midleft=(self.luminosity_dropdown_rect.left + 5, self.luminosity_dropdown_rect.centery))
                self.screen.blit(text_surface, text_rect)

                # Draw radius dropdown
                radius_label = self.subtitle_font.render("Radius (R☉)", True, self.BLACK)
                radius_label_rect = radius_label.get_rect(midleft=(self.width - self.customization_panel_width + 50, 405))
                self.screen.blit(radius_label, radius_label_rect)
                pygame.draw.rect(self.screen, self.WHITE, self.radius_dropdown_rect, 2)
                pygame.draw.rect(self.screen, self.BLUE if self.radius_dropdown_active else self.GRAY, 
                               self.radius_dropdown_rect, 1)
                dropdown_text = "Select Radius"
                if self.radius_dropdown_selected:
                    selected = next(((name, value) for name, value in self.radius_dropdown_options if name == self.radius_dropdown_selected), None)
                    if selected:
                        name, value = selected
                        if value is not None:
                            dropdown_text = f"{name} ({value:.2f} R☉)"
                        else:
                            dropdown_text = name
                text_surface = self.subtitle_font.render(dropdown_text, True, self.BLACK)
                text_rect = text_surface.get_rect(midleft=(self.radius_dropdown_rect.left + 5, self.radius_dropdown_rect.centery))
                self.screen.blit(text_surface, text_rect)

                # Draw activity level dropdown
                activity_label = self.subtitle_font.render("Activity Level", True, self.BLACK)
                activity_label_rect = activity_label.get_rect(midleft=(self.width - self.customization_panel_width + 50, 465))
                self.screen.blit(activity_label, activity_label_rect)
                pygame.draw.rect(self.screen, self.WHITE, self.activity_dropdown_rect, 2)
                pygame.draw.rect(self.screen, self.BLUE if self.activity_dropdown_active else self.GRAY, 
                               self.activity_dropdown_rect, 1)
                dropdown_text = "Select Activity Level"
                if self.activity_dropdown_selected:
                    selected = next(((name, value) for name, value in self.activity_dropdown_options if name == self.activity_dropdown_selected), None)
                    if selected:
                        name, value = selected
                        if value is not None:
                            dropdown_text = f"{name} ({value:.2f})"
                        else:
                            dropdown_text = name
                text_surface = self.subtitle_font.render(dropdown_text, True, self.BLACK)
                text_rect = text_surface.get_rect(midleft=(self.activity_dropdown_rect.left + 5, self.activity_dropdown_rect.centery))
                self.screen.blit(text_surface, text_rect)

                # Draw metallicity dropdown
                metallicity_label = self.subtitle_font.render("Metallicity [Fe/H]", True, self.BLACK)
                metallicity_label_rect = metallicity_label.get_rect(midleft=(self.width - self.customization_panel_width + 50, 525))
                self.screen.blit(metallicity_label, metallicity_label_rect)
                pygame.draw.rect(self.screen, self.WHITE, self.metallicity_dropdown_rect, 2)
                pygame.draw.rect(self.screen, self.BLUE if self.metallicity_dropdown_active else self.GRAY, 
                               self.metallicity_dropdown_rect, 1)
                dropdown_text = "Select Metallicity"
                if self.metallicity_dropdown_selected:
                    selected = next(((name, value) for name, value in self.metallicity_dropdown_options if name == self.metallicity_dropdown_selected), None)
                    if selected:
                        name, value = selected
                        if value is not None:
                            dropdown_text = f"{name} ({value:+.2f} [Fe/H])"
                        else:
                            dropdown_text = name
                text_surface = self.subtitle_font.render(dropdown_text, True, self.BLACK)
                text_rect = text_surface.get_rect(midleft=(self.metallicity_dropdown_rect.left + 5, self.metallicity_dropdown_rect.centery))
                self.screen.blit(text_surface, text_rect)

                # Draw spacetime grid
                self.draw_spacetime_grid()
        
        # Draw spacetime grid in the space area
        self.draw_spacetime_grid()
        
        # Draw orbit lines and bodies
        for body in self.placed_bodies:
            # Draw orbit line
            if body["type"] != "star" and body["name"] in self.orbit_points:
                points = self.orbit_points[body["name"]]
                if len(points) > 1:
                    pygame.draw.lines(self.screen, self.GRAY, False, points, 1)
            
            # Draw body
            if body["type"] == "star":
                # Stars don't rotate
                color = self.YELLOW
                pygame.draw.circle(self.screen, color, body["position"].astype(int), body["radius"])
            else:
                # Planets and moons rotate
                color = self.BLUE if body["type"] == "planet" else self.WHITE
                
                # Draw the rotating body
                self.draw_rotating_body(body, color)
            
            # Highlight selected body
            if self.selected_body and body["name"] == self.selected_body["name"]:
                # Draw a circle around the selected body
                pygame.draw.circle(self.screen, self.RED, body["position"].astype(int), body["radius"] + 5, 2)
        
        # Draw instructions if a tab is active
        if self.active_tab:
            instruction_text = self.subtitle_font.render(f"Click in the space to place a {self.active_tab}", True, self.WHITE)
            instruction_rect = instruction_text.get_rect(center=(self.width//2, self.tab_height + self.tab_margin))
            self.screen.blit(instruction_text, instruction_rect)
        else:
            instruction_text = self.subtitle_font.render("Select a tab to place celestial bodies", True, self.WHITE)
            instruction_rect = instruction_text.get_rect(center=(self.width//2, self.tab_height + self.tab_margin))
            self.screen.blit(instruction_text, instruction_rect)
        
        # Draw instruction to add at least one star and one planet
        if len(self.placed_bodies) > 0:
            stars = [b for b in self.placed_bodies if b["type"] == "star"]
            planets = [b for b in self.placed_bodies if b["type"] == "planet"]
            
            if len(stars) == 0:
                instruction = "Add at least one star to start simulation"
            elif len(planets) == 0:
                instruction = "Add at least one planet to start simulation"
            else:
                instruction = "Simulation will start automatically"
                
            instruction_text = self.subtitle_font.render(instruction, True, self.WHITE)
            instruction_rect = instruction_text.get_rect(center=(self.width//2, self.height - 30))
            self.screen.blit(instruction_text, instruction_rect)
        
        # Render dropdown menu last, so it appears on top of everything
        self.render_dropdown()
        
        pygame.display.flip()

    def render(self, engine: SimulationEngine):
        """Main render function"""
        if self.show_home_screen:
            self.render_home_screen()
        elif self.show_simulation_builder:
            self.render_simulation_builder()
        elif self.show_simulation:
            self.render_simulation(engine)
        
        pygame.display.flip() 

    def _format_value(self, value, unit, allow_scientific=True):
        if value is None:
            return "Custom"
        # For orbital period in days, never use scientific notation
        if unit == 'days':
            if abs(value) < 1:
                return f"{value:.4f} {unit}"
            elif abs(value) < 10:
                return f"{value:.3f} {unit}"
            elif abs(value) < 100:
                return f"{value:.2f} {unit}"
            else:
                return f"{value:.0f} {unit}"
        # Default adaptive formatting
        if allow_scientific and (abs(value) < 0.01 or abs(value) > 9999):
            return f"{value:.2e} {unit}"
        elif abs(value) < 1:
            return f"{value:.4f} {unit}"
        elif abs(value) < 10:
            return f"{value:.3f} {unit}"
        elif abs(value) < 100:
            return f"{value:.2f} {unit}"
        else:
            return f"{value:.1f} {unit}"

    # --- In all dropdown rendering code blocks (moons, planets, stars) ---
    # Replace formatting like f"{name} ({value:.3f} M🌕)" with:
    # f"{name} (" + self._format_value(value, 'M🌕') + ")"
    # And similarly for all other units (Earth masses, R🜨, km, AU, L☉, etc.)
    # For example:
    # dropdown_text = f"{name} (" + self._format_value(value, 'M🌕') + ")"
    # or
    # dropdown_text = f"{name} (" + self._format_value(value, 'Earth masses') + ")"
    # etc.