import pygame
import numpy as np
from typing import List, Tuple, Any, Dict, Optional
# Assuming simulation_engine provides these - adjust if names differ
# from simulation_engine import CelestialBody, SimulationEngine

# Dummy classes if simulation_engine isn't available
class CelestialBody: pass
class SimulationEngine:
    def __init__(self): self.bodies = []


# <<< NEW Dropdown Class >>>
class Dropdown:
    def __init__(self, rect: pygame.Rect, options: List[Tuple[str, Any, Optional[Any]]], font: pygame.font.Font,
                 default_option_name: str = "",
                 colors: Dict[str, Tuple[int, int, int]] = None,
                 padding: int = 5, option_height: int = 30):
        self.rect = rect
        self.options = options # List of (display_name, value, optional_third_param)
        self.font = font
        self.selected_option_name = default_option_name
        self.is_active = False # Hover state for the main box
        self.is_visible = False # Expanded state
        self.padding = padding
        self.option_height = option_height

        # Default colors if none provided
        self.colors = colors if colors else {
            "background": (255, 255, 255),
            "border": (200, 200, 200),
            "hover_bg": (240, 240, 240),
            "text": (0, 0, 0),
            "active_border": (0, 0, 255), # Blue border when active/hovered
            "inactive_border": (128, 128, 128) # Gray border otherwise
        }
        self.border_width = 1

        self._option_surfaces: List[pygame.Surface] = []
        self._option_rects: List[pygame.Rect] = []
        self._rendered_options_surface: Optional[pygame.Surface] = None
        self._create_option_surfaces() # Pre-render options

    def _create_option_surfaces(self):
        """Pre-renders the text for each option."""
        self._option_surfaces = []
        max_width = self.rect.width # Start with the base width
        total_height = len(self.options) * self.option_height

        temp_surface = pygame.Surface((1,1)) # Dummy surface for text measurement
        option_texts = [self._format_option_display_text(opt) for opt in self.options]

        # Find max width needed for text
        for text in option_texts:
             text_width = self.font.render(text, True, (0,0,0)).get_width() + 2 * self.padding
             max_width = max(max_width, text_width)

        # Update rect width if options require more space
        # self.rect.width = max_width # Optional: Resize main rect based on options

        self._rendered_options_surface = pygame.Surface((max_width, total_height), pygame.SRCALPHA)
        self._rendered_options_surface.fill(self.colors["background"])
        self._option_rects = []

        for i, text in enumerate(option_texts):
            option_rect = pygame.Rect(0, i * self.option_height, max_width, self.option_height)
            self._option_rects.append(option_rect)

            # Draw background and border for the option item
            pygame.draw.rect(self._rendered_options_surface, self.colors["background"], option_rect)
            pygame.draw.rect(self._rendered_options_surface, self.colors["border"], option_rect, self.border_width)

            # Render text
            text_surf = self.font.render(text, True, self.colors["text"])
            text_rect = text_surf.get_rect(midleft=(self.padding, option_rect.centery))
            self._rendered_options_surface.blit(text_surf, text_rect)

    def _format_option_display_text(self, option_data: Tuple[str, Any, Optional[Any]]) -> str:
        """ Formats the text to be displayed in the dropdown list for an option. """
        name, value, third_param = option_data if len(option_data) >= 3 else (*option_data, None)

        # Basic formatting - can be customized further if needed based on dropdown type
        if value is None:
            return f"{name} (Custom)"

        # Crude type checking for display
        if isinstance(value, float):
             # Simple float formatting, adjust precision as needed
            if abs(value) < 0.01 and value != 0:
                 return f"{name} ({value:.2e})" # Scientific for very small
            elif abs(value) >= 10000:
                 return f"{name} ({value:.2e})" # Scientific for very large
            else:
                 return f"{name} ({value:.2f})" # Standard decimal
        elif isinstance(value, int):
            return f"{name} ({value})"
        else: # Assume string or other representation is fine
             return f"{name} ({value})" # Fallback

    def get_selected_value(self) -> Any:
        """ Returns the actual value of the selected option, not the display name. """
        for name, value, *_ in self.options:
            if name == self.selected_option_name:
                return value
        return None # Or a default value if appropriate

    def handle_event(self, event: pygame.event.Event, current_mouse_pos: Tuple[int, int]) -> Tuple[bool, Optional[str]]:
        """
        Handles mouse events for the dropdown.
        Returns a tuple: (event_was_handled, selected_option_name_if_changed).
        """
        event_handled = False
        selected_option_change = None

        mouse_x, mouse_y = current_mouse_pos

        # Check hover state for the main box
        self.is_active = self.rect.collidepoint(mouse_x, mouse_y)

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(mouse_x, mouse_y):
                self.is_visible = not self.is_visible # Toggle visibility
                event_handled = True
            elif self.is_visible:
                # Check if click is on an option
                expanded_rect = pygame.Rect(self.rect.left, self.rect.bottom,
                                            self._rendered_options_surface.get_width(),
                                            self._rendered_options_surface.get_height())
                if expanded_rect.collidepoint(mouse_x, mouse_y):
                    relative_y = mouse_y - self.rect.bottom
                    option_index = relative_y // self.option_height
                    if 0 <= option_index < len(self.options):
                        self.selected_option_name = self.options[option_index][0]
                        selected_option_change = self.selected_option_name # Signal change
                        self.is_visible = False # Close after selection
                        event_handled = True
                else:
                     # Clicked outside expanded list, close it
                     self.is_visible = False
                     # Do not set event_handled = True here, allows clicks outside to be processed elsewhere

        return event_handled, selected_option_change


    def draw(self, surface: pygame.Surface):
        """Draws the dropdown on the given surface."""
        # Draw the main collapsed box
        border_color = self.colors["active_border"] if self.is_active else self.colors["inactive_border"]
        pygame.draw.rect(surface, self.colors["background"], self.rect)
        pygame.draw.rect(surface, border_color, self.rect, self.border_width)

        # Find the full data for the selected option to format display text correctly
        selected_option_data = next((opt for opt in self.options if opt[0] == self.selected_option_name), None)
        display_text = self.selected_option_name # Fallback
        if selected_option_data:
             display_text = self._format_option_display_text(selected_option_data)

        text_surf = self.font.render(display_text, True, self.colors["text"])
        text_rect = text_surf.get_rect(midleft=(self.rect.left + self.padding, self.rect.centery))
        surface.blit(text_surf, text_rect)

        # Draw dropdown arrow indicator (simple triangle)
        arrow_points = [
            (self.rect.right - 15, self.rect.centery - 3),
            (self.rect.right - 5, self.rect.centery - 3),
            (self.rect.right - 10, self.rect.centery + 3)
        ]
        pygame.draw.polygon(surface, self.colors["text"], arrow_points)


        # Draw the expanded options list if visible
        if self.is_visible and self._rendered_options_surface:
            options_pos = (self.rect.left, self.rect.bottom)
            surface.blit(self._rendered_options_surface, options_pos)

            # Optional: Highlight hovered option
            mouse_x, mouse_y = pygame.mouse.get_pos()
            expanded_rect = pygame.Rect(self.rect.left, self.rect.bottom,
                                            self._rendered_options_surface.get_width(),
                                            self._rendered_options_surface.get_height())
            if expanded_rect.collidepoint(mouse_x, mouse_y):
                 relative_y = mouse_y - self.rect.bottom
                 option_index = relative_y // self.option_height
                 if 0 <= option_index < len(self._option_rects):
                      hover_rect = self._option_rects[option_index].copy()
                      hover_rect.topleft = (options_pos[0] + hover_rect.left, options_pos[1] + hover_rect.top)
                      # Draw a semi-transparent hover highlight
                      hover_surf = pygame.Surface(hover_rect.size, pygame.SRCALPHA)
                      hover_surf.fill((*self.colors["hover_bg"][:3], 100)) # Use alpha
                      surface.blit(hover_surf, hover_rect.topleft)

# <<< END Dropdown Class >>>


class SolarSystemVisualizer:
    def __init__(self, width: int = 1200, height: int = 800):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("AIET - Solar System Simulator - Refactored")
        self.clock = pygame.time.Clock()
        self.scale = 100
        self.center = np.array([width/2, height/2])

        # Fonts
        self.font = pygame.font.Font(None, 36)
        self.title_font = pygame.font.Font(None, 72)
        self.subtitle_font = pygame.font.Font(None, 24)
        self.tab_font = pygame.font.Font(None, 20)
        self.button_font = pygame.font.Font(None, 36)
        self.tooltip_font = pygame.font.Font(None, 18) # Keep for potential future use

        # Screen states
        self.show_home_screen = True
        self.show_simulation_builder = False
        self.show_simulation = False
        self.show_customization_panel = False

        # Colors (Removed Preview Colors)
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
        self.ACTIVE_TAB_COLOR = (255, 100, 100)
        self.TOOLTIP_COLOR = self.LIGHT_GRAY

        # Home screen state
        self.ambient_colors = [(119, 236, 57)]
        self.current_color_index = 0
        self.color_change_counter = 0
        self.color_change_speed = 30

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

        # Simulation data
        self.placed_bodies = []
        self.body_counter = {"moon": 0, "planet": 0, "star": 0}
        self.G = 0.5
        self.time_step = 0.05
        self.orbit_points = {}
        self.orbit_history = {}
        self.orbit_grid_points = {}
        self.grid_size = 50
        self.rotation_speed = 0.1

        # Customization Panel state
        self.selected_body = None
        self.customization_panel_width = 400
        self.customization_panel = pygame.Rect(self.width - self.customization_panel_width, 0,
                                             self.customization_panel_width, self.height)
        self.close_button_size = 20
        self.close_button = pygame.Rect(self.width - self.close_button_size - 10, 10,
                                      self.close_button_size, self.close_button_size)

        # --- Refactored Dropdown Instantiation ---
        self.dropdowns: Dict[str, Dropdown] = {}
        dropdown_w = self.customization_panel_width - 100
        dropdown_h = 30
        dropdown_x = self.width - self.customization_panel_width + 50

        # Define options directly here or load from a config
        planet_mass_options = [ ("Mercury", 0.055), ("Venus", 0.815), ("Earth", 1.0), ("Mars", 0.107), ("Jupiter", 317.8), ("Saturn", 95.2), ("Uranus", 14.5), ("Neptune", 17.1), ("Custom", None)]
        planet_age_options = [("0.1 Gyr", 0.1), ("1.0 Gyr", 1.0), ("4.6 Gyr (Earth's age)", 4.6), ("6.0 Gyr", 6.0), ("Custom", None)]
        planet_radius_options = [("Mercury", 0.38), ("Venus", 0.95), ("Earth", 1.0), ("Mars", 0.53), ("Jupiter", 11.2), ("Saturn", 9.5), ("Uranus", 4.0), ("Neptune", 3.9), ("Custom", None)]
        planet_temp_options = [("Mercury", 440), ("Venus", 737), ("Earth", 288), ("Mars", 210), ("Jupiter", 165), ("Saturn", 134), ("Uranus", 76), ("Neptune", 72), ("Custom", None)]
        planet_gravity_options = [("Mercury", 3.7), ("Mars", 3.7), ("Venus", 8.87), ("Earth", 9.81), ("Uranus", 8.69), ("Neptune", 11.15), ("Saturn", 10.44), ("Jupiter", 24.79), ("Custom", None)]
        planet_orbit_dist_options = [("Mercury", 0.39), ("Venus", 0.72), ("Earth", 1.0), ("Mars", 1.52), ("Jupiter", 5.2), ("Saturn", 9.58), ("Uranus", 19.2), ("Neptune", 30.1), ("Custom", None)]
        planet_ecc_options = [("Circular Orbit", 0.0), ("Earth", 0.0167), ("Mars", 0.093), ("Mercury", 0.205), ("Pluto", 0.248), ("Custom", None)]
        planet_period_options = [("Mercury", 88), ("Venus", 225), ("Earth", 365.25), ("Mars", 687), ("Jupiter", 4333), ("Saturn", 10759), ("Uranus", 30687), ("Neptune", 60190), ("Custom", None)]
        planet_flux_options = [("Mercury", 6.67), ("Venus", 1.91), ("Earth", 1.0), ("Mars", 0.43), ("Jupiter", 0.037), ("Saturn", 0.011), ("Uranus", 0.0029), ("Neptune", 0.0015), ("Custom", None)]
        planet_density_options = [("Saturn", 0.69), ("Jupiter", 1.33), ("Neptune", 1.64), ("Uranus", 1.27), ("Mars", 3.93), ("Mercury", 5.43), ("Venus", 5.24), ("Earth", 5.51), ("Custom", None)]

        moon_mass_options = [ ("Deimos", 1.48e15, "kg"), ("Phobos", 1.07e16, "kg"), ("Europa", 0.0073, "M☾"), ("Enceladus", 0.0001, "M☾"), ("Titan", 0.0135, "M☾"), ("Ganymede", 0.0148, "M☾"), ("Callisto", 0.0107, "M☾"), ("Moon", 1.0, "M☾"), ("Custom", None, None)]
        moon_age_options = [("Solar System Moons (~4.6 Gyr)", 4.6), ("Custom", None)]
        moon_radius_options = [("Deimos", 6.2), ("Phobos", 11.3), ("Europa", 1560.8), ("Enceladus", 252.1), ("Titan", 2574.7), ("Ganymede", 2634.1), ("Callisto", 2410.3), ("Moon", 1737.4), ("Custom", None)]
        moon_orbit_dist_options = [("Phobos", 9378), ("Deimos", 23460), ("Europa", 670900), ("Enceladus", 237950), ("Titan", 1221870), ("Ganymede", 1070400), ("Callisto", 1882700), ("Moon", 384400), ("Custom", None)]
        moon_period_options = [("Phobos", 0.32), ("Deimos", 1.26), ("Europa", 3.55), ("Enceladus", 1.37), ("Titan", 15.95), ("Ganymede", 7.15), ("Callisto", 16.69), ("Moon", 27.3), ("Custom", None)]
        moon_temp_options = [("Europa", 102), ("Enceladus", 75), ("Ganymede", 110), ("Callisto", 134), ("Titan", 94), ("Moon", 220), ("Custom", None)]
        moon_gravity_options = [("Deimos", 0.003), ("Phobos", 0.0057), ("Enceladus", 0.113), ("Europa", 1.314), ("Titan", 1.352), ("Ganymede", 1.428), ("Callisto", 1.235), ("Moon", 1.62), ("Custom", None)]

        star_mass_options = [("0.08 M☉ (H-limit)", 0.08), ("0.5 M☉", 0.5), ("1.0 M☉ (Sun)", 1.0), ("1.5 M☉", 1.5), ("3.0 M☉", 3.0), ("5.0 M☉", 5.0), ("10.0 M☉", 10.0), ("20.0 M☉", 20.0), ("50.0 M☉", 50.0), ("100.0 M☉", 100.0), ("Custom", None)]
        star_age_options = [("1.0 Gyr", 1.0), ("Sun (4.6 Gyr)", 4.6), ("7.0 Gyr", 7.0), ("Custom", None)]
        star_spectral_options = [ ("O-type (~40K K)", 40000, (0, 0, 255)), ("B-type (~20K K)", 20000, (173, 216, 230)), ("A-type (~10K K)", 10000, (255, 255, 255)), ("F-type (~7.5K K)", 7500, (255, 255, 224)), ("G-type (~5.8K K)", 5800, (255, 255, 0)), ("K-type (~4.5K K)", 4500, (255, 165, 0)), ("M-type (~3K K)", 3000, (255, 0, 0)), ("Custom", None, None)]
        star_luminosity_options = [("Red Dwarf", 0.04), ("Orange Dwarf", 0.15), ("G-type (Sun)", 1.00), ("F-type", 2.00), ("A-type", 25.00), ("B-type Giant", 10000.00), ("O-type Supergiant", 100000.00), ("Custom", None)]
        star_radius_options = [("O-type", 10.0), ("B-type", 5.0), ("A-type", 2.0), ("F-type", 1.3), ("G-type (Sun)", 1.0), ("K-type", 0.8), ("M-type", 0.3), ("Custom", None)]
        star_activity_options = [("Low", 0.25), ("Moderate (Sun)", 0.5), ("High", 0.75), ("Very High", 1.0), ("Custom", None)]
        star_metallicity_options = [("-0.5 (Metal-poor)", -0.5), ("0.0 (Sun)", 0.0), ("+0.3 (Metal-rich)", 0.3), ("Custom", None)]

        # Group dropdowns by type for easier conditional rendering/handling
        self.dropdown_config = {
            'common': ['age'],
            'star': ['star_mass', 'star_age', 'spectral_class', 'luminosity', 'radius', 'activity', 'metallicity'],
            'planet': ['planet_mass', 'planet_age', 'planet_radius', 'temperature', 'gravity', 'orbit_distance', 'eccentricity', 'period', 'flux', 'density'],
            'moon': ['moon_mass', 'moon_age', 'moon_radius', 'moon_orbit_distance', 'moon_period', 'moon_temperature', 'moon_gravity']
        }

        # Create all dropdown instances (rectangles will be positioned in render loop)
        rect_stub = pygame.Rect(dropdown_x, 0, dropdown_w, dropdown_h) # Y will be set in render

        # Star
        self.dropdowns['star_mass'] = Dropdown(rect_stub.copy(), star_mass_options, self.subtitle_font, default_option_name="1.0 M☉ (Sun)")
        self.dropdowns['star_age'] = Dropdown(rect_stub.copy(), star_age_options, self.subtitle_font, default_option_name="Sun (4.6 Gyr)")
        self.dropdowns['spectral_class'] = Dropdown(rect_stub.copy(), star_spectral_options, self.subtitle_font, default_option_name="G-type (~5.8K K)")
        self.dropdowns['luminosity'] = Dropdown(rect_stub.copy(), star_luminosity_options, self.subtitle_font, default_option_name="G-type (Sun)")
        self.dropdowns['radius'] = Dropdown(rect_stub.copy(), star_radius_options, self.subtitle_font, default_option_name="G-type (Sun)")
        self.dropdowns['activity'] = Dropdown(rect_stub.copy(), star_activity_options, self.subtitle_font, default_option_name="Moderate (Sun)")
        self.dropdowns['metallicity'] = Dropdown(rect_stub.copy(), star_metallicity_options, self.subtitle_font, default_option_name="0.0 (Sun)")

        # Planet
        self.dropdowns['planet_mass'] = Dropdown(rect_stub.copy(), planet_mass_options, self.subtitle_font, default_option_name="Earth")
        self.dropdowns['planet_age'] = Dropdown(rect_stub.copy(), planet_age_options, self.subtitle_font, default_option_name="4.6 Gyr (Earth's age)")
        self.dropdowns['planet_radius'] = Dropdown(rect_stub.copy(), planet_radius_options, self.subtitle_font, default_option_name="Earth")
        self.dropdowns['temperature'] = Dropdown(rect_stub.copy(), planet_temp_options, self.subtitle_font, default_option_name="Earth")
        self.dropdowns['gravity'] = Dropdown(rect_stub.copy(), planet_gravity_options, self.subtitle_font, default_option_name="Earth")
        self.dropdowns['orbit_distance'] = Dropdown(rect_stub.copy(), planet_orbit_dist_options, self.subtitle_font, default_option_name="Earth")
        self.dropdowns['eccentricity'] = Dropdown(rect_stub.copy(), planet_ecc_options, self.subtitle_font, default_option_name="Earth")
        self.dropdowns['period'] = Dropdown(rect_stub.copy(), planet_period_options, self.subtitle_font, default_option_name="Earth")
        self.dropdowns['flux'] = Dropdown(rect_stub.copy(), planet_flux_options, self.subtitle_font, default_option_name="Earth")
        self.dropdowns['density'] = Dropdown(rect_stub.copy(), planet_density_options, self.subtitle_font, default_option_name="Earth")

        # Moon
        self.dropdowns['moon_mass'] = Dropdown(rect_stub.copy(), moon_mass_options, self.subtitle_font, default_option_name="Moon")
        self.dropdowns['moon_age'] = Dropdown(rect_stub.copy(), moon_age_options, self.subtitle_font, default_option_name="Solar System Moons (~4.6 Gyr)")
        self.dropdowns['moon_radius'] = Dropdown(rect_stub.copy(), moon_radius_options, self.subtitle_font, default_option_name="Moon")
        self.dropdowns['moon_orbit_distance'] = Dropdown(rect_stub.copy(), moon_orbit_dist_options, self.subtitle_font, default_option_name="Moon")
        self.dropdowns['moon_period'] = Dropdown(rect_stub.copy(), moon_period_options, self.subtitle_font, default_option_name="Moon")
        self.dropdowns['moon_temperature'] = Dropdown(rect_stub.copy(), moon_temp_options, self.subtitle_font, default_option_name="Moon")
        self.dropdowns['moon_gravity'] = Dropdown(rect_stub.copy(), moon_gravity_options, self.subtitle_font, default_option_name="Moon")


        # Input Field States (Keep for now, could be abstracted later)
        # We need to map dropdown "Custom" selections to activating these
        self.custom_input_map: Dict[str, str] = { # Maps dropdown name to input field state bool
             "planet_mass": "show_custom_mass_input",
             "planet_age": "show_custom_age_input",
             # ... Add mappings for ALL dropdowns that have a custom option
        }
        self.active_input_field: Optional[str] = None # Tracks which text input is active
        self.input_field_text: str = ""
        # Need rects for input fields if they are separate from dropdowns
        # Example:
        self.generic_input_rect = pygame.Rect(dropdown_x, 0, dropdown_w, dropdown_h) # Position dynamically
        # ... Define rects for all custom inputs as needed ...


    def _should_display_widget(self, widget_name: str, body_type: Optional[str]) -> bool:
        """ Checks if a dropdown/widget should be displayed for the given body type. """
        if not body_type:
            return False
        if widget_name in self.dropdown_config.get('common', []):
            return True
        if body_type == 'star' and widget_name in self.dropdown_config.get('star', []):
            return True
        if body_type == 'planet' and widget_name in self.dropdown_config.get('planet', []):
            return True
        if body_type == 'moon' and widget_name in self.dropdown_config.get('moon', []):
            return True
        # Add checks for input fields if they are named separately
        return False

    def _get_widget_label(self, widget_name: str) -> str:
        """ Returns the display label for a widget. """
        # Simple mapping, expand as needed
        labels = {
            'star_mass': "Mass (M☉)", 'planet_mass': "Mass (M⊕)", 'moon_mass': "Mass (M☾ / kg)",
            'star_age': "Age (Gyr)", 'planet_age': "Age (Gyr)", 'moon_age': "Age (Gyr)",
            'spectral_class': "Spectral Class (Temp)", 'luminosity': "Luminosity (L☉)",
            'radius': "Radius (R☉)", 'planet_radius': "Radius (R🜨)", 'moon_radius': "Radius (km)",
            'activity': "Activity Level", 'metallicity': "Metallicity [Fe/H]",
            'temperature': "Temperature (K)", 'moon_temperature': "Surface Temp (K)",
            'gravity': "Surface Gravity (m/s²)", 'moon_gravity': "Surface Gravity (m/s²)",
            'orbit_distance': "Orbit Distance (AU)", 'moon_orbit_distance': "Orbit Distance (km)",
            'eccentricity': "Eccentricity", 'period': "Period (days)", 'moon_period': "Period (days)",
            'flux': "Stellar Flux (EFU)", 'density': "Density (g/cm³)"
            # Add more labels...
        }
        return labels.get(widget_name, widget_name.replace('_', ' ').title()) # Default label

    def _update_body_from_dropdown(self, dropdown_name: str, selected_option_name: str):
        """ Updates the selected_body based on a dropdown selection. """
        if not self.selected_body: return

        dropdown = self.dropdowns[dropdown_name]
        value = dropdown.get_selected_value()

        if selected_option_name == "Custom":
             # Activate the corresponding custom input field
             # This part needs refinement based on how custom inputs are handled
             print(f"Custom selected for {dropdown_name}")
             # Example: if dropdown_name in self.custom_input_map:
             #     input_field_state_attr = self.custom_input_map[dropdown_name]
             #     setattr(self, input_field_state_attr, True)
             #     self.active_input_field = dropdown_name # Track which input is active
             #     # Initialize input text based on current body value
             pass # Needs implementation for custom inputs
        elif value is not None:
             # Map dropdown name to body attribute name
             attribute_map = {
                 'star_mass': 'mass', 'planet_mass': 'mass', 'moon_mass': 'mass', # Note potential unit conversion needed
                 'star_age': 'age', 'planet_age': 'age', 'moon_age': 'age',
                 'spectral_class': 'temperature', # Also need to handle color potentially
                 'luminosity': 'luminosity',
                 'radius': 'radius', 'planet_radius': 'radius', 'moon_radius': 'actual_radius', # Map to actual for moon
                 'activity': 'activity', 'metallicity': 'metallicity',
                 'temperature': 'temperature', 'moon_temperature': 'temperature',
                 'gravity': 'gravity', 'moon_gravity': 'surfaceGravity',
                 'orbit_distance': 'semiMajorAxis', 'moon_orbit_distance': 'orbit_radius', # Map to actual for moon
                 'eccentricity': 'eccentricity',
                 'period': 'orbital_period', 'moon_period': 'orbital_period',
                 'flux': 'stellarFlux', 'density': 'density'
             }
             attr_name = attribute_map.get(dropdown_name)
             if attr_name:
                  # Special handling for mass units
                 if attr_name == 'mass':
                     if dropdown_name == 'star_mass':
                         self.selected_body[attr_name] = value * 1000.0 # Convert M☉ to M⊕
                     elif dropdown_name == 'moon_mass':
                          # Find the original option to check for kg unit
                          option_data = next((opt for opt in dropdown.options if opt[0] == selected_option_name), None)
                          if option_data and len(option_data) == 3 and option_data[2] == 'kg':
                               self.selected_body[attr_name] = value / 7.35e22 # Convert kg to M☾
                          else:
                               self.selected_body[attr_name] = value # Assume M☾
                     else: # Planet mass
                          self.selected_body[attr_name] = value # Already M⊕
                 elif dropdown_name == 'moon_radius':
                      # Update visual radius based on actual radius
                      self.selected_body[attr_name] = value # Store actual km
                      # Scale visual radius (example scaling)
                      self.selected_body["radius"] = max(5, min(20, value / 100))
                 elif dropdown_name == 'moon_orbit_distance':
                     self.selected_body[attr_name] = value # Store actual km
                     # Scale visual orbit radius (example scaling)
                     self.selected_body["orbit_radius_visual"] = max(50, min(200, value / 1000)) # Need a separate visual attribute
                     # TODO: Need to update generate_orbit_grid to use orbit_radius_visual if present
                 else:
                      self.selected_body[attr_name] = value

                 print(f"Updated {attr_name} to {value} from {dropdown_name}") # Debug

                 # Trigger recalculations if necessary
                 if attr_name in ['mass', 'semiMajorAxis', 'orbit_distance', 'orbit_radius']:
                     if self.selected_body["type"] != "star":
                         self.generate_orbit_grid(self.selected_body)


    def handle_events(self) -> bool:
        """Handle pygame events. Returns False if the window should close."""
        current_mouse_pos = pygame.mouse.get_pos() # Get once per frame

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.show_home_screen:
                    # Check create button
                    button_center_x = self.width//2
                    button_center_y = self.height*3/4
                    button_width = 200; button_height = 50
                    button_rect = pygame.Rect(button_center_x - button_width//2, button_center_y - button_height//2, button_width, button_height)
                    if button_rect.collidepoint(event.pos):
                        self.show_home_screen = False
                        self.show_simulation_builder = True
                elif self.show_simulation_builder or self.show_simulation:
                    # --- Customization Panel Click Logic ---
                    if self.show_customization_panel and self.customization_panel.collidepoint(event.pos):
                        if self.close_button.collidepoint(event.pos):
                            self.show_customization_panel = False
                            self.selected_body = None
                            # Deactivate all dropdowns
                            for dropdown in self.dropdowns.values():
                                dropdown.is_visible = False
                            continue # Skip rest of panel handling

                        # --- Refactored Dropdown Event Handling ---
                        dropdown_handled = False
                        clicked_dropdown_name = None
                        for name, dropdown in self.dropdowns.items():
                             if self._should_display_widget(name, self.selected_body.get('type')):
                                handled, selection_change = dropdown.handle_event(event, current_mouse_pos)
                                if handled:
                                     dropdown_handled = True
                                     if selection_change: # An option was selected
                                         clicked_dropdown_name = name
                                         self._update_body_from_dropdown(name, selection_change)
                                     # Break here assumes only one dropdown can be interacted with per click
                                     break

                        # Close other dropdowns if one was clicked (either toggled or selected from)
                        if dropdown_handled and clicked_dropdown_name is not None:
                             for name, dropdown in self.dropdowns.items():
                                  if name != clicked_dropdown_name:
                                       dropdown.is_visible = False

                        # Handle clicks outside dropdowns but inside panel (e.g., text inputs) only if a dropdown didn't handle it
                        if not dropdown_handled:
                            # --- (Existing text input field activation/click logic here) ---
                            # Example:
                            # if self.generic_input_rect.collidepoint(event.pos) and self.show_custom_...:
                            #    self.active_input_field = 'relevant_field_name'
                            pass # Placeholder for input field logic

                        # Prevent clicks inside panel from triggering body placement/selection
                        continue # End panel click processing

                    # --- End Customization Panel Click Logic ---

                    # Handle tab clicks (outside panel)
                    tab_clicked = False
                    for tab_name, tab_rect in self.tabs.items():
                        if tab_rect.collidepoint(event.pos):
                            self.active_tab = tab_name if self.active_tab != tab_name else None
                            self.selected_body = None # Deselect body when changing/clicking tabs
                            self.show_customization_panel = False
                            tab_clicked = True
                            break
                    if tab_clicked: continue # Skip body placement/selection if tab was clicked

                    # Handle space area clicks (body placement or selection)
                    if self.space_area.collidepoint(event.pos):
                        # Check click on existing body first
                        clicked_body = None
                        for body in self.placed_bodies:
                            body_pos = body["position"].astype(int)
                            body_radius = body.get("radius", 10) # Use get with default
                            # Simple circular collision check
                            if np.linalg.norm(np.array(event.pos) - body_pos) <= body_radius:
                                clicked_body = body
                                break

                        if clicked_body:
                            self.selected_body = clicked_body
                            self.show_customization_panel = True
                            # Update dropdowns to reflect selected body's state (optional)
                            # for name, dropdown in self.dropdowns.items():
                            #     if self._should_display_widget(name, self.selected_body['type']):
                            #          pass # Logic to find matching option based on body's value
                        elif self.active_tab: # Place new body
                            self._place_new_body(event.pos)
                            # Potentially start simulation
                            stars = [b for b in self.placed_bodies if b["type"] == "star"]
                            planets = [b for b in self.placed_bodies if b["type"] == "planet"]
                            if len(stars) > 0 and len(planets) > 0 and self.show_simulation_builder:
                                self.show_simulation_builder = False
                                self.show_simulation = True
                        else: # Clicked empty space with no active tab
                            self.selected_body = None
                            self.show_customization_panel = False

                elif self.show_simulation: # Clicks in simulation mode
                     if self.space_area.collidepoint(event.pos):
                         # Left click returns to builder
                         self.show_simulation = False
                         self.show_simulation_builder = True


            elif event.type == pygame.KEYDOWN:
                 # --- (Existing text input handling logic K_RETURN, K_BACKSPACE, K_ESCAPE, text entry) ---
                 # This needs to be adapted to use self.active_input_field and self.input_field_text
                 # Example:
                 # if self.active_input_field:
                 #    if event.key == pygame.K_RETURN:
                 #        # Parse self.input_field_text, validate, update self.selected_body
                 #        # based on self.active_input_field name
                 #        self.active_input_field = None
                 #        self.input_field_text = ""
                 #    elif event.key == pygame.K_BACKSPACE:
                 #         self.input_field_text = self.input_field_text[:-1]
                 #    # ... etc ...
                 pass # Placeholder for detailed input handling


        return True # Game loop should continue

    def _place_new_body(self, pos: Tuple[int, int]):
        """Creates and adds a new body based on the active tab."""
        if not self.active_tab: return

        self.body_counter[self.active_tab] += 1
        body_type = self.active_tab

        # Default values (could be more sophisticated)
        default_radius_map = {"star": 20, "planet": 15, "moon": 10}
        default_mass_map = {"star": 1000.0, "planet": 1.0, "moon": 1.0} # Star in M⊕, Moon in M☾
        default_age = 4.6
        default_name = f"{body_type.capitalize()}_{self.body_counter[body_type]}"

        body = {
            "type": body_type,
            "position": np.array(pos, dtype=float),
            "velocity": np.array([0.0, 0.0]),
            "radius": default_radius_map.get(body_type, 10),
            "name": default_name,
            "mass": default_mass_map.get(body_type, 1.0),
            "parent": None,
            "orbit_radius": 0.0,
            "orbit_angle": 0.0,
            "orbit_speed": 0.0,
            "rotation_angle": 0.0,
            "rotation_speed": self.rotation_speed * (0.0 if body_type == "star" else 1.0 if body_type == "planet" else 2.0),
            "age": default_age,
            "habit_score": 0.0,
            # Add type-specific defaults as before
            **( {"luminosity": 1.0, "star_temperature": 5800, "star_color": (255, 255, 0)} if body_type == "star" else {} ),
            **( {"gravity": 9.81, "semiMajorAxis": 1.0, "eccentricity": 0.017, "orbital_period": 365.25, "stellarFlux": 1.0, "density": 5.51} if body_type == "planet" else {} ),
            **( {"actual_radius": 1737.4, "orbit_radius_actual": 384400, "temperature": 220, "surfaceGravity": 1.62, "orbital_period": 27.3} if body_type == "moon" else {} )
        }

        # Handle potential name clashes (simple increment)
        all_names = {b['name'] for b in self.placed_bodies}
        while body['name'] in all_names:
             self.body_counter[body_type] += 1
             body['name'] = f"{body_type.capitalize()}_{self.body_counter[body_type]}"


        self.placed_bodies.append(body)
        self.orbit_points[body["name"]] = []
        self.orbit_history[body["name"]] = []

        # Find parent and generate initial orbit AFTER adding to list
        if body_type != "star":
             self.generate_orbit_grid(body) # Find parent and set initial orbit params

        print(f"Placed {body['name']} at {pos}")


    # --- (update_ambient_colors, generate_orbit_grid, update_physics, etc. remain largely unchanged) ---
    # Minor change: generate_orbit_grid might need to use visual orbit radius for moons
    def update_ambient_colors(self):
        """Update the ambient colors for the title"""
        self.color_change_counter += 1
        if self.color_change_counter >= self.color_change_speed:
            self.color_change_counter = 0
            self.current_color_index = (self.current_color_index + 1) % len(self.ambient_colors)

    def generate_orbit_grid(self, body):
        """Generate a circular grid for the orbit path and set initial state."""
        if body["type"] == "star":
            body["parent"] = None # Stars have no parent
            return

        parent = None
        min_dist_sq = float('inf')

        if body["type"] == "planet":
            # Find nearest star
            for potential_parent in self.placed_bodies:
                if potential_parent["type"] == "star":
                    dist_sq = np.sum((potential_parent["position"] - body["position"])**2)
                    if dist_sq < min_dist_sq:
                        min_dist_sq = dist_sq
                        parent = potential_parent
        elif body["type"] == "moon":
            # Find nearest planet
             for potential_parent in self.placed_bodies:
                if potential_parent["type"] == "planet":
                    dist_sq = np.sum((potential_parent["position"] - body["position"])**2)
                    if dist_sq < min_dist_sq:
                        min_dist_sq = dist_sq
                        parent = potential_parent

        if parent:
            body["parent"] = parent["name"]
            # Use visual orbit radius for moons if it exists, otherwise calculate from positions
            orbit_radius_visual = body.get("orbit_radius_visual")
            if orbit_radius_visual is None:
                 orbit_radius_visual = np.linalg.norm(parent["position"] - body["position"])
                 body["orbit_radius_visual"] = orbit_radius_visual # Store calculated visual radius

            body["orbit_radius"] = orbit_radius_visual # Use visual radius for simulation path

            # Generate grid points based on visual radius
            grid_points = []
            for i in range(100):
                angle = i * 2 * np.pi / 100
                x = parent["position"][0] + orbit_radius_visual * np.cos(angle)
                y = parent["position"][1] + orbit_radius_visual * np.sin(angle)
                grid_points.append(np.array([x, y]))
            self.orbit_grid_points[body["name"]] = grid_points

            # Set initial angle based on current position relative to parent
            dx = body["position"][0] - parent["position"][0]
            dy = body["position"][1] - parent["position"][1]
            body["orbit_angle"] = np.arctan2(dy, dx)

            # Calculate orbital speed (based on VISUAL radius for simulation consistency)
            parent_mass = parent.get("mass", 1.0) # Default mass if missing? Risky.
            try:
                 # Avoid division by zero or sqrt of negative
                 if orbit_radius_visual > 1e-6 and parent_mass > 0:
                     speed_factor = 3.0 if body["type"] == "moon" else 1.0
                     body["orbit_speed"] = np.sqrt(self.G * parent_mass / orbit_radius_visual) / orbit_radius_visual * speed_factor
                 else:
                      body["orbit_speed"] = 0.0
            except (ValueError, ZeroDivisionError):
                 body["orbit_speed"] = 0.0 # Default to no speed on error


            # Set initial velocity for circular orbit (using visual radius)
            v = body["orbit_speed"] * orbit_radius_visual
            body["velocity"] = np.array([-v * np.sin(body["orbit_angle"]), v * np.cos(body["orbit_angle"])])
        else:
            # No parent found, clear orbit data
            body["parent"] = None
            body["orbit_radius"] = 0.0
            body["orbit_angle"] = 0.0
            body["orbit_speed"] = 0.0
            body["velocity"] = np.array([0.0, 0.0])
            if body["name"] in self.orbit_grid_points:
                del self.orbit_grid_points[body["name"]]

    def update_physics(self):
        """Update positions based on simple circular orbits around parent."""
        # Ensure parents are assigned first (can happen if bodies are added out of order)
        needs_regen = False
        for body in self.placed_bodies:
             if body['type'] != 'star' and not body.get('parent'):
                 needs_regen = True
                 break
        if needs_regen:
            print("WARN: Regenerating orbits due to missing parent links.")
            for body in self.placed_bodies:
                if body['type'] != 'star':
                     self.generate_orbit_grid(body)


        for body in self.placed_bodies:
            if body["type"] == "star" or not body.get("parent"):
                continue # Stars don't move, bodies without parents don't orbit

            # Find parent body in the current list
            parent = next((b for b in self.placed_bodies if b["name"] == body["parent"]), None)

            if parent:
                # Update orbit angle
                body["orbit_angle"] += body.get("orbit_speed", 0.0) * self.time_step

                # Calculate new position based on orbit angle and VISUAL radius
                orbit_radius_visual = body.get("orbit_radius", 0.0) # Use the radius driving simulation
                body["position"][0] = parent["position"][0] + orbit_radius_visual * np.cos(body["orbit_angle"])
                body["position"][1] = parent["position"][1] + orbit_radius_visual * np.sin(body["orbit_angle"])

                # Update velocity for circular orbit
                v = body.get("orbit_speed", 0.0) * orbit_radius_visual
                body["velocity"] = np.array([-v * np.sin(body["orbit_angle"]), v * np.cos(body["orbit_angle"])])

                # Update rotation angle
                body["rotation_angle"] = (body.get("rotation_angle", 0.0) + body.get("rotation_speed", 0.0) * self.time_step) % (2 * np.pi)

                # Store orbit points for trail
                if body["name"] in self.orbit_points:
                    self.orbit_points[body["name"]].append(body["position"].copy())
                    if len(self.orbit_points[body["name"]]) > 100:
                        self.orbit_points[body["name"]].pop(0)

                # Update orbit grid points if it's a moon (needs to follow planet)
                if body["type"] == "moon" and body["name"] in self.orbit_grid_points:
                    self.update_moon_orbit_grid(body, parent)


    def update_moon_orbit_grid(self, moon, planet):
        """Update the moon's orbit grid points to follow its parent planet."""
        orbit_radius_visual = moon.get("orbit_radius", 0.0) # Use visual radius
        if orbit_radius_visual <= 0: return # Don't draw if radius is invalid

        grid_points = []
        for i in range(100):
            angle = i * 2 * np.pi / 100
            x = planet["position"][0] + orbit_radius_visual * np.cos(angle)
            y = planet["position"][1] + orbit_radius_visual * np.sin(angle)
            grid_points.append(np.array([x, y]))

        self.orbit_grid_points[moon["name"]] = grid_points

    def draw_spacetime_grid(self):
        """Draw a static spacetime grid in the background"""
        space_top = self.tab_height + 2*self.tab_margin
        right_boundary = self.width - self.customization_panel_width if self.show_customization_panel else self.width
        grid_color_dark = (self.GRID_COLOR[0]//2, self.GRID_COLOR[1]//2, self.GRID_COLOR[2]//2)

        for y in range(int(space_top), self.height, self.grid_size):
            pygame.draw.line(self.screen, self.GRID_COLOR, (0, y), (right_boundary, y), 1)
            if y + self.grid_size//2 < self.height:
                pygame.draw.line(self.screen, grid_color_dark, (0, y + self.grid_size//2), (right_boundary, y + self.grid_size//2), 1)

        for x in range(0, int(right_boundary), self.grid_size):
            pygame.draw.line(self.screen, self.GRID_COLOR, (x, space_top), (x, self.height), 1)
            if x + self.grid_size//2 < right_boundary:
                pygame.draw.line(self.screen, grid_color_dark, (x + self.grid_size//2, space_top), (x + self.grid_size//2, self.height), 1)

    def render_home_screen(self):
        """Render the home screen with the title and create button"""
        self.screen.fill(self.BLACK)
        self.update_ambient_colors()
        title_text = self.title_font.render("AIET", True, self.ambient_colors[self.current_color_index])
        title_rect = title_text.get_rect(center=(self.width//2, self.height//3))
        self.screen.blit(title_text, title_rect)

        instruction_text = self.font.render("Create", True, self.ambient_colors[self.current_color_index])
        instruction_rect = instruction_text.get_rect(center=(self.width//2, self.height*3/4))
        button_width = instruction_text.get_width() + 40
        button_height = instruction_text.get_height() + 20
        button_rect = pygame.Rect( instruction_rect.centerx - button_width//2, instruction_rect.centery - button_height//2, button_width, button_height)

        pygame.draw.rect(self.screen, (50, 50, 50), button_rect, border_radius=10)
        pygame.draw.rect(self.screen, self.ambient_colors[self.current_color_index], button_rect, 2, border_radius=10)
        self.screen.blit(instruction_text, instruction_rect)
        pygame.display.flip()

    def format_age_display(self, age: float) -> str:
        """Format age for display, converting to Myr if less than 0.5 Gyr"""
        if age < 0.5:
            myr = age * 1000
            return f"{myr:.1f} Myr"
        return f"{age:.1f} Gyr"


    def render_simulation(self, engine: SimulationEngine):
        """Render the solar system simulation (live mode)."""
        # NOTE: This method now shares structure with render_simulation_builder
        # Could potentially be merged further, but kept separate for clarity between build/run modes
        self.screen.fill(self.DARK_BLUE)
        self.draw_spacetime_grid()
        self.update_physics() # Update physics in live sim mode

        # Draw orbit grid lines
        for body in self.placed_bodies:
            if body["type"] != "star" and body["name"] in self.orbit_grid_points:
                grid_points = self.orbit_grid_points[body["name"]]
                if len(grid_points) > 1:
                    color = self.LIGHT_GRAY if body["type"] == "planet" else (150, 150, 150)
                    # Ensure points are integers for drawing
                    int_points = [(int(p[0]), int(p[1])) for p in grid_points]
                    pygame.draw.lines(self.screen, color, True, int_points, 1)


        # Draw orbit trails and bodies
        for body in self.placed_bodies:
            # Draw orbit trail
            if body["type"] != "star" and body["name"] in self.orbit_points:
                points = self.orbit_points[body["name"]]
                if len(points) > 1:
                    color = self.GRAY if body["type"] == "planet" else (100, 100, 100)
                    # Ensure points are integers for drawing
                    int_points = [(int(p[0]), int(p[1])) for p in points]
                    pygame.draw.lines(self.screen, color, False, int_points, 1)

            # Draw body
            if body["type"] == "star":
                color = self.YELLOW
                pygame.draw.circle(self.screen, color, body["position"].astype(int), body["radius"])
            else:
                color = self.BLUE if body["type"] == "planet" else self.WHITE
                self.draw_rotating_body(body, color)

            # Highlight selected body (even in simulation mode)
            if self.selected_body and body["name"] == self.selected_body["name"]:
                pygame.draw.circle(self.screen, self.RED, body["position"].astype(int), body["radius"] + 5, 2)

         # Draw Top Bar (Tabs are visible but not interactive for placement)
        top_bar_height = self.tab_height + 2*self.tab_margin
        top_bar_rect = pygame.Rect(0, 0, self.width, top_bar_height)
        pygame.draw.rect(self.screen, self.WHITE, top_bar_rect)
        for tab_name, tab_rect in self.tabs.items():
             # Always draw tabs grayed out/inactive in simulation mode visually
             pygame.draw.rect(self.screen, self.GRAY, tab_rect, border_radius=5)
             tab_text = self.tab_font.render(tab_name.capitalize(), True, self.WHITE)
             tab_text_rect = tab_text.get_rect(center=tab_rect.center)
             self.screen.blit(tab_text, tab_text_rect)

        # Draw instructions for simulation mode
        instruction_text = self.subtitle_font.render("Simulation Running - Click space to return to Builder", True, self.WHITE)
        instruction_rect = instruction_text.get_rect(center=(self.width//2, self.height - 30))
        self.screen.blit(instruction_text, instruction_rect)

        # Draw customization panel if a body is selected (still viewable in sim mode)
        if self.show_customization_panel and self.selected_body:
            self._render_customization_panel() # Use helper

        pygame.display.flip()

    def draw_rotating_body(self, body, color):
        """Draw a celestial body with rotation"""
        radius = body.get("radius", 10) # Use get with default
        position = body.get("position", np.array([0,0])).astype(int)
        rotation_angle = body.get("rotation_angle", 0.0)

        # Basic circle
        pygame.draw.circle(self.screen, color, position, radius)

        # Draw rotation line only if radius is large enough
        if radius > 4:
            end_x = position[0] + radius * 0.8 * np.cos(rotation_angle)
            end_y = position[1] + radius * 0.8 * np.sin(rotation_angle)
            pygame.draw.line(self.screen, self.WHITE, position, (int(end_x), int(end_y)), 2)


    def _render_customization_panel(self):
        """Helper to render the customization panel content."""
        if not self.selected_body: return

        pygame.draw.rect(self.screen, self.WHITE, self.customization_panel)
        pygame.draw.rect(self.screen, self.BLACK, self.close_button, 2)
        pygame.draw.line(self.screen, self.BLACK, (self.close_button.left + 5, self.close_button.top + 5), (self.close_button.right - 5, self.close_button.bottom - 5), 2)
        pygame.draw.line(self.screen, self.BLACK, (self.close_button.left + 5, self.close_button.bottom - 5), (self.close_button.right - 5, self.close_button.top + 5), 2)

        title_text = self.font.render(f"Customize {self.selected_body['type'].capitalize()}", True, self.BLACK)
        title_rect = title_text.get_rect(center=(self.width - self.customization_panel_width // 2, 50))
        self.screen.blit(title_text, title_rect)

        # Draw Habitability if applicable
        if self.selected_body.get('type') != 'star':
            habitability_text = self.font.render(f"Habitability: {self.selected_body.get('habit_score', 0.0):.2f}%", True, (0, 150, 0)) # Darker Green
            habitability_rect = habitability_text.get_rect(center=(self.width - self.customization_panel_width // 2, 80))
            self.screen.blit(habitability_text, habitability_rect)


        # --- Render Dropdowns and Inputs ---
        current_y = 120 # Starting Y position below title/habitability
        widget_spacing = 60 # Space between widgets (label + dropdown/input)

        body_type = self.selected_body.get('type')

        # Determine which widgets to show based on body type
        widgets_to_show = self.dropdown_config.get('common', [])
        if body_type == 'star': widgets_to_show.extend(self.dropdown_config.get('star', []))
        elif body_type == 'planet': widgets_to_show.extend(self.dropdown_config.get('planet', []))
        elif body_type == 'moon': widgets_to_show.extend(self.dropdown_config.get('moon', []))

        # Sort or order widgets if needed (e.g., mass first) - using definition order for now
        widget_order = [ # Example explicit order
            'star_mass','planet_mass','moon_mass',
            'star_age','planet_age','moon_age',
            'spectral_class','luminosity', 'radius','planet_radius','moon_radius',
            'temperature','moon_temperature',
            'gravity','moon_gravity',
            'orbit_distance','moon_orbit_distance',
            'eccentricity',
            'period','moon_period',
            'flux','density',
            'activity','metallicity'
        ]
        ordered_widgets_to_show = [w for w in widget_order if w in widgets_to_show]


        for widget_name in ordered_widgets_to_show:
            if self._should_display_widget(widget_name, body_type):
                # Draw Label
                label_text = self._get_widget_label(widget_name)
                label_surf = self.subtitle_font.render(label_text, True, self.BLACK)
                label_rect = label_surf.get_rect(midleft=(self.width - self.customization_panel_width + 50, current_y - 15))
                self.screen.blit(label_surf, label_rect)

                # Draw Dropdown
                if widget_name in self.dropdowns:
                    dropdown = self.dropdowns[widget_name]
                    dropdown.rect.topleft = (self.width - self.customization_panel_width + 50, current_y)
                    dropdown.draw(self.screen)
                    current_y += widget_spacing

                # --- (Add logic here to draw associated custom input fields if active) ---
                # Example:
                # if widget_name in self.custom_input_map and getattr(self, self.custom_input_map[widget_name], False):
                #     self.generic_input_rect.topleft = (self.width - self.customization_panel_width + 50, current_y)
                #     # Draw the input rect and text (self.input_field_text if self.active_input_field == widget_name else formatted_current_value)
                #     pygame.draw.rect(self.screen, ...)
                #     self.screen.blit(...)
                #     current_y += widget_spacing # Add space for input field

        # Draw expanded dropdown lists last to ensure they are on top
        for dropdown in self.dropdowns.values():
            if dropdown.is_visible:
                 # Check if this dropdown *should* be visible for the current body before drawing expanded
                 # This feels redundant if is_visible is managed correctly, but as a safeguard:
                 # found_key = next((k for k, v in self.dropdowns.items() if v == dropdown), None)
                 # if found_key and self._should_display_widget(found_key, self.selected_body.get('type')):
                 dropdown.draw(self.screen) # Draw again to put list on top


    def render_simulation_builder(self):
        """Render the simulation builder screen."""
        self.screen.fill(self.DARK_BLUE)

        # Draw Top Bar & Tabs
        top_bar_height = self.tab_height + 2*self.tab_margin
        pygame.draw.rect(self.screen, self.WHITE, (0, 0, self.width, top_bar_height))
        for tab_name, tab_rect in self.tabs.items():
            color = self.ACTIVE_TAB_COLOR if tab_name == self.active_tab else self.GRAY
            border = 2 if tab_name == self.active_tab else 0
            pygame.draw.rect(self.screen, color, tab_rect, border_radius=5)
            if border > 0: pygame.draw.rect(self.screen, self.WHITE, tab_rect, border, border_radius=5)
            tab_text = self.tab_font.render(tab_name.capitalize(), True, self.WHITE)
            text_rect = tab_text.get_rect(center=tab_rect.center)
            self.screen.blit(tab_text, text_rect)

        # Draw spacetime grid (before bodies)
        self.draw_spacetime_grid()

        # Draw Placed Bodies (No trails needed in builder)
        for body in self.placed_bodies:
            # Draw orbit grid lines (predicted path)
            if body["type"] != "star" and body["name"] in self.orbit_grid_points:
                grid_points = self.orbit_grid_points[body["name"]]
                if len(grid_points) > 1:
                    color = self.LIGHT_GRAY if body["type"] == "planet" else (150, 150, 150)
                    int_points = [(int(p[0]), int(p[1])) for p in grid_points]
                    pygame.draw.lines(self.screen, color, True, int_points, 1)

             # Draw body
            if body["type"] == "star":
                color = self.YELLOW
                pygame.draw.circle(self.screen, color, body["position"].astype(int), body["radius"])
            else:
                color = self.BLUE if body["type"] == "planet" else self.WHITE
                self.draw_rotating_body(body, color) # Rotation is just visual here

            # Highlight selected body
            if self.selected_body and body["name"] == self.selected_body["name"]:
                pygame.draw.circle(self.screen, self.RED, body["position"].astype(int), body["radius"] + 5, 2)


        # Draw Customization Panel (if active) using helper
        if self.show_customization_panel and self.selected_body:
            self._render_customization_panel()

        # Draw Instructions
        instruction_text = ""
        if not self.active_tab:
            instruction_text = "Select a tab to place celestial bodies"
        else:
            # Check simulation conditions
            stars = [b for b in self.placed_bodies if b["type"] == "star"]
            planets = [b for b in self.placed_bodies if b["type"] == "planet"]
            if len(stars) == 0:
                instruction_text = f"Click to place {self.active_tab}. Add a STAR to enable simulation."
            elif len(planets) == 0 and self.active_tab != 'star': # Encourage placing a planet if a star exists
                 instruction_text = f"Click to place {self.active_tab}. Add a PLANET to enable simulation."
            elif len(planets) == 0 and self.active_tab == 'star':
                 instruction_text = f"Click to place {self.active_tab}. Add a PLANET to enable simulation."
            else: # Sim conditions met or placing another body
                 instruction_text = f"Click to place {self.active_tab}."

        if instruction_text:
            instruction_surf = self.subtitle_font.render(instruction_text, True, self.WHITE)
            # Position below tabs or at bottom depending on context? Let's try bottom.
            instruction_rect = instruction_surf.get_rect(center=(self.width//2, self.height - 30))
            # Draw semi-transparent background for better readability
            bg_rect = instruction_rect.inflate(10, 5)
            bg_surf = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            bg_surf.fill((0,0,0, 150))
            self.screen.blit(bg_surf, bg_rect.topleft)
            self.screen.blit(instruction_surf, instruction_rect)


        pygame.display.flip()


    def render(self, engine: SimulationEngine):
        """Main render function - dispatches to appropriate state renderer."""
        if self.show_home_screen:
            self.render_home_screen()
        elif self.show_simulation_builder:
            self.render_simulation_builder()
        elif self.show_simulation:
            self.render_simulation(engine)
        # Flip is handled within each specific render method now

# <<< Ensure _parse_input_value, _format_value methods are present >>>
# (Copied from previous version)

# --- Example main loop (if needed for testing) ---
if __name__ == '__main__':
    visualizer = SolarSystemVisualizer()
    # Dummy engine for rendering testing
    engine = SimulationEngine()
    running = True
    while running:
        running = visualizer.handle_events()
        # Pass mouse pos potentially needed by renderers now? No, handled internally.
        visualizer.render(engine)
        visualizer.clock.tick(60) # Limit frame rate
    pygame.quit()

