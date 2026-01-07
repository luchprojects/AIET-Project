import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional
try:
    from ml_habitability import MLHabitabilityCalculator
except ImportError:
    MLHabitabilityCalculator = None

@dataclass
class CelestialBody:
    name: str
    mass: float  # in Earth masses
    radius: float  # in Earth radii
    position: np.ndarray  # in AU
    velocity: np.ndarray  # in AU/year
    temperature: float  # in Kelvin
    atmosphere: Dict[str, float]  # composition by mass fraction
    type: str  # 'star', 'planet', etc.
    orbper: float = 365.25  # orbital period in days
    orbeccen: float = 0.0  # orbital eccentricity
    
    def __post_init__(self):
        self.acceleration = np.zeros(3)
        self.habitability_score = 0.0
        # For stellar parameters if it's a star
        self.lum = 1.0  # in Solar luminosities
        if self.type == 'star':
            # Stefan-Boltzmann calculation for luminosity if not provided
            # st_lum = (st_rad^2) * (st_teff / 5778.0)^4
            # radius is in Earth radii, convert to solar radii: 1 solar radius = 109.2 Earth radii
            solar_rad = self.radius / 109.2
            self.lum = (solar_rad**2) * (self.temperature / 5778.0)**4

class SimulationEngine:
    def __init__(self):
        self.bodies: List[CelestialBody] = []
        self.G = 39.478  # Gravitational constant in AU^3/(M_sun * year^2)
        self.time_step = 0.01  # in years
        try:
            self.ml_calculator = MLHabitabilityCalculator() if MLHabitabilityCalculator else None
        except Exception as e:
            print(f"Warning: Could not initialize ML calculator: {e}")
            self.ml_calculator = None
        
    def add_body(self, body: CelestialBody):
        self.bodies.append(body)
        
    def find_host_star(self, body: CelestialBody) -> Optional[CelestialBody]:
        """Find the most massive star that this body might be orbiting"""
        stars = [b for b in self.bodies if b.type == 'star']
        if not stars:
            return None
        # Return the most massive star for now
        return max(stars, key=lambda s: s.mass)

    def calculate_acceleration(self, body: CelestialBody) -> np.ndarray:
        """Calculate gravitational acceleration on a body due to all other bodies"""
        acceleration = np.zeros(3)
        for other in self.bodies:
            if other != body:
                # Convert mass from Earth masses to Solar masses for the calculation
                # if the other body is a star, mass might already be in solar masses?
                # Looking at main.py, sun mass is 1.0 (Solar mass), planets are in Earth masses.
                # This is inconsistent. Let's assume stars are in Solar masses and planets in Earth masses.
                # M_sun = 333,030 M_earth
                other_mass_solar = other.mass if other.type == 'star' else other.mass / 333030.0
                
                r = other.position - body.position
                r_mag = np.linalg.norm(r)
                if r_mag > 1e-5:  # Avoid division by zero and near-collisions
                    acceleration += self.G * other_mass_solar * r / (r_mag ** 3)
        return acceleration
    
    def update_positions(self):
        """Update positions and velocities using Verlet integration"""
        for body in self.bodies:
            body.acceleration = self.calculate_acceleration(body)
            body.velocity += body.acceleration * self.time_step
            body.position += body.velocity * self.time_step
            
    def calculate_habitability(self, body: CelestialBody) -> float:
        """Calculate habitability score using ML model if available, otherwise use basic physics"""
        if body.type != 'planet':
            return 0.0

        star = self.find_host_star(body)
        if not star:
            return 0.0

        # Calculate distance to star for insolation
        distance = np.linalg.norm(body.position - star.position)
        if distance < 1e-5:
            distance = 1.0 # default to 1 AU
            
        # pl_insol = st_lum / (distance^2)
        pl_insol = star.lum / (distance**2)

        if self.ml_calculator:
            features = {
                "pl_rade": body.radius,
                "pl_masse": body.mass,
                "pl_orbper": body.orbper,
                "pl_orbeccen": body.orbeccen,
                "pl_insol": pl_insol,
                "st_teff": star.temperature,
                "st_mass": star.mass,
                "st_rad": star.radius / 109.2, # Convert Earth radii back to Solar radii for ML
                "st_lum": star.lum
            }
            # The ML model returns percentage (0-100)
            return self.ml_calculator.predict(features)
        
        # Fallback to basic calculation if ML is not available
        score = 0.0
        # ... (rest of old logic)
        temp_factor = 1.0 - abs(body.temperature - 288) / 100
        score += max(0, temp_factor) * 0.3
        mass_factor = 1.0 - abs(body.mass - 1.0) / 2.0
        score += max(0, mass_factor) * 0.3
        if 'O2' in body.atmosphere and body.atmosphere['O2'] > 0.1:
            score += 0.2
        if 'H2O' in body.atmosphere and body.atmosphere['H2O'] > 0:
            score += 0.2
        return min(1.0, score) * 100.0 # Convert to percentage to match ML output
    
    def step(self, dt: float = 0.01):
        """Advance the simulation by one time step"""
        self.update_positions()
        for body in self.bodies:
            if body.type == 'planet':
                body.habitability_score = self.calculate_habitability(body) 