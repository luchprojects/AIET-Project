import numpy as np
from dataclasses import dataclass
from typing import Dict, List

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
    
    def __post_init__(self):
        self.acceleration = np.zeros(3)
        self.habitability_score = 0.0

class SimulationEngine:
    def __init__(self):
        self.bodies: List[CelestialBody] = []
        self.G = 39.478  # Gravitational constant in AU^3/(M_sun * year^2)
        self.time_step = 0.01  # in years
        
    def add_body(self, body: CelestialBody):
        self.bodies.append(body)
        
    def calculate_acceleration(self, body: CelestialBody) -> np.ndarray:
        """Calculate gravitational acceleration on a body due to all other bodies"""
        acceleration = np.zeros(3)
        for other in self.bodies:
            if other != body:
                r = other.position - body.position
                r_mag = np.linalg.norm(r)
                if r_mag > 0:  # Avoid division by zero
                    acceleration += self.G * other.mass * r / (r_mag ** 3)
        return acceleration
    
    def update_positions(self):
        """Update positions and velocities using Verlet integration"""
        for body in self.bodies:
            body.acceleration = self.calculate_acceleration(body)
            body.velocity += body.acceleration * self.time_step
            body.position += body.velocity * self.time_step
            
    def calculate_habitability(self, body: CelestialBody) -> float:
        """Calculate basic habitability score based on physical properties"""
        score = 0.0
        
        # Temperature factor (assuming Earth-like temperature range is ideal)
        temp_factor = 1.0 - abs(body.temperature - 288) / 100  # 288K is Earth's average
        score += max(0, temp_factor) * 0.3
        
        # Mass factor (assuming Earth-like mass is ideal)
        mass_factor = 1.0 - abs(body.mass - 1.0) / 2.0  # 1 Earth mass is ideal
        score += max(0, mass_factor) * 0.3
        
        # Atmosphere factor
        if 'O2' in body.atmosphere and body.atmosphere['O2'] > 0.1:
            score += 0.2
        if 'H2O' in body.atmosphere and body.atmosphere['H2O'] > 0:
            score += 0.2
            
        return min(1.0, score)
    
    def step(self, dt: float = 0.01):
        """Advance the simulation by one time step"""
        self.update_positions()
        for body in self.bodies:
            if body.type == 'planet':
                body.habitability_score = self.calculate_habitability(body) 