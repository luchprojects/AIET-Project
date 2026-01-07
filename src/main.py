import numpy as np
import pygame
from simulation_engine import SimulationEngine, CelestialBody
from visualization import SolarSystemVisualizer

def create_solar_system() -> SimulationEngine:
    """Create a sample solar system with a star and some planets"""
    engine = SimulationEngine()
    
    # Create the star (similar to our Sun)
    sun = CelestialBody(
        name="Sun",
        mass=1.0,  # 1 solar mass
        radius=109.0,  # 109 Earth radii
        position=np.array([0.0, 0.0, 0.0]),
        velocity=np.array([0.0, 0.0, 0.0]),
        temperature=5778,  # Kelvin
        atmosphere={"H": 0.73, "He": 0.25, "O": 0.01, "C": 0.01},
        type="star"
    )
    engine.add_body(sun)
    
    # Create Earth-like planet
    earth = CelestialBody(
        name="Earth",
        mass=1.0,  # 1 Earth mass
        radius=1.0,  # 1 Earth radius
        position=np.array([1.0, 0.0, 0.0]),  # 1 AU from sun
        velocity=np.array([0.0, 2.0, 0.0]),  # Circular orbit
        temperature=288,  # Kelvin
        atmosphere={"N2": 0.78, "O2": 0.21, "Ar": 0.01, "CO2": 0.0004},
        type="planet"
    )
    engine.add_body(earth)
    
    # Create Mars-like planet
    mars = CelestialBody(
        name="Mars",
        mass=0.107,  # 0.107 Earth masses
        radius=0.532,  # 0.532 Earth radii
        position=np.array([1.524, 0.0, 0.0]),  # 1.524 AU from sun
        velocity=np.array([0.0, 1.5, 0.0]),  # Circular orbit
        temperature=210,  # Kelvin
        atmosphere={"CO2": 0.95, "N2": 0.027, "Ar": 0.016},
        type="planet",
        orbper=687.0,
        orbeccen=0.0934
    )
    engine.add_body(mars)
    
    return engine

def main():
    # Initialize simulation
    engine = create_solar_system()
    visualizer = SolarSystemVisualizer()
    
    # Run orbit unit test on startup
    if hasattr(visualizer, 'run_orbit_unit_test'):
        visualizer.run_orbit_unit_test()
    
    # Main simulation loop
    running = True
    while running:
        # Handle events
        running = visualizer.handle_events()
        
        # Update simulation when in simulation state
        if visualizer.show_simulation:
            engine.step()
        
        # Render using the main render method (handles all states and tooltips)
        visualizer.render(engine)
    
    pygame.quit()

if __name__ == "__main__":
    main() 