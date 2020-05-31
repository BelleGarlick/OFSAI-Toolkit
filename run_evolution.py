import time

import pygame

from aiton_senna.evolution_simulation import EvolutionarySimulation
from fsai.objects.track import Track
from fsai.visualisation.draw_pygame import render

if __name__ == "__main__":
    pygame.init()
    screen_size = [700, 500]
    screen = pygame.display.set_mode(screen_size)

    simulation = EvolutionarySimulation()
    simulation.set_track(Track("../examples/data/tracks/brands_hatch.json"))

    last_time = time.time()
    while simulation.episode_count < 100:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                simulation.running = False

        now = time.time()
        dt = now - last_time
        last_time = now

        if not simulation.episode_running:
            simulation.new_episode(10)
            last_time = time.time()
        else:
            simulation.update(0.03)

        render(
            screen,
            screen_size,
            lines=[
                ((0, 0, 255), 2, simulation.blue_boundary),
                ((255, 255, 0), 2, simulation.yellow_boundary),
                ((255, 100, 0), 2, simulation.o),
            ],
            points=[
                # ((0, 255, 0), 2, simulation.fastest_points)
            ],
            cars=[ai.car for ai in simulation.get_alive_ai()],
            padding=0
        )
        pygame.display.flip()