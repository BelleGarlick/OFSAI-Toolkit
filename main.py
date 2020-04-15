import time
import pygame as pygame

from evolutionary_learner import EvolutionarySimulation
from fsai.visualisation.draw_pygame import render

CAR_COUNT = 10

pygame.init()
screen_size = [1000, 700]
screen = pygame.display.set_mode(screen_size)


simulation = EvolutionarySimulation(10, 52, [25, 3])
simulation.gen_cars(100)

simulation_running = True
last_time = time.time()

while simulation_running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    now = time.time()
    dt = now - last_time
    simulation.do_step(dt/15)

    render(
        screen,
        screen_size,
        lines=[
            ((0, 0, 255), 2, simulation.left_boundary),
            ((255, 255, 0), 2, simulation.right_boundary),
            ((255, 100, 0), 2, simulation.o)
        ],
        cars=[car for car in simulation.cars if car.alive],
        padding=0
    )

    pygame.display.flip()
    last_time = now

