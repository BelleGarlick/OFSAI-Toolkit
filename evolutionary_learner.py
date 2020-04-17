import copy
import math
import time
import pygame as pygame

from fsai.visualisation.draw_pygame import render
from typing import List

import numpy as np

from fsai import geometry
from fsai.objects.track import Track
from fsai.path_planning.waypoints import gen_waypoints, encode


class EvolutionarySimulation:
    def __init__(self, car_count: int, input_size: int, neurons: List[int]):
        self.layer_sizes = neurons
        self.input_size = input_size

        track = Track("examples/data/tracks/imola.json")
        self.initial_car = track.cars[0]

        self.right_boundary, self.left_boundary, self.o = track.get_boundary()
        self.all_boundary = np.vstack((self.right_boundary, self.left_boundary, self.o))

        self.step_size = 0.1
        self.best_weights = self.gen_random_weights(new=True)
        self.best_weights_distance = 0
        self.cars = self.gen_cars(car_count)
        self.car_count = car_count

        self.episode_length = 0
        self.episode_number = 0

        self.time_a = 0
        self.time_b = 0
        self.time_c = 0
        self.time_d = 0

    def gen_random_weights(self, new=False):
        input_size = self.input_size
        layers: List[np.ndarray] = []

        for layer_index in range(len(self.layer_sizes)):
            layer_size = self.layer_sizes[layer_index]
            if new:
                layers.append(
                    (np.random.rand(input_size + 1, layer_size) * 2 - 1)
                ),
            else:
                layers.append(
                    self.best_weights[layer_index] + (np.random.rand(input_size + 1, layer_size) * 2 - 1) * self.step_size
                ),
            input_size = layer_size
        return layers

    def gen_cars(self, count):
        cars = [copy.deepcopy(self.initial_car) for i in range(count)]
        for car in cars:
            car.alive = True
            car.weights = self.gen_random_weights()
        return cars

    def feed(self, input, weights):
        input_vector = input
        for weight in weights:
            output = np.transpose(weight).dot(np.append(input_vector, 1))
            # active
            input_vector = 1/(1 + np.exp(-output))
        return input_vector

    def do_step(self, time_delta):
        self.episode_length += time_delta

        alive_cars = [car for car in self.cars if car.alive]
        for car_index in range(len(alive_cars)):
            a = time.time()
            car = alive_cars[car_index]
            encoding = self.get_waypoint_encoding_for_car(car)
            b = time.time()
            output = self.feed(encoding, car.weights)
            car.steer = output[0] - output[1]
            car.throttle = output[2]
            car.brake = output[3]
            c =  time.time()
            car.physics.do_physics(time_delta)
            d = time.time()
            if self.has_intersected(car) or (
                    sum(car.physics.distances_travelled) < 10 and self.episode_length > 10):
                car.alive = False
            e = time.time()
            self.time_a += b-a
            self.time_b += c-b
            self.time_c += d-c
            self.time_d += e-d

        if len(alive_cars) == 0:
            furthest = self.cars[0]
            for car in self.cars:
                if car.physics.distance_travelled > furthest.physics.distance_travelled:
                    furthest = car

            self.best_weights = furthest.weights
            if furthest.physics.distance_travelled > self.best_weights_distance:
                self.best_weights_distance = furthest.physics.distance_travelled
            print("Episode {} Complete in {}s with distance: {}. Best distance: {}.".format(self.episode_number, self.episode_length, furthest.physics.distance_travelled, self.best_weights_distance))
            self.episode_length = 0
            self.episode_number += 1
            self.cars = self.gen_cars(self.car_count)

    def has_intersected(self, car):
        body_points = [
            (car.pos[0] + car.cg_to_front, car.pos[1] - (car.width + car.wheel_width) / 2),
            (car.pos[0] + car.cg_to_front, car.pos[1] + (car.width + car.wheel_width) / 2),
            (car.pos[0] - car.cg_to_rear, car.pos[1] + (car.width + car.wheel_width) / 2),
            (car.pos[0] - car.cg_to_rear, car.pos[1] - (car.width + car.wheel_width) / 2)
        ]

        for i in range(len(body_points)):
            body_points[i] = geometry.rotate(body_points[i], car.heading, car.pos)

        car_boundary = np.array([
            [body_points[0][0], body_points[0][1], body_points[1][0], body_points[1][1]],
            [body_points[1][0], body_points[1][1], body_points[2][0], body_points[2][1]],
            [body_points[2][0], body_points[2][1], body_points[3][0], body_points[3][1]],
            [body_points[3][0], body_points[3][1], body_points[0][0], body_points[0][1]],
        ])

        intersections = geometry.segment_intersections(car_boundary[0], self.all_boundary) + \
                        geometry.segment_intersections(car_boundary[1], self.all_boundary) + \
                        geometry.segment_intersections(car_boundary[2], self.all_boundary) + \
                        geometry.segment_intersections(car_boundary[3], self.all_boundary)

        return len(intersections) > 0


    def get_waypoint_encoding_for_car(self, car):
        positive_waypoints = 12
        negative_waypoints = 4
        waypoints = gen_waypoints(
            car_pos=car.pos,
            car_angle=car.heading,
            blue_boundary=self.left_boundary,
            yellow_boundary=self.right_boundary,
            orange_boundary=self.o,
            foresight=positive_waypoints,
            spacing=2,
            negative_foresight=negative_waypoints,
            radar_length=12,
            radar_count=5,
            radar_span=math.pi / 1.2,
            margin=car.width,
            smooth=True
        )

        # add inputs to the encoding
        encoding = np.array(encode(waypoints, 4))[:,:3]
        encoding[:,0] /= 6
        encoding[:,1] /= math.pi/2
        encoding[:,2] /= math.pi/2
        encoding = encoding.reshape(encoding.shape[0] * encoding.shape[1])

        # add car angle to input
        car_angle = car.heading + geometry.angle(waypoints[negative_waypoints].line) - math.pi/2
        encoding = np.append(encoding, geometry.clip([car_angle / (math.pi/2)], -1, 1)[0])

        # and position of car
        pos = waypoints[negative_waypoints].find_optimum_from_point(car.pos)
        encoding = np.append(encoding, pos)
        return encoding


if __name__ == "__main__":
    CAR_COUNT = 20

    pygame.init()
    screen_size = [1000, 700]
    screen = pygame.display.set_mode(screen_size)

    simulation = EvolutionarySimulation(CAR_COUNT, 53, [20, 12, 8, 4])

    simulation_running = True
    last_time = time.time()

    while simulation_running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        now = time.time()
        dt = now - last_time
        simulation.do_step(dt / 3)

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

