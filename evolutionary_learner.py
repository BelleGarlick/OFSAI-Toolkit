import copy
import math
import random
import time
import pygame as pygame

from fsai.visualisation.draw_pygame import render
from typing import List

import numpy as np

from fsai import geometry
from fsai.objects.track import Track
from fsai.path_planning.waypoints import gen_waypoints, encode


class EvolutionarySimulation:
    def __init__(self, tracks, car_count: int, input_size: int, neurons: List[int]):
        self.running = True

        self.layer_sizes = neurons
        self.input_size = input_size

        self.tracks = tracks
        self.initial_car, self.left_boundary, self.right_boundary, self.o, self.all_boundary = self.gen_track()

        self.episode_length = 0
        self.episode_number = 0
        self.episode_frame = 0

        self.step_size = 0.5
        self.best_weights = self.gen_random_weights(new=True)
        self.best_weights_distance = 0
        self.cars = self.gen_cars(car_count)
        self.car_count = car_count

        # waypoint length, waypoint to waypoint angle, acc_x, acc_y, vel_x, vel_y
        self.highest_input_data = [
            0, 0, 0, 0, 0, 0
        ]

        self.fastest_points = []

    def gen_track(self):
        track = Track(random.choice(self.tracks))
        initial_car = track.cars[0]

        left_boundary, right_boundary, o = track.get_boundary()
        all_boundary = np.vstack((right_boundary, left_boundary, o))
        return initial_car, left_boundary, right_boundary, o, all_boundary

    def gen_random_weights(self, new=False):
        input_size = self.input_size
        layers: List[np.ndarray] = []

        for layer_index in range(len(self.layer_sizes)):
            layer_size = self.layer_sizes[layer_index]
            if new:
                layers.append(
                    (np.random.rand(layer_size, input_size + 1) * 2 - 1)
                ),
            else:
                layers.append(
                    self.best_weights[layer_index] + (np.random.rand(layer_size, input_size + 1) * 2 - 1) * self.step_size
                ),
            input_size = layer_size
        return layers

    def gen_cars(self, count):
        cars = [copy.deepcopy(self.initial_car) for i in range(count)]
        # if self.episode_number % 5 == 4:
        #     cars = cars[0:1]
        for car in cars:
            car.alive = True
            car.weights = self.gen_random_weights()
            car.pos_marks = []
        self.step_size *= 0.99
        return cars

    def feed(self, input, weights):
        input_vector = input
        for weight in weights:
            output = np.dot(weight, np.append(input_vector, 1))
            # active
            input_vector = 1/(1 + np.exp(-output))
        return input_vector

    def do_step(self, time_delta):
        self.episode_length += time_delta
        if math.floor(self.episode_length * 4) > self.episode_frame + 1:
            self.episode_frame += 1

        alive_cars = [car for car in self.cars if car.alive]
        for car_index in range(len(alive_cars)):
            car = alive_cars[car_index]
            encoding = self.get_waypoint_encoding_for_car(car)
            output = self.feed(encoding, car.weights)
            car.steer = output[0] * 2 - 1
            car.throttle = output[1]
            car.brake = output[2]
            car.physics.update(time_delta)

            if self.has_intersected(car) or (
                    sum(car.physics.distances_travelled) < 4 and self.episode_length > 10):
                car.alive = False

            if self.episode_frame > len(car.pos_marks) - 1:
                car.pos_marks.append(copy.deepcopy(car.pos))

        if len(alive_cars) == 0:
            furthest = self.cars[0]
            for car in self.cars:
                if car.physics.distance_travelled > furthest.physics.distance_travelled:
                    furthest = car

            if furthest.physics.distance_travelled > self.best_weights_distance:
                self.best_weights_distance = furthest.physics.distance_travelled
                self.best_weights = furthest.weights
                self.fastest_points = furthest.pos_marks

            self.initial_car, self.left_boundary, self.right_boundary, self.o, self.all_boundary = self.gen_track()
            print("Episode {} Complete in {}s with distance: {}. Best distance: {}. Step Size: {}".format(self.episode_number, self.episode_length, furthest.physics.distance_travelled, self.best_weights_distance, self.step_size))

            if self.episode_number % 10 == 0:
                print(self.highest_input_data)
                self.highest_input_data = [0, 0, 0, 0, 0, 0]
            self.episode_length = 0
            self.episode_frame = 0
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
        negative_waypoints = 0
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
            smooth=True,
            force_perp_center_line=True
        )

        # add inputs to the encoding
        encoding = np.array(encode(waypoints, negative_waypoints))[:,:2]
        max_way_width = max(encoding[:,0])
        max_way2way_angle = max(encoding[:,1])
        encoding[:, 0] /= 10
        encoding[:, 1] /= 1.1
        encoding = encoding.reshape(encoding.shape[0] * encoding.shape[1])

        # # add car angle to input
        # car_angle = car.heading + geometry.angle(waypoints[negative_waypoints].line) - math.pi/2
        # encoding = np.append(encoding, geometry.clip([car_angle / (math.pi/2)], -1, 1)[0])

        # and position of car
        pos = waypoints[negative_waypoints].find_optimum_from_point(car.pos)
        encoding = np.append(encoding, [
            pos, sqrt(car.steer), car.throttle, car.brake,
            sqrt(car.physics.accel_c[0] / 24), sqrt(car.physics.accel_c[1] / 50),
            sqrt(car.physics.vel_c[0] / 26), sqrt(car.physics.vel_c[1] / 12)
        ])

        self.highest_input_data[0] = max(self.highest_input_data[0], max_way_width)
        self.highest_input_data[1] = max(self.highest_input_data[1], max_way2way_angle)
        self.highest_input_data[2] = max(self.highest_input_data[2], car.physics.accel_c[0])
        self.highest_input_data[3] = max(self.highest_input_data[3], car.physics.accel_c[1])
        self.highest_input_data[4] = max(self.highest_input_data[4], car.physics.vel_c[0])
        self.highest_input_data[5] = max(self.highest_input_data[5], car.physics.vel_c[1])

        return encoding


def sqrt(x):
    rt = math.sqrt(abs(x))
    if x < 0:
        return -rt
    return rt


if __name__ == "__main__":
    CAR_COUNT = 20

    pygame.init()
    screen_size = [700, 500]
    screen = pygame.display.set_mode(screen_size)

    simulation = EvolutionarySimulation([#"examples/data/tracks/monza.json",
                                         # "examples/data/tracks/azure_circuit.json",
                                         #"examples/data/tracks/imola.json",
                                         # "examples/data/tracks/dirtfish.json",
                                         "examples/data/tracks/brands_hatch.json",
                                         # "examples/data/tracks/cadwell_park.json",
                                         # "examples/data/tracks/silverstone_class.json",
                                         # "examples/data/tracks/wildcrest.json",
                                         # "examples/data/tracks/laguna_seca.json"
                                         ], CAR_COUNT, 34, [40, 30, 3])
    last_time = time.time()

    while simulation.running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                simulation.running = False

        now = time.time()
        dt = now - last_time
        simulation.do_step(dt)

        render(
            screen,
            screen_size,
            lines=[
                ((0, 0, 255), 2, simulation.left_boundary),
                ((255, 255, 0), 2, simulation.right_boundary),
                ((255, 100, 0), 2, simulation.o),
            ],
            points=[
                ((0, 255, 0), 2, simulation.fastest_points)
            ],
            cars=[car for car in simulation.cars if car.alive],
            padding=0
        )

        pygame.display.flip()
        last_time = now

