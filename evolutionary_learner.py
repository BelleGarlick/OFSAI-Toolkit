import copy
import math
import multiprocessing as mp
from typing import List

import numpy as np

from fsai import geometry
from fsai.objects.track import Track
from fsai.path_planning.waypoints import gen_waypoints, encode


class EvolutionarySimulation:
    def __init__(self, car_count: int, input_size: int, neurons: List[int]):
        self.layer_sizes = neurons
        self.input_size = input_size

        track = Track("examples/data/tracks/azure_circuit.json")
        self.initial_car = track.cars[0]
        self.left_boundary, self.right_boundary, self.o = track.get_boundary()

        self.step_size = 0.5
        self.current_weights = None
        self.current_weights = self.gen_random_weights()
        self.cars = self.gen_cars(car_count)
        self.car_count = car_count

        self.episode_length = 0
        self.episode_number = 0

    def gen_random_weights(self):
        input_size = self.input_size
        layers: List[np.ndarray] = []

        for layer_index in range(len(self.layer_sizes)):
            layer_size = self.layer_sizes[layer_index]
            if self.current_weights is None:
                layers.append(
                    (np.random.rand(input_size + 1, layer_size) * 2 - 1)
                ),
            else:
                layers.append(
                    self.current_weights[layer_index] + (np.random.rand(input_size + 1, layer_size) * 2 - 1) * self.step_size
                ),
            input_size = layer_size
        return layers

    def gen_cars(self, count):
        cars = [copy.deepcopy(self.initial_car) for i in range(count)]
        for car in cars:
            car.alive = True
            car.weights = self.gen_random_weights()
        self.step_size *= 0.95
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
        all_dead = True

        alive_cars = [car for car in self.cars if car.alive]
        if len(alive_cars) > 0:
            with mp.Pool(processes=len(alive_cars)) as pool:
                encodings = (pool.map(self.get_waypoint_encoding_for_car, alive_cars))

        for car_index in range(len(alive_cars)):
            car = alive_cars[car_index]
            encoding = encodings[car_index]
            output = self.feed(encoding, car.weights)
            car.steer = output[0] * 2 - 1
            car.throttle = output[1]
            car.brake = output[2]
            car.physics.do_physics(time_delta)

            if self.has_intersected(car, np.vstack((self.right_boundary, self.left_boundary, self.o))) or (
                    sum(car.physics.distances_travelled) < 10 and self.episode_length > 10):
                car.alive = False

        if len(alive_cars) == 0:
            furthest = self.cars[0]
            for car in self.cars:
                if car.physics.distance_travelled > furthest.physics.distance_travelled:
                    furthest = car

            print("Episode {} Complete in {}s with best distance: {}".format(self.episode_number, self.episode_length, furthest.physics.distance_travelled))
            self.current_weights = furthest.weights
            self.episode_length = 0
            self.episode_number += 1
            self.cars = self.gen_cars(self.car_count)

    def has_intersected(self, car, lines):
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

        intersections = geometry.segment_intersections(car_boundary[0], lines) + geometry.segment_intersections(car_boundary[1], lines) + geometry.segment_intersections(car_boundary[2], lines) + geometry.segment_intersections(car_boundary[3], lines)

        return len(intersections) > 0


    def get_waypoint_encoding_for_car(self, car):
        waypoints = gen_waypoints(
            car_pos=car.pos,
            car_angle=car.heading,
            blue_boundary=self.left_boundary,
            yellow_boundary=self.right_boundary,
            orange_boundary=self.o,
            foresight=8,
            spacing=2,
            negative_foresight=4,
            radar_length=12,
            radar_count=5,
            radar_span=math.pi / 1.2,
            margin=car.width,
            smooth=True
        )

        encoding = np.array(encode(waypoints, 4))
        encoding = encoding.reshape(encoding.shape[0] * encoding.shape[1])
        return encoding
