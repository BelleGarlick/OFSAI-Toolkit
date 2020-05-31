import copy
import math
import random
import time
import pygame as pygame

from fsai.mapping.boundary_estimation import get_delaunay_triangles
from fsai.visualisation.draw_opencv import render_area
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
        self.initial_car, self.left_boundary, self.right_boundary, self.o, self.all_boundary, self.blue_cones, self.yellow_cones, self.orange_cones, self.big_cones = self.gen_track()

        self.episode_length = 0
        self.episode_number = 0
        self.episode_frame = 0

        self.step_size = 0.2
        self.best_weights = self.gen_random_weights(new=True)
        # self.best_weights = np.load("/Users/samgarlick/Developer/GitHub/OS-FS-AI/best_weights.npy", allow_pickle=True)
        self.best_weights_distance = 0
        self.cars = self.gen_cars(car_count)
        self.car_count = car_count

        # waypoint length, waypoint to waypoint angle, acc_x, acc_y, vel_x, vel_y
        self.highest_input_data = [
            0, 0, 0, 0, 0, 0
        ]

        self.time_ab = 0
        self.time_bc = 0
        self.time_cd = 0
        self.time_de = 0

        self.fastest_points = []

    def gen_track(self):
        track = Track(random.choice(self.tracks))
        get_max_track_radias(track)
        initial_car = track.cars[0]

        left_boundary, right_boundary, o = track.get_boundary()
        all_boundary = np.vstack((right_boundary, left_boundary, o)) if len(o) > 0 else np.vstack((right_boundary, left_boundary))
        return initial_car, left_boundary, right_boundary, o, all_boundary, track.blue_cones, track.yellow_cones, track.orange_cones, track.big_cones

    def gen_random_weights(self, new=False):
        input_size = self.input_size
        layers: List[np.ndarray] = []

        for layer_index in range(len(self.layer_sizes)):
            layer_size = self.layer_sizes[layer_index]
            if new:
                layers.append(
                    (np.random.rand(layer_size, input_size + layer_size + 1) * 2 - 1)
                ),
            else:
                layers.append(
                    self.best_weights[layer_index] + (np.random.rand(layer_size, input_size + layer_size + 1) * 2 - 1) * self.step_size
                ),
            input_size = layer_size
        return np.array(layers)

    def gen_cars(self, count):
        cars = [copy.deepcopy(self.initial_car) for i in range(count)]

        for car in cars:
            car.alive = True
            car.last_outputs = [np.zeros((layer_size, 1)) for layer_size in self.layer_sizes]
            car.weights = self.gen_random_weights()
            car.pos_marks = []
        self.step_size *= 0.999
        return cars

    def feed(self, car, input, weights):
        input_vector = input
        for i in range(len(weights)):
            output = np.dot(weights[i], np.append(np.append(input_vector, car.last_outputs[i]), 1))
            # active
            input_vector = 1/(1 + np.exp(-output))
            car.last_outputs[i] = input_vector
        return input_vector

    def do_step(self, time_delta):
        self.episode_length += time_delta
        if math.floor(self.episode_length * 4) > self.episode_frame + 1:
            self.episode_frame += 1

        alive_cars = [car for car in self.cars if car.alive]
        for car_index in range(len(alive_cars)):
            car = alive_cars[car_index]
            encoding = self.get_waypoint_encoding_for_car(car)
            output = self.feed(car, encoding, car.weights)
            car.steer = output[0] * 2 - 1
            car.throttle = output[1]
            car.brake = output[2]

            car.physics.update(time_delta)

            if self.has_intersected(car) or (
                    sum(car.physics.distances_travelled) < 5 and self.episode_length > 10):
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

            self.initial_car, self.left_boundary, self.right_boundary, self.o, self.all_boundary, self.blue_cones, self.yellow_cones, self.orange_cones, self.big_cones = self.gen_track()
            print("Episode {} Complete in {}s with distance: {}. Best distance: {}. Step Size: {}".format(self.episode_number, self.episode_length, furthest.physics.distance_travelled, self.best_weights_distance, self.step_size))

            if self.episode_number % 10 == 0:
                print("Saved Weights as 'best_weights.npy'")
                np.save("best_weights", self.best_weights)
            self.episode_length = 0
            self.episode_frame = 0
            self.episode_number += 1
            self.cars = self.gen_cars(self.car_count)

    def has_intersected(self, car, line):
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

        filtered_lines = geometry.filter_lines_by_distance(car.pos, 8, self.all_boundary)

        intersections = geometry.segment_intersections(car_boundary[0], filtered_lines) + \
                        geometry.segment_intersections(car_boundary[1], filtered_lines) + \
                        geometry.segment_intersections(car_boundary[2], filtered_lines) + \
                        geometry.segment_intersections(car_boundary[3], filtered_lines)

        return len(intersections) > 0

    def get_waypoint_encoding_for_car(self, car):

        polygons = get_delaunay_triangles(self.blue_cones, self.yellow_cones, self.orange_cones, self.big_cones)
        image = render_area(
            camera_pos=car.pos,
            rotation=-car.heading - math.pi / 2,
            area=[50, 50],
            resolution=2,
            polygons=[
                ((255, 255, 255), (0, 0, 0), 0, polygons)
            ],
            lines=[
                ((255, 0, 0), 1, geometry.filter_lines_by_distance(car.pos, 60, self.left_boundary)),
                ((0, 255, 255), 1, geometry.filter_lines_by_distance(car.pos, 60, self.right_boundary)),
                ((0, 100, 255), 1, geometry.filter_lines_by_distance(car.pos, 60, self.o)),
            ],
            cars=[car],
            background=0
        )
        flattened_encoding = np.reshape(image/255, (50*50 * 3*4))
        car_data = np.array(
            [
                sqrt(car.steer), car.throttle, car.brake,
                sqrt(car.physics.accel_c[0] / 24), sqrt(car.physics.accel_c[1] / 50),
                sqrt(car.physics.vel_c[0] / 26), sqrt(car.physics.vel_c[1] / 12)
            ]
        )

        return np.hstack((flattened_encoding, car_data))


def get_max_track_radias(track):
    left_boundary, right_boundary, o = track.get_boundary()

    initial_car = track.cars[0]
    waypoints = gen_waypoints(
        car_pos=initial_car.pos,
        car_angle=initial_car.heading,
        blue_boundary=left_boundary,
        yellow_boundary=right_boundary,
        orange_boundary=o,
        full_track=True,
        spacing=0.5,
        radar_length=20,
        radar_count=19,
        radar_span=math.pi / 1.2,
        margin=initial_car.width,
        smooth=True
    )

    # add inputs to the encoding
    encoding = encode(waypoints, 0)
    max_curve = 0
    max_width = 0
    for e in encoding:
        max_width = max(max_width, abs(e[0]))
        max_curve = max(max_curve, abs(e[1]))

    # print("Max Width: {}, Max Curve: {}".format(max_width, max_curve))


def sqrt(x):
    rt = math.sqrt(abs(x))
    if x < 0:
        return -rt
    return rt


if __name__ == "__main__":
    CAR_COUNT = 6

    pygame.init()
    screen_size = [700, 500]
    screen = pygame.display.set_mode(screen_size)

    simulation = EvolutionarySimulation([#"examples/data/tracks/monza.json",
                                         # "examples/data/tracks/azure_circuit.json",
                                         # "examples/data/tracks/imola.json",
                                         #  "examples/data/tracks/dirtfish.json",
                                          "examples/data/tracks/training.json",
                                         # "examples/data/tracks/brands_hatch.json",
                                         # "examples/data/tracks/cadwell_park.json",
                                         #"examples/data/tracks/silverstone_class.json",
                                         # "examples/data/tracks/wildcrest.json",
                                         # "examples/data/tracks/laguna_seca.json"
                                         ], CAR_COUNT, 30007, [100, 30, 3])

    last_time = time.time()
    start = time.time()
    frames = 0
    while simulation.running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                simulation.running = False

        now = time.time()
        dt = now - last_time
        simulation.do_step(dt)
        frames += 1
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

