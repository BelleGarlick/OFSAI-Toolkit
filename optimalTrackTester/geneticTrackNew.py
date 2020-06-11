import time
import math

import pygame

from fsai.path_planning.waypoint import Waypoint
from fsai.path_planning.waypoints import gen_waypoints, encode, decimate_waypoints
from optimalTrackTester.geneticTestUtils import get_track_time, send_track_to_server, render_scene, get_track_from_server

RUNS_PER_SEGMENTS = 20
MAX_STEP_SIZE = 0.05
DELTA_STEP_SIZE = 0.6


class OptimalPathCreator:
    def __init__(self):
        self.name, self.uuid, self.initial_car, left_boundary, right_boundary, orange_boundary, self.intervals = get_track_from_server()
        self.boundary = left_boundary + right_boundary + orange_boundary

        self.waypoints = generate_waypoints(self.initial_car, left_boundary, right_boundary, orange_boundary)
        self.waypoint_count = len(self.waypoints)

        self.current_index = 0

        self.best_result = math.inf

    def run(self):
        pygame.init()
        screen_size = [800, 400]
        screen = pygame.display.set_mode(screen_size)

        start_step_time = time.time()

        running = True
        epoch = 0

        while running:
            new_best = False
            for i in range(RUNS_PER_SEGMENTS):
                step_size = MAX_STEP_SIZE * (DELTA_STEP_SIZE ** i)
                new_best = new_best or self.genetic_test(step_size)

            self.current_index += 1
            if self.current_index >= self.waypoint_count:
                self.current_index = 0

                print("{}".format(self.best_result))
                render_scene(screen, screen_size, self.waypoints)

                now = time.time()
                print("Epoch: {} Time: {}".format(epoch, now - start_step_time))
                start_step_time = now
                epoch += 1
                if epoch >= self.intervals:
                    running = False

        send_track_to_server(self.name, self.waypoints, self.best_result, self.uuid)

    def genetic_test(self, step_size):
        index = self.current_index
        initial_value = self.waypoints[index].optimum
        best = False
        self.waypoints[index].optimum = max(0, min(1, initial_value + step_size))
        test_time = get_track_time(self.waypoints)
        if test_time < self.best_result:
            self.best_result = test_time
            best = True
        else:
            self.waypoints[index].optimum = max(0, min(1, initial_value - step_size))
            test_time = get_track_time(self.waypoints)
            if test_time < self.best_result:
                self.best_result = test_time
                best = True
            else:
                self.waypoints[index].optimum = initial_value

        return best


def generate_waypoints(initial_car, left_boundary, right_boundary, orange_boundary):
    waypoints = gen_waypoints(
        car_pos=initial_car.pos,
        car_angle=initial_car.heading,
        blue_boundary=left_boundary,
        yellow_boundary=right_boundary,
        orange_boundary=orange_boundary,
        full_track=True,
        spacing=1,
        radar_length=30,
        radar_count=17,
        radar_span=math.pi / 1.1,
        margin=0,
        smooth=True
    )
    decimated_waypoints = []
    for i in range(len(waypoints)):
        if i % 3 == 0:
            decimated_waypoints.append(waypoints[i])
    # decimated_waypoints = decimate_waypoints(decimated_waypoints, threshold = 0.35, spread=1, max_gap = 4)
    for waypoint in decimated_waypoints:
        waypoint.optimum = 0.5 #random.random()
        waypoint.v = 0
    return decimated_waypoints


if __name__ == "__main__":
    OptimalPathCreator().run()
