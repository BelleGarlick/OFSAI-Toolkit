import copy
import random
import time
import math

import pygame

from fsai.path_planning.waypoints import gen_waypoints, encode, decimate_waypoints
from optimalTrackTester.geneticTestUtils import get_track_time, send_track_to_server, render_scene, \
    get_track_from_server, get_track_from_files, printProgressBar
from fsai import geometry

WAYPOINT_VARIATION_COUNT = 100
WAYPOINT_SELECTION = 25
WAYPOINT_OFFSPRING = 4
STEP_SIZE = 0.0001
RENDER = False
SEGMENTS = 4


class OptimalPathStandardEvolver:
    def __init__(self):
        self.name, self.uuid, self.initial_car, left_boundary, right_boundary, orange_boundary, self.intervals = get_track_from_files("azure_circuit")
        self.boundary = left_boundary + right_boundary + orange_boundary

        self.initial_waypoints = generate_waypoints(self.initial_car, left_boundary, right_boundary, orange_boundary)
        self.waypoint_count = len(self.initial_waypoints)

        car_mass = 0.74  # tonne
        gravity = 9.81
        frictional_force = 1.7

        self.max_lateral_frictional_force = frictional_force * gravity * car_mass
        self.max_speed = 104  # meters per second

        random.seed(0)

        self.time_a = 0
        self.time_b = 0
        self.time_c = 0
        self.time_d = 0
        self.time_e = 0
        self.time_f = 0
        self.count = 0

        self.waypoint_variations = []
        for i in range(WAYPOINT_VARIATION_COUNT):
            new_waypoints = copy.deepcopy(self.initial_waypoints)
            # for j in range(self.waypoint_count):
            #     new_waypoints[j].optimum = random.random()
            self.waypoint_variations.append({"track": new_waypoints, "time": -1})

        self.segment_bounds = generate_segment_bounds(len(new_waypoints), SEGMENTS)

    def run(self):
        if RENDER:
            pygame.init()
            screen_size = [800, 400]
            screen = pygame.display.set_mode(screen_size)
            render_scene(screen, screen_size, [w["track"] for w in self.waypoint_variations], line_width=1)

        now = time.time()
        for i in range(self.intervals):
            for segment in range(SEGMENTS):
                self.evaluate_waypoints()
                best_time, waypoint_variations = self.get_best_waypoints()
                self.populate_waypoints(self.segment_bounds[segment])

            if i % 100 == 0:
                if i % 100 == 0 and RENDER:
                    render_scene(screen, screen_size, [w["track"] for w in waypoint_variations], line_width=1)
                print("{}/{}: {}".format(i, self.intervals, best_time))

        print("Finished in {} seconds.".format(time.time() - now))
        best_time, best_waypoints = self.get_best_waypoints()

        send_track_to_server(self.name, best_waypoints[0], best_time, self.uuid)

    def evaluate_waypoints(self):
        for track in self.waypoint_variations:
            track["time"] = get_track_time(track["track"], self.max_lateral_frictional_force, self.max_speed)

    def get_best_waypoints(self):
        self.waypoint_variations = sorted(self.waypoint_variations, key=lambda k: k['time'])
        self.waypoint_variations = self.waypoint_variations[:WAYPOINT_SELECTION]

        return self.waypoint_variations[0]["time"], self.waypoint_variations

    def populate_waypoints(self, segment_bounds):
        new_waypoints = []#copy.deepcopy(self.waypoint_variations)

        # Average
        # total_average = len(self.waypoint_variations)
        # average = copy.deepcopy(self.waypoint_variations[0])
        # for i in range(len(average)):
        #     average["track"][i].optimum = 0
        #     for w in self.waypoint_variations:
        #         average["track"][i].optimum += w["track"][i].optimum / total_average
        # new_waypoints.append(average)

        start_bound, end_bound = segment_bounds
        # Variate
        for w in self.waypoint_variations:
            for _ in range(WAYPOINT_OFFSPRING):
                new_waypoints.append({
                    "track": geometry.variate_waypoints(w["track"], STEP_SIZE, start_bound, end_bound),
                    "time": -1
                })

        self.waypoint_variations = new_waypoints


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
        if i % 1 == 0:
            decimated_waypoints.append(waypoints[i])
    # decimated_waypoints = decimate_waypoints(decimated_waypoints, threshold = 0.35, spread=1, max_gap = 4)
    for waypoint in decimated_waypoints:
        waypoint.optimum = 0.5 #random.random()
        waypoint.v = 0
    return decimated_waypoints


def generate_segment_bounds(waypoint_count, bounds_count):
    segment_bounds = []
    segment_size = waypoint_count // bounds_count

    start = 0
    for i in range(bounds_count - 1):
        end = (i + 1) * segment_size
        segment_bounds.append((start, end))
        start = end

    segment_bounds.append((start + 1, waypoint_count - 1))
    return segment_bounds


if __name__ == "__main__":
    OptimalPathStandardEvolver().run()
