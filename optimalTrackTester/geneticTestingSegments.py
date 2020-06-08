import time
import json
import copy
import math
import random
from typing import List

import numpy as np
import pygame
import requests

from fsai.objects.track import Track
from fsai.path_planning.waypoint import Waypoint
from fsai.visualisation.draw_pygame import render
from fsai.path_planning.waypoints import gen_waypoints, encode, decimate_waypoints
from fsai import geometry

URL_NEW = "http://127.0.0.1/new/"
URL_POST = "http://127.0.0.1/save/"

DELTA_TIME = 0.01

RUNS_PER_SEGMENTS = 50
MAX_STEP_SIZE = 0.4
MIN_STEP_SIZE = 0.01
DELTA_STEP_SIZE = (MAX_STEP_SIZE - MIN_STEP_SIZE) / (RUNS_PER_SEGMENTS - 1)

SPREAD_DELTA = 1

MAX_SEGMENT_SIZE = 34
SEGMENT_SIZE_DELTA = 7
MIN_SEGMENT_SIZE = 7



class OptimalPathCreator:
    def __init__(self):
        self.name, self.uuid, self.initial_car, left_boundary, right_boundary, orange_boundary, self.intervals = get_track()
        self.boundary = left_boundary + right_boundary + orange_boundary

        self.waypoints = generate_waypoints(self.initial_car, left_boundary, right_boundary, orange_boundary)
        self.waypoint_count = len(self.waypoints)

        self.best_result = {
            "time": -1, "dis": 0, "pts": [], "car": self.initial_car
        }

        # User for profiling
        self.TOTAL_AB, self.TOTAL_BC, self.TOTAL_CD, self.TOTAL_DE, self.TOTAL_EF, self.TOTAL_FG, self.TOTAL_GH = 0, 0, 0, 0, 0, 0, 0

    def run(self):
        pygame.init()
        screen_size = [800, 400]
        screen = pygame.display.set_mode(screen_size)

        current_segment_size = MAX_SEGMENT_SIZE
        segment_index = 0

        first_best = False

        count = 0
        set_new_car = True

        start_step_time = time.time()

        running = True
        interval = 0

        while running:
            start_index, end_index = segment_index, segment_index + current_segment_size
            if end_index >= self.waypoint_count:
                start_index -= self.waypoint_count
                end_index -= self.waypoint_count
            render_scene(screen, screen_size, self.waypoints, self.best_result["pts"], start_index, end_index)

            for i in range(RUNS_PER_SEGMENTS):
                step_size = MAX_STEP_SIZE - (DELTA_STEP_SIZE * i)
                self.best_result, new_best, self.waypoints = self.genetic_test(self.best_result, self.waypoints, self.boundary, start_index, end_index, step_size, set_new_car)
                self.best_result["car"].physics.distance_travelled = 0
                if self.best_result["time"] != -1 and new_best:
                    set_new_car = False

                if new_best:
                    render_scene(screen, screen_size, self.waypoints, self.best_result["pts"], start_index, end_index)
                    print("{}: {} {}".format(i, self.best_result["time"], self.best_result["dis"]))

            segment_index += current_segment_size // 2

            if segment_index >= self.waypoint_count:
                segment_index = 0
                # segment_offset = not segment_offset
                current_segment_size -= SEGMENT_SIZE_DELTA
                if current_segment_size < MIN_SEGMENT_SIZE:
                    current_segment_size = MAX_SEGMENT_SIZE

                    now = time.time()
                    print("Steps: {} Time: {}".format(interval, now - start_step_time))
                    start_step_time = now
                    interval += 1
                    if interval >= self.intervals:
                        running = False
                print("Segments: {}".format(current_segment_size))

                print(self.TOTAL_AB)
                print(self.TOTAL_BC)
                print(self.TOTAL_CD)
                print(self.TOTAL_DE)
                print(self.TOTAL_EF)
                print(self.TOTAL_FG)
                print(self.TOTAL_GH)

                # if first_best:
                if count >= 0:
                    set_new_car = True
                    count = 0

        self.save_state()

    def save_state(self):
        best_result = self.best_result

        dict = {
            "time": best_result["time"],
            "distance": best_result["dis"],
            "car": {
                "pos": [best_result["car"].pos[0], best_result["car"].pos[1]],
                "heading": best_result["car"].heading,
                "vel": {
                    "x": best_result["car"].physics.velocity[0],
                    "y": best_result["car"].physics.velocity[1],
                },
                "acc": {
                    "x": best_result["car"].physics.accel[0],
                    "y": best_result["car"].physics.accel[1],
                },
                "yaw_rate": best_result["car"].physics.yaw_rate
            },
            "waypoints": [{
                "x1": waypoint.line[0],
                "y1": waypoint.line[1],
                "x2": waypoint.line[2],
                "y2": waypoint.line[3],
                "throttle": waypoint.throttle,
                "pos": waypoint.optimum
            } for waypoint in self.waypoints]
        }

        x = requests.post(URL_POST, data={
            "name": self.name,
            "uuid": self.uuid,
            "data": json.dumps(dict)
        })
        print("Sending data to server: " + str(x.status_code))

    def try_track(self, waypoints, initial_car, boundary):
        car = copy.deepcopy(initial_car)
        lap_2_car = None
        start_lap = 0
        episode_time = 0

        lap_time = -1
        lap_count = 0
        points = []
        timed_points = []

        alive = True
        ep_count = 0
        completed_lap = False

        while alive:
            time_a = time.time()
            intersected_a, intersected_b = has_intersected(car,
                waypoints[(car.waypoint_index + 1) % len(waypoints)].line,
                waypoints[(car.waypoint_index + 2) % len(waypoints)].line
            )
            if intersected_a:
                car.waypoint_index += 1
            elif intersected_b:
                car.waypoint_index += 2
            if car.waypoint_index > len(waypoints): car.waypoint_index -= len(waypoints)

            throttle_control_index = car.waypoint_index + 1
            target_waypoint_index = car.waypoint_index + 2

            time_b = time.time()
            if car.waypoint_index > len(waypoints) - 1:
                lap_count += 1
                car.waypoint_index = 0
                if lap_count == 1:
                    start_lap = episode_time
                    lap_2_car = copy.deepcopy(car)
                elif lap_count == 2:
                    lap_time = episode_time - start_lap
                    alive = False
                    completed_lap = True
                    points = timed_points

            time_c = time.time()
            if lap_count < 2:
                speed_target_waypoint = waypoints[throttle_control_index % len(waypoints)]
                target_waypoint = waypoints[target_waypoint_index % len(waypoints)]
                target_point = target_waypoint.get_optimum_point()

                time_d = time.time()
                delta_angle = angle_difference(geometry.angle_to(car.pos, target_point), car.heading) / car.max_steer
                if delta_angle > 1: delta_angle = 1
                if delta_angle < -1: delta_angle = -1

                time_e = time.time()
                car.steer = delta_angle
                car.throttle, car.brake = max(0, speed_target_waypoint.throttle), -min(0, speed_target_waypoint.throttle)

                last_position = [car.pos[0], car.pos[1]]
                car.physics.update(DELTA_TIME)
                current_point = [car.pos[0], car.pos[1]]

                time_f = time.time()
                if ep_count % 10 == 0:
                    point = [{"pos": current_point, "thr": speed_target_waypoint.throttle}]
                    points += point
                    if lap_count == 1:
                        timed_points += point

                if episode_time > 5 and sum(car.physics.distances_travelled) < 5:
                    alive = False

                delta_line = current_point + last_position
                filtered_boundary = geometry.filter_lines_by_distance(current_point, 10, boundary)
                time_g = time.time()
                intersections = geometry.segment_intersections(delta_line, filtered_boundary)
                if len(intersections) > 0:
                    alive = False
                time_h = time.time()

                self.TOTAL_AB += time_b - time_a
                self.TOTAL_BC += time_c - time_b
                self.TOTAL_CD += time_d - time_c
                self.TOTAL_DE += time_e - time_d
                self.TOTAL_EF += time_f - time_e
                self.TOTAL_FG += time_g - time_f
                self.TOTAL_GH += time_h - time_g

            ep_count += 1
            episode_time += DELTA_TIME
        distance = car.physics.distance_travelled

        return {
            "time": lap_time,
            "dis": 0 if distance < 2 else distance,
            "pts": points,
            "car": lap_2_car if lap_count > 1 else None
        }

    def genetic_test(self, best_values, waypoints: List[Waypoint], boundary, start_index, end_index, step_size, set_new_car: bool = False):
        new_best = False
        initial_waypoints = copy.deepcopy(waypoints)
        best_waypoints = copy.deepcopy(waypoints)

        waypoint_variations = variate_waypoint_fingerprint(initial_waypoints, step_size, start_index, end_index, best_values["time"] != -1)
        waypoint_variate = waypoint_variations["waypoints"]
        car_result = self.try_track(waypoint_variate, best_values["car"], boundary)

        if car_result["time"] != -1:
            if not set_new_car:
                car_result["car"] = best_values["car"]

            if best_values["time"] == -1 or car_result["time"] < best_values["time"]:
                best_waypoints = copy.deepcopy(waypoint_variate)
                best_values = car_result
                new_best = True
            elif time == best_values["time"] and car_result["dis"] > best_values["dis"]:
                best_waypoints = copy.deepcopy(waypoint_variate)
                best_values = car_result
                new_best = True
        else:
            # if the car did not put in a time and there is no best time
            if best_values["time"] == -1:
                if car_result["dis"] > best_values["dis"]:
                    best_waypoints = copy.deepcopy(waypoint_variate)
                    best_values["time"] = car_result["time"]
                    best_values["dis"] = car_result["dis"]
                    best_values["pts"] = car_result["pts"]
                    if car_result["car"] is not None:
                        best_values["car"] = car_result["car"]
                    new_best = True

        # best_waypoints = copy.deepcopy(waypoint_variate)
        # best_values["time"] = car_result["time"]
        # best_values["dis"] = car_result["dis"]
        # best_values["pts"] = car_result["pts"]

        return best_values, new_best, best_waypoints


def variate_waypoint_fingerprint(waypoints, step_size, start, end, variate_pos):
    for i in range(start, end):
        waypoint = waypoints[i]
        waypoint.throttle += (random.random() * 2 - 1) * step_size * 2
        waypoint.throttle = max(-1, min(1, waypoint.throttle))
        if variate_pos:
            waypoint.optimum += (random.random() * 2 - 1) * step_size
            waypoint.optimum = max(0, min(1, waypoint.throttle))

    return {
        "waypoints": waypoints
    }


def get_track():
    result = requests.get(URL_NEW)
    server_data = json.loads(result.text)
    name = server_data["name"]
    print(name)
    uuid = server_data["uuid"]
    intervals = server_data["intervals"]

    track = Track().from_json(server_data["data"])
    initial_car = track.cars[0]

    initial_car.waypoint_index = -1
    initial_car.last_interval_update = 0

    left_boundary, right_boundary, orange_boundary = track.get_boundary()

    if ".reversed" in name:
        initial_car.heading += math.pi
        left_boundary, right_boundary = right_boundary, left_boundary

    return name, uuid, initial_car, left_boundary, right_boundary, orange_boundary, intervals


def get_best_position_throttle(position_throttle_matrix):
    best = position_throttle_matrix[0]
    for i in range(1, len(position_throttle_matrix)):
        current = position_throttle_matrix[i]

        if current["time"] != -1:
            if best["time"] == -1 or current["time"] < best["time"]:
                best = current
            elif time == best["time"] and current["dis"] > best["dis"]:
                best = current
        else:
            # if the car did not put in a time and there is no best time
            if best["time"] == -1:
                if current["dis"] > best["dis"]:
                    best = current
    return best


def save_best(name, best_waypoints):
    with open(name + ".json", "w+") as file:
        lines = []
        for waypoint in best_waypoints:
            l = waypoint.line
            lines.append({"x1": l[0], "y1": l[1], "x2": l[2], "y2": l[3], "p": waypoint.optimum, "t": waypoint.throttle,
                          "b": waypoint.brake})
        file.write(json.dumps(lines))


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
        if i % 2 != 0:
            decimated_waypoints.append(waypoints[i])
    # decimated_waypoints = decimate_waypoints(decimated_waypoints, threshold = 0.35, spread=1, max_gap = 4)
    # decimated_waypoints = waypoints
    for waypoint in decimated_waypoints:
        waypoint.optimum = 0.5
        waypoint.throttle = 0.1
    return decimated_waypoints


def has_intersected(car, line_a, line_b):
    body_points = [
        (car.pos[0] + car.cg_to_front, car.pos[1] - (car.width + car.wheel_width) / 2),
        (car.pos[0] + car.cg_to_front, car.pos[1] + (car.width + car.wheel_width) / 2),
        (car.pos[0] - car.cg_to_rear, car.pos[1] + (car.width + car.wheel_width) / 2),
        (car.pos[0] - car.cg_to_rear, car.pos[1] - (car.width + car.wheel_width) / 2)
    ]

    for i in range(len(body_points)):
        body_points[i] = geometry.rotate(body_points[i], car.heading, car.pos)

    car_boundary = [
        [body_points[0][0], body_points[0][1], body_points[3][0], body_points[3][1]],
        [body_points[1][0], body_points[1][1], body_points[2][0], body_points[2][1]],
    ]

    intersections_a = geometry.segment_intersections(car_boundary[0], [line_a]) + \
                    geometry.segment_intersections(car_boundary[1], [line_a])

    intersections_b = geometry.segment_intersections(car_boundary[0], [line_b]) + \
                    geometry.segment_intersections(car_boundary[1], [line_b])

    return len(intersections_a) > 0, len(intersections_b) > 0


def angle_difference(angle_a, angle_b):
    difference = angle_a - angle_b

    while difference < -math.pi:
        difference += (math.pi * 2)
    while difference > math.pi:
        difference -= (math.pi * 2)
    return difference


def render_scene(screen, screen_size, waypoints, best_points, start_var, end_var):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    var_lines = []
    for i in range(start_var, end_var):
        var_lines.append(waypoints[i].line)

    render_points = [

    ]

    render_lines = [
        ((200, 255, 200), 1, [w.line for w in waypoints]),
        ((0, 0, 200), 1, var_lines)
    ]

    for i in range(len(waypoints)):
        point = waypoints[i].get_optimum_point()
        throttle = (waypoints[i].throttle + 1) / 2

        c = (int((1 - throttle) * 255), int((throttle) * 255), 0)
        render_points.append((c, 2, [point]))

    for i in range(1, len(best_points)):
        line = best_points[i]["pos"] + best_points[i - 1]["pos"]
        throttle = (best_points[i]["thr"] + 1) / 2

        c = (int((1 - throttle) * 255), int((throttle) * 255), 0)
        render_lines.append((c, 4, [line]))

    render(
        screen,
        screen_size,
        lines=render_lines,
        # points=[
        #     ((255, 0, 255), 2, best_points)
        # ],
        points=render_points,
        padding=0
    )

    pygame.display.flip()


if __name__ == "__main__":
    OptimalPathCreator().run()
