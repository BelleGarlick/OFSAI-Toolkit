import time
import json
import copy
import math
import random
from typing import List

import numpy as np
import pygame

from fsai.objects.track import Track
from fsai.path_planning.waypoint import Waypoint
from fsai.visualisation.draw_pygame import render
from fsai.path_planning.waypoints import gen_waypoints, encode, decimate_waypoints
from fsai import geometry

selected_index = 2
track_names = ['autodormo_internacional_do_algarve', 'azure_circuit', 'brands_hatch', 'brno', 'cadwell_park', 'chester_field', 'circuit_de_barcelona', 'cota', 'daytona_rally', 'daytona_speedway', 'dirtfish', 'donington', 'dubai_autodrome', 'fuji', 'glencern', 'green_wood', 'hockenheimring', 'hockenheimring-classic', 'hockenheimring-rally', 'imola', 'knockhill', 'knockhill_rally', 'laguna_seca', 'lankebanen_rally', 'le_mans', 'le_mans_karting', 'loheac', 'long_beach_street', 'lydden_hill', 'merc_benz_ice', 'mojave', 'monza', 'nordschleife', 'nurburgring_gp', 'oschersleben', 'oulton_park', 'red_bull_ring', 'road_america', 'rouen_les_essarts', 'ruapuna_park', 'sampala_ice_circuit', 'silverstone', 'silverstone_class', 'snetterton', 'sonoma_raceway', 'spa', 'spa_historic', 'sportsland_sugo', 'summerton', 'watkins_glen', 'wildcrest', 'willow_springs', 'zhuhai', 'zolder']


DELTA_TIME = 0.01

RUNS_PER_SEGMENTS = 50
MAX_STEP_SIZE = 0.6
MIN_STEP_SIZE = 0.01
DELTA_STEP_SIZE = (MAX_STEP_SIZE - MIN_STEP_SIZE) / (RUNS_PER_SEGMENTS - 1)

SPREAD_DELTA = 1

MAX_SEGMENT_SIZE = 45
SEGMENT_SIZE_DELTA = 10
MIN_SEGMENT_SIZE = 5


def run():
    name = track_names[selected_index]
    track = Track("../examples/data/tracks/{}.json".format(track_names[selected_index]))
    initial_car = track.cars[0]

    initial_car.waypoint_index = -1
    initial_car.last_interval_update = 0

    left_boundary, right_boundary, orange_boundary = track.get_boundary()
    boundary = left_boundary + right_boundary + orange_boundary
    print(name)

    pygame.init()
    screen_size = [800, 400]
    screen = pygame.display.set_mode(screen_size)

    current_segment_size = MAX_SEGMENT_SIZE
    segment_index = 0

    best_waypoints = generate_waypoints(initial_car, left_boundary, right_boundary, orange_boundary)

    running = True
    first_best = False

    best_result = {
        "time": -1,
        "dis": 0,
        "pts": [],
        "car": initial_car
    }

    count = 0
    set_new_car = True

    while running:
        start_index, end_index = segment_index, segment_index + current_segment_size
        if end_index >= len(best_waypoints):
            start_index -= len(best_waypoints)
            end_index -= len(best_waypoints)
        render_scene(screen, screen_size, best_waypoints, best_result["pts"], start_index, end_index)

        for i in range(RUNS_PER_SEGMENTS):
            step_size = MAX_STEP_SIZE - (DELTA_STEP_SIZE * i)
            best_result, new_best, best_waypoints = genetic_test(best_result, best_waypoints, boundary, start_index, end_index, step_size, set_new_car)
            best_result["car"].physics.distance_travelled = 0
            if best_result["time"] != -1 and new_best:
                set_new_car = False

            if new_best:
                render_scene(screen, screen_size, best_waypoints, best_result["pts"], start_index, end_index)
                print("{}: {} {}".format(i, best_result["time"], best_result["dis"]))

        segment_index += current_segment_size // 2

        if segment_index >= len(best_waypoints):
            segment_index = 0
            # segment_offset = not segment_offset
            current_segment_size -= SEGMENT_SIZE_DELTA
            if current_segment_size < MIN_SEGMENT_SIZE:
                current_segment_size = MAX_SEGMENT_SIZE
            print("Segments: {}".format(current_segment_size))

            # if first_best:
            if count >= 0:
                # save_state(best_result, best_waypoints)
                set_new_car = True
                count = 0


def save_state(best_result, waypoints):
    dict = {
        "time": best_result["time"],
        "distance": best_result["dis"],
        "car": {
            "pos": best_result["car"].pos,
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
        } for waypoint in waypoints]
    }

    print(dict)


def genetic_test(best_values, waypoints: List[Waypoint], boundary, start_index, end_index, step_size, set_new_car: bool = False):
    new_best = False
    initial_waypoints = copy.deepcopy(waypoints)
    best_waypoints = copy.deepcopy(waypoints)

    waypoint_variations = variate_waypoint_fingerprint(initial_waypoints, step_size, start_index, end_index, best_values["time"] != -1)
    waypoint_variate = waypoint_variations["waypoints"]
    car_result = try_waypoints(waypoint_variate, best_values["car"], boundary)

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
                if car_result["car"] != None:
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


def try_waypoints(waypoints, initial_car, boundary):
    time, distance, tracked_positions, car = try_track(initial_car, waypoints, boundary)

    return {
        "time": time,
        "dis": distance,
        "pts": tracked_positions,
        "car": car
    }


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


def try_track(initial_car, waypoints, boundary):
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
        if has_intersected(car, waypoints[(car.waypoint_index + 1) % len(waypoints)].line):
            car.waypoint_index += 1
        elif has_intersected(car, waypoints[(car.waypoint_index + 2) % len(waypoints)].line):
            car.waypoint_index += 2
        if car.waypoint_index > len(waypoints): car.waypoint_index -= len(waypoints)

        throttle_control_index = car.waypoint_index + 1
        target_waypoint_index = car.waypoint_index + 2

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

        if lap_count < 2:
            speed_target_waypoint = waypoints[throttle_control_index % len(waypoints)]
            target_waypoint = waypoints[target_waypoint_index % len(waypoints)]
            target_point = target_waypoint.get_optimum_point()

            delta_angle = angle_difference(geometry.angle_to(car.pos, target_point), car.heading) / car.max_steer
            if delta_angle > 1: delta_angle = 1
            if delta_angle < -1: delta_angle = -1

            car.steer = delta_angle
            car.throttle, car.brake = max(0, speed_target_waypoint.throttle), -min(0, speed_target_waypoint.throttle)

            last_position = [car.pos[0], car.pos[1]]
            car.physics.update(DELTA_TIME)
            current_point = [car.pos[0], car.pos[1]]

            if ep_count % 10 == 0:
                point = [{"pos": current_point, "thr": speed_target_waypoint.throttle}]
                points += point
                if lap_count == 1:
                    timed_points += point

            if episode_time > 5 and sum(car.physics.distances_travelled) < 5:
                alive = False

            delta_line = current_point + last_position
            filtered_boundary = geometry.filter_lines_by_distance(current_point, 10, boundary)
            intersections = geometry.segment_intersections(delta_line, filtered_boundary)
            if len(intersections) > 0:
                alive = False

        ep_count += 1
        episode_time += DELTA_TIME
    distance = car.physics.distance_travelled
    return lap_time, 0 if distance < 2 else distance, points, lap_2_car if lap_count > 1 else None


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
        if i % 2 != 3:
            decimated_waypoints.append(waypoints[i])
    # decimated_waypoints = decimate_waypoints(decimated_waypoints, threshold = 0.35, spread=1, max_gap = 4)
    # decimated_waypoints = waypoints
    for waypoint in decimated_waypoints:
        waypoint.optimum = 0.5
        waypoint.throttle = 0.1
    return decimated_waypoints


def has_intersected(car, line):
    body_points = [
        (car.pos[0] + car.cg_to_front, car.pos[1] - (car.width + car.wheel_width) / 2),
        (car.pos[0] + car.cg_to_front, car.pos[1] + (car.width + car.wheel_width) / 2),
        (car.pos[0] - car.cg_to_rear, car.pos[1] + (car.width + car.wheel_width) / 2),
        (car.pos[0] - car.cg_to_rear, car.pos[1] - (car.width + car.wheel_width) / 2)
    ]

    for i in range(len(body_points)):
        body_points[i] = geometry.rotate(body_points[i], car.heading, car.pos)

    car_boundary = [
        [body_points[0][0], body_points[0][1], body_points[1][0], body_points[1][1]],
        [body_points[1][0], body_points[1][1], body_points[2][0], body_points[2][1]],
        [body_points[2][0], body_points[2][1], body_points[3][0], body_points[3][1]],
        [body_points[3][0], body_points[3][1], body_points[0][0], body_points[0][1]],
    ]

    intersections = geometry.segment_intersections(car_boundary[0], [line]) + \
                    geometry.segment_intersections(car_boundary[1], [line]) + \
                    geometry.segment_intersections(car_boundary[2], [line]) + \
                    geometry.segment_intersections(car_boundary[3], [line])

    return len(intersections) > 0


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
    run()