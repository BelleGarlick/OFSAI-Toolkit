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

selected_index = 1
track_names = ['autodormo_internacional_do_algarve', 'azure_circuit', 'brands_hatch', 'brno', 'cadwell_park', 'chester_field', 'circuit_de_barcelona', 'cota', 'daytona_rally', 'daytona_speedway', 'dirtfish', 'donington', 'dubai_autodrome', 'fuji', 'glencern', 'green_wood', 'hockenheimring', 'hockenheimring-classic', 'hockenheimring-rally', 'imola', 'knockhill', 'knockhill_rally', 'laguna_seca', 'lankebanen_rally', 'le_mans', 'le_mans_karting', 'loheac', 'long_beach_street', 'lydden_hill', 'merc_benz_ice', 'mojave', 'monza', 'nordschleife', 'nurburgring_gp', 'oschersleben', 'oulton_park', 'red_bull_ring', 'road_america', 'rouen_les_essarts', 'ruapuna_park', 'sampala_ice_circuit', 'silverstone', 'silverstone_class', 'snetterton', 'sonoma_raceway', 'spa', 'spa_historic', 'sportsland_sugo', 'summerton', 'watkins_glen', 'wildcrest', 'willow_springs', 'zhuhai', 'zolder']


VARIATION_UNITS = 7
DELTA_TIME = 0.05
SPREAD_DELTA = 0.95


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

    epoch = 0

    best_waypoints = generate_waypoints(initial_car, left_boundary, right_boundary, orange_boundary)
    first_best = False
    iterating_index = 0

    running = True

    best_time, best_distance, best_points = -1, 0, []
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        best_time, best_distance, best_points, best_car, new_best = optimise_waypoints_at_index(best_time, best_distance, best_points, best_waypoints, iterating_index, initial_car, boundary)
        if best_car is not None:
            if not first_best:
                print("Now using best car with momentum")
                best_time = -1
                best_distance = 0
                first_best = True
            initial_car = best_car
            initial_car.physics.distance_travelled = 0

        render_scene(screen, screen_size, best_waypoints, iterating_index, best_points)
        iterating_index += 1
        if iterating_index % len(best_waypoints) == 0:
            iterating_index = 0
        print("{} {}".format(best_time, best_distance))


def optimise_waypoints_at_index(best_time, best_distance, best_points, waypoints: List[Waypoint], index, initial_car, boundary):
    current_index_position, current_index_velocity = waypoints[index].optimum, waypoints[index].throttle

    position_low, position_mid, position_high = 0.25, 0.5, 0.75
    throttle_low, throttle_mid, throttle_high = -0.5, 0, 0.5
    pos_throttle = None
    for iteration in range(10):
        pos_thrott_m = test_waypoints_at_index(waypoints, index, initial_car, boundary, [position_low, position_mid, position_high], [throttle_low, throttle_mid, throttle_high])

        pos_throttle = get_best_position_throttle(pos_thrott_m)

        # TODO This can be optimised since we know what the next bounds will be to an extent.
        # Get new position bounds
        half_pos = max(position_high - position_mid, position_mid - position_low) / 2
        if pos_throttle["pos"] == position_mid:
            position_low = position_mid - half_pos
            position_high = position_mid + half_pos
        elif pos_throttle["pos"] == position_low:
            position_low = position_low - half_pos
            position_mid = position_low
            position_high = position_low + half_pos
        elif pos_throttle["pos"] == position_high:
            position_low = position_high - half_pos
            position_mid = position_high
            position_high = position_high + half_pos
        else:
            position_low = position_mid - half_pos
            position_high = position_mid + half_pos

        # Get new position bounds
        half_throt = max(throttle_high - throttle_mid, throttle_mid - throttle_low) / 2
        if pos_throttle["thr"] == throttle_mid:
            throttle_low = position_mid - half_throt
            throttle_high = position_mid + half_throt
        elif pos_throttle["thr"] == throttle_low:
            throttle_low = throttle_low - half_throt
            throttle_mid = throttle_low
            throttle_high = throttle_low + half_throt
        elif pos_throttle["thr"] == position_high:
            throttle_low = throttle_high - half_throt
            throttle_mid = throttle_high
            throttle_high = throttle_high + half_throt
        else:
            throttle_low = throttle_mid - half_throt
            throttle_high = throttle_mid + half_throt

        position_low = min(1, max(0, position_low))
        position_mid = min(1, max(0, position_mid))
        position_high = min(1, max(0, position_high))
        throttle_low = min(1, max(-1, throttle_low))
        throttle_mid = min(1, max(-1, throttle_mid))
        throttle_high = min(1, max(-1, throttle_high))

    waypoints[index].optimum = current_index_position
    waypoints[index].throttle = current_index_velocity
    best_car = None
    new_best = False

    if pos_throttle["tim"] != -1:
        if best_time == -1 or pos_throttle["tim"] < best_time:
            print("New best as has a better time")
            waypoints[index].optimum = pos_throttle["pos"]
            waypoints[index].throttle = pos_throttle["thr"]
            best_time = pos_throttle["tim"]
            best_distance = pos_throttle["dis"]
            best_points = pos_throttle["pts"]
            best_car = pos_throttle["car"]
            new_best = True
        elif time == best_time and pos_throttle["dis"] > best_distance:
            waypoints[index].optimum = pos_throttle["pos"]
            waypoints[index].throttle = pos_throttle["thr"]
            best_time = pos_throttle["tim"]
            best_distance = pos_throttle["dis"]
            best_points = pos_throttle["pts"]
            best_car = pos_throttle["car"]
            new_best = True
            print("New best as has same time but better distance")
    else:
        # if the car did not put in a time and there is no best time
        if best_time == -1:
            if pos_throttle["dis"] > best_distance:
                waypoints[index].optimum = pos_throttle["pos"]
                waypoints[index].throttle = pos_throttle["thr"]
                best_time = pos_throttle["tim"]
                best_distance = pos_throttle["dis"]
                best_points = pos_throttle["pts"]
                best_car = pos_throttle["car"]
                new_best = True
                print("new best distance")

    return best_time, best_distance, best_points, best_car, new_best


def genetic_test(best_time, best_distance, best_points, waypoints: List[Waypoint], index, initial_car, boundary):
    current_index_position, current_index_velocity = waypoints[index].optimum, waypoints[index].throttle

    best_car = None
    new_best = False
    initial_waypoints = copy.deepcopy(waypoints)
    best_waypoints = copy.deepcopy(waypoints)
    for i in range(500):
        waypoint_variate = variate_waypoint_fingerprint(initial_waypoints, 0.1)
        car_result = test_waypoints(waypoint_variate, initial_car, boundary)

        if car_result["tim"] != -1:
            if best_time == -1 or car_result["tim"] < best_time:
                print("New best as has a better time")
                best_waypoints = copy.deepcopy(waypoint_variate)
                best_time = car_result["tim"]
                best_distance = car_result["dis"]
                best_points = car_result["pts"]
                best_car = car_result["car"]
                new_best = True
            elif time == best_time and car_result["dis"] > best_distance:
                best_waypoints = copy.deepcopy(waypoint_variate)
                best_time = car_result["tim"]
                best_distance = car_result["dis"]
                best_points = car_result["pts"]
                best_car = car_result["car"]
                new_best = True
                print("New best as has same time but better distance")
        else:
            # if the car did not put in a time and there is no best time
            if best_time == -1:
                if car_result["dis"] > best_distance:
                    best_waypoints = copy.deepcopy(waypoint_variate)
                    best_time = car_result["tim"]
                    best_distance = car_result["dis"]
                    best_points = car_result["pts"]
                    best_car = car_result["car"]
                    new_best = True
                    print("new best distance")

    return best_time, best_distance, best_points, best_car, new_best, best_waypoints


def variate_waypoint_fingerprint(waypoints, step_size):
    for waypoint in waypoints:
        waypoint.throttle += (random.random() * 2 - 1) * step_size
        waypoint.throttle = max(0, min(1, waypoint.throttle))
        waypoint.optimum += (random.random() * 2 - 1) * step_size
        waypoint.optimum = max(0, min(1, waypoint.throttle))
    return waypoints


def test_waypoints_at_index(waypoints, index, initial_car, boundary, positions, throttles):
    throttle_position_matrix = []
    for position in positions:
        for throttle in throttles:
            waypoints[index].optimum = position
            waypoints[index].throttle = throttle

            time, distance, tracked_positions, car = test_track(initial_car, waypoints, boundary)

            throttle_position_matrix.append({
                "pos": position,
                "thr": throttle,
                "tim": time,
                "dis": distance,
                "pts": tracked_positions,
                "car": car
            })
    return throttle_position_matrix


def test_waypoints(waypoints, initial_car, boundary):
    time, distance, tracked_positions, car = test_track(initial_car, waypoints, boundary)

    return {
        "tim": time,
        "dis": distance,
        "pts": tracked_positions,
        "car": car
    }


def get_best_position_throttle(position_throttle_matrix):
    best = position_throttle_matrix[0]
    for i in range(1, len(position_throttle_matrix)):
        current = position_throttle_matrix[i]

        if current["tim"] != -1:
            if best["tim"] == -1 or current["tim"] < best["tim"]:
                best = current
            elif time == best["tim"] and current["dis"] > best["dis"]:
                best = current
        else:
            # if the car did not put in a time and there is no best time
            if best["tim"] == -1:
                if current["dis"] > best["dis"]:
                    best = current
    return best


def variate_waypoints(waypoints, index=0, spread=0.1):
    spread_delta = spread / (VARIATION_UNITS - 1)
    speed_spread_delta = spread / ((VARIATION_UNITS - 1) * 2)
    all_waypoints = []
    # print("======== {}".format(spread_delta))
    for optimum_waypoint in range(VARIATION_UNITS):
        current_optimum = (spread_delta * optimum_waypoint) - (spread / 2)
        # print("{} {}".format(optimum_waypoint, spread_delta * optimum_waypoint))

        # print("======")
        for speed_variation in range(VARIATION_UNITS * 2 - 1):
            copied_waypoints = copy.deepcopy(waypoints)
            copied_waypoints[index].optimum += current_optimum
            copied_waypoints[index].optimum = max(0, min(1, copied_waypoints[index].optimum))

            # print("{} {} {}".format(speed_variation, spread, (spread_delta * 2 * speed_variation) - (spread)))
            copied_waypoints[index].throttle += (spread_delta * speed_variation) - (speed_spread_delta / 2)
            copied_waypoints[index].throttle = max(-1, min(1, copied_waypoints[index].optimum))

            all_waypoints.append(copied_waypoints)
    return all_waypoints


def test_track(initial_car, waypoints, boundary):
    car = copy.deepcopy(initial_car)
    episode_time = 0

    lap_time = -1
    points = []

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
            lap_time = episode_time
            alive = False
            car.waypoint_index = 0
            completed_lap = True
        else:
            speed_target_waypoint = waypoints[throttle_control_index % len(waypoints)]
            target_waypoint = waypoints[target_waypoint_index % len(waypoints)]
            target_point = target_waypoint.get_optimum_point()

            delta_angle = angle_difference(geometry.angle_to(car.pos, target_point), car.heading)
            delta_angle /= car.max_steer
            if delta_angle > 1: delta_angle = 1
            if delta_angle < -1: delta_angle = -1

            car.steer = delta_angle

            car.throttle = max(0, speed_target_waypoint.throttle)
            car.brake = -min(0, speed_target_waypoint.throttle)

            last_position = [car.pos[0], car.pos[1]]
            car.physics.update(DELTA_TIME)
            current_point = [car.pos[0], car.pos[1]]

            if ep_count % 4 == 0:
                points += [current_point]

            if len(car.physics.distances_travelled) > 50 and sum(car.physics.distances_travelled) < 5:
                alive = False

            delta_line = current_point + last_position
            filtered_boundary = geometry.filter_lines_by_distance(current_point, 10, boundary)
            intersections = geometry.segment_intersections(delta_line, filtered_boundary)
            if len(intersections) > 0:
                alive = False

        ep_count += 1
        episode_time += DELTA_TIME
    return lap_time, car.physics.distance_travelled, points, car if completed_lap else None


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
        if i % 2 == 0:
            decimated_waypoints.append(waypoints[i])
    decimated_waypoints = decimate_waypoints(decimated_waypoints, threshold = 0.35, spread=1, max_gap = 4)

    for waypoint in decimated_waypoints:
        waypoint.optimum = 0.5
        waypoint.throttle = 0.2
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


def render_scene(screen, screen_size, waypoint, iterating, best_points):
    render_lines = [
        ((200, 255, 200), 1, [w.line for w in waypoint]),
        ((0, 0, 255), 3, [waypoint[iterating].line]),
    ]

    for i in range(len(waypoint)):
        line = waypoint[i].get_optimum_point() + (waypoint[i - 1].get_optimum_point())
        throttle = (waypoint[i].throttle + 1) / 2

        c = (int((1 - throttle) * 255), int((throttle) * 255), 0)
        render_lines.append((c, 5, [line]))

    render(
        screen,
        screen_size,
        lines=render_lines,
        points=[
            ((255, 0, 255), 2, best_points)
        ],
        padding=0
    )

    pygame.display.flip()


if __name__ == "__main__":
    run()