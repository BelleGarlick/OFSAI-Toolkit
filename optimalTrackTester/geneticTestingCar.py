import copy
import json
import math
import random
import time

import numpy as np
import pygame

from fsai.objects.track import Track
from fsai.visualisation.draw_pygame import render
from fsai.path_planning.waypoints import gen_waypoints, encode
from fsai import geometry

CAR_COUNT = 500
FORESIGHT = 10
FORSIGHT_SPACING = 3
ANGLE_NORMALISATION = 1.5

STEP_SIZE_INITIAL = 0.3
STEP_SIZE_DELTA = 0.95

MAX_ANGLE_DISPLACEMENT = 0.9
MAX_CAR_SPEED = 58
MAX_CAR_ACCEL = 28

SPEED_TRACE_COLOR_DISPARITY = 0.05

# forsight + brake + throttle + steer + vel + accel
NETWORK_INPUT = FORESIGHT + 5
NETWORK_LAYER_SIZES = [20, 20, 3]


track_names = [
    "COTA",
    "hockenheimring",
    "nurburgring_gp",
    "sampala_ice_circuit",
    "sonoma_raceway",
    "spa",
    "le_mans",
    "snetterton",
    "zhuhai",
    "laguna_seca",
    "sportsland_sugo",
    "summerton",
    "dubai_autodrome",
    "BRNO",
    "daytona_rally",
    "daytona_speedway",
    "donington",
    "glencern",
    "hockenheimring-classic",
    "knockhill_rally",
    "lankebanen_rally",
    "le_mans_karting",
    "lydden_hill",
    "azure_circuit",
    "merc_benz_ice",
    "circuit_de_barcelona",
    "oschersleben",
    "silverstone",
    "knockhill",
    "brands_hatch",
    "cadwell_park",
    "dirtfish",
    "imola",
    "loheac",
    "monza",
    "red_bull_ring",
    "road_america",
    "silverstone_class",
    "wildcrest",
    "willow_springs",
    "autodormo_internacional_do_algarve",
    "oulton_park",
    "rouen_les_essarts",
    "zolder",
    "fuji",
    "green_wood",
    "hockenheimring-rally",
    "long_beach_street",
    "mojave",
    "watkins_glen",
    "nordschleife",
    "chester_field",
    "ruapuna_park",
    "spa_historic",
]

tracks = [{"name": name, "time": -1, "distance": 0, "reversed": False, "speed_trace": {"distance": [], "accel": [], "coast": [], "brake": []}} for name in track_names]
tracks += [{"name": name, "time": -1, "distance": 0, "reversed": True, "speed_trace": {"distance": [], "accel": [], "coast": [], "brake": []}} for name in track_names]

# with open("results.json", "r") as file:
#     json_data = json.loads(file.read())
#
# tracks = [track for track in json_data["tracks"] if track["time"] != -1]

def generate_cars(initial_car, car_count, step_size):
    cars = [copy.deepcopy(initial_car) for i in range(car_count)]
    for car in cars:
        car.waypoint_index = -1
        car.start_lap = -1
        car.end_lap = -1
        car.alive = True

        car.lap_count = 0
        car.start_lap = 0
        car.lap_time = -1
        car.in_lap_marker = True

        car.timed_speed_trace = {
            "distance": [],
            "accel": [],
            "coast": [],
            "brake": []
        }

        car.weights = []
        input_layer_size = NETWORK_INPUT + 1
        for i in range(len(NETWORK_LAYER_SIZES)):
            car.weights.append(
                initial_car.weights[i] + (np.random.rand(input_layer_size, NETWORK_LAYER_SIZES[i]) * 2 - 1) * step_size
            )
            input_layer_size = NETWORK_LAYER_SIZES[i] + 1
        car.weights = np.array(car.weights)

    return cars


def run():
    pygame.init()
    screen_size = [800, 400]
    screen = pygame.display.set_mode(screen_size)

    step_size = STEP_SIZE_INITIAL

    current_track_index = 0
    lines, initial_car = get_track(tracks[current_track_index]["name"])
    cars = generate_cars(initial_car, CAR_COUNT, step_size)

    last_time = time.time()
    running = True
    episode_length = 0
    episode_count = 1

    # Bests
    best_weights = initial_car.weights

    epoch = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        now = time.time()
        dt = (now - last_time) / 5
        dt = 0.05

        episode_length += dt

        p = []

        all_dead = True
        for i in range(CAR_COUNT):
            car = cars[i]
            if car.alive:
                all_dead = False

                target_points, encoding = get_target_points(car, lines, count=FORESIGHT, spacing=FORSIGHT_SPACING)

                if FORESIGHT == len(target_points):
                    p += target_points
                    if car.physics.abs_vel > MAX_CAR_SPEED:
                        print("New max car speed: {} ".format(car.physics.abs_vel))
                    if car.physics.abs_accel > MAX_CAR_ACCEL:
                        print("New max car accel: {} ".format(car.physics.abs_accel))
                    output = feed_network(car, np.asarray(
                        encoding + [car.brake, car.steer, car.throttle, min(1, sqrt(car.physics.abs_vel / MAX_CAR_SPEED)), min(1, sqrt(car.physics.abs_accel / MAX_CAR_ACCEL))]
                    ))

                    car.steer = output[0] * 2 - 1
                    car.throttle = output[1]
                    car.brake = output[2]
                    car.physics.update(dt)

                    if episode_length * 8 > len(car.timed_speed_trace["distance"]) + len(car.timed_speed_trace["accel"]) + len(car.timed_speed_trace["coast"]) + len(car.timed_speed_trace["brake"]):
                        if car.lap_count == 1:
                            if car.throttle > car.brake + SPEED_TRACE_COLOR_DISPARITY:
                                car.timed_speed_trace["accel"].append(car.pos)
                            elif car.brake > car.throttle + SPEED_TRACE_COLOR_DISPARITY:
                                car.timed_speed_trace["brake"].append(car.pos)
                            else:
                                car.timed_speed_trace["coast"].append(car.pos)
                        else:
                            car.timed_speed_trace["distance"].append(car.pos)

                    if geometry.distance(car.pos, initial_car.pos) < FORSIGHT_SPACING:
                        if not car.in_lap_marker:
                            car.lap_count += 1
                            car.in_lap_marker = True

                            if car.lap_count == 1:
                                car.start_lap = episode_length
                            if car.lap_count == 2:
                                car.lap_time = episode_length - car.start_lap
                                car.alive = False
                    else:
                        car.in_lap_marker = False

                    if sum(car.physics.distances_travelled) < 0.5 and episode_length > 4:
                        car.alive = False
                else:
                    car.alive = False

        if all_dead:
            current_track = tracks[current_track_index]
            track_name = current_track["name"]
            track_reversed = current_track["reversed"]

            new_best = False
            for i in range(1, len(cars)):
                if cars[i].lap_time > 0:
                    if tracks[current_track_index]["time"] == -1 or cars[i].lap_time < tracks[current_track_index]["time"]:
                        new_best = True
                        best_weights = cars[i].weights
                        tracks[current_track_index]["time"] = cars[i].lap_time
                        tracks[current_track_index]["distance"] = 0
                        tracks[current_track_index]["speed_trace"] = cars[i].timed_speed_trace
                        tracks[current_track_index]["speed_trace"]["distance"] = []

                else:
                    # if there is no time and the car travelled further than the current distance
                    if tracks[current_track_index]["time"] == -1 and cars[i].physics.distance_travelled > tracks[current_track_index]["distance"]:
                        new_best = True
                        best_weights = cars[i].weights
                        tracks[current_track_index]["distance"] = cars[i].physics.distance_travelled
                        tracks[current_track_index]["speed_trace"] = cars[i].timed_speed_trace

            track_distance = current_track["distance"]
            track_time = current_track["time"]

            print_episode_summary(epoch, episode_count, track_name, track_reversed, track_time, track_distance, new_best, step_size)

            current_track_index += 1
            if current_track_index >= len(tracks):
                current_track_index = 0
                epoch += 1
                episode_count = 0
                step_size *= STEP_SIZE_DELTA

                with open("results.json", "w+") as file:
                    file.write(json.dumps({"tracks": tracks, "step_size":  step_size, "epoch": epoch, "episode": episode_count}))

            lines, initial_car = get_track(tracks[current_track_index]["name"])
            if tracks[current_track_index]["reversed"]:
                initial_car.heading += math.pi

            initial_car.weights = best_weights
            np.save("best_weights", best_weights)
            cars = generate_cars(initial_car, CAR_COUNT, step_size)

            episode_count += 1
            episode_length = 0

        render(
            screen,
            screen_size,
            lines=[
                ((100, 100, 100), 2, lines)
            ],
            points=[
                ((225, 255, 255), 4, p),
                ((100, 0, 100), 4, tracks[current_track_index]["speed_trace"]["distance"]),
                ((0, 255, 0), 4, tracks[current_track_index]["speed_trace"]["accel"]),
                ((255, 255, 0), 4, tracks[current_track_index]["speed_trace"]["coast"]),
                ((255, 0, 0), 4, tracks[current_track_index]["speed_trace"]["brake"]),
            ],
            cars=[c for c in cars if c.alive],
            padding=10
        )

        pygame.display.flip()
        last_time = now


def get_track(file_name):
    track = Track("../examples/data/tracks/{}.json".format(file_name))
    initial_car = track.cars[0]
    initial_car.weights = []
    last_layer_size = NETWORK_INPUT + 1
    for i in range(len(NETWORK_LAYER_SIZES)):
        initial_car.weights.append(
            (np.random.rand(last_layer_size, NETWORK_LAYER_SIZES[i]) * 2 - 1)
        )
        last_layer_size = NETWORK_LAYER_SIZES[i] + 1
    initial_car.weights = np.array(initial_car.weights)
    # initial_car.weights = np.load("best_weights.npy", allow_pickle=True)
    left_boundary, right_boundary, o = track.get_boundary()

    lines = generate_target_line(initial_car, left_boundary, right_boundary, o)

    return lines, initial_car


def print_episode_summary(epoch, episode, track_name, track_reversed, track_time, track_distance, new_best, step_size):
    if new_best:
        string = "Epoch: {} Episode: {} Step Size: {} {}{}: ".format(epoch, episode, '%.3f' % step_size, track_name, "-reversed" if track_reversed else "")
        if track_time == -1:
            string += "{}m".format('%.3f' % track_distance)
        else:
            string += "{}s".format('%.3f' % track_time)

        print(string)


def generate_target_line(initial_car, left_boundary, right_boundary, orange_boundary):
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

    points = []
    for waypoint in waypoints:
        waypoint.optimum = 0.5
        points.append(waypoint.get_optimum_point())

    lines = []
    for i in range(len(points)):
        lines.append([points[i - 1][0], points[i - 1][1], points[i][0], points[i][1]])

    return lines


def feed_network(car, input):
    output = input
    for i in range(len(car.weights)):
        output = np.dot(
            np.hstack((output, 1)),
            car.weights[i]
        )
        output = 1 / (1 + np.exp(-output))  # activate
    return output


def modify_waypoints(all_waypoints, quickest_index, step_size):
    for waypoints in all_waypoints:
        for i in range(len(waypoints)):
            waypoints[i].optimum = all_waypoints[quickest_index][i].optimum + (random.random() * 2 - 1) * step_size
            if waypoints[i].optimum < 0:
                waypoints[i].optimum = 0
            if waypoints[i].optimum > 1:
                waypoints[i].optimum = 1
    return all_waypoints


def get_target_points(car, lines, count=10, spacing=2):
    intersections = []
    encoding = []

    last_point = car.pos
    last_angle = car.heading

    for _ in range(count):
        t_intersections = geometry.circle_line_intersections(last_point, spacing, lines)
        for p in t_intersections:
            angle_to = geometry.angle_to(last_point, p)
            angle = angle_difference(angle_to, last_angle)
            if abs(angle) < math.pi / 2:
                if abs(angle) > MAX_ANGLE_DISPLACEMENT and len(encoding) == 0:
                    return [], []
                intersections += [p]
                last_point = p
                last_angle = angle_to

                encoding.append(min(1.0, sqrt(angle / ANGLE_NORMALISATION)))
                continue

    return intersections, encoding


def angle_difference(angle_a, angle_b):
    difference = angle_a - angle_b

    while difference < -math.pi:
        difference += (math.pi * 2)
    while difference > math.pi:
        difference -= (math.pi * 2)
    return difference


def sqrt(x: float) -> float:
    rt = math.sqrt(abs(x))
    if x < 0:
        return -rt
    return rt


if __name__ == "__main__":
    run()
