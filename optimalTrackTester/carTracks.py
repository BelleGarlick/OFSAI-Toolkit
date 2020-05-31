import copy
import math
import random
import time

import cv2
import numpy as np

from fsai.objects.track import Track
from fsai.visualisation.draw_opencv import render
from fsai.path_planning.waypoints import gen_waypoints, encode

CAR_COUNT = 500
FORESIGHT = 10
FORSIGHT_SPACING = 3
ANGLE_NORMALISATION = 1.5

STEP_SIZE_INITIAL = 0.3
STEP_SIZE_DELTA = 0.995

MAX_ANGLE_DISPLACEMENT = 0.9
MAX_CAR_SPEED = 53
MAX_CAR_ACCEL = 23

SPEED_TRACE_COLOR_DISPARITY = 0.05

# forsight + brake + throttle + steer + vel + accel
NETWORK_INPUT = FORESIGHT + 5
NETWORK_LAYER_SIZES = [40, 40, 3]

tracks = [
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


def run():
    for track_name in tracks:
        print(track_name)
        track = Track("../examples/data/tracks/{}.json".format(track_name))
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

        lines, waypoints = generate_target_line(initial_car, left_boundary, right_boundary, o)

        image = render(
            [800, 600],
            lines=[
                ((100, 100, 100), 2, lines),
                # ((255, 255, 255), 2, [w.line for w in waypoints])
            ],
            points=[
                ((255, 0, 0), 4, track.blue_cones),
                ((0, 255, 255), 4, track.yellow_cones),
            ],
            cars=[initial_car],
            padding=10
        )
        cv2.imshow("", image)
        cv2.waitKey(0)




def print_episode_summary(track_name, track_reversed, track_time, track_distance, new_best, step_size):
    if new_best:
        string = "{}{}: ".format(track_name, "-reversed" if track_reversed else "")
        if track_time == -1:
            string += "{}m".format(track_distance)
        else:
            string += "{}s".format(track_time)

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

    return lines, waypoints


# def get_target_points(car, lines, count=10, spacing=2):
#     intersections = []
#     encoding = []
#
#     last_point = car.pos
#     last_angle = car.heading
#
#     for _ in range(count):
#         t_intersections = geometry.circle_line_intersections(last_point, spacing, lines)
#         for p in t_intersections:
#             angle_to = geometry.angle_to(last_point, p)
#             angle = angle_difference(angle_to, last_angle)
#             if abs(angle) < math.pi / 2:
#                 if abs(angle) > MAX_ANGLE_DISPLACEMENT and len(encoding) == 0:
#                     return [], []
#                 intersections += [p]
#                 last_point = p
#                 last_angle = angle_to
#
#                 encoding.append(min(1.0, sqrt(angle / ANGLE_NORMALISATION)))
#                 continue
#
#     return intersections, encoding



if __name__ == "__main__":
    run()
