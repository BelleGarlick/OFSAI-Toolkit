import math
import time

import cv2

from fsai.objects.track import Track
from fsai.path_planning.waypoint import Waypoint
from fsai.path_planning.waypoints import gen_waypoints, encode
from fsai.visualisation.draw_opencv import render
from fsai import geometry

negative_waypoints = 1

if __name__ == "__main__":
    track = Track("examples/data/tracks/azure_circuit.json")
    left_boundary, right_boundary, o = track.get_boundary()
    initial_car = track.cars[0]
    initial_car.pos[0] += 20
    initial_car.pos[1] -= 3
    initial_car.heading -= 0.1

    waypoints, l = gen_waypoints(
        car_pos=track.cars[0].pos,
        car_angle=track.cars[0].heading,
        blue_boundary=left_boundary,
        yellow_boundary=right_boundary,
        orange_boundary=o,
        foresight=40,
        spacing=2,
        negative_foresight=negative_waypoints,
        radar_length=12,
        radar_count=13,
        radar_span=math.pi / 1.2,
        margin=0,
        smooth=True
    )
    encoding = encode(waypoints, negative_waypoints)


    ## reconstruct waypoitns to check if it's valid
    reconstructed_lines = []
    # center
    center_enc = encoding[negative_waypoints]
    line = [0, -center_enc[0]/2, 0, center_enc[0]/2]
    reconstructed_lines.append(line)

    current_line_center = [0, 0]
    current_line_angle = 0

    points = []
    for i in range(negative_waypoints + 1, len(encoding)):
        line = encoding[i]
        prev_line = encoding[i-1]

        line_center = [
            current_line_center[0] + line[3] * math.cos(line[1] + current_line_angle),
            current_line_center[1] + line[3] * math.sin(line[1] + current_line_angle)
        ]
        points += [line_center]
        current_line_center = line_center


    image = render(
        [1000, 1000],
        lines=[
            ((0, 255, 0), 2, [waypoint.line for waypoint in waypoints]),
            ((255, 0, 0), 2, left_boundary),
            ((0, 255, 255), 2, right_boundary),
            ((255, 255, 0), 2, reconstructed_lines)
        ],
        points = [
            ((200, 200, 200), 3, points)
        ],
        cars=track.cars,
        padding=10,
        background=0
    )
    cv2.imshow("", image/255)
    cv2.waitKey(0)


