import math

import cv2

from fsai.objects.track import Track
from fsai.path_planning.waypoints import gen_local_waypoints, encode
from fsai.visualisation.draw_opencv import render, render_area

track = Track("examples/data/tracks/azure_circuit.json")
left_boundary, right_boundary, o = track.get_boundary()
car = track.cars[0]
car.steer = 0
car.pos.x += 48
car.pos.y += 15

waypoints = gen_local_waypoints(
        car_pos=car.pos,
        car_angle=car.heading,
        blue_boundary=left_boundary,
        yellow_boundary=right_boundary,
        orange_boundary=o,
        foresight=5,
        spacing=1.5,
        negative_foresight=4,
        smooth=True
)
encoding = encode(waypoints, 4)

cv2.imshow("", render(
    [1200, 900],
    lines=[
        ((255, 0, 0), 2, left_boundary),
        ((0, 255, 0), 2, right_boundary),
        ((0, 0, 255), 2, [waypoint.line for waypoint in waypoints])
    ],
    cars=[car]
) / 256)

cv2.waitKey(0)
cv2.imshow("", render_area(
    camera_pos=car.pos,
    rotation=-car.heading - math.pi/2,
    area=(20, 20),
    resolution=20,
    lines=[
        ((255, 0, 0), 2, left_boundary),
        ((0, 255, 0), 2, right_boundary),
        ((0, 0, 255), 2, [waypoint.line for waypoint in waypoints])
    ],
    cones=[
        ((255, 0, 0), 10, track.blue_cones),
        ((0, 255, 0), 10, track.yellow_cones),
    ],
    cars=[car]
) / 256)
cv2.waitKey(0)
