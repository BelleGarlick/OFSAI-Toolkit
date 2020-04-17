import math
import time

import cv2

from fsai.objects.track import Track
from fsai.path_planning.waypoint import Waypoint
from fsai.path_planning.waypoints import gen_waypoints, encode
from fsai.visualisation.draw_opencv import render
from fsai import geometry

if __name__ == "__main__":
    # track = Track("examples/data/tracks/azure_circuit.json")
    # left_boundary, right_boundary, o = track.get_boundary()
    # initial_car = track.cars[0]
    # initial_car.heading -= 0.5
    #
    # waypoints = gen_waypoints(
    #     car_pos=track.cars[0].pos,
    #     car_angle=track.cars[0].heading,
    #     blue_boundary=left_boundary,
    #     yellow_boundary=right_boundary,
    #     orange_boundary=o,
    #     foresight=10,
    #     spacing=1.5,
    #     negative_foresight=10,
    #     radar_length=12,
    #     radar_count=13,
    #     radar_span=math.pi / 1.2,
    #     margin=0,
    #     smooth=True
    # )
    # encoding = encode(waypoints, 10)
    # print(encoding)


    waypoint = Waypoint(line=[0, 0, 0, 16])
    p = waypoint.find_optimum_from_point([8, 2])
    print(p)

    image = render(
        [1000, 1000],
        lines=[
            ((0, 255, 0), 2, [waypoint.line]),
        ],
        padding=10,
        background=0
    )
    cv2.imshow("", image/255)
    cv2.waitKey(0)

