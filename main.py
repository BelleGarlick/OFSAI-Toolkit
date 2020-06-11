import math
import time

import cv2

from fsai.mapping.boundary_estimation import get_delaunay_triangles
from fsai.objects.track import Track
from fsai.path_planning.waypoint import Waypoint
from fsai.path_planning.waypoints import gen_waypoints, encode
from fsai.visualisation.draw_opencv import render, render_area
from fsai import geometry
from optimalTrackTester.geneticTestUtils import get_track_time


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

    for waypoint in waypoints:
        waypoint.optimum = 0.5 #random.random()
    return waypoints


if __name__ == "__main__":
    track = Track("examples/data/tracks/azure_circuit.json")
    initial_car = track.cars[0]
    left_boundary, right_boundary, o = track.get_boundary()
    waypoints = generate_waypoints(initial_car, left_boundary, right_boundary, o)

    time = get_track_time(waypoints, 1.7 * 9.81 * 0.74, 104.1667)

    waypoint_lines = []
    for w_index in range(len(waypoints)):
        cw = waypoints[w_index].get_optimum_point()
        nw = waypoints[w_index + 1 - len(waypoints)].get_optimum_point()

        v = 0.5
        try:
            v = waypoints[w_index].v / 26
        except:
            pass

        r = min(255, 510 - int(510 * v))
        g = min(255, 0 + int(510 * v))
        c = (g, r, 0)

        waypoint_lines.append((c, 2, [cw + nw]))

    print(waypoint_lines)
    print(time)
    image = render(
        [1000, 500],
        lines=waypoint_lines,
        padding=10,
        background=0
    )
    cv2.imshow("", image/255)
    cv2.waitKey(0)


