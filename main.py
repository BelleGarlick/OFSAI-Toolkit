import math
import time

import cv2

from fsai.mapping.boundary_estimation import get_delaunay_triangles
from fsai.objects.track import Track
from fsai.path_planning.waypoint import Waypoint
from fsai.path_planning.waypoints import gen_waypoints, encode
from fsai.visualisation.draw_opencv import render, render_area
from fsai import geometry

negative_waypoints = 0

def segment_intersections(line, segments):
    min_x_seg = min(line[0], line[2])
    min_y_seg = min(line[1], line[3])
    max_x_seg = max(line[0], line[2])
    max_y_seg = max(line[1], line[3])

    # convert fixed lines into y=mx+c ordinates
    a1 = line[3] - line[1]
    b1 = line[0] - line[2]
    c1 = a1 * line[0] + b1 * line[1]

    intersections = []
    delta = 0
    precision = 1000
    for index in range(len(segments)):
        seg_i = segments[index]
        seg_0 = seg_i[0]
        seg_1 = seg_i[1]
        seg_2 = seg_i[2]
        seg_3 = seg_i[3]

        a2 = seg_3 - seg_1
        b2 = seg_0 - seg_2
        c2 = a2 * seg_0 + b2 * seg_1
        delta = a1 * b2 - a2 * b1
        print(a2)
        print(b2)
        print(c2)
        print(delta)

        if delta != 0:
            x = (b2 * c1 - b1 * c2) / delta
            y = (a1 * c2 - a2 * c1) / delta
            print(x)
            print(y)

            # check the point exists within the
            min_x = round(precision * max(min(seg_0, seg_2), min_x_seg))
            min_y = round(precision * max(min(seg_1, seg_3), min_y_seg))
            max_x = round(precision * min(max(seg_0, seg_2), max_x_seg))
            max_y = round(precision * min(max(seg_1, seg_3), max_y_seg))

            if min_x <= round(precision * x) <= max_x and min_y <= round(precision * y) <= max_y:
                intersections.append([x, y])
    return intersections


if __name__ == "__main__":
    track = Track("examples/data/tracks/imola.json")
    initial_car = track.cars[0]
    left_boundary, right_boundary, o = track.get_boundary()

    polygons = get_delaunay_triangles(track.blue_cones, track.yellow_cones, track.orange_cones, track.big_cones)

    print(polygons)

    image = render(
        [1000, 1000],
        polygons=[
            ((255, 255, 255), (0, 0, 0), 0, polygons)
        ],
        lines=[
            ((255, 0, 0), 2, left_boundary),
            ((0, 255, 255), 2, right_boundary),
        ],
        cars=track.cars,
        padding=10,
        background=0
    )
    cv2.imshow("", image/255)
    cv2.waitKey(0)


