import math

import cv2

from fsai.objects.line import Line
from fsai.objects.point import Point
from fsai.path_planning.waypoints import gen_local_waypoints
from fsai.visualisation.track_2d import draw_track

# custom split in the track
blue_lines = [
    Line(a=Point(-2, 1), b=Point(-2, -5)),
    Line(a=Point(-2, -5), b=Point(-6, -9)),
    Line(a=Point(2, -10), b=Point(6, -14)),
]
yellow_lines = [
    Line(a=Point(2, 1), b=Point(2, -5)),
    Line(a=Point(2, -5), b=Point(6, -9)),
    Line(a=Point(-2, -10), b=Point(-6, -14)),
]
orange_lines = [
    Line(a=Point(-2, -10), b=Point(-2, -14)),
    Line(a=Point(2, -10), b=Point(2, -14))
]


# bundle the rendering in to one def
def show_waypoints(way_points):
    image = draw_track(
        lines=[
            (150, 150, 150), 2, [waypoint.line for waypoint in waypoints],
            ((255, 0, 0), 2, blue_lines),
            ((0, 255, 255), 2, yellow_lines),
            ((0, 100, 255), 2, orange_lines),
        ],
    )
    cv2.imshow("", image)
    cv2.waitKey(0)


""" Bias the way points to the left. Bias = 1 """
waypoints = gen_local_waypoints(
    Point(0, 0),
    -math.pi / 2,
    blue_boundary=blue_lines,
    yellow_boundary=yellow_lines,
    orange_boundary=orange_lines,
    foresight=6,
    negative_foresight=0,
    bias=1,
    margin=1
)
show_waypoints(waypoints)


""" Bias the way points to the right. Bias = -1 """
waypoints = gen_local_waypoints(
    Point(0, 0),
    -math.pi / 2,
    blue_boundary=blue_lines,
    yellow_boundary=yellow_lines,
    orange_boundary=orange_lines,
    foresight=6,
    negative_foresight=0,
    bias=-1,
    margin=1
)
show_waypoints(waypoints)


""" Bias the way points straight. Bias = 0, Bias Strength = 0.8 """
waypoints = gen_local_waypoints(
    Point(0, 0),
    -math.pi / 2,
    blue_boundary=blue_lines,
    yellow_boundary=yellow_lines,
    orange_boundary=orange_lines,
    foresight=6,
    negative_foresight=0,
    bias=0,
    bias_strength=0.8,
    margin=1
)
show_waypoints(waypoints)
