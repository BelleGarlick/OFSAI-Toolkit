import math
import time
from multiprocessing.pool import Pool
from typing import List, Tuple, Optional

import numpy as np

from fsai.objects.line import Line
from fsai.objects.point import Point
from fsai.objects.waypoint import Waypoint


def gen_local_waypoints(
        car_pos: Point,
        car_angle: float,
        blue_boundary: List[Line],
        yellow_boundary: List[Line],
        orange_boundary: List[Line],
        front: int = 10,
        back: int = 10) -> Tuple[List[Waypoint], List[Point]]:

    boundary = blue_boundary + yellow_boundary + orange_boundary

    now = time.time()
    waypoint_line, points = create_waypoint_at_pos(car_pos, boundary)
    print(time.time() - now)

    angle_point = Point(car_pos.x + 5, car_pos.y)
    print(car_angle)
    angle_point.rotate_around(car_pos, car_angle)

    return [Waypoint(line=waypoint_line)], [angle_point]


def create_waypoint_at_pos(point: Point, boundary):
    check_lines = [line for line in boundary if line.a.distance(point) < 10 or line.b.distance(point) < 10]
    lines: List[Tuple[Line, Line]] = __get_test_lines_around_point(point, count=10)
    smallest_line: Optional[Line] = None
    closest_distance = math.inf

    points = []

    for line in lines:
        points_a: List[Point] = __get_intersection_points(line[0], check_lines)
        points_b: List[Point] = __get_intersection_points(line[1], check_lines)

        if len(points_a) == 0:
            points_a = [line[0].a]
        if len(points_b) == 0:
            points_b = [line[1].a]

        point_a = point.get_closest_point(points_a)
        point_b = point.get_closest_point(points_b)

        distance = point_a[0].distance(point_b[0])
        if smallest_line is None or distance < closest_distance:
            smallest_line = Line(point_a[0], point_b[0])
            closest_distance = distance
        points = points + points_a + points_b
    return smallest_line, points


def __get_test_lines_around_point(
        origin: Point,
        count: int = 10,
        length: float = 10
) -> List[Tuple[Line, Line]]:
    lines = []

    deliminators = math.pi / count
    for i in range(count):
        p = Point(x=origin.x - length, y=origin.y)
        p.rotate_around(position=origin, angle=deliminators * i)
        lines.append((Line(a=p, b=origin), Line(a=p - (p - origin) * 2, b=origin)))

    return lines


def __get_intersection_points(
        line: Line,
        boundaries: List[Line]
):
    points: List[Point] = []
    for boundary in boundaries:
        point = line.intersects(boundary)
        if point is not None:
            points.append(point)
    return points

