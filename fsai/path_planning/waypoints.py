import math
import time
from multiprocessing.pool import Pool
from typing import List, Tuple, Optional

import numpy as np

from fsai.objects.line import Line
from fsai.objects.point import Point
from fsai.objects.waypoint import Waypoint


# TODO
# Bias the waypoitns
# Full Track
# Make sure waypoint.line.a is always on the elft
# reverse waypoitns
# Shorten line for car error

# TODO Maybe
# possibly make it so that you can only connect to blue on one side and yellow on the other


def gen_local_waypoints(
        car_pos: Point,
        car_angle: float,
        blue_boundary: List[Line],
        yellow_boundary: List[Line],
        orange_boundary: List[Line],
        forsight: int = 10,
        back: int = 10) -> Tuple[List[Waypoint], List[Point]]:

    boundary = blue_boundary + yellow_boundary + orange_boundary

    # create initial way point surrounding the car
    waypoint_line = create_waypoint_at_pos(car_pos, boundary)
    waypoint_line = Waypoint(line=waypoint_line)
    way_points_lines = [waypoint_line]

    forward_lines = []
    last_point = waypoint_line.get_optimum_point()
    for i in range(forsight):
        next_waypoint = get_next_waypoint(
                starting_point=last_point,
                direction=car_angle,
                boundary=boundary,
                spacing=3,
                max_length=20
        )

        forward_lines.append(next_waypoint)
        last_point = next_waypoint.get_optimum_point()

    return way_points_lines + forward_lines, []


def create_waypoint_at_pos(point: Point, boundary):
    check_lines = [line for line in boundary if line.a.distance(point) < 10 or line.b.distance(point) < 10]
    lines: List[Tuple[Line, Line]] = __get_test_lines_around_point(point, count=10)
    smallest_line: Optional[Line] = __get_intersection_line_from_test_lines(
        lines,
        check_lines
    )
    return smallest_line


def __create_radar_lines(
        initial_point: Point,
        initial_angle: float,
        spacing: float = 2,  # meters
        line_count: int = 7,
        angle_span: float = math.pi / 2,
        length: float = 10
):
    sub_lines: List[Tuple[Line, Line]] = []

    if line_count <= 1:
        p = Point(initial_point.x + spacing, initial_point.y)
        p.rotate_around(initial_point, initial_angle)

        la = Point(p.x, p.y - length)
        lb = Point(p.x, p.y + length)
        la.rotate_around(la, initial_angle)
        lb.rotate_around(lb, initial_angle)
        sub_lines.append((Line(a=la, b=p), Line(a=lb, b=p)))

    else:
        angle_change = angle_span / (line_count - 1)
        starting = initial_angle - angle_span / 2
        for i in range(line_count):
            current_angle = starting + (i * angle_change)
            p = Point(initial_point.x + spacing, initial_point.y)
            p.rotate_around(initial_point, current_angle)

            la = Point(p.x, p.y - length / 2)
            lb = Point(p.x, p.y + length / 2)
            la.rotate_around(p, current_angle)
            lb.rotate_around(p, current_angle)
            sub_lines.append((Line(a=la, b=p), Line(a=lb, b=p)))
    return sub_lines


def get_next_waypoint(
    starting_point: Point,
    direction: float,
    boundary: List[Line],
    spacing: float = 3,
    max_length: float = 20
) -> Waypoint:
    distance = (spacing**2 + max_length**2) ** (1/2)
    plausible_boundaries = [
        boundary_line for boundary_line in boundary
        if starting_point.distance(boundary_line.a) < distance or starting_point.distance(boundary_line.b) < distance]

    radar_lines = __create_radar_lines(
        initial_point=starting_point,
        initial_angle=direction,
        spacing=spacing,
        length=max_length
    )
    smallest_line: Optional[Line] = __get_intersection_line_from_test_lines(
        radar_lines,
        plausible_boundaries
    )
    return Waypoint(line=smallest_line)


def __get_intersection_line_from_test_lines(
        lines: List[Tuple[Line, Line]],
        track_boundary: List[Line]
) -> Optional[Line]:
    smallest_line: Optional[Line] = None
    closest_distance = math.inf

    for line in lines:
        center_points = line[0].b
        points_a: List[Point] = __get_intersection_points(line[0], track_boundary)
        points_b: List[Point] = __get_intersection_points(line[1], track_boundary)

        if len(points_a) == 0:
            points_a = [line[0].a]
        if len(points_b) == 0:
            points_b = [line[1].a]

        point_a = center_points.get_closest_point(points_a)
        point_b = center_points.get_closest_point(points_b)

        distance = point_a[0].distance(point_b[0])
        if smallest_line is None or distance < closest_distance:
            smallest_line = Line(point_a[0], point_b[0])
            closest_distance = distance
    return smallest_line


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

