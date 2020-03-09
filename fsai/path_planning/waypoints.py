import math
from typing import List, Tuple, Optional

import numpy as np

from fsai.objects.line import Line
from fsai.objects.point import Point
from fsai.objects.waypoint import Waypoint


# TODO
# Bias the waypoitns
# Make sure waypoint.line.a is always on the left
# reverse waypoitns
# ability to generate evenly spaced
# what happens if only one side is found


def gen_local_waypoints(
        car_pos: Point,
        car_angle: float,
        blue_boundary: List[Line],
        yellow_boundary: List[Line],
        orange_boundary: List[Line],
        forsight: int = 20,
        back: int = 10,
        spacing: float = 2,
        margin=0,
        overlap=False
) -> Tuple[List[Waypoint], List[Point]]:
    boundary = blue_boundary + yellow_boundary + orange_boundary

    # create initial way point surrounding the car
    lines, waypoint_line = create_waypoint_at_pos(car_pos, blue_boundary, yellow_boundary, orange_boundary)
    waypoint_line = Waypoint(line=waypoint_line)
    way_points_lines = [waypoint_line]

    initial_point = waypoint_line.get_optimum_point()

    forward_lines = create_frontal_waypoints(
        initial_point=initial_point, initial_angle=car_angle, count=forsight, spacing=spacing, overlap=overlap,
        blue_boundary=blue_boundary, yellow_boundary=yellow_boundary, orange_boundary=orange_boundary
    )

    reversed_lines = []
    # reversed_lines = create_frontal_waypoints(
    #     initial_point=initial_point, initial_angle=-car_angle, count=forsight, spacing=spacing, overlap=overlap, reversed=True,
    #     blue_boundary=blue_boundary, yellow_boundary=yellow_boundary, orange_boundary=orange_boundary
    # )

    # apply error margin
    all_waypoints = way_points_lines + forward_lines + reversed_lines
    all_waypoints = apply_error_margin(all_waypoints, margin)

    return all_waypoints, []


def create_waypoint_at_pos(point: Point, blue_boundary, yellow_boundary, orange_boundary):
    blue_lines = [line for line in blue_boundary if line.a.distance(point) < 10 or line.b.distance(point) < 10]
    yellow_lines = [line for line in yellow_boundary if line.a.distance(point) < 10 or line.b.distance(point) < 10]
    orange_lines = [line for line in orange_boundary if line.a.distance(point) < 10 or line.b.distance(point) < 10]

    lines: List[Tuple[Line, Line]] = __get_test_lines_around_point(point, count=10)
    smallest_line = __get_intersection_line_from_test_lines(
        lines,
        blue_lines,
        yellow_lines,
        orange_lines
    )
    return lines, smallest_line


def create_frontal_waypoints(initial_point: Point, initial_angle: float, count: int, spacing: float, blue_boundary: List[Line], yellow_boundary: List[Line], orange_boundary: List[Line], overlap=False, reversed=False):
    forward_lines = []
    last_point: Point = initial_point
    last_angle: float = initial_angle
    for i in range(count):
        next_waypoint = get_next_waypoint(
            starting_point=last_point,
            direction=last_angle,
            blue_boundary=blue_boundary,
            yellow_boundary=yellow_boundary,
            orange_boundary=orange_boundary,
            spacing=spacing,
            max_length=20,
            reversed=reversed
        )

        forward_lines.append(next_waypoint)

        next_point = next_waypoint.get_optimum_point()
        last_angle = last_point.angle_to(next_point)
        last_point = next_point

        if not overlap:
            if next_point.distance(initial_point) < spacing * 1.5 and i > 3:
                break
    return forward_lines


def __create_radar_lines(
        initial_point: Point,
        initial_angle: float,
        spacing: float = 2,  # meters
        line_count: int = 9,
        angle_span: float = math.pi / 1.4,
        length: float = 10,
        reversed=False
):
    sub_lines: List[Tuple[Line, Line]] = []

    if line_count <= 1:
        p = Point(initial_point.x + spacing, initial_point.y)
        p.rotate_around(initial_point, initial_angle)

        la = Point(p.x, p.y - length)
        lb = Point(p.x, p.y + length)
        if reversed: la, lb = lb, la

        la.rotate_around(la, initial_angle)
        lb.rotate_around(lb, initial_angle)

        if reversed: la, lb = lb, la
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
            if reversed: la, lb = lb, la

            la.rotate_around(p, current_angle)
            lb.rotate_around(p, current_angle)
            sub_lines.append((Line(a=la, b=p), Line(a=lb, b=p)))
    return sub_lines


def get_next_waypoint(
    starting_point: Point,
    direction: float,
    blue_boundary: List[Line],
    yellow_boundary: List[Line],
    orange_boundary: List[Line],
    spacing: float = 3,
    max_length: float = 20,
    reversed=False
) -> Waypoint:
    distance = (spacing**2 + max_length**2) ** (1/2)
    plausible_blue_boundaries = [
        boundary_line for boundary_line in blue_boundary
        if starting_point.distance(boundary_line.a) < distance or starting_point.distance(boundary_line.b) < distance]
    plausible_yellow_boundaries = [
        boundary_line for boundary_line in yellow_boundary
        if starting_point.distance(boundary_line.a) < distance or starting_point.distance(boundary_line.b) < distance]
    plausible_orange_boundaries = [
        boundary_line for boundary_line in orange_boundary
        if starting_point.distance(boundary_line.a) < distance or starting_point.distance(boundary_line.b) < distance]

    radar_lines = __create_radar_lines(
        initial_point=starting_point,
        initial_angle=direction,
        spacing=spacing,
        length=max_length,
        reversed=reversed
    )

    smallest_line: Optional[Line] = __get_intersection_line_from_test_lines(
        radar_lines,
        plausible_blue_boundaries,
        plausible_yellow_boundaries,
        plausible_orange_boundaries
    )
    return Waypoint(line=smallest_line)


def __get_intersection_line_from_test_lines(
        lines: List[Tuple[Line, Line]],
        blue_boundary: List[Line],
        yellow_boundary: List[Line],
        orange_boundary: List[Line],
) -> Optional[Line]:
    smallest_line: Optional[Line] = None
    closest_distance = math.inf
    points = []
    for line in lines:
        center_points = line[0].b
        # TODO Choose blue left or blue right
        points_a: List[Point] = __get_intersection_points(line[0], blue_boundary + orange_boundary)
        points_b: List[Point] = __get_intersection_points(line[1], yellow_boundary + orange_boundary)

        if len(points_a) == 0:
            points_a = [line[0].a]
        if len(points_b) == 0:
            points_b = [line[1].a]
        points = points + points_a + points_b
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
        p = Point(x=origin.x + length, y=origin.y)
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


def apply_error_margin(waypoints: List[Waypoint], margin: float) -> List[Waypoint]:
    # TODO make sure lines can't be reversed if margin is too large
    for waypoint in waypoints:
        normalised_point = waypoint.line.normalise()
        normalised_point = normalised_point * margin
        waypoint.line.a.add(normalised_point)
        waypoint.line.b.sub(normalised_point)
    return waypoints

