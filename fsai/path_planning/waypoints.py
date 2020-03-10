import math
from typing import List, Tuple, Optional

from fsai.objects.line import Line
from fsai.objects.point import Point
from fsai.objects.waypoint import Waypoint


# TODO
# ability to generate evenly spaced
# what happens if only one side is found -> does it lock? -> Does it hug left

# Read me todo
#  everything as normal
# add comments through out this
# propergate all values upwards
# add custom readme to packages for mapping

# Examples TODO
# Show basic, 10 font 5 back with custom spacing
# show margin
# show full track (/w & !/w overlap)
# missing boundary modes
# biasing effects
# generate evenly spaced
# decimate
# smooth lines out a lil to that they overlap less

# TODO
# double check all examples are still correct and workas intended

def gen_local_waypoints(
        car_pos: Point,
        car_angle: float,
        blue_boundary: List[Line],
        yellow_boundary: List[Line],
        orange_boundary: List[Line],
        forsight: int = 20,
        negative_forsight: int = 10,
        full_track: bool = False,
        spacing: float = 2,
        margin: float = 0,
        bias: float = 0,
        bias_strength=0.2,
        smooth=False
) -> List[Waypoint]:

    # create initial way point surrounding the car
    lines, waypoint_line = create_waypoint_at_pos(car_pos, car_angle, blue_boundary, yellow_boundary, orange_boundary)
    initial_waypoint = Waypoint(line=waypoint_line)

    initial_point = initial_waypoint.get_optimum_point()

    if full_track:
        forsight = 1000
        negative_forsight = 0

    forward_lines = create_frontal_waypoints(
        initial_point=initial_point, initial_angle=car_angle, count=forsight, spacing=spacing, overlap=not full_track,
        blue_boundary=blue_boundary, yellow_boundary=yellow_boundary, orange_boundary=orange_boundary, bias=bias, bias_strength=bias_strength
    )

    reversed_lines = create_frontal_waypoints(
        initial_point=initial_point, initial_angle=car_angle + math.pi, count=negative_forsight, spacing=spacing, overlap=not full_track, reverse=True,
        blue_boundary=blue_boundary, yellow_boundary=yellow_boundary, orange_boundary=orange_boundary, bias=bias, bias_strength=bias_strength
    )
    reversed_lines.reverse()

    # apply error margin
    all_waypoints = reversed_lines + [initial_waypoint] + forward_lines
    all_waypoints = apply_error_margin(all_waypoints, margin)

    return smoothify(all_waypoints, full_track) if smooth else all_waypoints


def create_waypoint_at_pos(point: Point, angle: float, blue_boundary, yellow_boundary, orange_boundary):
    blue_lines = [line for line in blue_boundary if line.a.distance(point) < 10 or line.b.distance(point) < 10]
    yellow_lines = [line for line in yellow_boundary if line.a.distance(point) < 10 or line.b.distance(point) < 10]
    orange_lines = [line for line in orange_boundary if line.a.distance(point) < 10 or line.b.distance(point) < 10]

    lines: List[Tuple[Line, Line]] = __get_test_lines_around_point(point, angle, count=10)
    smallest_line = __get_intersection_line_from_test_lines(
        lines,
        blue_lines,
        yellow_lines,
        orange_lines
    )
    return lines, smallest_line


def create_frontal_waypoints(initial_point: Point, initial_angle: float, count: int, spacing: float, blue_boundary: List[Line], yellow_boundary: List[Line], orange_boundary: List[Line], overlap=False, reverse=False, bias: float=0, bias_strength: float=0):
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
            reverse=reverse,
            bias=bias,
            bias_strength=bias_strength
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
        line_count: int = 13,
        angle_span: float = math.pi,
        length: float = 10,
        reverse=False
):
    sub_lines: List[Tuple[Line, Line]] = []

    if line_count <= 1:
        p = Point(initial_point.x + spacing, initial_point.y)
        p.rotate_around(initial_point, initial_angle)

        la = Point(p.x, p.y - length)
        lb = Point(p.x, p.y + length)
        if reverse: la, lb = lb, la

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
            if reverse: la, lb = lb, la

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
    reverse=False,
    bias: float = 0,
    bias_strength: float = 0.2
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
        reverse=reverse
    )

    smallest_line: Optional[Line] = __get_intersection_line_from_test_lines(
        radar_lines,
        plausible_blue_boundaries,
        plausible_yellow_boundaries,
        plausible_orange_boundaries,
        bias=bias,
        bias_strength=bias_strength
    )
    return Waypoint(line=smallest_line)


def __get_intersection_line_from_test_lines(
        lines: List[Tuple[Line, Line]],
        blue_boundary: List[Line],
        yellow_boundary: List[Line],
        orange_boundary: List[Line],
        bias: float = 0,
        bias_strength: float = 0.2
) -> Optional[Line]:
    smallest_line: Optional[Line] = None
    closest_distance = math.inf
    points = []
    count = 0
    line_count = len(lines)
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

        bias_angle = ((count + 0.5) / line_count * 2 - 1)
        distance -= distance * distribute(bias_angle - bias) * bias_strength
        if smallest_line is None or distance < closest_distance:
            smallest_line = Line(point_a[0], point_b[0])
            closest_distance = distance
        count += 1
    return smallest_line


def __get_test_lines_around_point(
        origin: Point,
        angle: float,
        count: int = 10,
        length: float = 10
) -> List[Tuple[Line, Line]]:
    lines = []

    total_angle_change = math.pi / 1.2
    angle_change = total_angle_change / count
    initial_angle = (angle - math.pi / 2) - (total_angle_change / 2)
    for i in range(count):
        p = Point(x=origin.x + length, y=origin.y)
        p.rotate_around(position=origin, angle=initial_angle + angle_change * i)
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


def smoothify(
        current_lines: List[Waypoint],
        full_track: bool
):
    smooth_waypoints = []

    if full_track:
        index_range = range(len(current_lines))
    else:
        index_range = range(1, len(current_lines) - 1, 1)

    for i in index_range:
        new_waypoint: Waypoint = current_lines[i].copy()

        pre = current_lines[i - 1].line.angle()
        cur = current_lines[i].line.angle()
        nex = current_lines[(i + 1) % len(current_lines)].line.angle()

        x, y = math.cos(pre), math.sin(pre)
        x, y = x + math.cos(cur), y + math.sin(cur)
        x, y = x + math.cos(nex), y + math.sin(nex)
        average_angle = math.atan2(y, x)
        delta_angle = cur - average_angle

        current_point = new_waypoint.get_optimum_point()
        new_waypoint.line.a.rotate_around(current_point, delta_angle)
        new_waypoint.line.b.rotate_around(current_point, delta_angle)
        smooth_waypoints.append(new_waypoint)

    if not full_track:
        smooth_waypoints = [current_lines[0]] + smooth_waypoints + [current_lines[-1]]

    return smooth_waypoints


def decimate_waypoints(waypoints: List[Waypoint], spread=0, max_gap: int = 3, threshold: float = 0.2) -> List[Waypoint]:
    all_waypoints: List[Tuple[Waypoint, float]] = [(waypoint, 0) for waypoint in waypoints]

    for i in range(1, len(all_waypoints) - 1):
        p = waypoints[i - 1].line.angle()
        c = waypoints[i].line.angle()
        n = waypoints[(i + 1) % len(waypoints)].line.angle()

        a = abs((p - c + math.pi) % (math.pi*2) - math.pi)
        a += abs((n - c + math.pi) % (math.pi*2) - math.pi)

        if a > 0.1:
            for j in range(-(spread + 1), spread + 1, 1):
                index = (i + j) % len(waypoints)
                all_waypoints[index] = (all_waypoints[index][0], max(a, all_waypoints[index][1]))

    decimated_waypoints = []
    last_added_waypoint = -max_gap
    for i in range(len(all_waypoints)):
        sticky = all_waypoints[i][0].sticky
        angled = all_waypoints[i][1] > threshold
        gap = i - last_added_waypoint >= max_gap
        ends = i == len(all_waypoints) - 1 or i == 0

        if sticky or angled or gap or ends:
            last_added_waypoint = i
            decimated_waypoints.append(all_waypoints[i][0])

    return decimated_waypoints


# /**
#  * Distribute values from -1 - 1 between 0 - 1. A gaussian distribution
#  * would also work but this is computationally cheaper
#  *
#  * @param x Input values to distribute
#  * @return distribution value
#  */
def distribute(x: float):
    return max(.0, 1 - abs(x))
