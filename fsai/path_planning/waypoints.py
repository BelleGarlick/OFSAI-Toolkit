import math
from typing import List, Tuple, Optional

import numpy as np

from fsai.objects.line import Line
from fsai.objects.point import Point
from fsai.objects.waypoint import Waypoint


# Read me todo
#  everything as normal with readme
# add comments through out this
# propergate all values upwards
# add custom readme to packages for mapping that explains htis in a lot of detail

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

BLUE_ON_LEFT = 0
YELLOW_ON_LEFT = 1


def gen_local_waypoints(
        car_pos: Point,
        car_angle: float,
        blue_boundary: List[Line],
        yellow_boundary: List[Line],
        orange_boundary: List[Line],
        foresight: int = 20,
        negative_foresight: int = 10,
        full_track: bool = False,
        spacing: float = 2,
        margin: float = 0,
        bias: float = 0,
        bias_strength=0.2,
        smooth=False
) -> List[Waypoint]:

    # create initial way point surrounding the car
    initial_waypoint = create_waypoint_at_pos(car_pos, car_angle, blue_boundary, yellow_boundary, orange_boundary)
    initial_point = initial_waypoint.get_optimum_point()

    # if full track then set the foresight really high, and negative to 0
    # we can set the foresight really high because waypoints wont overlap
    # if the full track is active
    if full_track:
        foresight = 100000
        negative_foresight = 0

    # generate waypoints in front of the origin
    forward_lines = create_frontal_waypoints(
        initial_point=initial_point,
        initial_angle=car_angle,
        count=foresight,
        spacing=spacing,
        overlap=not full_track,
        blue_boundary=blue_boundary,
        yellow_boundary=yellow_boundary,
        orange_boundary=orange_boundary,
        bias=bias,
        bias_strength=bias_strength
    )

    # generate waypoints behind the origin
    reversed_lines = create_frontal_waypoints(
        initial_point=initial_point,
        initial_angle=car_angle + math.pi,
        count=negative_foresight,
        spacing=spacing,
        overlap=not full_track,
        reverse=True,
        blue_boundary=blue_boundary,
        yellow_boundary=yellow_boundary,
        orange_boundary=orange_boundary,
        bias=bias,
        bias_strength=bias_strength
    )
    # waypoints should be reverse as they are created from the origin going outwards,
    # but we want them in order directional order, so we reverse them here
    reversed_lines.reverse()

    # merge all waypoints in order of negative foresight -> central waypoints -> forward waypoints
    all_waypoints = reversed_lines + [initial_waypoint] + forward_lines

    # apply error margin
    all_waypoints = apply_error_margin(all_waypoints, margin)

    # return lines: smoothed is needed
    return smoothify(all_waypoints, full_track) if smooth else all_waypoints


def create_waypoint_at_pos(point: Point, angle: float, blue_boundary, yellow_boundary, orange_boundary) -> Waypoint:
    blue_lines = [line for line in blue_boundary if line.a.distance(point) < 10 or line.b.distance(point) < 10]
    yellow_lines = [line for line in yellow_boundary if line.a.distance(point) < 10 or line.b.distance(point) < 10]
    orange_lines = [line for line in orange_boundary if line.a.distance(point) < 10 or line.b.distance(point) < 10]

    # TODO PROPERGATE VALUES UPWATDS
    lines: List[Tuple[Line, Line]] = __get_radar_lines_around_point(point, angle, count=10)
    smallest_line: Waypoint = __get_most_perpendicular_line_to_boundary(
        lines,
        blue_lines,
        yellow_lines,
        orange_lines
    )
    return smallest_line


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

    smallest_line: Optional[Waypoint] = __get_most_perpendicular_line_to_boundary(
        radar_lines,
        plausible_blue_boundaries,
        plausible_yellow_boundaries,
        plausible_orange_boundaries,
        bias=bias,
        bias_strength=bias_strength
    )
    return smallest_line


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


# TODO MAKE SURE ALL PARAMETERS
# TODO Full documentation
def __get_most_perpendicular_line_to_boundary(
        lines: List[Tuple[Line, Line]],
        blue_boundary: List[Line],
        yellow_boundary: List[Line],
        orange_boundary: List[Line],
        bias: float = 0,
        bias_strength: float = 0.2,
        left_colour: int = BLUE_ON_LEFT
) -> Optional[Waypoint]:
    # store the two variables used to find the shortest line
    smallest_line: Optional[Waypoint] = None
    closest_distance: float = math.inf

    # buffer the line_count so we don't need to keep calling it, adding lag
    line_count: int = len(lines)

    # loop through each line in the list of lines to find the more perpendicular line
    for i in range(line_count):
        line = lines[i]
        # set up defaults for the waypoints, optimum 0.5 (middle) and not sticky
        optimum, sticky = 0.5, False

        # dependant on what colour is on left, we can swap the way we search for points
        if left_colour == BLUE_ON_LEFT:
            left_intersections: List[Point] = __get_intersection_points(line[0], blue_boundary + orange_boundary)
            right_intersections: List[Point] = __get_intersection_points(line[1], yellow_boundary + orange_boundary)
        else:
            left_intersections: List[Point] = __get_intersection_points(line[0], yellow_boundary + orange_boundary)
            right_intersections: List[Point] = __get_intersection_points(line[1], blue_boundary + orange_boundary)

        # check the amount of intersections.
        # if a line does not intersect then we can use the end of the
        # end of the line as a dummy intersection point to draw the
        # line shortest line to.
        if len(left_intersections) == 0 and len(right_intersections) == 0:
            left_intersections = [line[0].a]
            right_intersections = [line[1].a]
            sticky = True
        elif len(left_intersections) == 0:
            left_intersections = [line[0].a]
            optimum = 1
            sticky = True
        elif len(right_intersections) == 0:
            right_intersections = [line[1].a]
            optimum = 0
            sticky = True

        # find the two points closest to the center of the line (line[0].b == line[1].b === center of line)
        center_points = line[0].b
        point_a = center_points.get_closest_point(left_intersections)
        point_b = center_points.get_closest_point(right_intersections)

        # calculate the distance of hte two points
        distance = point_a[0].distance(point_b[0])

        # Here we can apply the bias, explained in the method documentation. First we calculate how much bias to apply
        # for the current angle
        bias_angle = ((i + 0.5) / line_count * 2 - 1)
        bias_value = max(.0, 1 - abs(bias_angle - bias))
        # we can then create a new heuristic distance which is the current distance affected by the bias
        distance -= distance * bias_value * bias_strength
        # then we can compare hte smallest line with the heuristic
        if smallest_line is None or distance < closest_distance:
            # bae is schmol line so we update the smallest known line
            smallest_line = Waypoint(line=Line(point_a[0], point_b[0]), optimum=optimum, sticky=sticky)
            closest_distance = distance

    return smallest_line


# TODO DOUBLE CHECK THE ANGLES + PROPERGATE UPWARDS
def __get_radar_lines_around_point(
        origin: Point,
        angle: float,
        count: int = 10,
        length: float = 10,
        total_span: float = math.pi / 2
) -> List[Tuple[Line, Line]]:
    """
    This function will create a list of lines arching in-front of the vehicle. For example if you were 0,0 you might
    have 3 lines [(-2, 0), (0, 2)],  [(-1, 1), (1, 1)],  [(2, 0), (0, 2)]. These lines are made up of sub-lines,
    which stem from the center of the line. The reason this is done is that we can then calculate where the closest
    point of the sub-line[0] intersects with the left of hte track and where sub-line[1] intersects with the right of
    the track. By doing this we can then see which line has the closest pair of intersections and is therefore the
    most perpendicular to the track, making it a suitable waypoint.

    :param origin: The point at which the radar lines should arch around
    :param angle: The that the arching lines should be facing
    :param count: How many lines arch about the point
    :param length: The distance of the line from the origin
    :param total_span: The total spread of the lines created.
    :return: The arching lines ahead of the vehicle
    """
    # Store the list to test
    lines = []

    # calculate the change in angle from each line to the next
    angle_change = total_span / count
    # TODO Double check this is right, subtracting half pi seems bizzare - need an explenation in the commentts
    # Calculate the initial angle which is the left most starting point
    initial_angle = (angle - math.pi / 2) - (total_span / 2)

    # create a line infront of the car X (count) times.
    for i in range(count):
        # create and rotate the point the correct amount
        p = Point(x=origin.x + length, y=origin.y)
        p.rotate_around(position=origin, angle=initial_angle + angle_change * i)

        # create the line form the origin to this newly rotated point
        lines.append((Line(a=p, b=origin), Line(a=p - (p - origin) * 2, b=origin)))

    return lines


def __get_intersection_points(line: Line, boundaries: List[Line]):
    """
    Get all points where a given line intersects a list of given lines. This is used to detect where lines
    intersect boundary lines.

    :param line: Line to detect intersections upon
    :param boundaries: Boundary lines
    :return: List of point of all the intersections the given line has with the boundary.
    """
    points: List[Point] = []
    # loop through the boundary
    for boundary in boundaries:
        # find the intersection points
        point = line.intersects(boundary)
        if point is not None:
            # add the intersection points to the list
            points.append(point)
    return points


def apply_error_margin(waypoints: List[Waypoint], margin: float) -> List[Waypoint]:
    """
    We do not way the waypoints to span the whole width of the track. This is because that line generated
    may go right up to the line boundary. By applying an error margin we can shorten the length of each waypoint
    such that it allows for error either side of the track.

    The margin is applied to each side so if your car was 1.4m then you would have a margin of ~0.7. This is done
    by looping through each line, then for each line create a normalised copy of each line. A normalised line is a
    line segment that heads in the same direction but with a length of 1. By multipling this normalised line with
    the margin we create a line segment that is the length of our margin parallel to the waypoint. We can add and
    subtract this vector to the ends of our line in order to shorten the line.

    :param waypoints: List of waypoints to shorten
    :param margin: Margin size
    :return: List of marginalised waypoints
    """
    # Loop through each waypoint and apply error margin
    for waypoint in waypoints:
        # In the even the waypoint is too large then we cap it at half the waypoints length + error
        # This is to prevent the waypoint from becoming so negative that it flips it self.
        altered_margin = min(waypoint.line.length() / 2 - 0.01, margin)
        normalised_point = waypoint.line.normalise()  # calculate normalised vector
        normalised_point = normalised_point * altered_margin  # multiply the vector by the length of the margin
        waypoint.line.a.add(normalised_point)  # apply vector to line
        waypoint.line.b.sub(normalised_point)  # apply vector to the line

    # return the shortened waypoints.
    return waypoints


def smoothify(current_lines: List[Waypoint], full_track: bool):
    """
    Often the first list of waypoints generated may have waypoints that are a little zig-zaggy.
    This function will take a list of waypoints and return a new list of deep copies of waypoints
    in which the angles between the waypoints are more consistent and therefore smoother.

    This is done by going through each waypoint and calculating hte angle of hte surrounding waypoints
    then taking an average of them which hte iterated waypoint should be.

    :param current_lines: List of waypoints to smooth-out
    :param full_track: State whether the list of waypoint is a full track, if so all points will be optimised
                       other wise the ends of the list will not me altered.
    :return: Returns a list of waypoints that have been smoothed out
    """
    # create blank list to populate with smooth waypoints
    smooth_waypoints = []

    # create range based upon whether the track is full track ot not
    index_range = range(1, len(current_lines) - 1, 1)  # if not full track don't alter the ends else they'll be skewed
    if full_track:
        index_range = range(len(current_lines))

    # loop through each index in the range and and calculate a new angle for it
    for i in index_range:
        # create a new deep copy of the waypoint in which to mutate
        new_waypoint: Waypoint = current_lines[i].copy()

        # get the angle of the surrounding waypoints
        pre = current_lines[i - 1].line.angle()
        cur = current_lines[i].line.angle()
        nex = current_lines[(i + 1) % len(current_lines)].line.angle()

        # create a normalised vector for each angle which we wish to add, then add the vectors and get an average
        # angle from that. You must never average two angles the normal way. Average of 359 and 1 is 0, not 180...
        x, y = math.cos(pre), math.sin(pre)
        x, y = x + math.cos(cur), y + math.sin(cur)
        x, y = x + math.cos(nex), y + math.sin(nex)

        # calculate angle of the sum of the vectors
        average_angle = math.atan2(y, x)

        # calculate the difference between the average angle and current angle
        delta_angle = cur - average_angle

        # rotate the end of each line in the waypoint .line.a, .line.b around the center of the line
        center_point = new_waypoint.line.a + ((new_waypoint.line.b - new_waypoint.line.a) * 0.5)
        new_waypoint.line.a.rotate_around(center_point, delta_angle)
        new_waypoint.line.b.rotate_around(center_point, delta_angle)

        # add the newly rotated waypoint to the list of smooth waypoints
        smooth_waypoints.append(new_waypoint)

    # If we don't do the full track, then we don't iterate of the ends of the list. Here we add the lines back in
    if not full_track:
        smooth_waypoints = [current_lines[0]] + smooth_waypoints + [current_lines[-1]]

    # return the list of smooth AF waypoints.
    return smooth_waypoints


def decimate_waypoints(waypoints: List[Waypoint], threshold: float = 0.2, spread=0, max_gap: int = 3) -> List[Waypoint]:
    """
    When generating waypoints, the distances between each waypoint is fixed. This approach may not be
    beneficial when optimising the lines, ideally you would want more waypoints in a corner and fewer
    waypoints on a straight.

    This function removed redundant waypoints if the angle from one waypoint to another is lower than a given
    threshold. We can also provide a maximum gap between waypoints which ensures that waypoints are
    still 1 waypoint for every X waypoints removed. Lastly a spread can be provided which allows a
    corners angle to spread to surrounding waypoints. This is to keep surrounding waypoints in the corners
    which may be useful when optimising the target line.

    :param waypoints: The waypoints to decimate
    :param threshold: Minimum curved threshold required to keep waypoints.
    :param max_gap: Keeps 1 waypoint for every X (max_gap) waypoints removed.
    :param spread: How far the angle spreads to surrounding waypoints.
    :return: A new list of decimated waypoints
    """
    waypoint_count = len(waypoints)
    # Store a list of floats representing how much each waypoint
    # curves corresponding to the list of given waypoints.
    waypoint_curvature = np.zeros(waypoint_count)

    # loop through each waypoint in range 1:len-1 and calculate the curvature
    # of that waypoint. We dont calculate either end of the waypoints because
    # each end must be added to the list of final waypoints in order to make
    # sure the generated line knows where it should heard towards.
    for i in range(1, waypoint_count - 1):
        # get the angle of the surounding lines
        p: float = waypoints[i - 1].line.angle()
        c: float = waypoints[i].line.angle()
        n: float = waypoints[(i + 1) % waypoint_count].line.angle()

        # calculate the absolute change in angle bounded between -pi:pi
        a: float = abs((p - c + math.pi) % (math.pi*2) - math.pi)
        a += abs((n - c + math.pi) % (math.pi*2) - math.pi)

        # alter the surrounding waypoints to apple the spread
        for j in range(-(spread + 1), spread + 1, 1):
            index = (i + j) % waypoint_count
            waypoint_curvature[index] = max(a, waypoint_curvature[index])

    # list of the decimated waypoints to return to the users
    decimated_waypoints = []
    # this stores when the last waypoint was added, by keeping
    # track of this we know when we need to add a new waypoint
    # based upon the max_gap distance
    last_added_waypoint = -max_gap

    # loop though each waypoint and calculate conditions in which  we should add the waypoint
    for i in range(waypoint_count):
        # is the waypoint sticky? If a waypoint is sticky then it's optimum point cannot change
        # this is important to calculating the final line so these waypoints must not be removed
        sticky: bool = waypoints[i].sticky
        # check if the angle of the waypoint surpasses the given threshold
        angled: bool = waypoint_curvature[i] > threshold
        # if the gap between the last waypoint surpasses the max gap
        gap: bool = i - last_added_waypoint >= max_gap
        # is the waypoint at either end of the list of waypoints
        ends: bool = i == waypoint_count - 1 or i == 0

        # if any one of the prior conditions is true then we want to keep the wyapoint
        if sticky or angled or gap or ends:
            last_added_waypoint = i  # update the waypoint
            decimated_waypoints.append(waypoints[i])  # add to the list

    # here, has some optimised waypoints.
    return decimated_waypoints
