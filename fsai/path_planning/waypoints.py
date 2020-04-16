import math
from typing import List, Tuple, Optional

from fsai import geometry
from fsai.path_planning.waypoint import Waypoint

BLUE_ON_LEFT = 0
YELLOW_ON_LEFT = 1


def gen_waypoints(
        car_pos: List[float],
        car_angle: float,
        blue_boundary: List[List[float]],
        yellow_boundary: List[List[float]],
        orange_boundary: List[List[float]],
        foresight: int = 20,
        negative_foresight: int = 10,
        full_track: bool = False,
        spacing: float = 2,
        margin: float = 0,
        radar_length: float = 25,
        radar_count: int = 13,
        radar_span: float = math.pi / 1.1,
        bias: float = 0,
        bias_strength=0.2,
        left_boundary_colour: int = BLUE_ON_LEFT,
        smooth=False
) -> List[Waypoint]:
    """
    This method is used to create waypoints around a vehicle or for a full track. First an initial set of radar lines
    around the car position is made, then we search for the radar line which is hte best fit. Then we can create the
    surrounding waypoints using the 'create_waypoint_lines' method.

    :param car_pos: The initial point to generate the waypoints from
    :param car_angle: Orientation of the car, this will help angle the waypoints initially
    :param blue_boundary: Blue boundary of the track
    :param yellow_boundary: Yellow boundary of the track
    :param orange_boundary: Orange boundary of the track
    :param foresight: How many waypoints in front of the car to generate
    :param negative_foresight: How many waypoints behind the car to generate
    :param full_track: Should the entire track be generated or a fixed amount of waypoints
    :param spacing: How far apart should each waypoint be spaced
    :param margin: Error margin to shorten the track by. This is applied to both ends of the waypoints.
    :param radar_length: The maximum length a waypoint can be
    :param radar_count: How many radar lines should be tested upon to find the true waypoint
    :param radar_span: The total coverage the radar lines can exists between
    :param bias: Bias hte track to head certain directions
    :param bias_strength: How strongly to apply the bias
    :param left_boundary_colour: Which colour boundary is on the left [BLUE_ON_LEFT, YELLOW_ON_LEFT]
    :param smooth: If true then the waypoints will be smoothed to create a smoothing angle between waypoints
    :return: Return the list of generated waypoints
    """
    # create initial way point surrounding the car
    initial_waypoint = create_waypoint_at_pos(
        car_pos,
        car_angle,
        blue_boundary,
        yellow_boundary,
        orange_boundary,
        left_boundary_colour=left_boundary_colour,
        radar_line_count=radar_count,
        radar_line_length=radar_length,
        radar_span=radar_span
    )
    initial_point = initial_waypoint.get_optimum_point()

    # if full track then set the foresight really high, and negative to 0
    # we can set the foresight really high because waypoints wont overlap
    # if the full track is active
    if full_track:
        foresight = 10000
        negative_foresight = 0

    # generate waypoints in front of the origin
    forward_lines = create_waypoint_lines(
        initial_point=initial_point,
        initial_angle=car_angle,
        count=foresight,
        spacing=spacing,
        overlap=not full_track,
        blue_boundary=blue_boundary,
        yellow_boundary=yellow_boundary,
        orange_boundary=orange_boundary,
        bias=bias,
        bias_strength=bias_strength,
        max_radar_length=radar_length,
        radar_count=radar_count,
        radar_angle_span=radar_span,
        left_boundary_colour=left_boundary_colour
    )

    # generate waypoints behind the origin
    reversed_lines = create_waypoint_lines(
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
        bias_strength=bias_strength,
        max_radar_length=radar_length,
        radar_count=radar_count,
        radar_angle_span=radar_span,
        left_boundary_colour=left_boundary_colour
    )
    # way points should be reverse as they are created from the origin going outwards,
    # but we want them in order directional order, so we reverse them here
    reversed_lines.reverse()
    # merge all way points in order of negative foresight -> central way points -> forward way points
    all_waypoints = reversed_lines + [initial_waypoint] + forward_lines

    # apply error margin
    all_waypoints = apply_error_margin(all_waypoints, margin)

    # return lines: smoothed is needed
    smoothed = smoothify(all_waypoints, full_track) if smooth else all_waypoints
    return smoothed


def create_waypoint_at_pos(
        point: List[float],
        angle: float,
        blue_boundary: List[List[float]],
        yellow_boundary: List[List[float]],
        orange_boundary: List[List[float]],
        left_boundary_colour: int = BLUE_ON_LEFT,
        radar_line_count: int = 15,
        radar_line_length: float = 20,
        radar_span: float = math.pi/2) -> Waypoint:
    """
    This function will create the radar line around a given point. This is done by creating a list of waypoints
    rotating around hte point given such that each line passes through the origin point. Then each line is compared
    to find the best line for the track using the '__get_most_perpendicular_line_to_boundary' function. For more
    details on how the lines are generated see '__get_radar_lines_around_point'.

    :param point: Origin point to create points around
    :param angle: Heading angle of the vehicle
    :param blue_boundary: Blue track boundary
    :param yellow_boundary: Yellow track boundary
    :param orange_boundary: Orange track boundary
    :param left_boundary_colour: Which lines are on the left, in order to orientate the boundary correctly
    :param radar_line_count: How many lines arch about the point
    :param radar_line_length: The distance of each line
    :param radar_span: The total spread (either side of the direction the car faces) to create the lines around.
    :return: The best fit waypoint at a given point
    """
    # threshold all lines we check by making sure they're close enough. This is to limit the search range and reduce
    # computation time.
    radar_line_radius = radar_line_length / 2

    blue_lines = [line for line in blue_boundary if geometry.distance(point, line[0:2]) < radar_line_radius or geometry.distance(point, line[2:4]) < radar_line_radius]
    yellow_lines = [line for line in yellow_boundary if geometry.distance(point, line[0:2]) < radar_line_radius or geometry.distance(point, line[2:4]) < radar_line_radius]
    orange_lines = [line for line in orange_boundary if geometry.distance(point, line[0:2]) < radar_line_radius or geometry.distance(point, line[2:4]) < radar_line_radius]

    # Create a list of radar lines that pass through the origin point
    lines: List[Tuple[List[float], List[float]]] = __get_radar_lines_around_point(
        point,
        angle,
        count=radar_line_count,
        length=radar_line_length,
        total_span=radar_span
    )

    # find the best line out of the lines generated above
    smallest_line: Waypoint = __get_most_perpendicular_line_to_boundary(
        lines,
        blue_lines,
        yellow_lines,
        orange_lines,
        bias_strength=0,  # since we're creating the waypoints around a point we do not want any bias
        left_colour=left_boundary_colour
    )
    return smallest_line


def create_waypoint_lines(
        initial_point: List[float],
        initial_angle: float,
        count: int,
        spacing: float,
        blue_boundary: List[List[float]],
        yellow_boundary: List[List[float]],
        orange_boundary: List[List[float]],
        overlap=False,
        reverse=False,
        bias: float = 0,
        bias_strength: float = 0,
        max_radar_length: float = 20,
        radar_count: int = 13,
        radar_angle_span: float = math.pi,
        left_boundary_colour: int = BLUE_ON_LEFT
) -> List[Waypoint]:
    """
    This function is to be called to iteratively create new waypoints. This is done by calling the 'get_next_waypoint'
    repetitively. This function will end ones the count function is met or the lap has overlapped itself (if the over-
    lap parameter is set ot false).

    :param initial_point: The initial point to generate the waypoints from
    :param initial_angle: The angle the waypoints should head towards
    :param count: The amount of waypoints to generate
    :param spacing: How far each waypoint should be from the previous waypoint
    :param blue_boundary: The blue boundary of the track
    :param yellow_boundary: The yellow boundary of the track
    :param orange_boundary: The orange boundary of the track
    :param overlap: Are the way points allowed to overlap themselves.
    :param reverse: Are the way points to be reversed. Not line reversing a list, but reversing line.a <-> line.b
    :param bias: The bias of the track. See '__get_most_perpendicular_line_to_boundary'
    :param bias_strength: The bias strength of the track. See '__get_most_perpendicular_line_to_boundary'
    :param max_radar_length: Maximum length a way point line can be
    :param radar_count: The amount of plausible radar lines to create in order to find the best radar line
    :param radar_angle_span: Total radians that the stems line span from. See '__create_radar_lines'
    :param left_boundary_colour: Which colour boundary is on the left [BLUE_ON_LEFT, YELLOW_ON_LEFT]
    :return: A list of way points in the direction, from the origin point provided
    """
    # store all the waypoints generated here
    waypoint_lines: List[Waypoint] = []
    # store a buffer of the current point and angle to generate the next waypoint for
    last_point = initial_point
    last_angle: float = initial_angle

    for i in range(count):
        # calculate the next waypoint position given some parameters
        next_waypoint = get_next_waypoint(
            starting_point=last_point,
            direction=last_angle,
            blue_boundary=blue_boundary,
            yellow_boundary=yellow_boundary,
            orange_boundary=orange_boundary,
            spacing=spacing,
            max_length=max_radar_length,
            radar_count=radar_count,
            radar_angle_span=radar_angle_span,
            reverse=reverse,
            bias=bias,
            bias_strength=bias_strength,
            left_boundary_colour=left_boundary_colour
        )

        # add the new way point to the way point lines
        waypoint_lines.append(next_waypoint)

        # calculate the next angle and origin point
        next_point = next_waypoint.get_optimum_point()
        last_angle = geometry.angle_to(last_point, next_point)
        last_point = next_point

        # if we have want no overlap then we will prevent it by checking if the first way point and the current
        # way point get to close then we break the loop - this is only useful for calculating the full track
        if not overlap and i > 3:
            # see if the distances are close enough to close the track
            waypoint_center = geometry.line_center(next_waypoint.line)
            if geometry.distance(waypoint_center, initial_point) < spacing * 1.4:
                break
    return waypoint_lines


def get_next_waypoint(
    starting_point: List[float],
    direction: float,
    blue_boundary: List[List[float]],
    yellow_boundary: List[List[float]],
    orange_boundary: List[List[float]],
    spacing: float = 3,
    max_length: float = 20,
    radar_count: int = 13,
    radar_angle_span: float = math.pi,
    bias: float = 0,
    bias_strength: float = 0.2,
    left_boundary_colour: int = BLUE_ON_LEFT,
    reverse=False
) -> Waypoint:
    """
    This function is used to get the next waypoint given an origin, distance and boundaries. This is done by creating
    a set of radars lines (potential waypoints) using the '__create_radar_lines function', and then from this list
    searching for the most suitable waypoint line using the function '__get_most_perpendicular_line_to_boundary'
    which in turn returns the next waypoint.

    :param starting_point: Current point of the car/waypoint in which we wish to stem the next waypoint from
    :param direction: The current heading direction which us used to orientate the next waypoint
    :param blue_boundary: Blue boundary lines
    :param yellow_boundary: Yellow boundary lines
    :param orange_boundary: Orange boundary lines
    :param spacing: Spacing from the origin to the next waypoint
    :param max_length: Maximum length a waypoint line can be
    :param radar_count: The amount of plausible radar lines to create in order to find the best radar line
    :param radar_angle_span: Total radians that the stems line span from. See '__create_radar_lines'
    :param bias: The bias direction to head the waypoint. See '__get_most_perpendicular_line_to_boundary'
    :param bias_strength: How strongly the bias affects the waypoint. See '__get_most_perpendicular_line_to_boundary'
    :param left_boundary_colour: Which lines are on the left, in order to orientate the boundary correctly
    :param reverse: Should the lines be reversed, used when creating negative waypoints to flip line.a <-> line.b
    :return: The next waypoint given a set of parameters
    """
    # since we will be looking for the intersections with the boundaries, we will have to loop through each line in
    # boundary. This is potentially a very slow and costly process. We can speed it up by removing lines that are
    # unlikely to intercept the radar lines. We can do this by taking the distance of the origin to the end of each
    # line to create a distance which is the max radius the lines can exists and then remove all lines that are
    # too far from this distance, therefore only look for intersections with lines that could be within the
    # correct range
    # first calculate the potential distance which is the pythagoras of the spacing and the max-length/2. We divide
    # the max length by two, since the spacing distance is from the origin to the center of the radar line
    distance = (spacing**2 + (max_length / 2)**2) ** (1/2)

    # loop through blue lines finding potential intersections
    blue_lines = [line for line in blue_boundary if geometry.distance(starting_point, line[0:2]) < max_length or geometry.distance(starting_point, line[2:4]) < max_length]
    yellow_lines = [line for line in yellow_boundary if geometry.distance(starting_point, line[0:2]) < max_length or geometry.distance(starting_point, line[2:4]) < max_length]
    orange_lines = [line for line in orange_boundary if geometry.distance(starting_point, line[0:2]) < max_length or geometry.distance(starting_point, line[2:4]) < max_length]

    # create the radar lines, which are the plausible waypoint lines that the could be the ideal waypoint line
    radar_lines = __create_radar_lines(
        initial_point=starting_point,
        initial_angle=direction,
        spacing=spacing,
        line_count=radar_count,
        angle_span=radar_angle_span,
        length=max_length,
        reverse=reverse
    )

    # find the smalled line from the list of generated waypoints in radar_lines
    smallest_line: Optional[Waypoint] = __get_most_perpendicular_line_to_boundary(
        radar_lines,
        blue_lines,
        yellow_lines,
        orange_lines,
        bias=bias,
        bias_strength=bias_strength,
        left_colour=left_boundary_colour
    )
    return smallest_line


def __create_radar_lines(
        initial_point: List[float],
        initial_angle: float,
        spacing: float = 2,  # meters
        line_count: int = 13,
        angle_span: float = math.pi,
        length: float = 10,
        reverse=False
) -> List[Tuple[List[float], List[float]]]:
    """
    This function is the building blocks of the waypoint algorithm. Given a point and a heading direction, we wish to
    find the next waypoint. This is done by creating 'stem' lines, which span outwards from the initial point in the
    direction of the initial angle. These lines span over some radius, so we have a range of span lines ahead of the
    vehicle. Then we can create potential radar lines for each stem line. Each radar is connected to the respective
    stem line perpendicular to one another where the stem line connects to the center of the newly created radar line.
    These radar lines are the potential waypoints, aptly named radar because they're using to detect intersections with
    the boundaries.

    By creating these radar lines we can see the plausible places where the next waypoint should exists, and therefore
    work out which of these lines is the next waypoints ('__get_most_perpendicular_line_to_boundary' is used for this).
    There are a few parameters which govern how algorithm works. The spacing governs the length of the stem line, and
    therefore how far from the origin the radar line should spawn. Line count is how many radar lines are created
    doing this. More radar lines means more accurate waypoints, however it also means more calculation -> more lag.
    Angle span is the total angle you wish the stem lines to span across. Length is how long the radar lines should be.
    These radar lines are bounded between points based up on this length. If the lines are too long then we'll so
    lots of potential matches which means more calculation, but if it's too short then we may miss genuine matches.
    Reverse will flip the orientation of the line. We always want line.a to be on the left of the car. When the negative
    waypoints are being calculated then it is important to flip line or the left of the line or else line.a will be on
    the right of the car.

    :param initial_point: Initial point to stem from.
    :param initial_angle: angle of the car, heading direction of the waypoints
    :param spacing: How far should the radar lines be from the origin
    :param line_count: How many radar lines should be created
    :param angle_span: Total radians that the stem lines span across
    :param length: Length of each radar line
    :param reverse: Are the waypoints being generated behind the car, if so flip line.a:line.b
    :return: List of radar lines (potential waypoints)
    """
    # store the list of tuples of sub-lines
    sub_lines: List[Tuple[List[float], List[float]]] = []

    # since the stem lines span over some angle, we calculate a starting angle and delta angle
    angle_change = angle_span / (line_count - 1)
    starting = initial_angle - angle_span / 2

    # iterate x time for each line in the line count ot create
    for i in range(line_count):
        # first calculate which angle the line will stem from
        current_angle = starting + (i * angle_change)

        # then create a point which to create the radar lines from and rotate it around the initial point
        # this create the stem line, which the radar line can be drawn from
        p = geometry.rotate([initial_point[0] + spacing, initial_point[1]], current_angle, initial_point)

        # create left point and right point of the line respectively
        la = [p[0], p[1] - length / 2]
        lb = [p[0], p[1] + length / 2]

        # in the event the track is going backwards for the negative
        # waypoints, then we need to flip either end of the line
        if reverse:
            la, lb = lb, la

        # around the current point, in order to create the arching line which is
        # perpendicular line to the  outwards line
        la = geometry.rotate(la, current_angle, p)
        lb = geometry.rotate(lb, current_angle, p)

        # create a line which is the collection of sub-lines
        sub_lines.append(([la[0], la[1], p[0], p[1]], [lb[0], lb[1], p[0], p[1]]))
    return sub_lines


def __get_most_perpendicular_line_to_boundary(
        lines: List[Tuple[List[float], List[float]]],
        blue_boundary: List[List[float]],
        yellow_boundary: List[List[float]],
        orange_boundary: List[List[float]],
        bias: float = 0,
        bias_strength: float = 0.2,
        left_colour: int = BLUE_ON_LEFT
) -> Optional[Waypoint]:
    """
    This function will return a line from a list of line that is the most perpendicular to the boundary.
    These lines are made up of sub-lines, which stem from the center of the line. Then we can then calculate where
    the closest point of the sub-line[0] intersects with the left of hte track and where sub-line[1] intersects
    with the right of the track. By doing this we can then see which line has the closest pair of intersections
    and is therefore the most perpendicular to the track, making it a suitable waypoint.

    Additionally we can bias these waypoints. A bias of 0 means tend straight, -1 == tend left and 1 == tend right.
    We can also use the bias strength to determine how strongly to affect the biasing. This means in the event that
    there is a choice in the track, then we can bias where the waypoints tend in order to choose that path. We can
    alter and choose the shortest line like this: length = length * bias_value * bias_strength. Where the bias value
    is the bias at that angle.

    :param lines: The plausible waypoints to find the best waypoint from
    :param blue_boundary: Blue boundary of the track
    :param yellow_boundary: Yellow boundary of the track
    :param orange_boundary: Orange boundary of the track
    :param bias: Where the line should tend towards -> [-1: 1]
    :param bias_strength: How strongly the bias affects the decision -> [0: 1]
    :param left_colour: enum stating whether the blue or yellow is on the left of the track.
    :return: The most suitable waypoint for the given parameters
    """
    # buffer the line_count so we don't need to keep calling it, adding lag
    line_count: int = len(lines)
    blue_orange_boundary = blue_boundary + orange_boundary if len(orange_boundary) > 0 else blue_boundary
    yellow_orange_boundary = yellow_boundary + orange_boundary if len(orange_boundary) > 0 else yellow_boundary
    if left_colour == BLUE_ON_LEFT:
        left_boundary = blue_orange_boundary
        right_boundary = yellow_orange_boundary
    else:
        left_boundary = yellow_orange_boundary
        right_boundary = blue_orange_boundary

    # bias_value = [1 - abs(((f + 0.5) / line_count) * 2 - 1) for f in range(line_count)]
    # bias_value = 1 - (geometry.clip(bias_values, 0, 1) * bias_strength)

    # loop through each line in the list of lines to find the more perpendicular line
    distances = []
    for i in range(line_count):
        line = lines[i]
        # set up defaults for the waypoints, optimum 0.5 (middle) and not sticky
        optimum, sticky = 0.5, False

        # dependant on what colour is on left, we can swap the way we search for points
        left_intersections = geometry.segment_intersections(line[0], left_boundary)
        right_intersections = geometry.segment_intersections(line[1], right_boundary)

        # check the amount of intersections.
        # if a line does not intersect then we can use the end of the
        # end of the line as a dummy intersection point to draw the
        # line shortest line to.
        if len(left_intersections) == 0 and len(right_intersections) == 0:
            left_intersections = [line[0][0:2]]
            right_intersections = [line[1][0:2]]
            sticky = True
        elif len(left_intersections) == 0:
            left_intersections = [line[0][0:2]]
            optimum = 1
            sticky = True
        elif len(right_intersections) == 0:
            right_intersections = [line[1][0:2]]
            optimum = 0
            sticky = True

        # find the two points closest to the center of the line (line[0].b == line[1].b === center of line)
        center_points = line[0][2:4]
        point_a = geometry.closest_point(center_points, left_intersections)
        point_b = geometry.closest_point(center_points, right_intersections)

        # we can then create a new heuristic distance which is the current distance affected by the bias
        bias_angle = ((i + 0.5) / line_count * 2 - 1)
        bias_value = max(.0, 1 - abs(bias_angle - bias))
        # we can then create a new heuristic distance which is the current distance affected by the bias
        distance = geometry.distance(point_a, point_b)
        distance -= distance * bias_value * bias_strength

        distances.append({
            "line": [point_a[0], point_a[1], point_b[0], point_b[1]],
            "optimum": optimum,
            "sticky": sticky,
            "distance": distance
        })

    perp_line = distances[0]
    for i in range(1, len(distances)):
        if distances[i]["distance"] < perp_line["distance"]:
            perp_line = distances[i]

    return Waypoint(line=perp_line["line"], optimum=perp_line["optimum"], sticky=perp_line["sticky"])


def __get_radar_lines_around_point(
        origin: List[float],
        angle: float,
        count: int = 10,
        length: float = 10,
        total_span: float = math.pi / 2
) -> List[Tuple[List[float], List[float]]]:
    """
    This function will create a list of lines that rotate around an origin point (where each line passes through the
    origin). The lines produce potential waypoints around a point which we can selected as a waypoint for a particular
    point. These lines are made up of sub-lines, which stem from the center of the line. The reason this is done is
    that we can then calculate where the closest point of the sub-line[0] intersects with the left of hte track and
    where sub-line[1] intersects with the right of the track. By doing this we can then see which line has the closest
    pair of intersections and is therefore the most perpendicular to the track, making it a suitable waypoint.

    These line are created similar to butterfly wings such that the lines are generated in an area (total_span) in
    radians. This allows the waypoints to only spawn through the side of the vehicle rather than infront/behind.

    :param origin: The point at which the radar lines should arch around
    :param angle: The that the arching lines should be facing
    :param count: How many lines arch about the point
    :param length: The distance of the line from the origin
    :param total_span: The total spread (either side of the direction the car faces) to create the lines around.
    :return: The lines around a particular point
    """
    # Store the list to test
    lines = []

    # if only one line is to be generated that it must span 0 radians
    if count <= 1:
        count = 1  # make sure the min value is one, in the event it is given 0
        total_span = 0

    # calculate the change in angle from each line to the next
    angle_change = total_span / count

    # Calculate the initial angle which is the left most starting point. We rotate it -pi/2 such that the angle
    # starts rotated to the left of the vehicle (where blue lines are), then we subtract half the span, since we
    # wish to allow the lines to spawn in an area from [-span/2 -> span/2], hence why we take the current angle
    # then subtract half pi, then half the total span
    initial_angle = (angle - math.pi / 2) - (total_span / 2)

    # create a line infront of the car X (count) times.
    for i in range(count):
        # create and rotate the point the correct amount
        p = geometry.rotate([origin[0] + length / 2, origin[1]], angle=initial_angle + angle_change * i, around=origin)

        p_adjusted = p - (p - origin) * 2
        line = ([p[0], p[1], origin[0], origin[1]], [p_adjusted[0], p_adjusted[1], origin[0], origin[1]])
        # create the line form the origin to this newly rotated point
        lines.append(line)

    return lines


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
        altered_margin = min(geometry.length(waypoint.line) / 2 - 0.01, margin)
        normalised_point = geometry.normalise(waypoint.line)  # calculate normalised vector
        if normalised_point is not None:
            normalised_point = geometry.scale(normalised_point, altered_margin)  # multiply the vector by the length of the margin
            waypoint.line[0:2] = geometry.add(waypoint.line[0:2], normalised_point)  # apply vector to line
            waypoint.line[2:4] = geometry.sub(waypoint.line[2:4], normalised_point)  # apply vector to line

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

    # calculate angles
    angles = geometry.angles([line.line for line in current_lines])

    # loop through each index in the range and and calculate a new angle for it
    for i in index_range:
        # create a new deep copy of the waypoint in which to mutate
        new_waypoint: Waypoint = current_lines[i].copy()

        # get the angle of the surrounding waypoints
        pre = angles[i - 1]
        cur = angles[i]
        nex = angles[(i + 1) % len(current_lines)]

        # create a normalised vector for each angle which we wish to add, then add the vectors and get an average
        # angle from that. You must never average two angles the normal way. Average of 359 and 1 is 0, not 180...
        x, y = math.cos(pre) + math.cos(cur) + math.cos(nex), math.sin(pre) + math.sin(cur) + math.sin(nex)

        # calculate angle of the sum of the vectors
        average_angle = math.atan2(y, x)

        # calculate the difference between the average angle and current angle
        delta_angle = average_angle - cur

        # rotate the end of each line in the waypoint .line.a, .line.b around the center of the line
        center_point = geometry.add(new_waypoint.line[0:2], (geometry.scale(geometry.sub(new_waypoint.line[2:4], new_waypoint.line[0:2]), 0.5)))
        new_waypoint.line[0:2] = geometry.rotate(new_waypoint.line[0:2], delta_angle, center_point)
        new_waypoint.line[2:4] = geometry.rotate(new_waypoint.line[2:4], delta_angle, center_point)

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
    waypoint_curvature = []

    # loop through each waypoint in range 1:len-1 and calculate the curvature
    # of that waypoint. We dont calculate either end of the waypoints because
    # each end must be added to the list of final waypoints in order to make
    # sure the generated line knows where it should heard towards.
    for i in range(1, waypoint_count - 1):
        # get the angle of the surounding lines
        p: float = geometry.angle(waypoints[i - 1].line)
        c: float = geometry.angle(waypoints[i].line)
        n: float = geometry.angle(waypoints[(i + 1) % waypoint_count].line)

        # calculate the absolute change in angle bounded between -pi:pi
        a: float = abs((p - c + math.pi) % (math.pi*2) - math.pi)
        a += abs((n - c + math.pi) % (math.pi*2) - math.pi)

        # alter the surrounding waypoints to apple the spread
        for j in range(-(spread + 1), spread + 1, 1):
            index = (i + j) % waypoint_count
            waypoint_curvature.append(max(a, waypoint_curvature[index]))

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


def encode(waypoints: List[Waypoint], central_index: int):
    def delta_line_angle(line_a: List[float], line_b: List[float]):
        current_angle = geometry.angle(line_a)
        other_angle = geometry.angle(line_b)

        difference = current_angle - other_angle

        while difference < -math.pi:
            difference += (math.pi * 2)
        while difference > math.pi:
            difference -= (math.pi * 2)
        return difference

    X = []
    for f in range(central_index + 1, len(waypoints)):
        current_center = geometry.line_center(waypoints[f].line)
        prev_center = geometry.line_center(waypoints[f - 1].line)

        X += [[
            geometry.length(waypoints[f].line),
            delta_line_angle([prev_center[0], prev_center[1], current_center[0], current_center[1]], waypoints[f-1].line) + (math.pi / 2),
            delta_line_angle(waypoints[f].line, waypoints[f-1].line),
            geometry.distance(current_center, prev_center)
        ]]

    X = [[
        geometry.length(waypoints[central_index].line), 0, 0, 0
    ]] + X

    for n in range(central_index - 1, -1, -1):
        current_center = geometry.line_center(waypoints[n].line)
        prev_center = geometry.line_center(waypoints[n + 1].line)

        X = [[
            geometry.length(waypoints[n].line),
            delta_line_angle([prev_center[0], prev_center[1], current_center[0], current_center[1]], waypoints[n+1].line) - (math.pi / 2),
            -(delta_line_angle(waypoints[n+1].line, waypoints[n].line)),
            geometry.distance(current_center, prev_center)
        ]] + X
    return X


