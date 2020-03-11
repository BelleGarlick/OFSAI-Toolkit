import math
from typing import List, Tuple

import cv2
import numpy as np

from fsai.objects.car import Car
from fsai.objects.cone import CONE_COLOR_BIG_ORANGE, CONE_COLOR_ORANGE, CONE_COLOR_YELLOW, CONE_COLOR_BLUE, Cone
from fsai.objects.line import Line
from fsai.objects.point import Point
from fsai.objects.track import Track
from fsai.objects.waypoint import Waypoint

# TODO Recomment all of this


def draw_track(
        track: Track = None,

        cones: List[Cone] = None,
        blue_cones: List[Cone] = None,
        yellow_cones: List[Cone] = None,
        orange_cones: List[Cone] = None,
        big_cones: List[Cone] = None,

        blue_lines: List[Line] = None,
        yellow_lines: List[Line] = None,
        orange_lines: List[Line] = None,

        waypoints: List[Waypoint] = None,
        pedestrians: List[Point] = None,
        target_line: List[Point] = None,

        blue_line_colour: Tuple[int, int, int] = (255, 0, 0),
        yellow_line_colour: Tuple[int, int, int] = (0, 255, 255),
        orange_line_colour: Tuple[int, int, int] = (0, 0, 255),

        blue_cone_colour: Tuple[int, int, int] = (255, 0, 0),
        yellow_cone_colour: Tuple[int, int, int] = (0, 255, 255),
        orange_cone_colour: Tuple[int, int, int] = (0, 100, 255),
        big_cone_colour: Tuple[int, int, int] = (0, 0, 255),

        cars: List[Car] = None,

        background: int = 0,

        scale: int = 10,
        padding: int = 40
):
    """
    This method is used to draw the given items in the scene onto an image. Cones and lines can be provided,
    additionally the image can be scaled and padded for aesthetic reasons. Colours can be customised to suit
    you liking by altering the method parameters.

    :param track: If provided, the track data will be draw.
    :param cones: If provided, these cones will be drawn.
    :param blue_cones: If provided, these cones will be drawn.
    :param yellow_cones: If provided, these cones will be drawn.
    :param orange_cones: If provided, these cones will be drawn.
    :param big_cones: If provided, these cones will be drawn.
    :param blue_lines: If provided, these lines will be drawn.
    :param yellow_lines: If provided, these lines will be drawn.
    :param orange_lines: If provided, these lines will be drawn.
    :param waypoints: If provided, these waypoints will be draw.
    :param blue_line_colour: The colour in which to render blue lines.
    :param yellow_line_colour: The colour in which to render yellow lines.
    :param orange_line_colour: The colour in which to render orange lines.
    :param blue_cone_colour: The colour in which to render blue cones.
    :param yellow_cone_colour: The colour in which to render yellow cones.
    :param orange_cone_colour: The colour in which to render orange cones.
    :param big_cone_colour: The colour in which to render big orange cones.
    :param background: Background greyscale color to draw the scene onto (0 - 255) (int).
    :param scale: Scale the image of the track.
    :param padding: Add padding around the image.
    :return: cv2 image depicting the scene
    """
    # Merge all cones into one set
    if cones is None: cones = []
    if blue_cones is not None: cones = cones + blue_cones
    if yellow_cones is not None: cones = cones + yellow_cones
    if orange_cones is not None: cones = cones + orange_cones
    if big_cones is not None: cones = cones + big_cones
    if track is not None:
        if track.blue_cones is not None: cones = cones + track.blue_cones
        if track.yellow_cones is not None: cones = cones + track.yellow_cones
        if track.orange_cones is not None: cones = cones + track.orange_cones
        if track.big_cones is not None: cones = cones + track.big_cones

    # declare object lines
    if blue_lines is None: blue_lines = []
    if yellow_lines is None: yellow_lines = []
    if orange_lines is None: orange_lines = []

    if waypoints is None: waypoints = []
    if pedestrians is None: pedestrians = []
    if target_line is None: target_line = []

    if cars is None: cars = []
    cars = cars + track.cars

    # Work out bounds based upon all objects in scene
    min_x, min_y, max_x, max_y = __get_image_bounds(cones, blue_lines + yellow_lines + orange_lines, waypoints=waypoints)
    # calculate the scale/padding offset given the calculated boundaries
    x_offset, y_offset = -min_x * scale + padding, -min_y * scale + padding

    # create image
    image = np.zeros((
        int((max_y - min_y) * scale + 2 * padding),
        int((max_x - min_x) * scale + 2 * padding),
        3
    ))
    image.fill(background)

    # render lines
    for line in blue_lines:
        render_line(image, line, blue_line_colour, scale, x_offset, y_offset)

    for line in yellow_lines:
        render_line(image, line, yellow_line_colour, scale, x_offset, y_offset)

    for line in orange_lines:
        render_line(image, line, orange_line_colour, scale, x_offset, y_offset)

    for line in waypoints:
        render_line(image, line.line, (100, 100, 100), scale, x_offset, y_offset)
        render_point(image, line.get_optimum_point(), (0, 255, 0), scale, 4, x_offset, y_offset)

    # render cones
    for cone in cones:
        color = (255, 255, 255)
        if cone.color == CONE_COLOR_BLUE: color = blue_cone_colour
        if cone.color == CONE_COLOR_YELLOW: color = yellow_cone_colour
        if cone.color == CONE_COLOR_ORANGE: color = orange_cone_colour
        if cone.color == CONE_COLOR_BIG_ORANGE: color = big_cone_colour
        render_point(image, cone.pos, color, scale, 4, x_offset, y_offset)

    for car in cars:
        render_point(image, car.pos, (255, 0, 255), scale, 6, x_offset, y_offset)

    for point in pedestrians:
        if min_x<point.x<max_x and min_y<point.y < max_y:
            render_point(image, point, (255, 100, 255), scale, 4, x_offset, y_offset)

    for p in range(len(target_line) - 1):
        render_line(image, Line(a=target_line[p], b=target_line[p+1]), (255, 0, 255), scale, x_offset, y_offset)

    return image / 255


def __get_image_bounds(cones: List[Cone], lines: List[Line], waypoints: List[Waypoint]):
    """
    This method is used to work out the min, max boundaries of a track. These values are used to alter the draw
    positions of objects so that no matter where the track exists in world space it'll still be centered within
    the image.

    :param cones: List of all cones in the scene
    :param lines: List of all lines in the scene
    :return:
    """
    min_x, min_y, max_x, max_y = math.inf, math.inf, -math.inf, -math.inf
    for cone in cones:
        min_x = min(min_x, cone.pos.x)
        min_y = min(min_y, cone.pos.y)
        max_x = max(max_x, cone.pos.x)
        max_y = max(max_y, cone.pos.y)
    for line in lines:
        min_x = min([min_x, line.b.x, line.a.x])
        min_y = min([min_y, line.b.y, line.a.y])
        max_x = max([max_x, line.b.x, line.a.x])
        max_y = max([max_y, line.b.y, line.a.y])
    for waypoint in waypoints:
        min_x = min([min_x, waypoint.line.b.x, waypoint.line.a.x])
        min_y = min([min_y, waypoint.line.b.y, waypoint.line.a.y])
        max_x = max([max_x, waypoint.line.b.x, waypoint.line.a.x])
        max_y = max([max_y, waypoint.line.b.y, waypoint.line.a.y])

    if max_x == -math.inf:
        max_x, max_y, min_x, min_y = 0, 0, 0, 0

    return min_x, min_y, max_x, max_y


def render_point(
        image,
        point: Point,
        colour: Tuple[int, int, int],
        scale: float,
        radius: float,
        offset_x: int,
        offset_y: int):
    """
    Render a point onto the image provided.

    :param image: Image to render upon.
    :param point: Point position to draw upon.
    :param colour: Color of the point
    :param scale: Scale to render the point.
    :param radius: Radius of the point
    :param offset_x: x offset point.
    :param offset_y: y offset point.
    :return: image with the rendered lines.
    """
    return cv2.circle(
        image,
        (int(round(point.x * scale + offset_x)), int(round(point.y * scale + offset_y))),
        radius,
        colour,
        -1
    )


def render_line(image, line: Line, colour: Tuple[int, int, int], scale: float, x_offset: int, y_offset: int):
    """
    Draw line on the image with the given parameters. Used for drawing track boundary lines.

    :param image: Image to render upon.
    :param line: Line to render.
    :param colour: Colour of each line.
    :param scale: Scale to render the line.
    :param x_offset: x offset line.
    :param y_offset: y offset line.
    :return: image with the rendered lines.
    """
    image = cv2.line(
        image,
        (int(round(line.a.x * scale + x_offset)), int(round(line.a.y * scale + y_offset))),
        (int(round(line.b.x * scale + x_offset)), int(round(line.b.y * scale + y_offset))),
        colour,
        2
    )
    return image
