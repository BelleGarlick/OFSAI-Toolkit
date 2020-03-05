import math
from typing import List, Tuple

import cv2
import numpy as np

from fsai.objects.cone import CONE_COLOR_BIG_ORANGE, CONE_COLOR_ORANGE, CONE_COLOR_YELLOW, CONE_COLOR_BLUE, Cone
from fsai.objects.line import Line
from fsai.objects.point import Point
from fsai.objects.track import Track


def draw_track(
        track: Track = None,
        cones: List[Cone] = None,
        blue_cones: List[Cone] = None,
        yellow_cones: List[Cone] = None,
        orange_cones: List[Cone] = None,
        big_orange_cones: List[Cone] = None,

        blue_lines: List[Line] = None,
        yellow_lines: List[Line] = None,
        orange_lines: List[Line] = None,

        blue_line_colour: Tuple[int, int, int] = (255, 0, 0),
        yellow_line_colour: Tuple[int, int, int] = (0, 255, 255),
        orange_line_colour: Tuple[int, int, int] = (0, 0, 255),

        blue_cone_colour: Tuple[int, int, int] = (255, 0, 0),
        yellow_cone_colour: Tuple[int, int, int] = (0, 255, 255),
        orange_cone_colour: Tuple[int, int, int] = (0, 100, 255),
        big_orange_cone_colour: Tuple[int, int, int] = (0, 0, 255),

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
    :param big_orange_cones: If provided, these cones will be drawn.
    :param blue_lines: If provided, these lines will be drawn.
    :param yellow_lines: If provided, these lines will be drawn.
    :param orange_lines: If provided, these lines will be drawn.
    :param blue_line_colour: The colour in which to render blue lines.
    :param yellow_line_colour: The colour in which to render yellow lines.
    :param orange_line_colour: The colour in which to render orange lines.
    :param blue_cone_colour: The colour in which to render blue cones.
    :param yellow_cone_colour: The colour in which to render yellow cones.
    :param orange_cone_colour: The colour in which to render orange cones.
    :param big_orange_cone_colour: The colour in which to render big orange cones.
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
    if big_orange_cones is not None: cones = cones + big_orange_cones
    if track is not None:
        if track.blue_cones is not None: cones = cones + track.blue_cones
        if track.yellow_cones is not None: cones = cones + track.yellow_cones
        if track.orange_cones is not None: cones = cones + track.orange_cones
        if track.big_orange_cones is not None: cones = cones + track.big_orange_cones

    # declare object lines
    if blue_lines is None: blue_lines = []
    if yellow_lines is None: yellow_lines = []
    if orange_lines is None: orange_lines = []

    # Work out bounds based upon all objects in scene
    min_x, min_y, max_x, max_y = __get_image_bounds(cones, blue_lines + yellow_lines + orange_lines)
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
    render_lines(image, blue_lines, blue_line_colour, scale, x_offset, y_offset)
    render_lines(image, yellow_lines, yellow_line_colour, scale, x_offset, y_offset)
    render_lines(image, orange_lines, orange_line_colour, scale, x_offset, y_offset)

    # render cones
    for cone in cones:
        color = (255, 255, 255)
        if cone.color == CONE_COLOR_BLUE: color = blue_cone_colour
        if cone.color == CONE_COLOR_YELLOW: color = yellow_cone_colour
        if cone.color == CONE_COLOR_ORANGE: color = orange_cone_colour
        if cone.color == CONE_COLOR_BIG_ORANGE: color = big_orange_cone_colour
        image = render_point(image, cone.pos, color, scale, 4, x_offset, y_offset)

    return image / 255


def __get_image_bounds(cones: List[Cone], lines: List[Line]):
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
    for lines in lines:
        min_x = min([min_x, lines.b.x, lines.a.x])
        min_y = min([min_y, lines.b.y, lines.a.y])
        max_x = max([max_x, lines.b.x, lines.a.x])
        max_y = max([max_y, lines.b.y, lines.a.y])

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


def render_lines(image, lines: List[Line], colour: Tuple[int, int, int], scale: float, x_offset: int, y_offset: int):
    """
    Draw line on the image with the given parameters. Used for drawing track boundary lines.

    :param image: Image to render upon.
    :param lines: List of the lines to render.
    :param colour: Colour of each line.
    :param scale: Scale to render the line.
    :param x_offset: x offset line.
    :param y_offset: y offset line.
    :return: image with the rendered lines.
    """
    for line in lines:
        image = cv2.line(
            image,
            (int(round(line.a.x * scale + x_offset)), int(round(line.a.y * scale + y_offset))),
            (int(round(line.b.x * scale + x_offset)), int(round(line.b.y * scale + y_offset))),
            colour,
            2
        )
    return image
