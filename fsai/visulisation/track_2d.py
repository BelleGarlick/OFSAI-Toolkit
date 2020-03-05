import math

import cv2
import numpy as np

from fsai.objects.cone import CONE_COLOR_BIG_ORANGE, CONE_COLOR_ORANGE, CONE_COLOR_YELLOW, CONE_COLOR_BLUE


def draw_track(
        cones=None,

        blue_lines=None,
        yellow_lines=None,
        orange_lines=None,

        background: int = 0,

        scale: int = 10,
        padding: int = 40
):
    if cones is None: cones = []

    if blue_lines is None: blue_lines = []
    if yellow_lines is None: yellow_lines = []
    if orange_lines is None: orange_lines = []

    # TODO Work out bounds based upon all objects in scene
    min_x, min_y, max_x, max_y = math.inf, math.inf, -math.inf, -math.inf
    for cone in cones:
        min_x = min(min_x, cone.pos.x)
        min_y = min(min_y, cone.pos.y)
        max_x = max(max_x, cone.pos.x)
        max_y = max(max_y, cone.pos.y)
    for lines in blue_lines + yellow_lines + orange_lines:
        min_x = min(min_x, lines.a.x)
        min_y = min(min_y, lines.a.y)
        max_x = max(max_x, lines.a.x)
        max_y = max(max_y, lines.a.y)
        min_x = min(min_x, lines.b.x)
        min_y = min(min_y, lines.b.y)
        max_x = max(max_x, lines.b.x)
        max_y = max(max_y, lines.b.y)

    x_offset, y_offset = -min_x * scale + padding, -min_y * scale + padding
    if max_x == -math.inf:
        max_x, max_y, min_x, min_y = 0, 0, 0, 0

    image = np.zeros((
        int((max_y - min_y) * scale + 2 * padding),
        int((max_x - min_x) * scale + 2 * padding),
        3
    ))
    image.fill(background)

    render_lines(image, blue_lines, (255, 0, 0), scale, x_offset, y_offset)
    render_lines(image, yellow_lines, (0, 255, 255), scale, x_offset, y_offset)
    render_lines(image, orange_lines, (0, 100, 255), scale, x_offset, y_offset)

    # render cones
    for cone in cones:
        color = (255, 255, 255)
        if cone.color == CONE_COLOR_BLUE: color = (255, 0, 0)
        if cone.color == CONE_COLOR_YELLOW: color = (0, 255, 255)
        if cone.color == CONE_COLOR_ORANGE: color = (0, 100, 255)
        if cone.color == CONE_COLOR_BIG_ORANGE: color = (0, 0, 255)
        image = render_point(image, cone, color, scale, 4, x_offset, y_offset)

    return image / 255


def render_point(image, cone, colour, scale, radius, offset_x, offset_y):
    return cv2.circle(
        image,
        (int(round(cone.pos.x * scale + offset_x)), int(round(cone.pos.y * scale + offset_y))),
        radius,
        colour,
        -1
    )


def render_lines(image, lines, colour, scale, x_offset, y_offset):
    for line in lines:
        image = cv2.line(
            image,
            (int(round(line.a.x * scale + x_offset)), int(round(line.a.y * scale + y_offset))),
            (int(round(line.b.x * scale + x_offset)), int(round(line.b.y * scale + y_offset))),
            colour,
            2
        )
    return image
