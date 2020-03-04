import math

import cv2
import numpy as np


def draw(
        blue_cones=None,
        yellow_cones=None,
        orange_cones=None,
        big_orange_cones=None,

        blue_lines=None,
        yellow_lines=None,
        orange_lines=None,

        background: int = 255,

        scale: int = 10,
        padding: int = 10
):
    if blue_cones is None: blue_cones = []
    if yellow_cones is None: yellow_cones = []
    if orange_cones is None: orange_cones = []
    if big_orange_cones is None: big_orange_cones = []

    if blue_lines is None: blue_lines = []
    if yellow_lines is None: yellow_lines = []
    if orange_lines is None: orange_lines = []

    # TODO Work out bounds based upon all objects in scene
    min_x, min_y, max_x, max_y = math.inf, math.inf, -math.inf, -math.inf
    all_cones = blue_cones + yellow_cones + orange_cones + big_orange_cones
    for cone in all_cones:
        min_x = min(min_x, cone.pos.x)
        min_y = min(min_y, cone.pos.y)
        max_x = max(max_x, cone.pos.x)
        max_y = max(max_y, cone.pos.y)
    x_offset, y_offset = -min_x * scale + padding, -min_y * scale + padding

    image = np.zeros((
        int((max_y - min_y) * scale + 2 * padding),
        int((max_x - min_x) * scale + 2 * padding),
        3
    ))
    image.fill(background)

    render_lines(image, blue_lines, (255, 0, 0), scale, x_offset, y_offset)
    render_lines(image, yellow_lines, (0, 255, 255), scale, x_offset, y_offset)
    render_lines(image, orange_lines, (0, 100, 255), scale, x_offset, y_offset)

    if blue_cones is not None: render_points(image, blue_cones, (255, 0, 0), scale, 4, x_offset, y_offset)
    if yellow_cones is not None: render_points(image, yellow_cones, (0, 255, 255), scale, 4, x_offset, y_offset)
    if orange_cones is not None: render_points(image, orange_cones, (0, 100, 255), scale, 4, x_offset, y_offset)
    if big_orange_cones is not None: render_points(image, big_orange_cones, (0, 100, 255), scale, 4, x_offset, y_offset)

    return image / 255


def render_points(image, points, colour, scale, radius, offset_x, offset_y):
    for blue_cone in points:
        image = cv2.circle(
            image,
            (int(round(blue_cone.pos.x * scale + offset_x)), int(round(blue_cone.pos.y * scale + offset_y))),
            radius,
            colour,
            -1
        )
    return image


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
