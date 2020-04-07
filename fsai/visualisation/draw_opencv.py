from typing import List, Tuple

import cv2
import numpy as np

from fsai.car.car import Car
from fsai.objects.cone import Cone
from fsai.objects.line import Line
from fsai.objects.point import Point
from fsai.utils.visualise_2d import calculate_translations


def render(
        image_size: Tuple[int, int],
        cones: List[Tuple[Tuple[int, int, int], float, List[Cone]]] = None,
        points: List[Tuple[Tuple[int, int, int], float, List[Point]]] = None,
        lines: List[Tuple[Tuple[int, int, int], float, List[Line]]] = None,
        cars: List[Car] = None,
        background: int = 0,
        padding: int = 40
):
    """
    This method is used to draw items into an open cv image

    :param image_size: The size of the image size. Used to scale the objects correctly.
    :param cones: If provided cone objects can be drawn. Given in the format [(colour, radius, [cones])]
    :param points: If provided points objects can be drawn. Given in the format [(colour, radius, [points])]
    :param lines: If provided points objects can be drawn. Given in the format [(colour, width, [lines])]
    :param cars: The list of cars to render into the scene. Given in the format [car]
    :param background: Set the background color in the cv2 scene
    :param padding: Add some padding between the window border and the rendered scene for aesthetic purposes
    """
    if cars is None: cars = []
    if points is None: points = []
    if cones is None: cones = []
    if lines is None: lines = []

    x_offset, y_offset, scale = calculate_translations(
        cones, points, lines, image_size, padding
    )

    # x_offset, y_offset = -min_x * scale + padding, -min_y * scale + padding
    image = np.zeros((image_size[1], image_size[0], 3))
    image.fill(background)

    # render lines into the scene
    for line_data in lines:
        colour, radius, line_list = line_data
        for line in line_list:
            render_line(image, line, colour, scale, x_offset, y_offset)

    # draw cones into the scene
    for cone_data in cones:
        colour, radius, cone_list = cone_data
        for cone in cone_list:
            render_point(image, cone.pos, colour, scale, radius, x_offset, y_offset)

    # Render points into the scene
    for point_data in points:
        colour, radius, points_list = point_data
        for point in points_list:
            render_point(image, point, colour, scale, radius, x_offset, y_offset)

    # render the cars into the scene
    for car in cars:
        render_car(image, car, scale, x_offset, y_offset)

    return image


def render_car(image, car: Car, scale, x_offset, y_offset):
    body_points = [
        Point(car.pos.x + car.cg_to_front, car.pos.y - car.width / 2),
        Point(car.pos.x + car.cg_to_front, car.pos.y + car.width/2),
        Point(car.pos.x - car.cg_to_rear, car.pos.y - car.width/2),
        Point(car.pos.x - car.cg_to_rear, car.pos.y + car.width/2)
    ]
    for point in body_points:
        point.rotate_around(car.pos, car.heading)
        render_point(image, point, (255, 0, 255), scale, 6, x_offset, y_offset)


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
