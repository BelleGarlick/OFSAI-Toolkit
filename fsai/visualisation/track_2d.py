import math
from typing import List, Tuple

import cv2
import numpy as np
import pygame

from fsai.car.car import Car
from fsai.objects.cone import Cone
from fsai.objects.line import Line
from fsai.objects.point import Point
from fsai.objects.track import Track

# TODO Recomment all of this


def draw_cv(
        track: Track = None,
        cones: List[Tuple[Tuple[int, int, int], float, List[Cone]]] = None,
        points: List[Tuple[Tuple[int, int, int], float, List[Point]]] = None,
        lines: List[Tuple[Tuple[int, int, int], float, List[Line]]] = None,
        text: List[Tuple[Tuple[int, int, int], Tuple[int, int], str]] = None,
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
    :param background: Background greyscale color to draw the scene onto (0 - 255) (int).
    :param scale: Scale the image of the track.
    :param padding: Add padding around the image.
    :return: cv2 image depicting the scene
    """
    if cars is None: cars = []
    if track is not None: cars = cars + track.cars
    if points is None: points = []

    # Work out bounds based upon all objects in scene
    # calculate the scale/padding offset given the calculated boundaries
    min_x, min_y, max_x, max_y = __get_image_bounds(cones, points, lines)
    x_offset, y_offset = -min_x * scale + padding, -min_y * scale + padding

    # create image
    image = np.zeros((
        int((max_y - min_y) * scale + 2 * padding),
        int((max_x - min_x) * scale + 2 * padding),
        3
    ))
    image.fill(background)

    for line_data in lines:
        colour, radius, line_list = line_data
        for line in line_list:
            render_line(image, line, colour, scale, x_offset, y_offset)

    for cone_data in cones:
        colour, radius, cone_list = cone_data
        for cone in cone_list:
            render_point(image, cone.pos, colour, scale, radius, x_offset, y_offset)
    # Render points
    for point_data in points:
        colour, radius, points_list = point_data
        for point in points_list:
            render_point(image, point, colour, scale, radius, x_offset, y_offset)

    for car in cars:
        render_car(image, car, scale, x_offset, y_offset)

    for text_data in text:
        colour, org, string = text_data
        # font
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Line thickness of 2 px
        thickness = 2
        cv2.putText(image, string, org, font,
                    1, colour, thickness, cv2.LINE_AA)

    return image / 255


def render_car(image, car, scale, x_offset, y_offset):
    body_points = [
        Point(car.pos.x + car.cg_to_front, car.pos.y - car.width / 2),
        Point(car.pos.x + car.cg_to_front, car.pos.y + car.width/2),
        Point(car.pos.x - car.cg_to_rear, car.pos.y - car.width/2),
        Point(car.pos.x - car.cg_to_rear, car.pos.y + car.width/2)
    ]
    for point in body_points:
        point.rotate_around(car.pos, car.heading)
        render_point(image, point, (255, 0, 255), scale, 6, x_offset, y_offset)


def __get_image_bounds(cones: List[Tuple[Tuple, float, List[Cone]]], points:  List[Tuple[Tuple, float, List[Point]]], lines:  List[Tuple[Tuple, float, List[Line]]]):
    """
    This method is used to work out the min, max boundaries of a track. These values are used to alter the draw
    positions of objects so that no matter where the track exists in world space it'll still be centered within
    the image.

    :param cones: List of all cones in the scene
    :param lines: List of all lines in the scene
    :return:
    """
    min_x, min_y, max_x, max_y = math.inf, math.inf, -math.inf, -math.inf
    for cone_tuples in cones:
        for cone in cone_tuples[2]:
            min_x = min(min_x, cone.pos.x)
            min_y = min(min_y, cone.pos.y)
            max_x = max(max_x, cone.pos.x)
            max_y = max(max_y, cone.pos.y)

    for point_tuples in points:
        for point in point_tuples[2]:
            min_x = min(min_x, point.x)
            min_y = min(min_y, point.y)
            max_x = max(max_x, point.x)
            max_y = max(max_y, point.y)

    for line_tuple in lines:
        for line in line_tuple[2]:
            min_x = min([min_x, line.b.x, line.a.x])
            min_y = min([min_y, line.b.y, line.a.y])
            max_x = max([max_x, line.b.x, line.a.x])
            max_y = max([max_y, line.b.y, line.a.y])

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


def draw_pygame(
        pygame_screen,
        pygame_screen_size,
        cones: List[Tuple[Tuple[int, int, int], float, List[Cone]]] = None,
        points: List[Tuple[Tuple[int, int, int], float, List[Point]]] = None,
        lines: List[Tuple[Tuple[int, int, int], float, List[Line]]] = None,
        text: List[Tuple[Tuple[int, int, int], Tuple[int, int], str]] = None,
        cars: List[Car] = None,
        background: int = 0,

        padding: int = 40
):
    """
    This method is used to draw the given items in the scene onto an image. Cones and lines can be provided,
    additionally the image can be scaled and padded for aesthetic reasons. Colours can be customised to suit
    you liking by altering the method parameters.

    :param track: If provided, the track data will be draw.
    :param cones: If provided, these cones will be drawn.
    :param background: Background greyscale color to draw the scene onto (0 - 255) (int).
    :param scale: Scale the image of the track.
    :param padding: Add padding around the image.
    :return: cv2 image depicting the scene
    """
    if cars is None: cars = []
    if points is None: points = []

    # # Work out bounds based upon all objects in scene
    # # calculate the scale/padding offset given the calculated boundaries
    min_x, min_y, max_x, max_y = __get_image_bounds(cones, points, lines)
    width = (max_x - min_x)
    height = (max_y - min_y) + (2 * padding)
    x_scale = (pygame_screen_size[0] - (2 * padding)) / width
    y_scale = (pygame_screen_size[1] - (2 * padding)) / height
    scale = max(x_scale, y_scale)

    # x_offset, y_offset = -min_x * scale + padding, -min_y * scale + padding

    pygame_screen.fill(background)

    for line_data in lines:
        colour, radius, line_list = line_data
        for line in line_list:
            altered_x1 = int((line.a.x - min_x) * scale + padding)
            altered_y1 = int((line.a.y - min_y) * scale + padding)
            altered_x2 = int((line.b.x - min_x) * scale + padding)
            altered_y2 = int((line.b.y - min_y) * scale + padding)

            pygame.draw.line(pygame_screen, colour, (altered_x1, altered_y1), (altered_x2, altered_y2), radius)

    for cone_data in cones:
        colour, radius, cone_list = cone_data
        for cone in cone_list:
            altered_x = int((cone.pos.x - min_x) * scale + padding)
            altered_y = int((cone.pos.y - min_y) * scale + padding)
            pygame.draw.circle(pygame_screen, colour, (altered_x, altered_y), radius)

    # Render points
    for point_data in points:
        colour, radius, points_list = point_data
        for point in points_list:
            render_point_pygame(pygame_screen, point, colour, scale, radius, -min_x, -min_y, padding)

    for car in cars:
        render_car(pygame_screen, car, scale, -min_x, -min_y, padding)

    # for text_data in text:
    #     colour, org, string = text_data
    #     # font
    #     # font = cv2.FONT_HERSHEY_SIMPLEX
    #
    #     pygame_screen.draw.text(string, (20, 100))
    #     # cv2.putText(image, string, org, font, 1, colour, thickness, cv2.LINE_AA)


def render_car(pygame_scene, car, scale, x_offset, y_offset, padding):
    body_points = [
        Point(car.pos.x + car.cg_to_front, car.pos.y - car.width / 2),
        Point(car.pos.x + car.cg_to_front, car.pos.y + car.width/2),
        Point(car.pos.x - car.cg_to_rear, car.pos.y + car.width/2),
        Point(car.pos.x - car.cg_to_rear, car.pos.y - car.width/2)
    ]
    for point in body_points:
        point.rotate_around(car.pos, car.heading)
    render_point_pygame(pygame_scene, body_points, (255, 0, 255), scale, x_offset, y_offset, padding)


def render_point_pygame(
        pygame_scene,
        point: Point,
        colour: Tuple[int, int, int],
        scale: float,
        radius: float,
        offset_x: int,
        offset_y: int,
        padding: float):
    altered_x = int((point.x + offset_x) * scale + padding)
    altered_y = int((point.y + offset_y) * scale + padding)
    pygame.draw.circle(pygame_scene, colour, (altered_x, altered_y), radius)


def render_point_pygame(
        pygame_scene,
        points: List[Point],
        colour: Tuple[int, int, int],
        scale: float,
        offset_x: int,
        offset_y: int,
        padding: float):

    polygon_points = []
    for point in points:
        polygon_points.append((
            int((point.x + offset_x) * scale + padding),
            int((point.y + offset_y) * scale + padding)
        ))
    pygame.draw.polygon(pygame_scene, colour, polygon_points)
