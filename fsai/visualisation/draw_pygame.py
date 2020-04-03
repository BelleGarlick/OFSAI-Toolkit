from typing import List, Tuple
import pygame
from fsai.car.car import Car
from fsai.objects.cone import Cone
from fsai.objects.line import Line
from fsai.objects.point import Point
from fsai.utils.visualise_2d import calculate_translations


def render(
        pygame_screen,
        screen_size,
        cones: List[Tuple[Tuple[int, int, int], float, List[Cone]]] = None,
        points: List[Tuple[Tuple[int, int, int], float, List[Point]]] = None,
        lines: List[Tuple[Tuple[int, int, int], float, List[Line]]] = None,
        cars: List[Car] = None,
        background: int = 0,
        padding: int = 40
):
    """
    This method is used to draw items into the given pygame scene.

    :param pygame_screen: The scene in which to draw upon.
    :param screen_size: The size of the pygame screen size. Used to scale the objects correctly.
    :param cones: If provided cone objects can be drawn. Given in the format [(colour, radius, [cones])]
    :param points: If provided points objects can be drawn. Given in the format [(colour, radius, [points])]
    :param lines: If provided points objects can be drawn. Given in the format [(colour, width, [lines])]
    :param cars: The list of cars to render into the scene. Given in the format [car]
    :param background: Set the background color in the pygame scene
    :param padding: Add some padding between the window border and the rendered scene for aesthetic purposes
    """
    if cars is None: cars = []
    if points is None: points = []

    x_offset, y_offset, scale = calculate_translations(
        cones, points, lines, screen_size, padding
    )

    pygame_screen.fill(background)

    # render lines into the scene
    for line_data in lines:
        colour, radius, line_list = line_data
        for line in line_list:
            altered_x1 = int(line.a.x * scale + x_offset)
            altered_y1 = int(line.a.y * scale + y_offset)
            altered_x2 = int(line.b.x * scale + x_offset)
            altered_y2 = int(line.b.y * scale + y_offset)

            pygame.draw.line(pygame_screen, colour, (altered_x1, altered_y1), (altered_x2, altered_y2), radius)

    # draw cones into the scene
    for cone_data in cones:
        colour, radius, cone_list = cone_data
        for cone in cone_list:
            altered_x = int(cone.pos.x * scale + x_offset)
            altered_y = int(cone.pos.y * scale + y_offset)
            pygame.draw.circle(pygame_screen, colour, (altered_x, altered_y), radius)

    # Render points into the scene
    for point_data in points:
        colour, radius, points_list = point_data
        render_point_pygame(pygame_screen, points, colour, radius, x_offset, y_offset, scale)

    # render the cars into the scene
    for car in cars:
        render_car_pygame(pygame_screen, car, x_offset, y_offset, scale)


def render_car_pygame(
        pygame_scene,
        car,
        x_offset,
        y_offset,
        scale):
    """
    This function is used to render a car into the pygame scene. When given a car object and some additional parameters
    the chassis of the car will be rendered into the scene.

    :param pygame_scene: Pygame scene in which to render the car
    :param car: The car in which to render into the scene
    :param x_offset: An offset to apply to the x axis
    :param y_offset: An offset to apply to the y axis
    :param scale: A value to scale all points by
    :return:
    """
    # draw chassis of the car.
    body_points = [
        Point(car.pos.x + car.cg_to_front, car.pos.y - car.width / 2),
        Point(car.pos.x + car.cg_to_front, car.pos.y + car.width/2),
        Point(car.pos.x - car.cg_to_rear, car.pos.y + car.width/2),
        Point(car.pos.x - car.cg_to_rear, car.pos.y - car.width/2)
    ]

    # rotate the chassis to orientate it to the car's heading.
    for point in body_points:
        point.rotate_around(car.pos, car.heading)

    # render a polygon into the scene that makes up the chassis of the car.
    render_polygon_pygame(pygame_scene, body_points, (255, 0, 255), x_offset, y_offset, scale)


def render_point_pygame(
        pygame_scene,
        points_list: List[Point],
        colour: Tuple[int, int, int],
        radius: float,
        offset_x: int,
        offset_y: int,
        scale: float):
    """
    This function is used to render a polygon into the pygame scene. When given a load of points in global space, this
    function will scale, apply padding, applying a radius and offset each point by the given parameters an draw it into
    the given scene.

    :param pygame_scene: The pygame scene in which to draw onto
    :param points_list: The list of points which make up the polygon
    :param colour: The colour of the polygon
    :param radius: The radius of each point
    :param offset_x: An offset to apply to the x axis
    :param offset_y: An offset to apply to the y axis
    :param scale: A value to scale all points by
    :return:
    """
    for point in points_list:
        altered_x = int(point.x * scale + offset_x)
        altered_y = int(point.y * scale + offset_y)
        pygame.draw.circle(pygame_scene, colour, (altered_x, altered_y), radius)


def render_polygon_pygame(
        pygame_scene,
        points: List[Point],
        colour: Tuple[int, int, int],
        offset_x: int,
        offset_y: int,
        scale: float):
    """
    This function is used to render a polygon into the pygame scene. When given a load of points in global space, this
    function will scale, apply padding and offset each point by the given parameters an draw it into the given scene in
    order to draw the polygon that is the collection of points.

    :param pygame_scene: The pygame scene in which to draw onto
    :param points: The list of points which make up the polygon
    :param colour: The colour of the polygon
    :param offset_x: An offset to apply to the x axis
    :param offset_y: An offset to apply to the y axis
    :param scale: A value to scale all points by
    :return: None
    """
    polygon_points = []
    for point in points:
        # Translate points
        polygon_points.append((
            int(point.x * scale + offset_x),
            int(point.y * scale + offset_y)
        ))

    # Draw
    pygame.draw.polygon(pygame_scene, colour, polygon_points)
