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
            render_line(pygame_screen, line, colour, scale, 1, 0, Point(0, 0), x_offset, y_offset)

    # draw cones into the scene
    for cone_data in cones:
        colour, radius, cone_list = cone_data
        for cone in cone_list:
            render_point(pygame_screen, cone.pos, colour, scale, 1, 0, Point(0, 0), radius, x_offset, y_offset)

    # Render points into the scene
    for point_data in points:
        colour, radius, points_list = point_data
        for point in points_list:
            render_point(pygame_screen, point, colour, scale, 1, 0, Point(0, 0), radius, x_offset, y_offset)

    # render the cars into the scene
    for car in cars:
        render_car(pygame_screen, car, scale, 1, 0, Point(0, 0), x_offset, y_offset)


def render_area(
        pygame_screen,
        camera_pos: Point,
        rotation: float,
        area: Tuple[int, int],
        resolution: int = 20,
        cones: List[Tuple[Tuple[int, int, int], float, List[Cone]]] = None,
        points: List[Tuple[Tuple[int, int, int], float, List[Point]]] = None,
        lines: List[Tuple[Tuple[int, int, int], float, List[Line]]] = None,
        cars: List[Car] = None,
        background: int = 255
):
    if cars is None: cars = []
    if points is None: points = []
    if cones is None: cones = []
    if lines is None: lines = []

    x_offset = area[0] / 2 - camera_pos.x
    y_offset = area[1] / 2 - camera_pos.y

    pygame_screen.fill(background)

    # render lines into the scene
    for line_data in lines:
        colour, radius, line_list = line_data
        for line in line_list:
            render_line(pygame_screen, line, colour, 1, resolution, rotation, camera_pos, x_offset, y_offset)

    # draw cones into the scene
    for cone_data in cones:
        colour, radius, cone_list = cone_data
        for cone in cone_list:
            render_point(pygame_screen, cone.pos, colour, 1, resolution, rotation, camera_pos, radius, x_offset, y_offset)

    # Render points into the scene
    for point_data in points:
        colour, radius, points_list = point_data
        for point in points_list:
            render_point(pygame_screen, point, colour, 1, resolution, rotation, camera_pos, radius, x_offset, y_offset)

    # render the cars into the scene
    for car in cars:
        render_car(pygame_screen, car, 1, resolution, rotation, camera_pos, x_offset, y_offset)

    return pygame_screen


def render_car(image, car: Car, scale, resolution, rotation, camera_pos, x_offset, y_offset):
    body_points = [
        Point(car.pos[0] + car.cg_to_front, car.pos[1] - car.width / 2),
        Point(car.pos[0] + car.cg_to_front, car.pos[1] + car.width / 2),
        Point(car.pos[0] - car.cg_to_rear, car.pos[1] + car.width / 2),
        Point(car.pos[0] - car.cg_to_rear, car.pos[1] - car.width / 2)
    ]
    rear_left_tire_points = [
        Point(car.pos[0] - car.cg_to_rear_axle + car.wheel_radius / 2, car.pos[1] - car.width / 2 - car.wheel_width),
        Point(car.pos[0] - car.cg_to_rear_axle + car.wheel_radius / 2, car.pos[1] - car.width / 2),
        Point(car.pos[0] - car.cg_to_rear_axle - car.wheel_radius / 2, car.pos[1] - car.width / 2),
        Point(car.pos[0] - car.cg_to_rear_axle - car.wheel_radius / 2, car.pos[1] - car.width / 2 - car.wheel_width)
    ]
    rear_right_tire_points = [
        Point(car.pos[0] - car.cg_to_rear_axle + car.wheel_radius / 2, car.pos[1] + car.width / 2 + car.wheel_width),
        Point(car.pos[0] - car.cg_to_rear_axle + car.wheel_radius / 2, car.pos[1] + car.width / 2),
        Point(car.pos[0] - car.cg_to_rear_axle - car.wheel_radius / 2, car.pos[1] + car.width / 2),
        Point(car.pos[0] - car.cg_to_rear_axle - car.wheel_radius / 2, car.pos[1] + car.width / 2 + car.wheel_width)
    ]
    front_left_tire_points = [
        Point(car.pos[0] + car.cg_to_front_axle + car.wheel_radius / 2, car.pos[1] - car.width / 2 - car.wheel_width),
        Point(car.pos[0] + car.cg_to_front_axle + car.wheel_radius / 2, car.pos[1] - car.width / 2),
        Point(car.pos[0] + car.cg_to_front_axle - car.wheel_radius / 2, car.pos[1] - car.width / 2),
        Point(car.pos[0] + car.cg_to_front_axle - car.wheel_radius / 2, car.pos[1] - car.width / 2 - car.wheel_width)
    ]
    front_right_tire_points = [
        Point(car.pos[0] + car.cg_to_front_axle + car.wheel_radius / 2, car.pos[1] + car.width / 2 + car.wheel_width),
        Point(car.pos[0] + car.cg_to_front_axle + car.wheel_radius / 2, car.pos[1] + car.width / 2),
        Point(car.pos[0] + car.cg_to_front_axle - car.wheel_radius / 2, car.pos[1] + car.width / 2),
        Point(car.pos[0] + car.cg_to_front_axle - car.wheel_radius / 2, car.pos[1] + car.width / 2 + car.wheel_width)
    ]

    for point in body_points:
        point.rotate_around(car.pos, car.heading)
    for point in rear_left_tire_points:
        point.rotate_around(car.pos, car.heading)
    for point in rear_right_tire_points:
        point.rotate_around(car.pos, car.heading)
    for point in front_left_tire_points:
        point.rotate_around(car.pos, car.heading)
        point.rotate_around(Point(car.pos[0] + car.cg_to_front_axle, car.pos[1] - (car.width / 2) - (car.wheel_width / 2)), car.steer * car.max_steer)
    for point in front_right_tire_points:
        point.rotate_around(car.pos, car.heading)
        point.rotate_around(Point(car.pos[0] + car.cg_to_front_axle, car.pos[1] + (car.width / 2) + (car.wheel_width / 2)), car.steer * car.max_steer)

    render_polygon(image, body_points, (0, 0, 255), scale, resolution, rotation, camera_pos, x_offset, y_offset)
    render_polygon(image, rear_left_tire_points, (100, 100, 100), scale, resolution, rotation, camera_pos, x_offset, y_offset)
    render_polygon(image, rear_right_tire_points, (100, 100, 100), scale, resolution, rotation, camera_pos, x_offset, y_offset)
    render_polygon(image, front_left_tire_points, (100, 100, 100), scale, resolution, rotation, camera_pos, x_offset, y_offset)
    render_polygon(image, front_right_tire_points, (100, 100, 100), scale, resolution, rotation, camera_pos, x_offset, y_offset)


def render_polygon(
        pygame_scene,
        points: List[Point],
        color,
        scale,
        resolution,
        rotation,
        rotation_center,
        offset_x: int,
        offset_y: int):
    """
    This function is used to render a polygon into the pygame scene. When given a load of points in global space, this
    function will scale, apply padding and offset each point by the given parameters an draw it into the given scene in
    order to draw the polygon that is the collection of points.

    :param pygame_scene: The pygame scene in which to draw onto
    :param points: The list of points which make up the polygon
    :param color: The colour of the polygon
    :param scale: Scalar value to scale the points by
    :param resolution: Resolution of the scene
    :param rotation: radians to rotate the points around
    :param rotation_center: Rotational center to rotate the points around
    :param offset_x: An offset to apply to the x axis
    :param offset_y: An offset to apply to the y axis
    :param scale: A value to scale all points by
    :return: None
    """
    rotated_points = []
    for point in points:
        p = point.copy()
        p.rotate_around(rotation_center, rotation)
        rotated_points.append(p)

    polygon_points = []
    for point in rotated_points:
        # Translate points
        polygon_points.append((
            int((point.x * scale + offset_x) * resolution),
            int((point.y * scale + offset_y) * resolution)
        ))

    pygame.draw.polygon(pygame_scene, color, polygon_points)


def render_point(
        pygame_scene,
        point: Point,
        colour: Tuple[int, int, int],
        scale: float,
        resolution: float,
        rotation: float,
        rotation_center: Point,
        radius: float,
        x_offset: int,
        y_offset: int):
    """
    Render a point onto the image provided.

    :param pygame_scene: Image to render upon.
    :param point: Point position to draw upon.
    :param colour: Color of the point
    :param scale: Scale to render the point.
    :param radius: Radius of the point
    :param resolution: Resolution of the image to render the line upon
    :param rotation: Radians to rate the point
    :param rotation_center: Point to rotate the point around
    :param x_offset: x offset point.
    :param y_offset: y offset point.
    :return: image with the rendered lines.
    """
    p = point.copy()
    p.rotate_around(rotation_center, rotation)
    p.x = (p.x * scale + x_offset) * resolution
    p.y = (p.y * scale + y_offset) * resolution
    pygame.draw.circle(
        pygame_scene,
        colour,
        (int(round(p.x)), int(round(p.y))),
        radius
    )


def render_line(
        pygame_screen,
        line: Line,
        colour: Tuple[int, int, int],
        scale: float,
        resolution: float,
        rotation,
        rotation_center,
        x_offset: int,
        y_offset: int):
    """
    Draw line on the image with the given parameters. Used for drawing track boundary lines.

    :param pygame_screen: Screen to render upon.
    :param line: Line to render.
    :param colour: Colour of each line.
    :param scale: Scale to render the line.
    :param resolution: Resolution of the line
    :param rotation: How much the line should be rotated
    :param rotation_center: The point of which to rotate the line around
    :param x_offset: x offset line.
    :param y_offset: y offset line.
    :return: image with the rendered lines.
    """
    a = line.a.copy()
    a.rotate_around(rotation_center, rotation)
    a.x = (a.x * scale + x_offset) * resolution
    a.y = (a.y * scale + y_offset) * resolution
    b = line.b.copy()
    b.rotate_around(rotation_center, rotation)
    b.x = (b.x * scale + x_offset) * resolution
    b.y = (b.y * scale + y_offset) * resolution

    pygame.draw.line(
        pygame_screen,
        colour,
        (int(round(a.x)), int(round(a.y))),
        (int(round(b.x)), int(round(b.y))),
        2
    )
