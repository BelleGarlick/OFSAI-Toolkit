from typing import List, Tuple

import cv2
import numpy as np

from fsai.car.car import Car
from fsai.objects import geometry
from fsai.objects.cone import Cone
from fsai.utils.visualise_2d import calculate_translations


def render(
        image_size: Tuple[int, int],
        cones: List[Tuple[Tuple[int, int, int], float, np.ndarray]] = None,
        points: List[Tuple[Tuple[int, int, int], float, np.ndarray]] = None,
        lines: List[Tuple[Tuple[int, int, int], float, List[np.ndarray]]] = None,
        cars: List[Car] = None,
        background: int = 255,
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

    origin = np.asarray([0, 0])

    # render lines into the scene
    for line_data in lines:
        colour, radius, line_list = line_data
        for line in line_list:
            render_line(image, line, colour, scale, 1, 0, origin, x_offset, y_offset)

    # draw cones into the scene
    for cone_data in cones:
        colour, radius, cone_list = cone_data
        for cone in cone_list:
            render_point(image, cone.pos, colour, scale, 1, 0, origin, radius, x_offset, y_offset)

    # Render points into the scene
    for point_data in points:
        colour, radius, points_list = point_data
        for point in points_list:
            render_point(image, point, colour, scale, 1, 0, origin, radius, x_offset, y_offset)

    # render the cars into the scene
    for car in cars:
        render_car(image, car, scale, 1, 0, origin, x_offset, y_offset)

    return image


def render_area(
        camera_pos: np.ndarray,
        rotation: float,
        area: Tuple[int, int],
        resolution: int = 20,
        cones: List[Tuple[Tuple[int, int, int], float, List[Cone]]] = None,
        points: List[Tuple[Tuple[int, int, int], float, np.ndarray]] = None,
        lines: List[Tuple[Tuple[int, int, int], float, List[np.ndarray]]] = None,
        cars: List[Car] = None,
        background: int = 255
):
    if cars is None: cars = []
    if points is None: points = []
    if cones is None: cones = []
    if lines is None: lines = []

    x_offset = area[0] / 2 - camera_pos[0]
    y_offset = area[1] / 2 - camera_pos[1]

    # x_offset, y_offset = -min_x * scale + padding, -min_y * scale + padding
    image = np.zeros((area[1] * resolution, area[0] * resolution, 3))
    image.fill(background)

    # render lines into the scene
    for line_data in lines:
        colour, radius, line_list = line_data
        for line in line_list:
            render_line(image, line, colour, 1, resolution, rotation, camera_pos, x_offset, y_offset)

    # draw cones into the scene
    for cone_data in cones:
        colour, radius, cone_list = cone_data
        for cone in cone_list:
            render_point(image, cone, colour, 1, resolution, rotation, camera_pos, radius, x_offset, y_offset)

    # Render points into the scene
    for point_data in points:
        colour, radius, points_list = point_data
        for point in points_list:
            render_point(image, point, colour, 1, resolution, rotation, camera_pos, radius, x_offset, y_offset)

    # render the cars into the scene
    for car in cars:
        render_car(image, car, 1, resolution, rotation, camera_pos, x_offset, y_offset)

    return image


def render_car(image, car: Car, scale, resolution, rotation, camera_pos, x_offset, y_offset):
    body_points = np.array([
        (car.pos[0] + car.cg_to_front, car.pos[1] - car.width / 2),
        (car.pos[0] + car.cg_to_front, car.pos[1] + car.width / 2),
        (car.pos[0] - car.cg_to_rear, car.pos[1] + car.width / 2),
        (car.pos[0] - car.cg_to_rear, car.pos[1] - car.width / 2)
    ])
    rear_left_tire_points = np.array([
        (car.pos[0] - car.cg_to_rear_axle + car.wheel_radius / 2, car.pos[1] - car.width / 2 - car.wheel_width),
        (car.pos[0] - car.cg_to_rear_axle + car.wheel_radius / 2, car.pos[1] - car.width / 2),
        (car.pos[0] - car.cg_to_rear_axle - car.wheel_radius / 2, car.pos[1] - car.width / 2),
        (car.pos[0] - car.cg_to_rear_axle - car.wheel_radius / 2, car.pos[1] - car.width / 2 - car.wheel_width)
    ])
    rear_right_tire_points = np.array([
        (car.pos[0] - car.cg_to_rear_axle + car.wheel_radius / 2, car.pos[1] + car.width / 2 + car.wheel_width),
        (car.pos[0] - car.cg_to_rear_axle + car.wheel_radius / 2, car.pos[1] + car.width / 2),
        (car.pos[0] - car.cg_to_rear_axle - car.wheel_radius / 2, car.pos[1] + car.width / 2),
        (car.pos[0] - car.cg_to_rear_axle - car.wheel_radius / 2, car.pos[1] + car.width / 2 + car.wheel_width)
    ])
    front_left_tire_points = np.array([
        (car.pos[0] + car.cg_to_front_axle + car.wheel_radius / 2, car.pos[1] - car.width / 2 - car.wheel_width),
        (car.pos[0] + car.cg_to_front_axle + car.wheel_radius / 2, car.pos[1] - car.width / 2),
        (car.pos[0] + car.cg_to_front_axle - car.wheel_radius / 2, car.pos[1] - car.width / 2),
        (car.pos[0] + car.cg_to_front_axle - car.wheel_radius / 2, car.pos[1] - car.width / 2 - car.wheel_width)
    ])
    front_right_tire_points = np.array([
        (car.pos[0] + car.cg_to_front_axle + car.wheel_radius / 2, car.pos[1] + car.width / 2 + car.wheel_width),
        (car.pos[0] + car.cg_to_front_axle + car.wheel_radius / 2, car.pos[1] + car.width / 2),
        (car.pos[0] + car.cg_to_front_axle - car.wheel_radius / 2, car.pos[1] + car.width / 2),
        (car.pos[0] + car.cg_to_front_axle - car.wheel_radius / 2, car.pos[1] + car.width / 2 + car.wheel_width)
    ])

    for i in range(len(body_points)):
        body_points[i] = geometry.rotate(body_points[i], car.heading, car.pos)
    for i in range(len(rear_left_tire_points)):
        rear_left_tire_points[i] = geometry.rotate(rear_left_tire_points[i], car.heading, car.pos)
    for i in range(len(rear_right_tire_points)):
        rear_right_tire_points[i] = geometry.rotate(rear_right_tire_points[i], car.heading, car.pos)
    for i in range(len(front_left_tire_points)):
        front_left_tire_points[i] = geometry.rotate(front_left_tire_points[i], car.heading, car.pos)
        front_left_tire_points[i] = geometry.rotate(front_left_tire_points[i], car.steer * car.max_steer, np.array([car.pos[0] + car.cg_to_front_axle, car.pos[1] - (car.width / 2) - (car.wheel_width / 2)]))
    for i in range(len(front_right_tire_points)):
        front_right_tire_points[i] = geometry.rotate(front_right_tire_points[i], car.heading, car.pos)
        front_right_tire_points[i] = geometry.rotate(front_right_tire_points[i], car.steer * car.max_steer, np.array([car.pos[0] + car.cg_to_front_axle, car.pos[1] + (car.width / 2) + (car.wheel_width / 2)]))

    render_polygon(image, body_points, (0, 0, 255), scale, resolution, rotation, camera_pos, x_offset, y_offset)
    render_polygon(image, rear_left_tire_points, (100, 100, 100), scale, resolution, rotation, camera_pos, x_offset, y_offset)
    render_polygon(image, rear_right_tire_points, (100, 100, 100), scale, resolution, rotation, camera_pos, x_offset, y_offset)
    render_polygon(image, front_left_tire_points, (100, 100, 100), scale, resolution, rotation, camera_pos, x_offset, y_offset)
    render_polygon(image, front_right_tire_points, (100, 100, 100), scale, resolution, rotation, camera_pos, x_offset, y_offset)


def render_polygon(
        image,
        points: np.ndarray,
        color,
        scale,
        resolution,
        rotation,
        rotation_center,
        offset_x: int,
        offset_y: int):

    rotated_points = geometry.rotate_points(points, rotation, rotation_center)

    pts = np.array([[
             (p[0] * scale + offset_x) * resolution,
             (p[1] * scale + offset_y) * resolution
         ] for p in rotated_points], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.fillPoly(image, [pts], color, 8)


def render_point(
        image,
        point: np.ndarray,
        colour: Tuple[int, int, int],
        scale: float,
        resolution: float,
        rotation: float,
        rotation_center: np.ndarray,
        radius: float,
        x_offset: int,
        y_offset: int):
    """
    Render a point onto the image provided.

    :param image: Image to render upon.
    :param point: Point position to draw upon.
    :param colour: Color of the point
    :param scale: Scale to render the point.
    :param radius: Radius of the point
    :param x_offset: x offset point.
    :param y_offset: y offset point.
    :return: image with the rendered lines.
    """
    p = geometry.rotate(point, rotation, rotation_center)
    p[0] = (p[0] * scale + x_offset) * resolution
    p[1] = (p[1] * scale + y_offset) * resolution
    return cv2.circle(
        image,
        (int(round(p[0])), int(round(p[1]))),
        radius,
        colour,
        -1
    )


def render_line(
        image,
        line: np.ndarray,
        colour: Tuple[int, int, int],
        scale: float,
        resolution: float,
        rotation,
        rotation_center,
        x_offset: int,
        y_offset: int):
    """
    Draw line on the image with the given parameters. Used for drawing track boundary lines.

    :param image: Image to render upon.
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
    a = geometry.rotate(line[0:2], rotation, rotation_center)
    a[0] = (a[0] * scale + x_offset) * resolution
    a[1] = (a[1] * scale + y_offset) * resolution
    b = geometry.rotate(line[2:4], rotation, rotation_center)
    b[0] = (b[0] * scale + x_offset) * resolution
    b[1] = (b[1] * scale + y_offset) * resolution

    image = cv2.line(
        image,
        (int(round(a[0])), int(round(a[1]))),
        (int(round(b[0])), int(round(b[1]))),
        colour,
        2
    )
    return image
