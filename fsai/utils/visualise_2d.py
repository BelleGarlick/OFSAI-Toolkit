import math
from typing import List, Tuple

import numpy as np

from fsai.objects.cone import Cone


def calculate_translations(cones, points, lines, image_size, padding):
    min_x, min_y, max_x, max_y = __get_image_bounds(cones, points, lines)
    width = (max_x - min_x)
    height = (max_y - min_y)
    x_scale = (image_size[0] - (2 * padding)) / width
    y_scale = (image_size[1] - (2 * padding)) / height
    scale = min(x_scale, y_scale)
    translate_x = (image_size[0] - (width * scale) - (2 * padding)) // 2
    translate_y = (image_size[1] - (height * scale) - (2 * padding)) // 2
    x_offset = (-min_x) * scale + padding + translate_x
    y_offset = (-min_y) * scale + padding + translate_y

    return x_offset, y_offset, scale


def __get_image_bounds(
        cones: List[Tuple[Tuple, float, List[Cone]]],
        points:  List[Tuple[Tuple, float, np.ndarray]],
        lines:  List[Tuple[Tuple, float, np.ndarray]]):
    """
    This method is used to work out the min, max boundaries of a track. These values are used to alter the draw
    positions of objects so that no matter where the track exists in world space it'll still be centered within
    the image.

    :param cones: List of all cones in the scene
    :param points: List of all points in the scene
    :param lines: List of all lines in the scene
    :return:
    """
    min_x, min_y, max_x, max_y = math.inf, math.inf, -math.inf, -math.inf
    cones = [] if cones is None else cones
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
            min_x = min([min_x, line[2], line[0]])
            min_y = min([min_y, line[3], line[1]])
            max_x = max([max_x, line[2], line[0]])
            max_y = max([max_y, line[3], line[1]])

    if max_x == -math.inf:
        max_x, max_y, min_x, min_y = 0, 0, 0, 0

    return min_x, min_y, max_x, max_y
