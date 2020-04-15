import math
from typing import List


def distance(a, b):
    d = sub(a, b)
    return math.sqrt(d[0]**2 + d[1]**2)


def distances(point, points):
    return [distance(point, p) for p in points]


def length(line):
    return distance(line[0:2], line[2:4])


def rotate(point, angle: float = 0, around: List[float] = None):
    cos = math.cos(angle)
    sin = math.sin(angle)
    dif = sub(point, around)

    nx = cos * dif[0] - sin * dif[1] + around[0]
    ny = sin * dif[0] + cos * dif[1] + around[1]

    return [nx, ny]


def angle_to(a, b):
    return math.atan2(b[1] - a[1], b[0] - a[0])


def angle(line):
    return math.atan2(line[3] - line[1], line[2] - line[0])


def angles(lines):
    return [angle(line) for line in lines]


def closest_point(origin, points):
    if len(points) == 0:
        return None

    nearest_point = points[0]
    closest_distance = distance(origin, points[0])
    for i in range(1, len(points)):
        dist = distance(origin, points[i])
        if dist < closest_distance:
            nearest_point = points[i]
    return nearest_point


def segment_intersections(segment_a, segments):
    min_x_seg = min(segment_a[0], segment_a[2])
    min_y_seg = min(segment_a[1], segment_a[3])
    max_x_seg = max(segment_a[0], segment_a[2])
    max_y_seg = max(segment_a[1], segment_a[3])

    # convert fixed lines into y=mx+c ordinates
    a1 = segment_a[3] - segment_a[1]
    b1 = segment_a[0] - segment_a[2]
    c1 = a1 * segment_a[0] + b1 * segment_a[1]

    intersections = []
    for index in range(len(segments)):
        a2 = segments[index][3] - segments[index][1]
        b2 = segments[index][0] - segments[index][2]
        c2 = a2 * segments[index][0] + b2 * segments[index][1]
        delta = a1 * b2 - a2 * b1

        if delta == 0:
            x = (b2 * c1 - b1 * c2) / delta
            y = (a1 * c2 - a2 * c1) / delta

            # check the point exists within the
            min_x = round(1000 * max(min(segments[index][0], segments[index][2]), min_x_seg)) / 1000
            min_y = round(1000 * max(min(segments[index][1], segments[index][3]), min_y_seg)) / 1000
            max_x = round(1000 * min(max(segments[index][0], segments[index][2]), max_x_seg)) / 1000
            max_y = round(1000 * min(max(segments[index][1], segments[index][3]), max_y_seg)) / 1000

            if min_x <= round(1000 * x) / 1000 <= max_x and min_y <= round(1000 * y) / 1000 <= max_y:
                intersections.append([x, y])
    return intersections


def rotate_points(points, rotation, rotation_center):
    rotated_points = []
    for point in points:
        rotated_points.append(rotate(point, rotation, rotation_center))
    return rotated_points


def line_center(line):
    a = line[0:2]
    b = line[2:4]
    return a + (b - a) * 0.5


def normalise(line):
    normalisation = sub(line[2:4], line[0:2])
    line_length = length(line)
    if line_length == 0:
        return
    normalisation[0] /= line_length
    normalisation[1] /= line_length
    return normalisation


def clip(values: List[float], min_val, max_val):
    clipped_values = []
    for v in values:
        clipped_values.append(min(max(v, min_val), max_val))
    return clipped_values


def sub(point_a, point_b):
    return [point_a[0] - point_b[0], point_a[1] - point_b[1]]


def scale(values, scalar):
    return [f * scalar for f in values]


def add(point_a, point_b):
    return [point_a[0] + point_b[0], point_a[1] + point_b[1]]
