import math

import numpy as np


def distance(a: np.ndarray, b: np.ndarray):
    d = a - b
    return np.sqrt(d.dot(d))


def distances(point, points):
    d = points - point
    sqrs = d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1]
    return np.sqrt(sqrs)


def length(line: np.ndarray):
    return distance(line[0:2], line[2:4])


def rotate(point, angle: float = 0, around: np.ndarray = np.zeros((1, 2))):
    cos = math.cos(angle)
    sin = math.sin(angle)
    sub = point - around

    nx = cos * sub[0] - sin * sub[1] + around[0]
    ny = sin * sub[0] + cos * sub[1] + around[1]

    return np.array([nx, ny])


def angle_to(a: np.ndarray, b: np.ndarray):
    return math.atan2(b[1] - a[1], b[0] - a[0])


def angle(line: np.ndarray):
    return math.atan2(line[3] - line[1], line[2] - line[0])


def closest_point(origin: np.ndarray, points):
    dist = np.sum((points - origin)**2, axis=1)
    return points[np.argmin(dist)]


def segment_intersections(segment_a, segments):
    min_x_seg = min(segment_a[0], segment_a[2])
    min_y_seg = min(segment_a[1], segment_a[3])
    max_x_seg = max(segment_a[0], segment_a[2])
    max_y_seg = max(segment_a[1], segment_a[3])

    # convert fixed lines into y=mx+c ordinates
    a1 = segment_a[3] - segment_a[1]
    b1 = segment_a[0] - segment_a[2]
    c1 = a1 * segment_a[0] + b1 * segment_a[1]

    a2 = segments[:, 3] - segments[:, 1]
    b2 = segments[:, 0] - segments[:, 2]
    c2 = a2 * segments[:, 0] + b2 * segments[:, 1]
    delta = a1 * b2 - a2 * b1

    x = (b2 * c1 - b1 * c2) / delta
    y = (a1 * c2 - a2 * c1) / delta

    intersections = []
    for index in range(len(x)):
        # check the point exists within the
        min_x = round(1000 * max(min(segments[index][0], segments[index][2]), min_x_seg)) / 1000
        min_y = round(1000 * max(min(segments[index][1], segments[index][3]), min_y_seg)) / 1000
        max_x = round(1000 * min(max(segments[index][0], segments[index][2]), max_x_seg)) / 1000
        max_y = round(1000 * min(max(segments[index][1], segments[index][3]), max_y_seg)) / 1000

        if x[index] != np.nan and y[index] != np.nan and min_x <= round(
                1000 * x[index]) / 1000 <= max_x and min_y <= round(1000 * y[index]) / 1000 <= max_y:
            intersections.append(np.array([x[index], y[index]]))
    return intersections


def rotate_points(points, rotation, rotation_center):
    rotated_points = []
    for point in points:
        rotated_points.append(rotate(point, rotation, rotation_center))
    return rotated_points


def line_center(line: np.ndarray):
    a = line[0:2]
    b = line[2:4]
    return a + (b - a) * 0.5



def normalise(line: np.ndarray) -> np.ndarray:
    normalisation: np.ndarray = line[2:4] - line[0:2]
    length = np.sqrt(normalisation.dot(normalisation))
    if length == 0:
        return
    normalisation[0] /= length
    normalisation[1] /= length
    return normalisation