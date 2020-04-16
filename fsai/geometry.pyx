import math
from typing import List


cpdef float distance(a, b):
    cdef float[2] d = sub(a, b)
    return math.sqrt(d[0]*d[0] + d[1]*d[1])


cpdef distances(point, points):
    return [distance(point, p) for p in points]


cpdef float length(line):
    return distance(line[0:2], line[2:4])


cpdef rotate(point, float angle = 0, around: List[float] = None):
    cdef float cos = math.cos(angle)
    cdef float sin = math.sin(angle)
    cdef float[2] dif = sub(point, around)

    cdef float nx = cos * dif[0] - sin * dif[1] + around[0]
    cdef float ny = sin * dif[0] + cos * dif[1] + around[1]

    return [nx, ny]


cpdef float angle_to(a, b):
    return math.atan2(b[1] - a[1], b[0] - a[0])


cpdef float angle(line):
    return math.atan2(line[3] - line[1], line[2] - line[0])


cpdef angles(lines):
    return [angle(line) for line in lines]


cpdef closest_point(origin, points):
    if len(points) == 0:
        return None

    cdef float[2] nearest_point = points[0]
    cdef float closest_distance = distance(origin, points[0])
    cdef float dist = 0
    for i in range(1, len(points)):
        dist = distance(origin, points[i])
        if dist < closest_distance:
            nearest_point = points[i]
    return nearest_point


cpdef segment_intersections(segment_a, segments):
    cdef float min_x_seg = cmin(segment_a[0], segment_a[2])
    cdef float min_y_seg = cmin(segment_a[1], segment_a[3])
    cdef float max_x_seg = cmax(segment_a[0], segment_a[2])
    cdef float max_y_seg = cmax(segment_a[1], segment_a[3])

    # convert fixed lines into y=mx+c ordinates
    cdef float a1 = segment_a[3] - segment_a[1]
    cdef float b1 = segment_a[0] - segment_a[2]
    cdef float c1 = a1 * segment_a[0] + b1 * segment_a[1]

    intersections = []
    cdef float a2 = 0
    cdef float b2 = 0
    cdef float c2 = 0
    cdef float delta = 0
    cdef float x = 0
    cdef float y = 0
    cdef float min_x = 0
    cdef float min_y = 0
    cdef float max_x = 0
    cdef float max_y = 0
    for index in range(len(segments)):
        a2 = segments[index][3] - segments[index][1]
        b2 = segments[index][0] - segments[index][2]
        c2 = a2 * segments[index][0] + b2 * segments[index][1]
        delta = a1 * b2 - a2 * b1

        if delta != 0:
            x = (b2 * c1 - b1 * c2) / delta
            y = (a1 * c2 - a2 * c1) / delta

            # check the point exists within the
            min_x = round(1000 * cmax(cmin(segments[index][0], segments[index][2]), min_x_seg))
            min_y = round(1000 * cmax(cmin(segments[index][1], segments[index][3]), min_y_seg))
            max_x = round(1000 * cmin(cmax(segments[index][0], segments[index][2]), max_x_seg))
            max_y = round(1000 * cmin(cmax(segments[index][1], segments[index][3]), max_y_seg))

            if min_x <= round(1000 * x) <= max_x and min_y <= round(1000 * y) <= max_y:
                intersections.append([x, y])
    return intersections


cpdef rotate_points(points, rotation, rotation_center):
    rotated_points = []
    for point in points:
        rotated_points.append(rotate(point, rotation, rotation_center))
    return rotated_points


cpdef line_center(line):
    a = line[0:2]
    b = line[2:4]
    return add(a, scale(sub(b, a), 0.5))


cpdef normalise(line):
    cdef float[2] normalisation = sub(line[2:4], line[0:2])
    cdef float line_length = length(line)
    if line_length == 0:
        return None
    normalisation[0] /= line_length
    normalisation[1] /= line_length
    return normalisation


cpdef clip(values: List[float], float min_val, float max_val):
    clipped_values = []
    for v in values:
        clipped_values.append(cmin(cmax(v, min_val), max_val))
    return clipped_values


cpdef sub(point_a, point_b):
    return [point_a[0] - point_b[0], point_a[1] - point_b[1]]


cpdef scale(values, float scalar):
    return [f * scalar for f in values]


cpdef add(point_a, point_b):
    return [point_a[0] + point_b[0], point_a[1] + point_b[1]]


cpdef cmin(float a, float b):
    if a < b:
        return a
    else:
        return b


cpdef cmax(float a, float b):
    if a > b:
        return a
    else:
        return b