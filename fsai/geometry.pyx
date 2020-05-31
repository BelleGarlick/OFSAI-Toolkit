from libc.math cimport atan2, round, sin, cos, sqrt
import cython

cpdef float distance(a, b):
    return cdistance(a[0], a[1], b[0], b[1])


cpdef distances(point, points):
    return [cdistance(point[0], point[1], p[0], p[1]) for p in points]


cpdef float length(line):
    return cdistance(line[0], line[1], line[2], line[3])


cpdef rotate(point, float angle, around):
    cdef float cos_f = cos(angle)
    cdef float sin_f = sin(angle)
    cdef float[2] dif = sub(point, around)

    cdef float nx = cos_f * dif[0] - sin_f * dif[1] + around[0]
    cdef float ny = sin_f * dif[0] + cos_f * dif[1] + around[1]
    return [nx, ny]


cpdef float angle_to(a, b):
    return atan2(b[1] - a[1], b[0] - a[0])


cpdef float angle(line):
    return atan2(line[3] - line[1], line[2] - line[0])


cpdef angles(lines):
    return [angle(line) for line in lines]


cpdef closest_point(origin, points):
    if len(points) == 0:
        return None

    cdef float[2] nearest_point = points[0]
    cdef float closest_distance = distance(origin, points[0])
    cdef float dist = 0
    cdef size_t i
    for i in range(1, len(points)):
        dist = distance(origin, points[i])
        if dist < closest_distance:
            nearest_point = points[i]
    return nearest_point


@cython.cdivision(True)
cpdef segment_intersections(line, segments):
    cdef float lin_0 = line[0]
    cdef float lin_1 = line[1]
    cdef float lin_2 = line[2]
    cdef float lin_3 = line[3]
    cdef float min_x_seg = cmin(lin_0, lin_2)
    cdef float min_y_seg = cmin(lin_1, lin_3)
    cdef float max_x_seg = cmax(lin_0, lin_2)
    cdef float max_y_seg = cmax(lin_1, lin_3)

    # convert fixed lines into y=mx+c ordinates
    cdef float a1 = lin_3 - lin_1
    cdef float b1 = lin_0 - lin_2
    cdef float c1 = a1 * lin_0 + b1 * lin_1

    intersections = []
    cdef float a2, b2, c2
    cdef float delta = 0
    cdef float x, y
    cdef float min_x, min_y, max_x, max_y
    cdef size_t index
    cdef float seg_0, seg_1, seg_2, seg_3
    cdef float precision = 1000
    for index in range(len(segments)):
        seg_i = segments[index]
        seg_0 = seg_i[0]
        seg_1 = seg_i[1]
        seg_2 = seg_i[2]
        seg_3 = seg_i[3]

        a2 = seg_3 - seg_1
        b2 = seg_0 - seg_2
        c2 = a2 * seg_0 + b2 * seg_1
        delta = a1 * b2 - a2 * b1

        if delta != 0:
            x = (b2 * c1 - b1 * c2) / delta
            y = (a1 * c2 - a2 * c1) / delta

            # check the point exists within the
            min_x = round(precision * cmax(cmin(seg_0, seg_2), min_x_seg))
            min_y = round(precision * cmax(cmin(seg_1, seg_3), min_y_seg))
            max_x = round(precision * cmin(cmax(seg_0, seg_2), max_x_seg))
            max_y = round(precision * cmin(cmax(seg_1, seg_3), max_y_seg))

            if min_x <= round(precision * x) <= max_x and min_y <= round(precision * y) <= max_y:
                intersections.append([x, y])
    return intersections


cpdef rotate_points(points, float rotation, rotation_center):
    rotated_points = []

    cdef size_t p
    for p in range(len(points)):
        rotated_points.append(rotate(points[p], rotation, rotation_center))
    return rotated_points


cpdef line_center(line):
    a = line[0:2]
    b = line[2:4]
    return add(a, scale(sub(b, a), 0.5))


@cython.cdivision(True)
cpdef normalise(line):
    cdef float[2] normalisation = sub(line[2:4], line[0:2])
    cdef float line_length = length(line)
    if line_length == 0:
        return None
    normalisation[0] /= line_length
    normalisation[1] /= line_length
    return normalisation


cpdef clip(values, float min_val, float max_val):
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


cpdef filter_lines_by_distance(point, float distance, lines):
    filtered_lines = []
    cdef float px = point[0]
    cdef float py = point[1]

    cdef size_t line_index
    cdef float lax, lay, lbx, lby
    for line_index in range(len(lines)):
        l = lines[line_index]
        lax = l[0]
        lay = l[1]
        lbx = l[2]
        lby = l[3]
        if cdistance(px, py, lax, lay) < distance or cdistance(px, py, lbx, lby) < distance:
            filtered_lines.append(lines[line_index])
    return filtered_lines



cdef float cmin(float a, float b):
    if a < b:
        return a
    else:
        return b


cdef float cmax(float a, float b):
    if a > b:
        return a
    else:
        return b


cdef float cdistance(float a1, float a2, float b1, float b2):
    cdef float d0 = a1 - b1
    cdef float d1 = a2 - b2
    return sqrt(d0*d0 + d1*d1)


@cython.cdivision(True)
cpdef circle_line_intersections(point, float radius, lines):
    intersections = []

    cdef float c_x = point[0]
    cdef float c_y = point[1]

    cdef float dx, dy, A, B, C, det, t, p_x, p_y, a_x, a_y, b_x, b_y, min_x_line, min_y_line, max_x_line, max_y_line

    for line in lines:
        a_x = line[0]
        a_y = line[1]
        b_x = line[2]
        b_y = line[3]

        # line = ax, ay, bx, by
        min_x_line = cmin(a_x, b_x)
        min_y_line = cmin(a_y, b_y)
        max_x_line = cmax(a_x, b_x)
        max_y_line = cmax(a_y, b_y)

    #//        minXline = (Math.round(minXline * 1000d)) / 1000d;
    #//        minYline = (Math.round(minYline * 1000d)) / 1000d;
    #//        maxXline = (Math.round(maxXline * 1000d)) / 1000d;
    #//        maxYline = (Math.round(maxYline * 1000d)) / 1000d;

        dx = b_x - a_x
        dy = b_y - a_y

        A = dx * dx + dy * dy
        B = 2 * (dx * (a_x - c_x) + dy * (a_y - c_y))
        C = (a_x - c_x) * (a_x - c_x) + (a_y - c_y) * (a_y - c_y) - radius * radius

        det = B * B - 4 * A * C;
        if (A <= 0.0000001) or (det < 0):
            # No real solutions.
            pass
        elif (det == 0):
            # One solution.
            t = -B / (2 * A)

            p_x = a_x + t * dx
            p_y = a_y + t * dy
            if (p_x > min_x_line and p_x < max_x_line and p_y > min_y_line and p_y < max_y_line):
                intersections.append([p_x, p_y])
        else:
            # Two solutions.
            t = (float)((-B + sqrt(det)) / (2 * A));
            p_x = a_x + t * dx
            p_y = a_y + t * dy
            if (p_x > min_x_line and p_x < max_x_line and p_y > min_y_line and p_y < max_y_line):
                intersections.append([p_x, p_y])

            t = (float)((-B - sqrt(det)) / (2 * A));
            p_x = a_x + t * dx
            p_y = a_y + t * dy
            if (p_x > min_x_line and p_x < max_x_line and p_y > min_y_line and p_y < max_y_line):
                intersections.append([p_x, p_y])

    return intersections
