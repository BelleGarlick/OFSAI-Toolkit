import math
import sys
from typing import List
import sympy as sym
from fsai.objects.point import Point
from fsai.path_planning.waypoint import Waypoint


def calculate_minimum_curve(waypoints: List[Waypoint] = None, full_track: bool = False, epochs: int = 1000):
    if full_track:
        if len(waypoints) < 5:
            print("Calculating the minimum curve of a full requires at least 5 waypoints.")
        else:
            # Loop through all waypoints to start calculating the minimum curve
            for i in range(epochs):
                total_angle = 0
                for i in range(len(waypoints)):
                    if not waypoints[i].sticky:
                        p2 = waypoints[i - 2].get_optimum_point()
                        p1 = waypoints[i - 1].get_optimum_point()
                        n1 = waypoints[(i + 1) % len(waypoints)].get_optimum_point()
                        n2 = waypoints[(i + 2) % len(waypoints)].get_optimum_point()

                        waypoints[i].optimum, angle = __calculate_optimal_p(p2, p1, waypoints[i], n1, n2)
                        total_angle += angle
                print(total_angle)
    else:
        print("Unimplemented for partial track")


def __calculate_optimal_p(pre_2: Point, pre_1: Point, waypoint: Waypoint, nex_1: Point, nex_2: Point):
    total_angle = None
    optimal_value = 0.5

    steps = 25
    ubound = min(1, waypoint.optimum + 0.1)
    lbound = max(0, waypoint.optimum - 0.1)
    for i in range(steps):
        lmbound = (ubound - lbound) * 0.3 + lbound
        umbound = (ubound - lbound) * 0.7 + lbound

        bounds = [lbound, lmbound, umbound, ubound]
        for bound_index in range(4):
            p = bounds[bound_index]
            waypoint.optimum = p
            angle = __calculate_total_angle(
                pre_2.x, pre_2.y, pre_1.x, pre_1.y, waypoint.line.a.x, waypoint.line.a.y, waypoint.line.b.x, waypoint.line.b.y, p, nex_1.x, nex_1.y, nex_2.x, nex_2.y)

            bounds[bound_index] = angle

        total_angle = min(bounds)

        min_bound = bounds.index(min(bounds))
        if min_bound == 0:
            optimal_value = lbound
            ubound = lmbound
        if min_bound == 1:
            optimal_value = lmbound
            ubound = umbound
        if min_bound == 2:
            optimal_value = umbound
            lbound = lmbound
        if min_bound == 3:
            optimal_value = ubound
            lbound = umbound

    return optimal_value, total_angle


def __calculate_total_angle(p1_x, p1_y, p2_x, p2_y, wa_x, wa_y, wb_x, wb_y, op: float, p4_x, p4_y, p5_x, p5_y):
    a = ((math.atan2((wa_y + ((wb_y - wa_y) * op)) - p2_y, (wa_x + ((wb_x - wa_x) * op)) - p2_x) - math.atan2(p1_y - p2_y, p1_x - p2_x)) + (2 * math.pi)) % (2 * math.pi) - math.pi
    b = ((math.atan2(p4_y - (wa_y + ((wb_y - wa_y) * op)), p4_x - (wa_x + ((wb_x - wa_x) * op))) - math.atan2(p2_y - (wa_y + ((wb_y - wa_y) * op)), p2_x - (wa_x + ((wb_x - wa_x) * op)))) + (2 * math.pi)) % (2 * math.pi) - math.pi
    c = ((math.atan2(p5_y - p4_y, p5_x - p4_x) - math.atan2((wa_y + ((wb_y - wa_y) * op)) - p4_y, (wa_x + ((wb_x - wa_x) * op)) - p4_x)) + (2 * math.pi)) % (2 * math.pi) - math.pi
    avg = (a + b + c) / 3

    return abs(a - avg) + abs(b - avg) + abs(c - avg)

    op = sym.Symbol("op")
    p1_x = sym.Symbol("p1_x")
    p1_y = sym.Symbol("p1_y")
    p2_x = sym.Symbol("p2_x")
    p2_y = sym.Symbol("p2_y")
    wa_x = sym.Symbol("wa_x")
    wa_y = sym.Symbol("wa_y")
    wb_x = sym.Symbol("wb_x")
    wb_y = sym.Symbol("wb_y")
    p4_x = sym.Symbol("p4_x")
    p4_y = sym.Symbol("p4_y")
    p5_x = sym.Symbol("p5_x")
    p5_y = sym.Symbol("p5_y")
    print(sym.diff(((sym.atan2((wa_y + ((wb_y - wa_y) * op)) - p2_y, (wa_x + ((wb_x - wa_x) * op)) - p2_x) - sym.atan2(p1_y - p2_y, p1_x - p2_x)) + (2 * sym.pi)) - sym.pi, op))
    # x = sym.Symbol("x")
    # op, p1_x, p1_y, p2_x, p2_y, wa_x, wa_y, wb_x, wb_y, p4_x, p4_y, p5_x, p5_y = sym.Symbol("op p1_x p1_y p2_x p2_y wa_x wa_y wb_x wb_y p4_x p4_y p5_x p5_y")
    # c = sym.Symbol("c")
    # print(sym.diff(sym.atan2(b, c), b))
    # sys.exit()
    return total_angle

# def __calculate_total_angle(p1, p2, p3, p4, p5):
#     a = __angle(p1, p2, p3)
#     b = __angle(p2, p3, p4)
#     c = __angle(p3, p4, p5)
#     avg = (a + b + c) / 3
#
#     return abs(a - avg) + abs(b - avg) + abs(c - avg)


def __angle(a: Point, b, c):
    angle = ((math.atan2(c.y - b.y, c.x - b.x) - math.atan2(a.y - b.y, a.x - b.x)) + (2 * math.pi)) % (2 * math.pi) - math.pi
    return angle
