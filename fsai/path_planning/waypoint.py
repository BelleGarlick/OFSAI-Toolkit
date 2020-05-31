from typing import List

import numpy as np

from fsai import geometry


class Waypoint:
    def __init__(self, line: List[float], sticky=False, optimum=0.5):
        self.line: List[float] = line
        self.optimum: float = optimum
        self.sticky = sticky
        self.throttle = 0

    def get_optimum_point(self):
        return geometry.add(self.line[0:2], geometry.scale(geometry.sub(self.line[2:4], self.line[0:2]), self.optimum))

    def copy(self):
        return Waypoint(line=self.line.copy(), sticky=self.sticky, optimum=self.optimum)

    def find_optimum_from_point(self, point):
        diff = geometry.sub(self.line[0:2], self.line[2:4])
        if diff[0] == 0 and diff[1] == 0:
            return 0.5
        elif diff[0] == 0:
            rel = self.line[1] - point[1]
            return min(max(rel / diff[1], 0), 1)
        elif diff[1] == 0:
            rel = self.line[0] - point[0]
            return min(max(rel / diff[0], 0), 1)
        else:
            self_m = diff[1] / diff[0]
            self_c = self.line[1] - self_m * self.line[0]

            perp_m = -1/self_m
            perp_c = point[1] - perp_m * point[0]

            intersection_x = (perp_c - self_c) / (self_m - perp_m)
            intersection_y = perp_m * intersection_x + perp_c

            intersection_point = [intersection_x, intersection_y]
            rel = geometry.sub(self.line[0:2], intersection_point)

            return min(max(rel[0] / diff[0], 0), 1)
        return 0.5

    def __str__(self):
        return "Waypoint: " + str(self.line)