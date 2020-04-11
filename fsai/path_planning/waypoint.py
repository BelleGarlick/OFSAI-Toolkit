import numpy as np


class Waypoint:
    def __init__(self, line: np.ndarray, sticky=False, optimum=0.5):
        self.line: np.ndarray = line
        self.optimum: float = optimum
        self.sticky = sticky

    def get_optimum_point(self):
        return self.line[0:2] + ((self.line[2:4] - self.line[0:2]) * self.optimum)

    def copy(self):
        return Waypoint(line=self.line.copy(), sticky=self.sticky, optimum=self.optimum)
