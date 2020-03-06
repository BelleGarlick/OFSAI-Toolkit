from fsai.objects.line import Line


class Waypoint:
    def __init__(self, line: Line):
        self.line: Line = line
        self.optimum: float = 0.5
        self.sticky = False

    def get_optimum_point(self):
        return self.line.a + ((self.line.b - self.line.a) * self.optimum)
