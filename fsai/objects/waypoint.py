from fsai.objects.line import Line


class Waypoint:
    def __init__(self, line: Line, sticky=False, optimum=0.5):
        self.line: Line = line
        self.optimum: float = optimum
        self.sticky = sticky

    def get_optimum_point(self):
        return self.line.a + ((self.line.b - self.line.a) * self.optimum)

    def copy(self):
        return Waypoint(line=self.line.copy(), sticky=self.sticky, optimum=self.optimum)
