from fsai.objects.point import Point

CONE_COLOR_BLUE = 0
CONE_COLOR_YELLOW = 1
CONE_COLOR_BIG_ORANGE = 2
CONE_COLOR_ORANGE = 3


class Cone:
    def __init__(self, point: Point, color: int):
        self.point: Point = point
        self.color = color

    def copy(self):
        return Cone(Point(self.point.x, self.point.y), self.color)

    def __str__(self):
        return "Point: ({}), Color: {}".format(self.point, self.color)