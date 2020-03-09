from typing import Optional

from fsai.objects.point import Point


class Line:
    def __init__(self, a: Point, b: Point):
        self.a = a
        self.b = b

    def normalise(self) -> Point:
        normalisation = self.b - self.a
        length = normalisation.distance(Point(0, 0))
        normalisation.x /= length
        normalisation.y /= length
        return normalisation

    def intersects(self, line) -> Optional[Point]:
        # convert fixed lines into y=mx+c ordinates
        a1 = line.b.y - line.a.y
        b1 = line.a.x - line.b.x
        c1 = a1 * line.a.x + b1 * line.a.y

        a2 = self.b.y - self.a.y
        b2 = self.a.x - self.b.x
        c2 = a2 * self.a.x + b2 * self.a.y

        delta = a1 * b2 - a2 * b1
        if delta == 0:
            return None

        x = (b2 * c1 - b1 * c2) / delta
        y = (a1 * c2 - a2 * c1) / delta

        # check the point exists within the
        min_x = max(min(line.a.x, line.b.x), min(self.a.x, self.b.x))
        min_y = max(min(line.a.y, line.b.y), min(self.a.y, self.b.y))
        max_x = min(max(line.a.x, line.b.x), max(self.a.x, self.b.x))
        max_y = min(max(line.a.y, line.b.y), max(self.a.y, self.b.y))

        if min_x <= x <= max_x and min_y <= y <= max_y:
            return Point(x=x, y=y)
        return None
