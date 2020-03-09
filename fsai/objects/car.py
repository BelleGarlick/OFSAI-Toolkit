from fsai.objects.point import Point


class Car:
    def __init__(self, pos=Point(0, 0), orientation=0):
        self.pos = pos
        self.orientation = orientation
