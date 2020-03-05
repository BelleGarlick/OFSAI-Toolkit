import math


class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __str__(self):
        return "x: {}, y: {}".format(self.x, self.y)

    def rotate_around(self, position, angle):
        nx = math.cos(angle) * (self.x - position.x) - math.sin(angle) * (self.y - position.y) + position.x
        ny = math.sin(angle) * (self.x - position.x) + math.cos(angle) * (self.y - position.y) + position.y

        self.x = nx
        self.y = ny

    def add(self, point):
        self.x += point.x
        self.y += point.y

    def sub(self, point):
        self.x -= point.x
        self.y -= point.y

    def distance(self, point):
        """
        Get the distance between two points
        :param point: The current position of this object
        :return: Distance between the two objects
        """
        return math.hypot(self.x - point.x, self.y - point.y)
