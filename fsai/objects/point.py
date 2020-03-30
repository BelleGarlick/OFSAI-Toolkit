import math
from typing import List


class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __str__(self):
        return "x: {}, y: {}".format(self.x, self.y)

    def __add__(self, point):
        return Point(self.x + point.x, self.y + point.y)

    def __sub__(self, point):
        return Point(self.x - point.x, self.y - point.y)

    def __mul__(self, scalar):
        return Point(self.x * scalar, self.y * scalar)

    def __truediv__(self, scalar):
        return Point(self.x / scalar, self.y / scalar)

    def __abs__(self):
        return Point(abs(self.x), abs(self.y))

    def __len__(self):
        return math.hypot(self.x, self.y)

    def copy(self):
        return Point(self.x, self.y)

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

    def normalize(self):
        length = len(self)
        return Point(
            self.x / length,
            self.y / length
        )

    def distance(self, point):
        """
        Get the distance between two points
        :param point: The current position of this object
        :return: Distance between the two objects
        """
        return math.hypot(self.x - point.x, self.y - point.y)

    def get_closest_point(self, points: List):
        """
        Returns the closest point from a list of points to this object in the form of a list
        :param points: List of points to find the closest from
        :return: Returns a list of points, this is to make it easier to translate into cython.
        """
        if len(points) == 0:
            return []
        else:
            closest_distance = self.distance(points[0])
            closest_point = points[0]

            for point in points[1:]:
                distance = self.distance(point)
                if distance < closest_distance:
                    closest_distance = distance
                    closest_point = point

            return [closest_point]

    def angle_to(self, point):
        return math.atan2(point.y - self.y, point.x - self.x)
