import math


class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __str__(self):
        return "x: {}, y: {}".format(self.x, self.y)

    def rotate_around(self, position, car_angle):
        nx = math.cos(car_angle) * (self.x - position.x) - math.sin(car_angle) * (self.y - position.y) + position.x
        ny = math.sin(car_angle) * (self.x - position.x) + math.cos(car_angle) * (self.y - position.y) + position.y

        self.x = nx
        self.y = ny

    def add(self, point):
        self.x += point.x
        self.y += point.y

    def sub(self, point):
        self.x -= point.x
        self.y -= point.y

    def distance(self, point):
        return math.hypot(self.x - point.x, self.y - point.y)