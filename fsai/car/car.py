import math

from fsai.car.physics.basic import CarPhysics
from fsai.objects.point import Point


class Car:
    def __init__(
            self,
            pos=Point(0, 0),
            heading=0,
    ):
        self.pos: Point = pos
        self.heading = heading
        self.physics = CarPhysics(self)

        self.mass = 1000

        self.width = 1.2
        self.max_steer = 0.6
        self.wheel_radius = 0.3
        self.cg_to_front_axle = 0.75
        self.cg_to_rear_axle = 0.75
        self.cg_height = 0.55
        self.cg_to_front = 1.25
        self.cg_to_rear = 1.25

        self.throttle = 0
        self.brake = 0
        self.steer = 0
