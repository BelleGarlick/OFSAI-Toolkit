import math
import numpy as np

from fsai.objects.point import Point


def sign(x):
    return -1 if x < 0 else 1


def clamp(x, min_x, max_x):
    return min(max(x, min_x), max_x)


class CarPhysics:
    def __init__(self, car):
        self.car = car

        self.engine_power = 10000
        self.brake_power = 13000

        self.velocity = Point(0, 0)  # x = long, y == lat
        self.local_acceleration = Point(0, 0)  # x = long, y == lat
        self.angular_speed = 0

        self.aero_drag = 2.5
        self.roll_drag = 5

    def update(self, dt: float):
        local_velocity = self.velocity.copy()
        local_velocity.rotate_around(Point(0, 0), -self.car.heading)

        self.car.heading += (self.car.steer * self.car.max_steer * 8) * dt

        self.local_acceleration = Point(0, 0)
        self.local_acceleration = self.get_total_force(local_velocity, self.car.throttle * self.engine_power - self.car.brake * self.brake_power) / self.car.mass

        local_velocity += self.local_acceleration * dt
        local_velocity.x = max(0, local_velocity.x)

        self.velocity = local_velocity.copy()
        self.velocity.rotate_around(Point(0, 0), self.car.heading)
        self.car.pos += self.velocity * dt

    def get_total_force(self, local_velocity, wheel_torque):
        """
        All force calculated in local space
        :return: Force upon a vehicle
        """
        aero_drag = Point(
            -self.aero_drag * local_velocity.x * abs(local_velocity.x),
            -self.aero_drag * local_velocity.y * abs(local_velocity.y)
        )
        roll_drag = local_velocity * -self.roll_drag

        wheel_traction = Point(wheel_torque / self.car.wheel_radius, 0)
        return wheel_traction + aero_drag + roll_drag
