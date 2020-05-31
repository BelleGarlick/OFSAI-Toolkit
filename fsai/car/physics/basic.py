import math
import numpy as np
from fsai import geometry


def sign(x):
    return -1 if x < 0 else 1


def clamp(x, min_x, max_x):
    return min(max(x, min_x), max_x)


class CarPhysics:
    def __init__(self, car):
        self.car = car

        self.engine_power = 10000
        self.brake_power = 13000

        self.velocity = np.asarray([0, 0])  # x = long, y == lat
        self.accel_c = np.asarray([0, 0])  # x = long, y == lat
        self.vel_c = [0, 0]
        self.angular_speed = 0

        self.aero_drag = 2.5
        self.roll_drag = 5

        self.distance_travelled = 0
        self.distances_travelled = []
        self.abs_vel = 0
        self.abs_accel = 0

    def update(self, dt: float):
        self.vel_c = geometry.rotate(self.velocity, -self.car.heading, [0, 0])

        self.car.heading += (self.car.steer * self.car.max_steer * 8) * dt

        self.accel_c = np.asarray([0, 0])
        self.accel_c = geometry.scale(self.get_total_force(self.car.throttle * self.engine_power - self.car.brake * self.brake_power), 1 / self.car.mass)
        self.abs_accel = geometry.distance(self.accel_c, [0, 0])

        self.vel_c[0] += self.accel_c[0] * dt
        self.vel_c[1] += self.accel_c[1] * dt
        self.vel_c[0] = max(0, self.vel_c[0])

        self.abs_vel = geometry.distance(self.vel_c, [0, 0])
        self.distance_travelled += self.abs_vel * dt
        self.distances_travelled += [self.abs_vel * dt]
        self.distances_travelled = self.distances_travelled[-100:]

        self.velocity = geometry.rotate(self.vel_c, self.car.heading, [0, 0])
        self.car.pos = geometry.add(self.car.pos, geometry.scale(self.velocity, dt))

    def get_total_force(self, wheel_torque):
        """
        All force calculated in local space
        :return: Force upon a vehicle
        """
        aero_drag = [
            -self.aero_drag * self.vel_c[0] * abs(self.vel_c[0]),
            -self.aero_drag * self.vel_c[1] * abs(self.vel_c[1])
        ]
        roll_drag = geometry.scale(self.vel_c, -self.roll_drag)

        wheel_traction = [wheel_torque / self.car.wheel_radius, 0]
        return geometry.add(geometry.add(wheel_traction, aero_drag), roll_drag)
