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
        self.local_acceleration = np.asarray([0, 0])  # x = long, y == lat
        self.angular_speed = 0

        self.aero_drag = 2.5
        self.roll_drag = 5

        self.distance_travelled = 0
        self.distances_travelled = []

    def update(self, dt: float):
        local_velocity = geometry.rotate(self.velocity, -self.car.heading, [0, 0])

        self.car.heading += (self.car.steer * self.car.max_steer * 8) * dt

        self.local_acceleration = np.asarray([0, 0])
        self.local_acceleration = geometry.scale(self.get_total_force(local_velocity, self.car.throttle * self.engine_power - self.car.brake * self.brake_power), 1 / self.car.mass)

        local_velocity[0] += self.local_acceleration[0] * dt
        local_velocity[1] += self.local_acceleration[1] * dt
        local_velocity[0] = max(0, local_velocity[0])

        absVel = geometry.distance(local_velocity, [0, 0])
        self.distance_travelled += absVel * dt
        self.distances_travelled += [absVel * dt]
        self.distances_travelled = self.distances_travelled[-100:]

        self.velocity = geometry.rotate(local_velocity, self.car.heading, [0, 0])
        self.car.pos = geometry.add(self.car.pos, geometry.scale(self.velocity, dt))

    def get_total_force(self, local_velocity, wheel_torque):
        """
        All force calculated in local space
        :return: Force upon a vehicle
        """
        aero_drag = [
            -self.aero_drag * local_velocity[0] * abs(local_velocity[0]),
            -self.aero_drag * local_velocity[1] * abs(local_velocity[1])
        ]
        roll_drag = geometry.scale(local_velocity, -self.roll_drag)

        wheel_traction = [wheel_torque / self.car.wheel_radius, 0]
        return geometry.add(geometry.add(wheel_traction, aero_drag), roll_drag)
