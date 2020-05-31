import math
import numpy as np


def sign(x):
    return -1 if x < 0 else 1


def clamp(x, min_x, max_x):
    return min(max(x, min_x), max_x)


class CarPhysics:
    def __init__(
            self,
            car,
    ):
        from fsai.car.car import Car
        self.car: Car = car

        self.ebrake = 0

        # variable stats about the car
        self.velocity: np.ndarray = np.zeros(2)  # m/s in world coords
        self.accel: np.ndarray = np.zeros(2)  # accel in world space
        self.absVel = 0

        self.vel_c: np.ndarray = np.zeros(2)
        self.accel_c: np.ndarray = np.zeros(2)
        
        self.__abs_vel: float = 0  # absolute velocity in m/s
        self.yaw_rate: float = 0  # angular vel in radians
        self.__steer_angle: float = 0  # actual amount of front wheel steering  [-max_steer, max_steer]

        self.weight_transfer = 0.2
        self.gravity = 9.81
        self.air_resist = 2.5
        self.roll_resist = 4
        self.engine_force = 10000
        self.brake_force = 14000
        self.inertia_scale = 1
        self.corner_stiffness_front = 4
        self.corner_stiffness_rear = 4.2
        self.tire_grip = 5
        self.lock_grip = 0.7
        self.e_brake_force = 2000

        self.distance_travelled = 0
        self.distances_travelled = []

    def update(self, dt: float):
        # Shorthand
        sn = math.sin(self.car.heading)
        cs = math.cos(self.car.heading)
        steer_angle = self.car.steer * self.car.max_steer

        # Get velocity in local car coordinates
        self.vel_c[0] = cs * self.velocity[0] + sn * self.velocity[1]
        self.vel_c[1] = cs * self.velocity[1] - sn * self.velocity[0]

        # Weight on axles based on centre of gravity and weight shift due to forward/reverse acceleration
        axle_weight_ratio_front, axle_weight_ratio_rear = self.__get_axle_weight_ratios()
        axle_weight_front = self.car.mass * (axle_weight_ratio_front * self.gravity - self.weight_transfer * self.accel_c[0] * self.car.cg_height / self.__get_wheel_base())
        axle_weight_rear = self.car.mass * (axle_weight_ratio_rear * self.gravity + self.weight_transfer * self.accel_c[0] * self.car.cg_height / self.__get_wheel_base())

        # Resulting velocity of the wheels as result of the yaw rate of the car body.
        # v = yaw rate * r where r is distance from axle to CG and yawRate (angular velocity) in rad/s.
        yaw_speed_front = self.car.cg_to_front_axle * self.yaw_rate
        yaw_speed_rear = -self.car.cg_to_rear_axle * self.yaw_rate

        # Calculate slip angles for front and rear wheels (a.k.a. alpha)
        slip_angle_front = math.atan2(self.vel_c[1] + yaw_speed_front, abs(self.vel_c[0])) - sign(self.vel_c[0]) * steer_angle
        slip_angle_rear = math.atan2(self.vel_c[1] + yaw_speed_rear,  abs(self.vel_c[0]))

        tire_grip_front = self.tire_grip
        tire_grip_rear = self.tire_grip * (1.0 - self.ebrake * (1.0 - self.lock_grip))  # reduce rear grip when ebrake is on

        friction_force_front_cy = clamp(-self.corner_stiffness_front * slip_angle_front, -tire_grip_front, tire_grip_front) * axle_weight_front
        friction_force_rear_cy = clamp(-self.corner_stiffness_rear * slip_angle_rear, -tire_grip_rear, tire_grip_rear) * axle_weight_rear

        # Get amount of brake/throttle from our inputs
        brake = min(self.car.brake * self.brake_force + self.ebrake * self.e_brake_force, self.brake_force)
        throttle = self.car.throttle * self.engine_force

        # Resulting force in local car coordinates.
        # This is implemented as a RWD car only.
        traction_force_cx = throttle - brake * sign(self.vel_c[0])
        traction_force_cy = 0

        drag_force_cx = -self.roll_resist * self.vel_c[0] - self.air_resist * self.vel_c[0] * abs(self.vel_c[0])
        drag_force_cy = -self.roll_resist * self.vel_c[1] - self.air_resist * self.vel_c[1] * abs(self.vel_c[1])

        # total force in car coordinates
        total_force_cx = drag_force_cx + traction_force_cx
        total_force_cy = drag_force_cy + traction_force_cy + math.cos(steer_angle) * friction_force_front_cy + friction_force_rear_cy

        # acceleration along car axes
        self.accel_c[0] = total_force_cx / self.car.mass  # forward/reverse accel
        self.accel_c[1] = total_force_cy / self.car.mass  # sideways accel

        # acceleration in world coordinates
        self.accel[0] = cs * self.accel_c[0] - sn * self.accel_c[1]
        self.accel[1] = sn * self.accel_c[0] + cs * self.accel_c[1]

        # update velocity
        self.velocity[0] += self.accel[0] * dt
        self.velocity[1] += self.accel[1] * dt

        self.absVel = math.hypot(self.velocity[0], self.velocity[1])
        self.distance_travelled += self.absVel * dt
        self.distances_travelled += [self.absVel * dt]
        self.distances_travelled = self.distances_travelled[-100:]

        # calculate rotational forces
        angular_torque = (friction_force_front_cy + traction_force_cy) * self.car.cg_to_front_axle - friction_force_rear_cy * self.car.cg_to_rear_axle

        # Sim gets unstable at very slow speeds, so just stop the car
        if abs(self.absVel) < 0.5 and not throttle:
            self.velocity[0], self.velocity[1], self.absVel = 0, 0, 0
            angular_torque, self.yaw_rate = 0, 0

        angular_accel = angular_torque / self.__get_inertia()

        self.yaw_rate += angular_accel * dt
        self.car.heading += self.yaw_rate * dt

        # finally we can update position
        self.car.pos[0] += self.velocity[0] * dt
        self.car.pos[1] += self.velocity[1] * dt

        return self.absVel * dt

    def current_speed_mph(self):
        return self.absVel * 2.23694

    def current_speed_kmph(self):
        return self.absVel * 3.6

    def current_speed_mps(self):
        return self.absVel

    def __get_inertia(self):
        return self.car.mass * self.inertia_scale

    def __get_wheel_base(self):
        return self.car.cg_to_front_axle + self.car.cg_to_rear_axle

    def __get_axle_weight_ratios(self):
        wheel_base = self.__get_wheel_base()
        front_ratio = self.car.cg_to_rear_axle / wheel_base
        rear_ratio = self.car.cg_to_front_axle / wheel_base
        return front_ratio, rear_ratio

