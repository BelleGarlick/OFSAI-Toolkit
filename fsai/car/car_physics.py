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

        self.throttle: float = 0
        self.brake: float = 0
        self.e_brake: float = 0
        self.__steer: float = 0  # steering [-1 - 1]

        # variable stats about the car
        self.__velocity: np.ndarray = np.zeros(2)  # m/s in world coords
        self.__velocity_c: np.ndarray = np.zeros(2)  # m/s in local space (x forwards, y sideways)
        self.__accel: np.ndarray = np.zeros(2)  # accel in world space
        self.__accel_c: np.ndarray = np.zeros(2)  # accel in local space
        
        self.__abs_vel: float = 0  # absolute velocity in m/s
        self.__yaw_rate: float = 0  # angular vel in radians
        self.__steer_angle: float = 0  # actual amount of front wheel steering  [-max_steer, max_steer]

        self.weight_transfer = 0.2
        self.gravity = 9.81
        self.air_resist = 2.5
        self.roll_resist = 5
        self.engine_force = 10000
        self.brake_force = 12000
        self.inertia_scale = 1
        self.corner_stiffness_front = 0.5
        self.corner_stiffness_rear = .52
        self.tire_grip = 50
        self.lock_grip = 0.7
        self.e_brake_force = 2000
        
    def update(self, throttle: float, break_val: float, steer: float, dt: float):
        self.throttle = throttle
        self.brake = break_val
        self.__steer = steer
        self.__steer_angle = self.__steer * self.car.max_steer

        self.do_physics(dt)
        
    def do_physics(self, dt: float):
        sn = math.sin(self.car.heading)
        cs = math.cos(self.car.heading)

        # Get velocity in local car coordinates
        self.__velocity_c[0] = cs * self.__velocity[0] + sn * self.__velocity[1]
        self.__velocity_c[1] = cs * self.__velocity[1] - sn * self.__velocity[0]

        # Weight on axles based on centre of gravity and weight shift due to forward/reverse acceleration
        wheel_base = self.__get_wheel_base()
        axle_wheel_ratio_front, axle_wheel_ratio_rear = self.__get_axle_weight_ratios()
        axle_weight_front = self.car.mass * (axle_wheel_ratio_front * self.gravity - self.weight_transfer * self.__accel_c[0] * self.car.cg_height / wheel_base)  # front weight
        axle_weight_rear = self.car.mass * (axle_wheel_ratio_rear * self.gravity + self.weight_transfer * self.__accel_c[0] * self.car.cg_height / wheel_base)  # rear weight

        # Resulting velocity of the wheels as result of the yaw rate of the car body.
        # v = yaw_rate * r where r is distance from axle to CG and yaw_rate (angular velocity) in rad/s.
        yaw_speed_front = self.car.cg_to_front_axle * self.__yaw_rate
        yaw_speed_rear = -self.car.cg_to_rear_axle * self.__yaw_rate

        # Calculate slip angles for front and rear wheels (a.k.a. alpha)
        slip_angle_front = math.atan2(self.__velocity_c[1] + yaw_speed_front, abs(self.__velocity_c[0])) - sign(self.__velocity_c[0]) * self.__steer_angle  # front slip angle
        slip_angle_rear = math.atan2(self.__velocity_c[1] + yaw_speed_rear, abs(self.__velocity_c[0]))  # rear slip

        # angle
        tire_grip_front = self.tire_grip
        tire_grip_rear = self.tire_grip * (1.0 - self.e_brake * (1.0 - self.lock_grip))  # reduce rear grip when ebrake is on
        friction_force_front_cy = clamp(-self.corner_stiffness_front * slip_angle_front, -tire_grip_front, tire_grip_front) * axle_weight_front  # friction front
        friction_force_rear_cy = clamp(-self.corner_stiffness_rear * slip_angle_rear, -tire_grip_rear, tire_grip_rear) * axle_weight_rear  # friction rear

        #  Get amount of brake/throttle from our inputs
        brake = min(self.brake * self.brake_force + self.e_brake * self.e_brake_force, self.brake_force)
        throttle = self.throttle * self.engine_force

        #  Resulting force in local car coordinates.
        #  This is implemented as a RWD car only.
        traction_force_cx = throttle - brake * sign(self.__velocity_c[0])
        traction_force_cy = 0

        drag_force_cx = (-self.roll_resist * self.__velocity_c[0]) - (self.air_resist * self.__velocity_c[0] * abs(self.__velocity_c[0]))
        drag_force_cy = (-self.roll_resist * self.__velocity_c[1]) - (self.air_resist * self.__velocity_c[1] * abs(self.__velocity_c[1]))

        # total force in car coordinates
        total_force_c = np.zeros(2)
        total_force_c[0] = drag_force_cx + traction_force_cx   # forward/reverse accel
        total_force_c[1] = drag_force_cy + traction_force_cy   # forward/reverse accel

        # acceleration along car axes
        self.__accel_c[0] = total_force_c[0] / self.car.mass  # [0] = car acceleration
        self.__accel_c[1] = total_force_c[1] / self.car.mass

        # acceleration in world coordinates
        self.__accel[0] = cs * self.__accel_c[0] - sn * self.__accel_c[1]
        self.__accel[1] = sn * self.__accel_c[0] + cs * self.__accel_c[1]
        # update velocity
        self.__velocity[0] += self.__accel[0] * dt
        self.__velocity[1] += self.__accel[1] * dt
        self.__abs_vel = math.hypot(self.__velocity[1], self.__velocity[0])

        # calculate rotational forces
        angular_torque = (friction_force_front_cy + total_force_c[1]) * self.car.cg_to_front_axle - friction_force_rear_cy * self.car.cg_to_rear_axle

        #  Sim gets unstable at very slow speeds, so just stop the car
        if abs(self.__abs_vel) < 0.01:
            self.__velocity[0] = self.__velocity[1] = self.__abs_vel = 0
            angular_torque = self.__yaw_rate = 0

        angular_accel = angular_torque / self.__get_inertia()

        self.__yaw_rate += angular_accel * dt
        self.car.heading += self.__yaw_rate * dt

        #  finally we can update position
        self.car.pos.x += self.__velocity[0] * dt
        self.car.pos.y += self.__velocity[1] * dt

    def current_speed_mph(self):
        return self.__velocity_c[0] * 2.23694

    def current_speed_kmph(self):
        return self.__velocity_c[0] * 3.6

    def current_speed_mps(self):
        return self.__velocity_c[0]

    def __get_inertia(self):
        return self.car.mass * self.inertia_scale

    def __get_wheel_base(self):
        return self.car.cg_to_front_axle + self.car.cg_to_rear_axle

    def __get_axle_weight_ratios(self):
        wheel_base = self.__get_wheel_base()
        # front_ratio = self.car.cg_to_front_axle / wheel_base
        # rear_ratio = self.car.cg_to_rear_axle / wheel_base
        front_ratio = self.car.cg_to_rear_axle / wheel_base
        rear_ratio = self.car.cg_to_front_axle / wheel_base
        return front_ratio, rear_ratio

