import math
import numpy as np

from fsai.objects.point import Point


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
        #
        # self.throttle: float = 0
        # self.brake: float = 0
        # self.e_brake: float = 0
        # self.__steer: float = 0  # steering [-1 - 1]
        #
        # # variable stats about the car
        # self.__velocity: np.ndarray = np.zeros(2)  # m/s in world coords
        # self.__velocity_c: np.ndarray = np.zeros(2)  # m/s in local space (x forwards, y sideways)
        # self.__accel: np.ndarray = np.zeros(2)  # accel in world space
        # self.__accel_c: np.ndarray = np.zeros(2)  # accel in local space
        #
        # self.__abs_vel: float = 0  # absolute velocity in m/s
        # self.__yaw_rate: float = 0  # angular vel in radians
        # self.__steer_angle: float = 0  # actual amount of front wheel steering  [-max_steer, max_steer]
        #
        # self.weight_transfer = 0.2
        # self.gravity = 9.81
        # self.engine_force = 10000
        # self.brake_force = 12000
        self.inertia_scale = 1
        self.corner_stiffness_front = 0.5
        self.corner_stiffness_rear = 0.5
        self.tire_grip = 50
        # self.lock_grip = 0.7
        # self.e_brake_force = 2000

        self.mass = 1000  # kg
        self.engine_power = 10000

        self.velocity = Point(0, 0)  # x = long, y == lat
        self.local_acceleration = Point(0, 0)  # x = long, y == lat
        self.angular_speed = 0

        self.aero_drag = 2.5
        self.roll_drag = 5
        
    def update(self, throttle: float, break_val: float, steer: float, dt: float):
        local_velocity = self.velocity.copy()
        local_velocity.rotate_around(Point(0, 0), -self.car.heading)

        # self.car.heading += steer * 0.1
        total_force = self.get_total_force(local_velocity, throttle * self.engine_power)  # relative

        angular_torque = self.get_angular_torque(local_velocity, steer * self.car.max_steer, total_force.y)
        angular_accel = angular_torque / self.__get_inertia()

        # if abs(local_velocity.x) < 0.1:
        #     local_velocity = Point(0, 0)
        #
        #     angular_torque = self.__yaw_rate = 0

        self.angular_speed += angular_accel * dt
        self.car.heading += self.angular_speed * dt


        self.local_acceleration = Point(0, 0)
        self.local_acceleration = total_force / self.mass
        local_velocity += self.local_acceleration * dt

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

    def get_axis_loads(self):
        # wheel_base = self.__get_wheel_base()
        # axle_wheel_ratio_front, axle_wheel_ratio_rear = self.__get_axle_weight_ratios()
        # axle_weight_front = self.car.mass * (axle_wheel_ratio_front * self.gravity - self.weight_transfer * self.__accel_c[0] * self.car.cg_height / wheel_base)  # front weight
        # axle_weight_rear = self.car.mass * (axle_wheel_ratio_rear * self.gravity + self.weight_transfer * self.__accel_c[0] * self.car.cg_height / wheel_base)  # rear weight
        wheel_base = self.car.cg_to_rear_axle + self.car.cg_to_front_axle
        axle_weight_front = self.car.mass * (self.car.cg_to_rear_axle / wheel_base)
        axle_weight_rear = self.car.mass * (self.car.cg_to_front_axle / wheel_base)
        return axle_weight_front, axle_weight_rear

    def get_slip_angles(self, local_velocity, steer_angle):
        yaw_speed_front = self.car.cg_to_front_axle * self.angular_speed
        yaw_speed_rear = -self.car.cg_to_rear_axle * self.angular_speed

        # Calculate slip angles for front and rear wheels (a.k.a. alpha)
        slip_angle_front = math.atan2(local_velocity.y + yaw_speed_front, abs(local_velocity.x)) - sign(local_velocity.x) * steer_angle  # front slip angle
        slip_angle_rear = math.atan2(local_velocity.y + yaw_speed_rear, abs(local_velocity.x))  # rear slip
        return slip_angle_front, slip_angle_rear

    def __get_inertia(self):
        return self.car.mass * self.inertia_scale

    def get_lateral_force(self, slip_front, slip_rear):
        axle_weight_front, axle_weight_rear = self.get_axis_loads()
        friction_force_front_cy = clamp(-self.corner_stiffness_front * slip_front, -self.tire_grip, self.tire_grip) * axle_weight_front  # friction front
        friction_force_rear_cy = clamp(-self.corner_stiffness_rear * slip_rear, -self.tire_grip, self.tire_grip) * axle_weight_rear  # friction rear
        return friction_force_front_cy, friction_force_rear_cy

    def get_angular_torque(self, local_velocity, steer_angle, lateral_force):
        slip_front, slip_rear = self.get_slip_angles(local_velocity, steer_angle)
        force_lateral_front, force_lateral_rear = self.get_lateral_force(slip_front, slip_rear)
        force_lateral_front += lateral_force
        force_lateral_rear += lateral_force

        angular_torque = (math.cos(steer_angle) * self.car.cg_to_front_axle * force_lateral_front) - (self.car.cg_to_rear_axle * force_lateral_rear)
        return angular_torque * self.mass