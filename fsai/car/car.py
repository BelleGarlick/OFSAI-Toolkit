import numpy as np

from fsai.car.physics.advanced import CarPhysics


class Car:
    def __init__(
            self,
            pos: np.ndarray=None,
            heading=0,
    ):
        self.pos = pos if pos is not None else np.array([0, 0])
        self.heading = heading
        self.physics = CarPhysics(self)

        self.mass = 1000

        self.width = 0.8
        self.max_steer = 0.6
        self.wheel_radius = 0.5
        self.wheel_width = 0.3
        self.cg_to_front_axle = 0.75
        self.cg_to_rear_axle = 0.75
        self.cg_height = 0.55
        self.cg_to_front = 1.25
        self.cg_to_rear = 1.25

        self.throttle = 0
        self.brake = 0
        self.steer = 0
