import copy
import math
import fsai.geometry as geometry
import numpy as np

from fsai.visualisation.draw_opencv import render_area


class AI:
    def __init__(self, simulation):
        self.simulation = simulation

        self.alive = True
        self.car = copy.deepcopy(simulation.base_car)

        self.distance = 0
        self.model = simulation.gen_model()

    def update(self, dt: float):
        image, car_data = self.get_model_input_data()
        predictions = self.model.predict([
            np.array([image]), np.array([car_data])
        ])
        predictions = predictions[0]
        self.car.steer = predictions[0] * 2 - 1
        self.car.throttle = predictions[1]
        self.car.brake = predictions[2]

        self.distance += self.car.physics.update(dt)

        self.kill()

    def kill(self):
        if has_intersected(self.car, self.simulation.all_boundaries):
            self.alive = False

    def get_model_input_data(self):
        image = render_area(
            camera_pos=self.car.pos,
            rotation=-self.car.heading - math.pi / 2,
            area=[40, 60],
            resolution=2,
            lines=[
                ((255, 0, 0), 1, geometry.filter_lines_by_distance(self.car.pos, 60, self.simulation.blue_boundary)),
                ((0, 255, 255), 1, geometry.filter_lines_by_distance(self.car.pos, 60, self.simulation.yellow_boundary)),
                ((0, 100, 255), 1, geometry.filter_lines_by_distance(self.car.pos, 60, self.simulation.o)),
            ],
            cars=[self.car],
            background=0
        )

        car_data = np.array(
            [
                sqrt(self.car.steer), self.car.throttle, self.car.brake,
                sqrt(self.car.physics.accel_c[0] / 24), sqrt(self.car.physics.accel_c[1] / 50),
                sqrt(self.car.physics.vel_c[0] / 26), sqrt(self.car.physics.vel_c[1] / 12)
            ]
        )

        return image, car_data


def has_intersected(car, all_boundary_lines):
    body_points = [
        (car.pos[0] + car.cg_to_front, car.pos[1] - (car.width + car.wheel_width) / 2),
        (car.pos[0] + car.cg_to_front, car.pos[1] + (car.width + car.wheel_width) / 2),
        (car.pos[0] - car.cg_to_rear, car.pos[1] + (car.width + car.wheel_width) / 2),
        (car.pos[0] - car.cg_to_rear, car.pos[1] - (car.width + car.wheel_width) / 2)
    ]

    for i in range(len(body_points)):
        body_points[i] = geometry.rotate(body_points[i], car.heading, car.pos)

    car_boundary = np.array([
        [body_points[0][0], body_points[0][1], body_points[1][0], body_points[1][1]],
        [body_points[1][0], body_points[1][1], body_points[2][0], body_points[2][1]],
        [body_points[2][0], body_points[2][1], body_points[3][0], body_points[3][1]],
        [body_points[3][0], body_points[3][1], body_points[0][0], body_points[0][1]],
    ])

    filtered_lines = geometry.filter_lines_by_distance(car.pos, 8, all_boundary_lines)

    intersections = geometry.segment_intersections(car_boundary[0], filtered_lines) + \
                    geometry.segment_intersections(car_boundary[1], filtered_lines) + \
                    geometry.segment_intersections(car_boundary[2], filtered_lines) + \
                    geometry.segment_intersections(car_boundary[3], filtered_lines)

    return len(intersections) > 0


def sqrt(x):
    rt = math.sqrt(abs(x))
    if x < 0:
        return -rt
    return rt