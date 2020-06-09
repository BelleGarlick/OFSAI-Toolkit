import time
import json
import copy
import math
import random
from typing import List

import numpy as np
import pygame
import requests

from fsai.objects.track import Track
from fsai.path_planning.waypoint import Waypoint
from fsai.visualisation.draw_pygame import render
from fsai.path_planning.waypoints import gen_waypoints, encode, decimate_waypoints
from fsai import geometry

URL_NEW = "http://127.0.0.1:8080/new/"
URL_POST = "http://127.0.0.1:8080/save/"

RUNS_PER_SEGMENTS = 20
MAX_STEP_SIZE = 0.05
DELTA_STEP_SIZE = 0.6


class OptimalPathCreator:
    def __init__(self):
        self.name, self.uuid, self.initial_car, left_boundary, right_boundary, orange_boundary, self.intervals = get_track()
        self.boundary = left_boundary + right_boundary + orange_boundary

        self.waypoints = generate_waypoints(self.initial_car, left_boundary, right_boundary, orange_boundary)
        self.waypoint_count = len(self.waypoints)

        self.current_index = 0

        self.best_result = math.inf

    def run(self):
        pygame.init()
        screen_size = [800, 400]
        screen = pygame.display.set_mode(screen_size)

        start_step_time = time.time()

        running = True
        epoch = 0

        while running:
            new_best = False
            for i in range(RUNS_PER_SEGMENTS):
                step_size = MAX_STEP_SIZE * (DELTA_STEP_SIZE ** i)
                new_best = new_best or self.genetic_test(step_size)

            self.current_index += 1
            if self.current_index >= self.waypoint_count:
                self.current_index = 0

                print("{}".format(self.best_result))
                render_scene(screen, screen_size, self.waypoints, self.current_index)

                now = time.time()
                print("Epoch: {} Time: {}".format(epoch, now - start_step_time))
                start_step_time = now
                epoch += 1
                if epoch >= self.intervals:
                    running = False

        self.save_state()

    def save_state(self):
        best_result = self.best_result

        dict = {
            "time": best_result,
            "waypoints": [{
                "x1": waypoint.line[0],
                "y1": waypoint.line[1],
                "x2": waypoint.line[2],
                "y2": waypoint.line[3],
                "throttle": waypoint.throttle,
                "pos": waypoint.optimum
            } for waypoint in self.waypoints]
        }

        x = requests.post(URL_POST, data={
            "name": self.name,
            "uuid": self.uuid,
            "data": json.dumps(dict)
        })
        print("Sending data to server: " + str(x.status_code))

    def try_track(self, waypoints) -> float:

        # get waypoint - waypoint angle
        radius = []
        velocities = []
        for i in range(self.waypoint_count):

            p = waypoints[i - 1].get_optimum_point()
            c = waypoints[i].get_optimum_point()
            n = waypoints[(i + 1) % self.waypoint_count].get_optimum_point()

            r = geometry.findCircleRadius(p[0], p[1], c[0], c[1], n[0], n[1])
            waypoints[i].r = r
            radius.append({
                "R": r,
                "s": geometry.distance(c, n)
            })

        for i in range(self.waypoint_count):
            R = radius[(i) % self.waypoint_count]["R"]
            AyMax = 1.2 * 9.81 * 1
            VMax = 26 # mps = 60

            Vd = math.sqrt(R * AyMax)
            Vd = min(Vd, VMax)
            V = Vd * 1
            velocities.append(V)

        velocities = self.smooth_velocity(velocities)
        velocities = self.smooth_velocity(velocities)

        for i in range(len(velocities)):
            waypoints[i].v = velocities[i]

        total_time = 0
        for i in range(self.waypoint_count):
            if velocities[i] == 0:
                return math.inf
            total_time += radius[i]["s"] / velocities[i]
        return total_time

    def smooth_velocity(self, velocities):
        new_velocities = []
        for i in range(len(velocities)):
            pv = velocities[i - 1]
            cv = velocities[i]
            nv = velocities[(i + 1) % self.waypoint_count]
            v = cv
            if nv < cv:
                v = (cv + nv + nv + nv) / 4
            if pv < cv:
                v = (cv + pv + pv) / 3
            new_velocities.append(v)
        return new_velocities

    def genetic_test(self, step_size):
        index = self.current_index
        initial_value = self.waypoints[index].optimum
        best = False
        self.waypoints[index].optimum = max(0, min(1, initial_value + step_size))
        test_time = self.try_track(self.waypoints)
        if test_time < self.best_result:
            self.best_result = test_time
            best = True
        else:
            self.waypoints[index].optimum = max(0, min(1, initial_value - step_size))
            test_time = self.try_track(self.waypoints)
            if test_time < self.best_result:
                self.best_result = test_time
                best = True
            else:
                self.waypoints[index].optimum = initial_value

        return best


def get_track():
    result = requests.get(URL_NEW)
    server_data = json.loads(result.text)
    name = server_data["name"]
    print(name)
    uuid = server_data["uuid"]
    intervals = server_data["intervals"]

    track = Track().from_json(server_data["data"])
    initial_car = track.cars[0]

    initial_car.waypoint_index = -1
    initial_car.last_interval_update = 0

    left_boundary, right_boundary, orange_boundary = track.get_boundary()

    if ".reversed" in name:
        initial_car.heading += math.pi
        left_boundary, right_boundary = right_boundary, left_boundary

    return name, uuid, initial_car, left_boundary, right_boundary, orange_boundary, intervals


def generate_waypoints(initial_car, left_boundary, right_boundary, orange_boundary):
    waypoints = gen_waypoints(
        car_pos=initial_car.pos,
        car_angle=initial_car.heading,
        blue_boundary=left_boundary,
        yellow_boundary=right_boundary,
        orange_boundary=orange_boundary,
        full_track=True,
        spacing=1,
        radar_length=30,
        radar_count=17,
        radar_span=math.pi / 1.1,
        margin=0,
        smooth=True
    )
    decimated_waypoints = []
    for i in range(len(waypoints)):
        if i % 3 == 0:
            decimated_waypoints.append(waypoints[i])
    # decimated_waypoints = decimate_waypoints(decimated_waypoints, threshold = 0.35, spread=1, max_gap = 4)
    for waypoint in decimated_waypoints:
        waypoint.optimum = 0.5 #random.random()
        waypoint.v = 0
    return decimated_waypoints


def render_scene(screen, screen_size, waypoints, start_var):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    var_lines = []
    var_lines.append(waypoints[start_var].line)

    render_points = [

    ]

    render_lines = [
        ((200, 255, 200), 1, [w.line for w in waypoints]),
        ((0, 0, 200), 1, var_lines)
    ]

    # for i in range(len(waypoints)):
    #     point = waypoints[i].get_optimum_point()
    #     throttle = (waypoints[i].throttle + 1) / 2
    #
    #     c = (int((1 - throttle) * 255), int((throttle) * 255), 0)
    #     render_points.append((c, 2, [point]))

    for i in range(0, len(waypoints)):
        line = waypoints[i].get_optimum_point() + waypoints[i - 1].get_optimum_point()

        v = waypoints[i].v / 26

        r = min(255, 510 - int(510 * v))
        g = min(255, 0 + int(510 * v))
        c = (r, g, 0)
        render_lines.append((c, 4, [line]))

    render(
        screen,
        screen_size,
        lines=render_lines,
        # points=[
        #     ((255, 0, 255), 2, best_points)
        # ],
        points=render_points,
        padding=0
    )

    pygame.display.flip()


if __name__ == "__main__":
    OptimalPathCreator().run()
