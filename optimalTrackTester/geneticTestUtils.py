import json
import math
import time

import pygame
import requests

from fsai import geometry
from fsai.objects.track import Track
from fsai.visualisation.draw_pygame import render

URL_NEW = "http://127.0.0.1:8080/new/"
URL_POST = "http://127.0.0.1:8080/save/"


TRACK_PATH = "/Users/samgarlick/Developer/GitHub/OS-FS-AI/examples/data/tracks/{}.json"


def get_track_from_server():
    result = requests.get(URL_NEW)
    server_data = json.loads(result.text)
    name = server_data["name"]
    print(name)
    uuid = server_data["uuid"]
    intervals = server_data["intervals"]

    track = Track().from_json(server_data["data"])
    initial_car = track.cars[0]

    left_boundary, right_boundary, orange_boundary = track.get_boundary()

    if ".reversed" in name:
        initial_car.heading += math.pi
        left_boundary, right_boundary = right_boundary, left_boundary

    return name, uuid, initial_car, left_boundary, right_boundary, orange_boundary, intervals


def get_track_from_files(track_name):
    uuid = -1
    intervals = 50000
    print(track_name)

    track = Track(TRACK_PATH.format(track_name.replace(".reversed", "")))
    initial_car = track.cars[0]

    left_boundary, right_boundary, orange_boundary = track.get_boundary()

    if ".reversed" in track_name:
        initial_car.heading += math.pi
        left_boundary, right_boundary = right_boundary, left_boundary

    return track_name, uuid, initial_car, left_boundary, right_boundary, orange_boundary, intervals


def send_track_to_server(track_name, waypoints, best_time, uuid):
    best_result = best_time

    dict = {
        "time": best_result,
        "waypoints": [{
            "x1": waypoint.line[0],
            "y1": waypoint.line[1],
            "x2": waypoint.line[2],
            "y2": waypoint.line[3],
            "pos": waypoint.optimum,
            "v": waypoint.optimum
        } for waypoint in waypoints]
    }

    x = requests.post(URL_POST, data={
        "name": track_name,
        "uuid": uuid,
        "data": json.dumps(dict)
    })
    print("Sending data to server: " + str(x.status_code))


def get_track_time(waypoints, max_frictional_force, max_speed):
    # get waypoint - waypoint angle
    waypoint_length = len(waypoints)

    optimum_points = [w.get_optimum_point() for w in waypoints]
    optimum_points = [optimum_points[-1]] + optimum_points + [optimum_points[0]]

    radii, lengths = geometry.get_corner_radii_for_points(optimum_points)

    velocities = geometry.get_max_velocities_from_corners(radii, max_frictional_force, max_speed)

    for i in range(10):
        velocities = smooth_velocity(velocities)

    for i in range(len(velocities)):
        waypoints[i].v = velocities[i]

    total_time = 0
    for i in range(waypoint_length):
        if velocities[i] == 0:
            return math.inf
        total_time += lengths[i] / velocities[i]

    return total_time


def smooth_velocity(velocities):
    vel_length = len(velocities)
    new_velocities = []
    for i in range(len(velocities)):
        pv = velocities[i - 1]
        cv = velocities[i]
        nv = velocities[(i + 1) % vel_length]
        v = cv
        if nv < cv:
            print("{} {}".format(nv, cv))
            v = nv + (cv - nv) * 0.95  # higher value means better brakes
            cv = v
        if pv < cv:
            v = pv + (cv - pv) * 0.3  # higher bias means faster acceleration
        new_velocities.append(v)
    return new_velocities


def render_scene(screen, screen_size, waypoints_list, line_width=4):
    should_close = False
    for event in pygame.event.get():
        should_close = event.type == pygame.QUIT

    render_lines = [
        ((200, 255, 200), 1, [w.line for w in waypoints_list[0]])
    ]

    waypoints = waypoints_list[0]
    for i in range(0, len(waypoints)):
        p0 = geometry.add(waypoints[i].line[0:2], geometry.scale(geometry.sub(waypoints[i].line[2:4], waypoints[i].line[0:2]), 0.5))
        p1 = geometry.add(waypoints[i-1].line[0:2], geometry.scale(geometry.sub(waypoints[i-1].line[2:4], waypoints[i-1].line[0:2]), 0.5))
        line = p0 + p1
        c = (0, 0, 255)
        render_lines.append((c, line_width, [line]))

    for waypoints in waypoints_list:
        for i in range(0, len(waypoints)):
            line = waypoints[i].get_optimum_point() + waypoints[i - 1].get_optimum_point()

            v = 0.5
            try:
                v = waypoints[i].v / 26
            except:
                pass

            r = min(255, 510 - int(510 * v))
            g = min(255, 0 + int(510 * v))
            c = (r, g, 0)
            render_lines.append((c, line_width, [line]))

    render(
        screen,
        screen_size,
        lines=render_lines,
        padding=0
    )

    pygame.display.flip()

    return should_close

def printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()
