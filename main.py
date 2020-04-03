# import time
# import pygame as pygame
#
# from fsai.objects.line import Line
# from fsai.objects.point import Point
# from fsai.objects.track import Track
# from fsai.visualisation.draw_pygame import render
#
#
# track = Track("examples/data/tracks/azure_circuit.json")
# for cone in track.blue_cones + track.yellow_cones + track.orange_cones + track.big_cones:
#     vector = Line(Point(0, 0), cone.pos)
#     normalised = vector.normalise()
#     length = vector.length()
#     cone.pos = normalised * (length * 1.3)
#
# blue_lines, yellow_lines, orange_lines = track.get_boundary()
#
# screen_size = [1000, 800]
# pygame.init()
# screen = pygame.display.set_mode(screen_size)
#
#
# last_update = time.time()
#
# throttle = 0
#
# down_down = False
# left_down = False
# right_down = False
#
#
# running = True
# while running:
#     steer = 0
#
#     now = time.time()
#     dt = now - last_update
#
#     for event in pygame.event.get():
#         running = event.type != pygame.QUIT
#         if event.type == pygame.KEYDOWN:
#             left_down = left_down or event.key == pygame.K_a
#             throttle = throttle or event.key == pygame.K_w
#             right_down = right_down or event.key == pygame.K_d
#             down_down = down_down or event.key == pygame.K_s
#         if event.type == pygame.KEYUP:
#             if event.key == pygame.K_a:
#                 left_down = False
#             if event.key == pygame.K_w:
#                 throttle = 0
#             if event.key == pygame.K_d:
#                 right_down = False
#             if event.key == pygame.K_s:
#                 down_down = False
#     if left_down: steer = -1
#     if right_down: steer = 1
#
#     car = track.cars[0]
#
#     track.cars[0].throttle = throttle
#     track.cars[0].brake = int(down_down)
#     track.cars[0].steer = steer
#     track.cars[0].physics.update(dt)
#
#     # draw and show the track
#     render(
#         screen,
#         screen_size,
#         cones=[
#             ((255, 255, 0), 5, track.yellow_cones),
#             ((0, 0, 255), 5, track.blue_cones)
#         ],
#         lines=[
#             ((0, 0, 255), 2, blue_lines),
#             ((255, 255, 0), 2, yellow_lines),
#             ((255, 100, 0), 2, orange_lines)
#         ],
#         cars=track.cars
#     )
#
#     pygame.display.flip()
#     last_update = now
# pygame.quit()
import math

import cv2

from fsai.car.car import Car
from fsai.objects.line import Line
from fsai.objects.point import Point
from fsai.path_planning.waypoints import gen_local_waypoints, decimate_waypoints
from fsai.visualisation.draw_opencv import render

track_data_file = "/Users/samgarlick/Downloads/Track Data Export.csv"

track = {}

with open(track_data_file, "r") as file:
    for line in file.readlines()[3:]:
        line_data, value, m = line.split(",")
        value = float(value)
        line_data, index = line_data.replace("]", "").split("[")
        index = int(index)
        tokens = line_data.split(".")[1:]

        if tokens[0] not in track:
            track[tokens[0]] = {}
        if tokens[1] not in track[tokens[0]]:
            track[tokens[0]][tokens[1]] = []

        track[tokens[0]][tokens[1]].append(value)


print(track.keys())
print(track["racingLine"].keys())
print(track["centreLine"].keys())

#
# left_boundary_points = []
# right_boundary_points = []
# for i in range(len(track["trackOutline"]["xTrackEdgeLeft"])):
#     left_boundary_points.append(Point(
#         track["trackOutline"]["xTrackEdgeLeft"][i],
#         track["trackOutline"]["yTrackEdgeLeft"][i],
#     ))
#
# for i in range(len(track["trackOutline"]["xTrackEdgeRight"])):
#     right_boundary_points.append(Point(
#         track["trackOutline"]["xTrackEdgeRight"][i],
#         track["trackOutline"]["yTrackEdgeRight"][i],
#     ))
#
# left_boundary, right_boundary = [], []
# for i in range(len(left_boundary_points)):
#     left_boundary.append(Line(left_boundary_points[i-1], left_boundary_points[i]))
# for i in range(len(right_boundary_points)):
#     right_boundary.append(Line(right_boundary_points[i-1], right_boundary_points[i]))
#
# car = Car(pos=Point(695, -400), heading=math.pi/6 - math.pi)
# waypoints = gen_local_waypoints(
#     car_pos=car.pos,
#     car_angle=car.heading,
#     blue_boundary=left_boundary,
#     yellow_boundary=right_boundary,
#     orange_boundary=[],
#     full_track=True,
#     spacing=3,
#     radar_length=40,
#     smooth=True
# )
#
# rendered_image = render(
#         image_size=(6000, 2400),
#         lines=[
#             ((255, 0, 0), 6, left_boundary),
#             ((0, 255, 255), 6, right_boundary),
#             ((100,100,100), 2, [waypoint.line for waypoint in waypoints])
#         ],
#         cars=[car],
#         padding=10
#     )
#
# cv2.imshow("", rendered_image)
# cv2.imwrite("/Users/samgarlick/Developer/GitHub/OS-FS-AI/output.png", rendered_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
