import math

import cv2

from fsai.car.car import Car
from fsai.objects.line import Line
from fsai.objects.point import Point
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

print(track["trackOutline"]["xTrackEdgeRight"][-1])
        # track["trackOutline"]["yTrackEdgeLeft"][i],)

left_boundary_points = []
right_boundary_points = []
for i in range(len(track["trackOutline"]["xTrackEdgeLeft"])):
    left_boundary_points.append(Point(
        track["trackOutline"]["xTrackEdgeLeft"][i],
        track["trackOutline"]["yTrackEdgeLeft"][i],
    ))

for i in range(len(track["trackOutline"]["xTrackEdgeRight"])):
    right_boundary_points.append(Point(
        track["trackOutline"]["xTrackEdgeRight"][i],
        track["trackOutline"]["yTrackEdgeRight"][i],
    ))

left_boundary, right_boundary = [], []
for i in range(len(left_boundary_points)):
    left_boundary.append(Line(left_boundary_points[i-1], left_boundary_points[i]))
for i in range(len(right_boundary_points)):
    right_boundary.append(Line(right_boundary_points[i-1], right_boundary_points[i]))

total = 0
heading = 2.076
cuurrent_point = Point(track["trackOutline"]["xTrackEdgeRight"][0], track["trackOutline"]["yTrackEdgeRight"][0])
lines = []
for i in range(1, len(track["racingLine"]["sLap"])-1):
    length = track["racingLine"]["sLap"][i] - track["racingLine"]["sLap"][i-1]
    angle = track["racingLine"]["cLap"][i]
    heading += angle
    new_point = Point(0, length)
    new_point.rotate_around(Point(0, 0), heading)
    new_point += cuurrent_point
    lines.append(Line(cuurrent_point, new_point))
    cuurrent_point = new_point
    total += length

car = Car(pos=Point(695, -400), heading=math.pi/6 - math.pi)
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

rendered_image = render(
     image_size=(4000, 2400),
     lines=[
         ((255, 0, 0), 6, left_boundary),
         ((0, 255, 255), 6, right_boundary),
         # ((100,100,100), 2, [waypoint.line for waypoint in waypoints]),
         ((0,0,255), 2, lines)
     ],
     cars=[car],
     padding=10
 )

# cv2.imshow("", rendered_image)
cv2.imwrite("/Users/samgarlick/Developer/GitHub/OS-FS-AI/output.png", rendered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
