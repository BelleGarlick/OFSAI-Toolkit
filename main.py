import cv2

from fsai.objects.line import Line
from fsai.objects.point import Point
from fsai.objects.track import Track
from fsai.path_planning.waypoints import gen_local_waypoints
from fsai.visualisation.track_2d import draw_track

track = Track("examples/data/tracks/azure_circuit.json")
blue_lines, yellow_lines, orange_lines = track.get_boundary()
print(track.to_json())

for i in range(200):
    waypoints, points = gen_local_waypoints(
        track.car_pos,
        i/20,
        front=10,
        back=10,
        blue_boundary=blue_lines,
        yellow_boundary=yellow_lines,
        orange_boundary=orange_lines
    )
    scene = draw_track(
        track=track,
        waypoints=waypoints,
        blue_lines=blue_lines,
        yellow_lines=yellow_lines,
        orange_lines=orange_lines,
        pedestrians=points
    )
    cv2.imshow("", scene)
    cv2.waitKey(0)

