import random

import cv2

from fsai.objects.track import Track
from fsai.path_planning.target_line import get_points_from_waypoints
from fsai.path_planning.waypoints import gen_local_waypoints, decimate_waypoints
from fsai.visualisation.track_2d import draw_track


track = Track("examples/data/tracks/cadwell_park.json")
blue_lines, yellow_lines, orange_lines = track.get_boundary()

waypoints = gen_local_waypoints(
    track.cars[0].pos,
    track.cars[0].orientation,
    full_track=True,
    blue_boundary=blue_lines,
    yellow_boundary=yellow_lines,
    orange_boundary=orange_lines,
    spacing=1,
    smooth=True,
    margin=1
)

waypoints = decimate_waypoints(waypoints)

scene = draw_track(
    track=track,
    waypoints=waypoints,
    blue_lines=blue_lines,
    yellow_lines=yellow_lines,
    orange_lines=orange_lines,
    target_line=get_points_from_waypoints(waypoints)
)
cv2.imshow("", scene)
cv2.waitKey(0)
cv2.destroyAllWindows()
