import math

import cv2

from fsai.objects.track import Track
from fsai.path_planning.waypoints import gen_local_waypoints, YELLOW_ON_LEFT
from fsai.visualisation.draw_opencv import draw_track


# Load track and boundary
track = Track("examples/data/tracks/monza.json")
blue_lines, yellow_lines, orange_lines = track.get_boundary()

""" Generate waypoints such that the yellow boundary is on the left, rather than blue """
waypoints = gen_local_waypoints(
    track.cars[0].pos,
    track.cars[0].heading + math.pi,  # Rotate the car so that it faces the direction with yellow on the left
    blue_boundary=blue_lines,
    yellow_boundary=yellow_lines,
    orange_boundary=orange_lines,
    foresight=20,  # Generate lines ahead of car only
    negative_foresight=0,
    spacing=1.5,
    margin=1.5,
    left_boundary_colour=YELLOW_ON_LEFT  # Generate lines where yellow is the left boundary
)

# render the track
scene = draw_track(
    track=track,
    lines=[
        (150, 150, 150), 2, [waypoint.line for waypoint in waypoints],
        ((255, 0, 0), 2, blue_lines),
        ((0, 255, 255), 2, yellow_lines),
        ((0, 100, 255), 2, orange_lines),
    ],
)
cv2.imshow("", scene)
cv2.waitKey(0)
cv2.destroyAllWindows()
