import cv2

from fsai.objects.track import Track
from fsai.path_planning.waypoints import gen_local_waypoints, decimate_waypoints
from fsai.visualisation.draw_opencv import render


# Load track and boundary
track = Track("examples/data/tracks/red_bull_ring.json")
blue_lines, yellow_lines, orange_lines = track.get_boundary()


# Generate full track of waypoints - full waypoints
waypoints = gen_local_waypoints(
    track.cars[0].pos,
    track.cars[0].heading,
    blue_boundary=blue_lines,
    yellow_boundary=yellow_lines,
    orange_boundary=orange_lines,
    full_track=True,  # full track
    spacing=1.5,  # 1.5m spacing
    margin=1,  # Apply a margin of 1 meters either side of the track
    smooth=True
)

# decimated waypoints
waypoints = decimate_waypoints(
    waypoints,
    spread=1,  # how many surrounding waypoints are kept
    max_gap=5,  # max gap between waypoints
    threshold=0.25  # min radian change between waypoints
)

# render the track
scene = render(
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
