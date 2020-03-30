import cv2

from fsai.objects.track import Track
from fsai.path_planning.waypoints import gen_local_waypoints
from fsai.visualisation.track_2d import draw_track


# Load track and boundary
track = Track("examples/data/tracks/willow_springs.json")
blue_lines, yellow_lines, orange_lines = track.get_boundary()

""" Generate the waypoints when only the blue boundaries are known """
waypoints = gen_local_waypoints(
    track.cars[0].pos,
    track.cars[0].heading,
    blue_boundary=blue_lines,
    yellow_boundary=[],
    orange_boundary=orange_lines,
    foresight=20,
    spacing=2.5,  # 3m spacing
    margin=1.5  # Apply a margin of 1 meters either side of the track
)

# render the track
scene = draw_track(
    track=track,
    lines=[
        (150, 150, 150), 2, [waypoint.line for waypoint in waypoints],
        ((255, 0, 0), 2, blue_lines),
        ((0, 255, 255), 2, []),
        ((0, 100, 255), 2, orange_lines),
    ],
)
cv2.imshow("", scene)
cv2.waitKey(0)
cv2.destroyAllWindows()


""" Generate waypoints when only the yellow boundaries are known """
waypoints = gen_local_waypoints(
    track.cars[0].pos,
    track.cars[0].heading,
    blue_boundary=[],
    yellow_boundary=yellow_lines,
    orange_boundary=orange_lines,
    foresight=20,
    spacing=2.5,  # 3m spacing
    margin=1.5  # Apply a margin of 1 meters either side of the track
)

# render the track
scene = draw_track(
    track=track,
    waypoints=waypoints,
    lines=[
        ((255, 0, 0), 2, []),
        ((0, 255, 255), 2, yellow_lines),
        ((0, 100, 255), 2, orange_lines),
    ],
)
cv2.imshow("", scene)
cv2.waitKey(0)
cv2.destroyAllWindows()
