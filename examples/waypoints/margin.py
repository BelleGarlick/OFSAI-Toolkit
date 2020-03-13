import cv2

from fsai.objects.track import Track
from fsai.path_planning.waypoints import gen_local_waypoints
from fsai.visualisation.track_2d import draw_track


# Load track and boundary
track = Track("examples/data/tracks/monza.json")
blue_lines, yellow_lines, orange_lines = track.get_boundary()

# gen waypoints, with a 1.5 meter margin
waypoints = gen_local_waypoints(
    track.cars[0].pos,
    track.cars[0].orientation,
    blue_boundary=blue_lines,
    yellow_boundary=yellow_lines,
    orange_boundary=orange_lines,
    foresight=20,  # 20 waypoints ahead
    negative_foresight=5,  # 5 waypoints behind
    spacing=1.5,  # 1.5m spacing
    margin=1.5  # Apply a margin of 1.5 meters either side of the track
)

# render the track
scene = draw_track(
    track=track,
    waypoints=waypoints,
    blue_lines=blue_lines,
    yellow_lines=yellow_lines,
    orange_lines=orange_lines
)
cv2.imshow("", scene)
cv2.waitKey(0)
cv2.destroyAllWindows()
