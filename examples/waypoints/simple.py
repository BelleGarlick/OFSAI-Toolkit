import cv2

from fsai.objects.track import Track
from fsai.path_planning.waypoints import gen_local_waypoints
from fsai.visualisation.track_2d import draw_track


# Load track and boundary
track = Track("examples/data/tracks/loheac.json")
blue_lines, yellow_lines, orange_lines = track.get_boundary()

# Gen 10 waypoints ahead of vehicle, 5 behind, spaced by 2 meters
waypoints = gen_local_waypoints(
    track.cars[0].pos,
    track.cars[0].orientation,
    blue_boundary=blue_lines,
    yellow_boundary=yellow_lines,
    orange_boundary=orange_lines,
    foresight=10,  # 10 waypoints ahead
    negative_foresight=5,  # 5 waypoints behind
    spacing=2
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
