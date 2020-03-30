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
    track.cars[0].heading,
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
