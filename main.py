import cv2

from fsai.objects.track import Track
from fsai.path_planning.waypoints import gen_local_waypoints
from fsai.visualisation.track_2d import draw_track


track = Track("examples/data/tracks/cadwell_park.json")
blue_lines, yellow_lines, orange_lines = track.get_boundary()

waypoints, points = gen_local_waypoints(
    track.car_pos,
    track.car_orientation,
    forsight=10,
    back=10,
    blue_boundary=blue_lines,
    yellow_boundary=yellow_lines,
    orange_boundary=orange_lines,
    margin=0.5
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
cv2.destroyAllWindows()

