import cv2

from fsai.objects.track import Track
from fsai.visualisation.track_2d import draw_track

track = Track("examples/data/tracks/loheac.json")
blue_lines, yellow_lines, orange_lines = track.get_boundary()

image = draw_track(
    cones=track.blue_cones + track.yellow_cones + track.big_orange_cones,
    blue_lines=blue_lines,
    yellow_lines=yellow_lines,
    orange_lines=orange_lines,
    blue_line_colour=(100, 255, 100),
    yellow_line_colour=(255, 0, 255),
    big_orange_cone_colour=(0, 255, 0)
)
cv2.imshow("", image)
cv2.waitKey(0)
