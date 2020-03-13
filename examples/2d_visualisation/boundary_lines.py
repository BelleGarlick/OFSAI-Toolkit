import cv2

from fsai.objects.track import Track
from fsai.visualisation.track_2d import draw_track

track = Track("examples/data/tracks/imola.json")
blue_lines, yellow_lines, orange_lines = track.get_boundary()

image = draw_track(
    track=track,
    blue_lines=blue_lines,
    yellow_lines=yellow_lines,
    orange_lines=orange_lines,
    background=100,
    scale=20,
    padding=50
)
cv2.imshow("", image)
cv2.waitKey(0)
