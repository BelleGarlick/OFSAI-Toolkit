import cv2

from fsai.objects.track import Track
from fsai.visualisation.track_2d import draw_track

track = Track("examples/data/tracks/imola.json")
blue_lines, yellow_lines, orange_lines = track.get_boundary()

image = draw_track(
    track=track,
    lines=[
        ((255, 0, 0), 2, blue_lines),
        ((0, 255, 255), 2, yellow_lines),
        ((0, 100, 255), 2, orange_lines),
    ],
    background=100,
    scale=20,
    padding=50
)
cv2.imshow("", image)
cv2.waitKey(0)
