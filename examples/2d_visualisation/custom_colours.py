import cv2

from fsai.objects.track import Track
from fsai.visualisation.track_2d import draw_track

track = Track("examples/data/tracks/loheac.json")
blue_lines, yellow_lines, orange_lines = track.get_boundary()

image = draw_track(
    cones=[
        ((255, 0, 0), 5, track.blue_cones),
        ((0, 255, 255), 5, track.yellow_cones),
        ((0, 0, 255), 5, track.big_cones)
    ],
    lines=[
        ((100, 255, 100), 2, blue_lines),
        ((255, 0, 255), 2, yellow_lines),
        ((0, 255, 0), 2, orange_lines),
    ]
)
cv2.imshow("", image)
cv2.waitKey(0)
