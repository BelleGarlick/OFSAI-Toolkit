import cv2

from fsai.objects.track import Track
from fsai.visualisation.track_2d import draw_track


# load track track into cone objects
track = Track("examples/data/tracks/laguna_seca.json")

# get boundary from the track
blue_lines, yellow_lines, orange_lines = track.get_boundary()

# draw and show the track
cv2.imshow("Track Boundary Example", draw_track(
    lines=[
        ((255, 0, 0), 2, blue_lines),
        ((0, 255, 255), 2, yellow_lines),
        ((0, 100, 255), 2, orange_lines),
    ],
))
cv2.waitKey(0)
cv2.destroyAllWindows()


