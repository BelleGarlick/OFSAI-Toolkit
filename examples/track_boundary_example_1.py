import cv2

from fsai.objects.track import Track
from fsai.visulisation.track_2d import draw_track


# load track track into cone objects
track = Track("examples/data/tracks/laguna_seca.json")

# get boundary from the track
blue_lines, yellow_lines, orange_lines = track.get_boundary()

# draw and show the track
cv2.imshow("Track Boundary Example", draw_track(
    blue_lines=blue_lines,
    yellow_lines=yellow_lines,
    orange_lines=orange_lines
))
cv2.waitKey(0)
cv2.destroyAllWindows()


