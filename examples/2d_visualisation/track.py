import cv2

from fsai.objects.track import Track
from fsai.visualisation.track_2d import draw_track

track = Track("examples/data/tracks/laguna_seca.json")
image = draw_track(track=track)
cv2.imshow("", image)
cv2.waitKey(0)
