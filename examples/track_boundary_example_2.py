import cv2

from fsai.mapping.boundary_estimation import create_boundary
from fsai.objects.cone import CONE_COLOR_YELLOW, Cone, CONE_COLOR_BLUE
from fsai.visulisation.track_2d import draw_track


blue_cones = [
    Cone(x=6, y=0, color=CONE_COLOR_BLUE),
    Cone(x=4, y=4, color=CONE_COLOR_BLUE),
    Cone(x=0, y=6, color=CONE_COLOR_BLUE),
    Cone(x=-4, y=4, color=CONE_COLOR_BLUE),
    Cone(x=-6, y=0, color=CONE_COLOR_BLUE),
    Cone(x=-4, y=-4, color=CONE_COLOR_BLUE),
    Cone(x=4, y=-4, color=CONE_COLOR_BLUE),
    Cone(x=0, y=-6, color=CONE_COLOR_BLUE)
]

yellow_cones = [
    Cone(x=3, y=0, color=CONE_COLOR_YELLOW),
    Cone(x=0, y=3, color=CONE_COLOR_YELLOW),
    Cone(x=-3, y=0, color=CONE_COLOR_YELLOW),
    Cone(x=0, y=-3, color=CONE_COLOR_YELLOW)
]

# get boundary from the track
blue_lines, yellow_lines, orange_lines = create_boundary(
    blue_cones=blue_cones,
    yellow_cones=yellow_cones,
)

# draw and show the track
cv2.imshow("Track Boundary Example", draw_track(
    cones=blue_cones+yellow_cones,
    blue_lines=blue_lines,
    yellow_lines=yellow_lines,
    orange_lines=orange_lines
))
cv2.waitKey(0)
cv2.destroyAllWindows()


