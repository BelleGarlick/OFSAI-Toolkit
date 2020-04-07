import cv2

from fsai.objects.track import Track
from fsai.path_planning.waypoints import gen_local_waypoints, encode
from fsai.visualisation.draw_opencv import render

track = Track("examples/data/tracks/azure_circuit.json")
left_boundary, right_boundary, o = track.get_boundary()
car = track.cars[0]
car.pos.x += 35
car.pos.y += 20

waypoints = gen_local_waypoints(
        car_pos=car.pos,
        car_angle=car.heading,
        blue_boundary=left_boundary,
        yellow_boundary=right_boundary,
        orange_boundary=o,
        foresight=5,
        spacing=1.5,
        negative_foresight=4,
        smooth=True
)
encoding = encode(waypoints, 4)

cv2.imshow("", render(
    [600, 400],
    lines=[
        ((255, 0, 0), 2, left_boundary),
        ((0, 255, 0), 2, right_boundary),
        ((0, 0, 255), 2, [waypoint.line for waypoint in waypoints])
    ],
    cars=[car]
))
cv2.waitKey(0)