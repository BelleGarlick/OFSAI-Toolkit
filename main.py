import math
import time

import cv2

from fsai.objects.track import Track
from fsai.path_planning.waypoints import gen_waypoints

if __name__ == "__main__":
    track = Track("examples/data/tracks/azure_circuit.json")
    left_boundary, right_boundary, o = track.get_boundary()
    car = track.cars[0]
    car.steer = 0
    car.pos[0] += 48
    car.pos[1] += 15

    now = time.time()
    count = 1000
    for i in range(count):
        waypoints = gen_waypoints(
            car_pos=car.pos,
            car_angle=car.heading,
            blue_boundary=left_boundary,
            yellow_boundary=right_boundary,
            orange_boundary=o,
            foresight=10,
            spacing=2,
            radar_count=5,
            radar_length=10,
            negative_foresight=5,
            margin=0.2,
            smooth=True
        )
    print((time.time() - now) / count)
    # encoding = encode(waypoints, 4)
    #
    # cv2.imshow("", render(
    #     [1200, 900],
    #     lines=[
    #         ((255, 0, 0), 2, left_boundary),
    #         ((0, 255, 0), 2, right_boundary),
    #         ((0, 0, 255), 2, [waypoint.line for waypoint in waypoints])
    #     ],
    #     cars=[car]
    # ) / 256)
    #
    # cv2.waitKey(0)
    # cv2.imshow("", render_area(
    #     camera_pos=car.pos,
    #     rotation=-car.heading - math.pi/2,
    #     area=(20, 20),
    #     resolution=20,
    #     lines=[
    #         ((255, 0, 0), 2, left_boundary),
    #         ((0, 255, 0), 2, right_boundary),
    #         ((0, 0, 255), 2, [waypoint.line for waypoint in waypoints])
    #     ],
    #     cones=[
    #         ((255, 0, 0), 10, track.blue_cones),
    #         ((0, 255, 0), 10, track.yellow_cones),
    #     ],
    #     cars=[car]
    # ) / 256)
    # cv2.waitKey(0)
