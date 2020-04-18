import math
import time

from fsai.objects.track import Track
from fsai.path_planning.waypoints import gen_waypoints


time_a = 0
time_b = 0
time_c = 0


if __name__ == "__main__":
    track = Track("examples/data/tracks/azure_circuit.json")
    car = track.cars[0]
    left_boundary, right_boundary, o = track.get_boundary()
    epochs = 1000
    start = time.time()

    for i in range(epochs):
        waypoints = gen_waypoints(
            car_pos=car.pos,
            car_angle=car.heading,
            blue_boundary=left_boundary,
            yellow_boundary=right_boundary,
            orange_boundary=o,
            foresight=8,
            spacing=2,
            negative_foresight=4,
            radar_length=12,
            radar_count=5,
            radar_span=math.pi / 1.2,
            margin=car.width,
            smooth=True
        )
    end = time.time()
    total_time = end - start
    average_time = total_time / epochs
    print("Total: {} Average: {} FPS: {}".format(total_time, average_time, 1/average_time))


# Numpy: Total: 260.0598840713501 Average: 0.2600598840713501 FPS: 3.8452681910972464
# Python Total: 104.2193124294281 Average: 0.1042193124294281 FPS: 9.595150617378598
# Cython: Total: 2.3972067832946777 Average: 0.0023972067832946776 FPS: 417.152165165167
