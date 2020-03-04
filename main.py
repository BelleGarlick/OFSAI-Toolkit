import json

import cv2

from fsai.mapping.boundary_estimation import create_boundary
from fsai.objects.cone import Cone, CONE_COLOR_BLUE, CONE_COLOR_YELLOW, CONE_COLOR_ORANGE, CONE_COLOR_BIG_ORANGE
from fsai.perception.cone_detection.yolo import YOLO
from fsai.visulisation.image_annotations import annotate

class_names = ["yellow", "blue", "orange", "big_orange"]
class_colours = [(0, 255, 255), (255, 0, 0), (0, 100, 255), (0, 0, 255)]

vision = YOLO(class_names, tiny=True, weights="saved_weights.h5", iou_threshold=0.2,
              score_threshold=0.1)
# vision.train(
#     "/Users/samgarlick/Developer/GitHub/OS-FS-AI/annotations.txt",
#     "/Users/samgarlick/Developer/GitHub/OS-FS-AI/",
#     "saved_weights.h5",
#     epochs=20,
#     learning_rate=1e-2,
#     flip_dataset=True
# )

for i in range(400):
    image = cv2.imread("/Users/samgarlick/Developer/GitHub/OS-FS-AI/data/images/{}.jpg".format(i))
    if image is not None:
        image = cv2.resize(image, (image.shape[1]//2, image.shape[0]//2))

        results = vision.detect(image)
        image = annotate(
            image=image,
            label_annotations=results
        )

        cv2.imshow("", image)  # opencv-python-headless
        cv2.waitKey(0)

# with open("/Users/samgarlick/GitHub/ML-Laptime-Optimisation/Track Creator/tracks/daytona_rally.json") as file:
#     track = json.loads(file.read())
#
# blue_cones = [Cone(x=c["x"], y=c["y"], color=CONE_COLOR_BLUE) for c in track["blue_cones"]]
# yellow_cones = [Cone(x=c["x"], y=c["y"], color=CONE_COLOR_YELLOW) for c in track["yellow_cones"]]
# orange_cones = [Cone(x=c["x"], y=c["y"], color=CONE_COLOR_ORANGE) for c in track["orange_cones"]]
# big_orange_cones = [Cone(x=c["x"], y=c["y"], color=CONE_COLOR_BIG_ORANGE) for c in track["big_orange_cones"]]
#
# blue_boundaries, yellow_boundaries, orange_boundaries = create_boundary(
#     blue_cones,
#     yellow_cones,
#     orange_cones,
#     big_orange_cones
# )
#
# image = draw(
#     blue_cones,
#     yellow_cones,
#     orange_cones,
#     big_orange_cones,
#     blue_lines=blue_boundaries,
#     yellow_lines=yellow_boundaries,
#     orange_lines=orange_boundaries,
#     background=0,
#     padding=10
# )
#
# cv2.imshow("", image)
# cv2.waitKey(0)
