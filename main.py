# from flask import Flask
#
# from fsai.tools.image_annotator import image_annotator_blueprint
# from fsai.utils.annotations import auto_detect_annotation_loader
# import cv2
# from fsai.perception.cone_detection.yolo import YOLO
#
# classes = ["blue_cone", "yellow_cone", "orange_cone", "big_cone", "person", "car"]
# yolo = YOLO(
#     classes=classes,
#     size=416,
#     max_predictions=20,
#     score_threshold=0.03,
#     iou_threshold=0.3,
#     weights="/Users/samgarlick/Developer/GitHub/OS-FS-AI/data/model.h5",
#     tiny=True
# )
#
#
# def custom_image_loader(image_path, classes):
#     image_path = image_path.replace(".xml", ".jpg")
#     image_path = image_path.replace("/pc/", "/pci/")
#     image = cv2.imread(image_path)
#     return [[float(y) for y in x] for x in yolo.detect(image)]
#
#
# app = Flask(__name__)
# app.register_blueprint(image_annotator_blueprint(
#     "/Users/samgarlick/Developer/GitHub/OS-FS-AI/data/pci/",
#     "/Users/samgarlick/Developer/GitHub/OS-FS-AI/data/pc/",
#     "/Users/samgarlick/Developer/GitHub/OS-FS-AI/data/opci/",
#     "/Users/samgarlick/Developer/GitHub/OS-FS-AI/data/opc/",
#     classes,
#     ["#0000ff", "#dddd00", "#ff9900", "#ff3300", "#ff00ff", "#00ee00"],
#     annotation_loader=custom_image_loader
# ))
#
# # app.register_blueprint(image_annotator_blueprint(
# #     "/Users/samgarlick/Developer/GitHub/OS-FS-AI/data/fixed_images/",
# #     "/Users/samgarlick/Developer/GitHub/OS-FS-AI/data/fixed_annotations/",
# #     "/Users/samgarlick/Developer/GitHub/OS-FS-AI/data/images/",
# #     "/Users/samgarlick/Developer/GitHub/OS-FS-AI/data/annotations/",
# #     ["blue_cone", "yellow_cone", "orange_cone", "big_cone", "person", "car"],
# #     ["#0000ff", "#dddd00", "#ff9900", "#ff3300", "#ff00ff", "#00ee00"]
# # ))
#
#
# if __name__ == "__main__":
#     app.run("0.0.0.0", port=5000, debug=True)


# import time
# import pygame as pygame
#
# from fsai.objects.line import Line
# from fsai.objects.point import Point
# from fsai.objects.track import Track
# from fsai.path_planning.waypoints import gen_local_waypoints
# from fsai.visualisation.track_2d import draw_pygame
#
#
# track = Track("examples/data/tracks/azure_circuit.json")
# for cone in track.blue_cones + track.yellow_cones + track.orange_cones + track.big_cones:
#     vector = Line(Point(0, 0), cone.pos)
#     normalised = vector.normalise()
#     length = vector.length()
#     cone.pos = normalised * (length * 1.3)
#
# blue_lines, yellow_lines, orange_lines = track.get_boundary()
#
# screen_size = [1000, 800]
# # pygame.init()
# # screen = pygame.display.set_mode(screen_size)
#
#
# last_update = time.time()
#
# throttle = 0
#
# down_down = False
# left_down = False
# right_down = False
#
#
# running = True
# while running:
#     steer = 0
#
#     now = time.time()
#     dt = now - last_update
#
#     for event in pygame.event.get():
#         running = event.type != pygame.QUIT
#         if event.type == pygame.KEYDOWN:
#             left_down = left_down or event.key == pygame.K_a
#             throttle = throttle or event.key == pygame.K_w
#             right_down = right_down or event.key == pygame.K_d
#             down_down = down_down or event.key == pygame.K_s
#         if event.type == pygame.KEYUP:
#             if event.key == pygame.K_a:
#                 left_down = False
#             if event.key == pygame.K_w:
#                 throttle = 0
#             if event.key == pygame.K_d:
#                 right_down = False
#             if event.key == pygame.K_s:
#                 down_down = False
#     if left_down: steer = -1
#     if right_down: steer = 1
#
#     car = track.cars[0]
#
#     track.cars[0].throttle = throttle
#     track.cars[0].brake = int(down_down)
#     track.cars[0].steer = steer
#     track.cars[0].physics.update(dt)
#
#     waypoints = gen_local_waypoints(
#         car.pos,
#         car.heading,
#         blue_lines,
#         yellow_lines,
#         orange_lines,
#         foresight=20,
#         negative_foresight=0,
#         margin=((car.width / 2) * 1.2),
#         spacing=2,
#         smooth=True
#     )
#
#
#     # draw and show the track
#     draw_pygame(
#         screen,
#         screen_size,
#         cones=[
#             ((255, 255, 0), 5, track.yellow_cones),
#             ((0, 0, 255), 5, track.blue_cones)
#         ],
#         lines=[
#             ((0, 0, 255), 2, blue_lines),
#             ((255, 255, 0), 2, yellow_lines),
#             ((255, 100, 0), 2, orange_lines),
#             ((100, 100, 100), 2, [waypoint.line for waypoint in waypoints])
#         ],
#         text=[
#             ((255, 255, 255), (50, 50), str(track.cars[0].pos))
#         ],
#         cars=track.cars
#     )
#
#     pygame.display.flip()
#     last_update = now
# pygame.quit()


import json
from flask import Flask
from fsai.tools.image_annotator import image_annotator_blueprint

"""
This is an example of how you can customise the image annotator tool to allow you to load a custom format.
In this example a json format stored as such:
{
    "width": <width>,
    "height": <height>,
    "boxes": [
        {"class", "x1", "x2", "y1", "y2"},
        {"class", "x1", "x2", "y1", "y2"}
    ]
}
"""

input_images_path = "../images/"  # input unannotated images
input_annotations_path = "../annotations/"  # dir for annotations pre-human annotation
output_images_path = "../annotated-images/"  # dir for images once they're annotated
output_annotations_path = "../annotated-annotations/"  # dir for human annotated images

classes = ["blue_cone", "yellow_cone", "orange_cone", "big_cone"]
class_colours = ["#0000ff", "#dddd00", "#ff9900", "#ff3300"]
files_extensions = ["json"]


def custom_image_loader(label_path, image_path, class_names):
    """
    This is an example of a custom image loader that could be used to load json into the correct format for this tool.

    :param label_path: Path to the annotation file
    :param image_path: Path to the image file (not used by this function)
    :param class_names: List of class names
    :return: Correctly formatted annotations
    """
    # Customise load function
    annotation_json = json.loads(label_path)
    iw, ih = annotation_json["width"], annotation_json["height"]

    annotations = []
    # loop through all boxes in the json to formatted the data into the correct c, x, y, w, h format
    for box in annotation_json["boxes"]:
        annotations.append((
            class_names.index(box["class"]),
            (box["x1"] + box["x2"] / 2) / iw,  # calc x from x1, x2
            (box["y1"] + box["y2"] / 2) / ih,  # calc y from y1, y2
            (box["x2"] - box["x1"]) / iw,  # calc w from x1, x2
            (box["y2"] - box["y1"]) / ih  # calc h from y1, y2
        ))

    return annotations


def export_as_json(output_label_path, image_width, image_height, annotations):
    """
    This is an example of an output function which converts the ordinates used by this tool and saves it too the format
    outlined above.

    :param output_label_path: Path to save the function
    :param image_width: Width of the image
    :param image_height: Height of the image
    :param annotations: Human annotated labels
    """
    output = {
        "width": image_width,
        "height": image_height,
        "boxes": []
    }

    for box in annotations:
        class_num, x, y, w, h = box

        # translate x, y, w, h -> x1, y1, x2, y2
        output["boxes"].append({
            "class": classes[class_num],
            "x1": (x - w / 2) * image_width,
            "y1": (y - h / 2) * image_height,
            "x2": (x + w / 2) * image_width,
            "y2": (y + h / 2) * image_height
        })

    # save the json
    with open(output_label_path, "w+") as label_file:
        label_file.write(json.dumps(output, indent=4))


# set up the flask server with the custom functions
app = Flask(__name__)
app.register_blueprint(image_annotator_blueprint(
    input_images_path,
    input_annotations_path,
    output_images_path,
    output_annotations_path,
    classes,
    colors=class_colours,
    annotation_loader=custom_image_loader,
    export_function=export_as_json
))


if __name__ == "__main__":
    app.run("0.0.0.0", port=5000, debug=True)
