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
