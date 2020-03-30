import json
import os
import random
from typing import List

import cv2
from flask import Blueprint, render_template, abort, send_file, request

from fsai.utils.annotations import load_darknet_annotations, export_as_darknet


def image_annotator_blueprint(
        images_path: str,
        images_labels_path: str,
        output_image_path,
        output_labels_path,
        classes: List[str],
        colors: List[str],
        file_formats: List[str] = None,
        annotation_loader=load_darknet_annotations,
        export_function=export_as_darknet
):
    annotator_blueprint = Blueprint('fsai_image_annotator', __name__)

    if images_path[-1] != "/": images_path += "/"
    if images_labels_path[-1] != "/": images_labels_path += "/"
    if output_image_path[-1] != "/": output_image_path += "/"
    if output_labels_path[-1] != "/": output_labels_path += "/"

    if file_formats is None: label_types = ["xml", "txt"]

    @annotator_blueprint.route('/')
    def main_page():
        return render_template("image_annotator.html", class_colours=colors, labels=zip(classes, colors, [i for i in range(len(classes))]), numClasses=len(classes), cursor_line_colour="#dd00ff", cursor_line_width=4)

    @annotator_blueprint.route('/new/')
    def get_unnanoted_image():
        image_data = {
            "image_found": False,
            "reason": "Unknown error please try again later",
            "name": "",
            "width": 0,
            "height": 0,
            "annotations": []
        }

        possible_images = [x for x in os.listdir(images_path) if x[0] is not "."]

        if len(possible_images) > 0:
            image_name = random.choice(possible_images)
            image_file_name = "".join(image_name.split(".")[:-1])
            label_path = None
            for label_type in label_types:
                file_type_path = images_labels_path + image_file_name + "." + label_type
                if os.path.exists(file_type_path):
                    label_path = file_type_path

            if label_path is not None:
                image_data["image_found"] = True
                image_data["name"] = image_name

                img = cv2.imread(images_path + image_name)
                image_data["height"], image_data["width"] = img.shape[0:2]
                image_data["annotations"] = annotation_loader(label_path, labels)
            else:
                image_data["reason"] = "Couldn't find annotation for image {} in path {}".format(image_file_name, images_labels_path)

        else:
            image_data["reason"] = "No images found in: {}".format(images_path)

        return json.dumps(image_data)

    @annotator_blueprint.route('/img/<name>/')
    def get_image(name):
        if "/" not in name and "\\" not in name:
            if os.path.exists(images_path + name):
                return send_file(images_path + name)
        return abort(401)

    @annotator_blueprint.route('/save/', methods=["POST"])
    def save_image():
        raw_annotation_string = request.form["image_data"]
        annotation_data = json.loads(raw_annotation_string)

        success = {
            "success": True,
            "reason": "An unknown error occurred. Please try later."
        }

        if os.path.exists(output_image_path):
            if os.path.exists(output_labels_path):
                pass
                if os.path.exists(images_path + annotation_data["name"]):
                    label_path = images_labels_path + "".join(annotation_data["name"].split(".")[:-1]) + ".txt"
                    os.rename(images_path + annotation_data["name"], output_image_path + annotation_data["name"])
                    if os.path.exists(label_path):
                        os.remove(label_path)

                    export_function(
                        output_labels_path + "".join(annotation_data["name"].split(".")[:-1]) + ".txt",
                        annotation_data["annotations"]
                    )

            else:
                success["success"] = False
                success["reason"] = "Output labels path 'output_labels_path' not found in blueprint configuration."
        else:
            success["success"] = False
            success["reason"] = "Output images path 'output_image_path' not found in blueprint configuration."

        return success

    return annotator_blueprint
