import json
import os
import random
from typing import List

import cv2
from flask import Blueprint, render_template, abort, send_file, request, session, redirect, url_for

from fsai.utils.annotations import load_darknet_annotations, export_as_darknet


def image_annotator_blueprint(
        images_path: str,
        images_labels_path: str,
        output_image_path,
        output_labels_path,
        classes: List[str],
        colors: List[str],
        file_formats: List[str] = None,
        output_file_format: str = "txt",
        session_key=None,
        redirect_url=None,
        annotation_loader=load_darknet_annotations,
        export_function=export_as_darknet,
        download=None
):
    annotator_blueprint = Blueprint('fsai_image_annotator', __name__)

    if images_path[-1] != "/": images_path += "/"
    if images_labels_path[-1] != "/": images_labels_path += "/"
    if output_image_path[-1] != "/": output_image_path += "/"
    if output_labels_path[-1] != "/": output_labels_path += "/"

    if file_formats is None: file_formats = ["xml", "txt"]

    @annotator_blueprint.route('/')
    def main_page():
        if session_key is None or (session_key in session and session[session_key] == "valid"):
            return render_template(
                "image_annotator.html",
                class_colours=colors,
                labels=zip(classes, colors, [i for i in range(len(classes))]),
                numClasses=len(classes),
                cursor_line_colour="#dd00ff",
                cursor_line_width=4,
                show_download=download is not None,
                download_route=url_for("fsai_image_annotator.download_route"),
                save_route=url_for("fsai_image_annotator.save_image"),
                new_route=url_for("fsai_image_annotator.get_unnanoted_image")
            )
        if redirect_url is not None:
            return redirect(redirect_url)
        return abort(401)

    @annotator_blueprint.route('/new/')
    def get_unnanoted_image():
        if session_key is None or (session_key in session and session[session_key] == "valid"):
            image_data = {
                "image_found": False,
                "reason": "Unknown error please try again later",
                "name": "",
                "width": 0,
                "height": 0,
                "annotations": []
            }

            if os.path.exists(images_path):
                possible_images = [x for x in os.listdir(images_path) if x[0] is not "."]

                if len(possible_images) > 0:
                    image_name = random.choice(possible_images)
                    image_file_name = "".join(image_name.split(".")[:-1])
                    label_path = get_annotation_file_path(images_labels_path, image_file_name, file_formats)

                    image_data["name"] = image_name
                    image_data["url"] = url_for("fsai_image_annotator.get_image", name=image_name)
                    image_data["image_found"] = True
                    img = cv2.imread(images_path + image_name)
                    image_data["height"], image_data["width"] = img.shape[0:2]

                    image_data["annotations"] = annotation_loader(label_path, images_path + image_name, classes)

                else:
                    image_data["reason"] = "No images found in: {}".format(images_path)
            else:
                image_data["reason"] = "Path: {} does not exists.".format(images_path)

            return json.dumps(image_data)
        if redirect_url is not None:
            return redirect(redirect_url)
        return abort(401)

    @annotator_blueprint.route('/img/<name>/')
    def get_image(name):
        if session_key is None or (session_key in session and session[session_key] == "valid"):
            if "/" not in name and "\\" not in name:
                if os.path.exists(images_path + name):
                    return send_file(images_path + name)
                return abort(404)
        if redirect_url is not None:
            return redirect(redirect_url)
        return abort(401)

    @annotator_blueprint.route('/save/', methods=["POST"])
    def save_image():
        if session_key is None or (session_key in session and session[session_key] == "valid"):
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
                        image_name = "".join(annotation_data["name"].split(".")[:-1])
                        previous_annotation_name = get_annotation_file_path(images_labels_path, image_name, file_formats)
                        output_path = output_labels_path + image_name + "." + output_file_format

                        os.rename(images_path + annotation_data["name"], output_image_path + annotation_data["name"])
                        if os.path.exists(previous_annotation_name):
                            os.remove(previous_annotation_name)

                        export_function(
                            output_path,
                            annotation_data["width"],
                            annotation_data["height"],
                            annotation_data["annotations"]
                        )

                else:
                    success["success"] = False
                    success["reason"] = "Output labels path 'output_labels_path' not found in blueprint configuration."
            else:
                success["success"] = False
                success["reason"] = "Output images path 'output_image_path' not found in blueprint configuration."

            return success
        if redirect_url is not None:
            return redirect(redirect_url)
        return abort(401)

    @annotator_blueprint.route("/download/")
    def download_route():
        if session_key is None or (session_key in session and session[session_key] == "valid"):
            if download is not None:
                return download()
            else:
                return "Downloading has not been setup."
        if redirect_url is not None:
            return redirect(redirect_url)
        return abort(401)

    return annotator_blueprint


def get_annotation_file_path(parent_dir_path, image_name, posible_formats):
    for label_type in posible_formats:
        file_type_path = parent_dir_path + image_name + "." + label_type
        if os.path.exists(file_type_path):
            return file_type_path
    return None
