import xml.etree.ElementTree as ET
from typing import Tuple, List


def auto_detect_annotation_loader(path, image_path, classes: List[str]) -> List[Tuple[int, float, float, float, float]]:
    with open(path) as file:
        if "<annotation>" in file.read():
            return load_voc_xml_annotations(path, image_path, classes)
        else:
            return load_darknet_annotations(path, image_path, classes)


def load_darknet_annotations(path, image_path, classes: List[str]) -> List[Tuple[int, float, float, float, float]]:
    annotations = []
    with open(path, "r") as file:
        for annotation in file.readlines():
            class_num, x, y, w, h = annotation.split()

            class_num = int(float(class_num))
            x = float(x)
            y = float(y)
            w = float(w)
            h = float(h)

            annotations.append((class_num, x, y, w, h))
    return annotations


def load_voc_xml_annotations(path, image_path, classes: List[str]) -> List[Tuple[int, float, float, float, float]]:
    annotations = []

    annotation_tree = ET.parse(path)
    root_tag = annotation_tree.getroot().tag
    if root_tag != "annotation":
        raise Exception('Invalid XML Root Tag', "Expected 'annotation' found {}".format(root_tag))

    image_size = annotation_tree.findall('size')[0]
    image_width = int(image_size.findall('width')[0].text)
    image_height = int(image_size.findall('height')[0].text)

    annotations_elements = annotation_tree.findall('object')
    for annotation_element in annotations_elements:
        name = None
        x_min, x_max, y_min, y_max = None, None, None, None

        for child in annotation_element:
            if child.tag == "name": name = child.text
            if child.tag == "bndbox":
                for box_ordinate in child:
                    if box_ordinate.tag == "xmin": x_min = float(box_ordinate.text)
                    if box_ordinate.tag == "xmax": x_max = float(box_ordinate.text)
                    if box_ordinate.tag == "ymin": y_min = float(box_ordinate.text)
                    if box_ordinate.tag == "ymax": y_max = float(box_ordinate.text)

        if name in classes:
            class_num = classes.index(name)
            x = (x_min + x_max) / (2 * image_width)
            y = (y_min + y_max) / (2 * image_height)
            width = abs(x_min - x_max) / image_width
            height = abs(y_min - y_max) / image_width

            annotations.append((class_num, x, y, width, height))
    return annotations


def export_as_darknet(output_annotations_path: str, image_width: int, image_height: int, annotations: List[Tuple[int, float, float, float, float]]):
    with open(path, "w+") as label_file:
        lines = []
        for box in annotations:
            lines.append(" ".join([str(s) for s in box]))
        label_file.write("\n".join(lines))
