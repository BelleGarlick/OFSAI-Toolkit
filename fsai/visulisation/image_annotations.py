import random

import cv2


# TODO Show labels
def draw(labels_path=None, label_annotations=None, image_path=None, image=None, colours=None, line_width=3):
    if colours is None: colours = []

    if image_path is not None:
        image = cv2.imread(image_path)

    if image is not None:
        if labels_path is not None:
            with open(labels_path) as file:
                annotations = file.readlines()
                for annotation in annotations:
                    class_num, x, y, w, h = [float(x) for x in annotation.split()][0:5]
                    image = __draw_rectangle(image, class_num, x, y, w, h, colours, line_width)

        if label_annotations is not None:
            for annotation in label_annotations:
                class_num, x, y, w, h = annotation[0:5]
                image = __draw_rectangle(image, class_num, x, y, w, h, colours, line_width)

    return image


def __draw_rectangle(image, class_num, x, y, w, h, colours, line_width):
    x1 = (x - w / 2) * image.shape[1]
    y1 = (y - h / 2) * image.shape[0]
    x2 = (x + w / 2) * image.shape[1]
    y2 = (y + h / 2) * image.shape[0]

    colour = __get_colour(colours, class_num)

    image = cv2.rectangle(
        image,
        (int(x1), int(y1)),
        (int(x2), int(y2)),
        colour,
        line_width
    )

    return image


def __get_colour(colours, class_num):
    if class_num >= len(colours):
        random.seed(class_num)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        random.seed(None)
        return color
    else:
        return colours[int(class_num)]
