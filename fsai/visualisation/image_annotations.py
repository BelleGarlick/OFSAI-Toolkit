import random
from typing import List, Tuple
import cv2
from fsai.utils.annotations import auto_detect_annotation_loader


def draw_annotations(
        labels_path=None,
        label_annotations=None,
        image_path=None,
        image=None,
        colors=None,
        class_names=None,
        line_width=3,
        show_names=True,
        annotation_load_function=auto_detect_annotation_loader
):
    """
    This function is to be used to visualise image annotations within python using OpenCV. This could be used as the
    output of a camera stream or an image saved in a file. This allows you to visually check how well images have been
    labelled.

    :param labels_path: If given labels will load from a file at this given path.
    :param annotation_load_function: This mutable function dictates how the labels should be loaded into darknet
    :param label_annotations: Pre-loaded annotations can be passed here
    :param image_path: The path to an image to load and render labels upon
    :param image: A pre-loaded OpenCV image to render upon
    :param colors: A list of colours for the labels to render as.
    :param class_names: Labels for the classes, passed to the annotation loader.
    :param show_names: If True then the class labels will be render along side the drawn annotations.
    :param line_width: The line width to render the boxes with.
    :return: Returns the labeled image
    """
    # if no colours are provided, set the colours to an empty array. This is not be a default param due to mutability
    if colors is None: colors = []

    # If an image path is provided, load the image from the path
    if image_path is not None:
        image = cv2.imread(image_path)

    # Check there is an image to draw onto
    if image is not None:
        if labels_path is not None:
            annotations = annotation_load_function(labels_path, image_path, class_names)
            for box in annotations:
                class_num, x, y, w, h = box
                image = __draw_rectangle(image, class_num, x, y, w, h, class_names, colors, line_width, show_names)

        if label_annotations is not None:
            for annotation in label_annotations:
                class_num, x, y, w, h = annotation[0:5]
                image = __draw_rectangle(image, class_num, x, y, w, h, class_names, colors, line_width, show_names)

    return image


def __draw_rectangle(image, class_num, x, y, w, h, class_names, colours, line_width, show_names):
    """
    This function is used to render a box around a given darknet format position within an image using OpenCV to show
    the the labelled area.

    :param image: The image to draw upon
    :param class_num: The class index to draw
    :param x: The center x position of the box / image width
    :param y: The center y position of the box / image height
    :param w: The width of the box / image width
    :param h: The height of the box / image height
    :param colours: A list of colours used to draw the boxes in (Optional)
    :param line_width: The line width of the box
    :param show_names: Determines whether class names are visually displayed with the annotation box
    :return: The draw image.
    """
    # calc new bounds
    x1 = int((x - w / 2) * image.shape[1])
    y1 = int((y - h / 2) * image.shape[0])
    x2 = int((x + w / 2) * image.shape[1])
    y2 = int((y + h / 2) * image.shape[0])

    # get the colour at the given index
    colour = __get_colour(colours, class_num)

    # draw the rectangle
    image = cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        colour,
        line_width
    )

    # If class names are provided, render the class names
    if show_names:
        text = class_names[class_num] if class_num < len(class_names) else "Unnamed"
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Render background for text
        image = cv2.rectangle(image, (x1 - ((line_width + 1) // 2), y1 - 30), (x1 + 17 * len(text), y1), colour, -1)
        # render text in the image
        cv2.putText(image, text, (x1, y1 - line_width), font, 1, (255, 255, 255), 2)

    return image


def __get_colour(colours: List[Tuple[int, int, int]], class_num: int):
    """
    Given a list of colours and an index this function will return the colour in the index. In the event that the index
    is greater than the list of colours then a seeded colour is returned.

    :param colours: List of colours.
    :param class_num: Index to get the colour from.
    :return: A colour at an index.
    """
    if class_num >= len(colours):
        # seed the image to get a custom image
        random.seed(class_num)
        color = (random.randint(0, 200), random.randint(0, 200), random.randint(0, 200))

        # reset the seed
        random.seed(None)
        return color
    else:
        return colours[int(class_num)]
