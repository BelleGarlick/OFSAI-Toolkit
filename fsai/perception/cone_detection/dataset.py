import os
import random
from typing import List

import cv2
import numpy as np
import tensorflow as tf

from fsai.utils.annotations import auto_detect_annotation_loader


@tf.function
def transform_targets_for_output(y_true, grid_size, anchor_idxs):
    # y_true: (N, boxes, (x1, y1, x2, y2, class, best_anchor))
    N = tf.shape(y_true)[0]

    # y_true_out: (N, grid, grid, anchors, [x, y, w, h, obj, class])
    y_true_out = tf.zeros(
        (N, grid_size, grid_size, tf.shape(anchor_idxs)[0], 6))

    anchor_idxs = tf.cast(anchor_idxs, tf.int32)

    indexes = tf.TensorArray(tf.int32, 1, dynamic_size=True)
    updates = tf.TensorArray(tf.float32, 1, dynamic_size=True)
    idx = 0
    for i in tf.range(N):
        for j in tf.range(tf.shape(y_true)[1]):
            if tf.equal(y_true[i][j][2], 0):
                continue
            anchor_eq = tf.equal(
                anchor_idxs, tf.cast(y_true[i][j][5], tf.int32))

            if tf.reduce_any(anchor_eq):
                box = y_true[i][j][0:4]
                box_xy = (y_true[i][j][0:2] + y_true[i][j][2:4]) / 2

                anchor_idx = tf.cast(tf.where(anchor_eq), tf.int32)
                grid_xy = tf.cast(box_xy // (1/grid_size), tf.int32)

                # grid[y][x][anchor] = (tx, ty, bw, bh, obj, class)
                indexes = indexes.write(
                    idx, [i, grid_xy[1], grid_xy[0], anchor_idx[0][0]])
                updates = updates.write(
                    idx, [box[0], box[1], box[2], box[3], 1, y_true[i][j][4]])
                idx += 1

    return tf.tensor_scatter_nd_update(y_true_out, indexes.stack(), updates.stack())


def transform_targets(y_train, anchors, anchor_masks, size):
    y_outs = []
    grid_size = size // 32

    # calculate anchor index for true boxes
    anchors = tf.cast(anchors, tf.float32)
    anchor_area = anchors[..., 0] * anchors[..., 1]
    box_wh = y_train[..., 2:4] - y_train[..., 0:2]
    box_wh = tf.tile(tf.expand_dims(box_wh, -2),
                     (1, 1, tf.shape(anchors)[0], 1))
    box_area = box_wh[..., 0] * box_wh[..., 1]
    intersection = tf.minimum(box_wh[..., 0], anchors[..., 0]) * \
        tf.minimum(box_wh[..., 1], anchors[..., 1])
    iou = intersection / (box_area + anchor_area - intersection)
    anchor_idx = tf.cast(tf.argmax(iou, axis=-1), tf.float32)
    anchor_idx = tf.expand_dims(anchor_idx, axis=-1)

    y_train = tf.concat([y_train, anchor_idx], axis=-1)

    for anchor_idxs in anchor_masks:
        y_outs.append(transform_targets_for_output(
            y_train, grid_size, anchor_idxs))
        grid_size *= 2

    return tuple(y_outs)


def load_dataset(
        annotations_path: str,
        images_path: str,
        classes: List[str],
        anchors,
        anchor_masks,
        input_size: int = 416,
        max_predictions: int = 5,
        val_split: float = 0.1,
        batch_size: int = 8
):
    x = []
    y = []

    if os.path.exists(annotations_path):
        for annotation_file in os.listdir(annotations_path):
            if annotation_file[0] == ".": continue
            try:
                annotation_path = os.path.join(annotations_path, annotation_file)

                image_path = ".".join(annotation_file.split(".")[:-1]) + ".jpg"

                image = cv2.imread(os.path.join(images_path, image_path))   # load and normalise
                max_size = max(image.shape)

                letter_boxed_image = letterbox_image(image, input_size)
                x.append(letter_boxed_image / 255)  # add normalised image
                oh, ow, depth = image.shape

                boxes = auto_detect_annotation_loader(annotation_path, classes)
                label = []

                for box in boxes:
                    class_num, cx, cy, w, h = [float(b) for b in box]
                    x1 = (cx - w/2) * ow
                    x2 = (cx + w/2) * ow
                    y1 = (cy - h/2) * oh
                    y2 = (cy + h/2) * oh
                    x1 = (x1 + ((max_size-ow)/2)) / max_size
                    y1 = (y1 + ((max_size-oh)/2)) / max_size
                    x2 = (x2 + ((max_size-ow)/2)) / max_size
                    y2 = (y2 + ((max_size-oh)/2)) / max_size
                    label.append([x1, y1, x2, y2, class_num])

                for i in range(max_predictions - len(label)):
                    label.append([0, 0, 0, 0, 0])

                y.append(label[:max_predictions])
            except Exception as e:
                print("Error loading annotation: {} - skipping to next annotation.".format(annotation_file))
                print(e)

            # with open(annotations_path, "r") as annotations:
            #     for line in annotations.readlines():
            #         line_tokens = line.split()
            #         if len(line_tokens) >= 2:
            #             image_path = line_tokens[0]
            #
            #             image = cv2.imread(images_path + image_path)   # load and normalise
            #             max_size = max(image.shape)
            #
            #             letter_boxed_image = letterbox_image(image, input_size)
            #             x.append(letter_boxed_image / 255)  # add normalised image
            #             oh, ow, depth = image.shape
            #
            #             boxes = [data.split(",") for data in line_tokens[1:]]
    else:
        print("Annotations path not found")

    # shuffle data set
    temp = list(zip(x, y))
    random.shuffle(temp)
    x, y = zip(*temp)

    # split data set
    val_split = int(val_split * len(x))
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    print(x.shape)
    print(y.shape)
    val_x, train_x = x[:val_split], x[val_split:]
    val_y, train_y = y[:val_split], y[val_split:]

    # convert to tensors
    val_x = tf.convert_to_tensor(val_x, tf.float32)
    train_x = tf.convert_to_tensor(train_x, tf.float32)
    val_y = tf.convert_to_tensor(val_y, tf.float32)
    train_y = tf.convert_to_tensor(train_y, tf.float32)

    # create data set
    train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    val_dataset = tf.data.Dataset.from_tensor_slices((val_x, val_y))

    # set batch size
    train_dataset = train_dataset.batch(batch_size)
    val_dataset = val_dataset.batch(batch_size)

    # transform data to yolo format
    train_dataset = train_dataset.map(lambda x, y: (x, transform_targets(y, anchors, anchor_masks, input_size)))
    val_dataset = val_dataset.map(lambda x, y: (x, transform_targets(y, anchors, anchor_masks, input_size)))
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return train_dataset, val_dataset


def letterbox_image(image, size: int):
    """resize image with unchanged aspect ratio using padding"""
    iw, ih, _ = image.shape
    scale = min(size / iw, size / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = cv2.resize(image, (nh, nw))

    if image.shape[0] < size:
        delta = int((size - image.shape[0]) / 2)
        buffer = np.empty((delta, image.shape[1], 3))
        buffer.fill(128)

        image = np.vstack((buffer, image))
        image = np.vstack((image, buffer))

    if image.shape[1] < size:
        delta = int((size - image.shape[1]) / 2)
        buffer = np.empty((image.shape[0], delta, 3))
        buffer.fill(128)

        image = np.hstack((buffer, image))
        image = np.hstack((image, buffer))

    if size != image.shape[0] or size != image.shape[1]:
        image = cv2.resize(image, (size, size))
    return image
