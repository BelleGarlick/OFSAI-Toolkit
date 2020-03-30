from typing import List

import numpy as np
import tensorflow as tf

import tensorflowjs as tfjs
from tensorflow_core.python.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, TensorBoard

from fsai.perception.cone_detection import dataset
from fsai.perception.cone_detection.dataset import letterbox_image
from fsai.perception.cone_detection.models import default_yolo_tiny_anchors, get_tiny_yolo_v3_model, yolo_loss, \
    get_yolo_v3_model


class YOLO:
    def __init__(
            self,
            classes: List[str],
            size: int = 416,
            max_predictions: int = 20,
            weights: str = None,
            anchors: List[List[int]] = None,
            anchor_masks: List[List[int]] = None,
            score_threshold=0.3,
            iou_threshold=0.4,
            tiny=True
    ):
        self.classes = classes
        self.num_class = len(self.classes)
        self.size = size
        self.max_predictions = max_predictions
        self.tiny = tiny
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold

        self.anchors = anchors
        self.anchor_masks = anchor_masks

        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)

        if anchor_masks is None: self.anchor_masks = [[3, 4, 5], [0, 1, 2]]
        if anchors is None: self.anchors = default_yolo_tiny_anchors
        self.anchors = np.asarray(self.anchors) / size
        self.masks = np.asarray(self.anchor_masks)

        self.model = self.__gen_model()

        self.weights_path = weights
        self.set_weights_path(weights)

    def set_weights_path(self, weights_path):
        self.weights_path = weights_path
        if weights_path is not None:
            self.model.load_weights(weights_path)
        else:
            self.model = self.__gen_model()

    def __gen_model(self, training=False):
        if self.tiny:
            return get_tiny_yolo_v3_model(
                size=self.size,
                max_boxes=self.max_predictions,
                anchors=self.anchors,
                masks=self.anchor_masks,
                iou_threshold=self.iou_threshold,
                score_threshold=self.score_threshold,
                classes=self.num_class,
                training=training
            )
        else:
            return get_yolo_v3_model(
                size=self.size,
                max_boxes=self.max_predictions,
                anchors=self.anchors,
                masks=self.anchor_masks,
                iou_threshold=self.iou_threshold,
                score_threshold=self.score_threshold,
                classes=self.num_class,
                training=training
            )

    def detect(self, image):
        height, width, depth = image.shape
        max_size = max(image.shape)
        image = letterbox_image(image, self.size) / 255
        image = tf.expand_dims(image, 0)
        boxes, scores, classes, nums = self.model(image)
        results = []
        for i in range(nums[0]):
            # re calculate annotation positions
            x1 = (boxes[0][i][0].numpy() * max_size - ((max_size-width)/2)) / width
            y1 = (boxes[0][i][1].numpy() * max_size - ((max_size-height)/2)) / height
            x2 = (boxes[0][i][2].numpy() * max_size - ((max_size-width)/2)) / width
            y2 = (boxes[0][i][3].numpy() * max_size - ((max_size-height)/2)) / height

            # aabb -> darknet
            result = [
                int(classes[0][i].numpy()),
                (x1 + x2) / 2,
                (y1 + y2) / 2,
                (x2 - x1),
                (y2 - y1),
                scores[0][i].numpy()
            ]

            if max(result) > 0:
                results.append(result)
        return results

    def train(
        self,
        annotations_path: str,
        images_path: str,
        output_weights_path: str,
        validation_split: float = 0.1,
        epochs: int = 50,
        batch_size: int = 8,
        learning_rate: float = 1e-3,
        flip_dataset=True
    ):
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("====== Training Model ======")
        print("Loading dataset...")
        train_dataset, val_dataset = dataset.load_dataset(
            annotations_path,
            images_path,
            classes=self.classes,
            anchors=self.anchors,
            anchor_masks=self.anchor_masks,
            input_size=self.size,
            batch_size=batch_size,
            max_predictions=self.max_predictions,
            val_split=validation_split
        )
        print("Dataset loaded....")

        model = self.__gen_model(training=True)
        print("Model built...")
        if self.weights_path is not None:
            model.load_weights(self.weights_path)
            print("Saved model loaded...")

        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
            loss=[yolo_loss(self.anchors[mask], classes=self.num_class) for mask in self.anchor_masks]
        )
        print("Model compiled...")

        callbacks = [
            ReduceLROnPlateau(verbose=1),
            EarlyStopping(patience=3, verbose=1),
            ModelCheckpoint('checkpoints/yolov3_train_{epoch}.tf', verbose=1, save_weights_only=True),
            TensorBoard(log_dir='logs')
        ]
        print("Begin training...")
        history = model.fit(train_dataset,
                            epochs=epochs,
                            callbacks=callbacks,
                            validation_data=val_dataset)
        model.save_weights(output_weights_path)
        print("====== Model trained ======")

        return history

    def export_to_tfjs(self, output_model):
        tfjs.converters.save_keras_model(self.model, output_model)

    def export_to_tflite(self, output_model):
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        tflite_model = converter.convert()
        file = open(output_model, 'wb')
        file.write(tflite_model)
