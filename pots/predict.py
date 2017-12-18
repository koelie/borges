"""Classify images as vases or bowls"""
import logging
import argparse
import sys
import os
import time
import json
from collections import Counter, defaultdict

import cv2
import numpy as np
from . import models

log = logging.getLogger(__name__)


class Classifier(object):
    """Vase / Bowl classifier"""

    def __init__(self, model_description):
        """Classifies images as vases or bowls

        Parameters
        ----------
        model_description : dict
            A dict with keys:
            - class_map : dict mapping 0/1 to class names
            - image_size : size the classifier was trained on
            - weights_fn : path to network weights file
        """
        self.class_map = model_description['class_map']
        self.model = models.cnn(model_description['image_size'])
        self.model.load_weights(model_description['weights_fn'])

    def predict(self, image_fn):
        """Predicts whether input image is a bowl or a vase

        Parameters
        ----------
        image_fn : str
            path to input image

        Returns
        -------
        class_confs : dict
            dict with confidence for each class
        """
        image = cv2.imread(image_fn, cv2.IMREAD_COLOR)

        # check input and resize if necessary
        if image.shape[2] != (self.model.input_shape[3]):
            raise ValueError("Incorrect number of channels")
        model_sz = self.model.input_shape[1:3]
        image = image.astype(np.float32) / 255.

        image = cv2.resize(
            image, model_sz, interpolation=cv2.INTER_LINEAR
        )

        # predict and remove batch dim
        pred = self.model.predict(image[None])[0, 0]
        class_confs = {
            self.class_map["0"]: 1 - pred,
            self.class_map["1"]: pred
        }
        return class_confs


if __name__ == '__main__':
    # pylint: disable=invalid-name
    parser = argparse.ArgumentParser(
        description='Predict image')
    parser.add_argument(
        'model_json', type=str,
        help='path to model json file to use',
    )
    parser.add_argument(
        'images', type=str, nargs='+',
        help='one or more images to segment',
    )
    parser.add_argument(
        '-d', '--debug', action='store_true',
        help='enable debug logging',
    )

    args = parser.parse_args()
    loglvl = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        stream=sys.stdout,
        level=loglvl,
        format='[%(asctime)s: %(levelname)s] %(message)s'
    )

    log.info('Loading model')
    with open(args.model_json) as f:
        model_desc = json.load(f)

    cls = Classifier(model_desc)
    class_count = Counter()
    class_max = {}
    
    for i, image_fn in enumerate(args.images):
        tic = time.time()
        confs = cls.predict(image_fn)
        duration = time.time() - tic

        classes, vals = zip(*confs.items())
        log.info(
            'Predicted %d/%d: %s (%2.3fs), %s: %04.1f%%, %s: %04.1f%%',
            i, len(args.images),
            os.path.basename(image_fn), duration,
            classes[0], vals[0]*100,
            classes[1], vals[1]*100,
        )
        
        pred = 0 if vals[0] > vals[1] else 1
        pred_class = classes[pred]
        pred_conf = vals[pred]
        class_count[pred_class] += 1
        if pred_class not in class_max or pred_conf > class_max[pred_class][0]:
            class_max[pred_class] = pred_conf, image_fn

    for cls in classes:
        log.info("Total predictions for %s: %d", cls, class_count[cls])
        if cls in class_max:
            log.info("Max conf for %s: %s (%f)", cls, class_max[cls][1], class_max[cls][0])
