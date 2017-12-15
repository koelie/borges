"""Segment images with trained model and CRF post-processing"""
import logging
import argparse
import sys
import os
import time
import json

import cv2
import numpy as np
from . import models

log = logging.getLogger(__name__)


class Classifier(object):
    def __init__(self, model_desc):
        """Classifies images as vases or bowls
        
        Parameters
        ----------
        """
        self.class_map = model_desc['class_map']
        self.model = models.classification_model(model_desc['image_size'])
        self.model.load_weights(model_desc['weights_fn'])

    def predict(self, image_fn):
        """Segments given image
        
        """
        image = cv2.imread(image_fn, cv2.IMREAD_COLOR)

        # check input and resize if necessary
        h, w, depth = image.shape
        if depth != (self.model.input_shape[3]):
            raise ValueError("Incorrect number of channels")
        model_sz = self.model.input_shape[1:3]
        image = image.astype(np.float32) / 255.

        image = cv2.resize(
            image, model_sz[::-1], interpolation=cv2.INTER_LINEAR
        )

        # predict and remove batch dim
        pred = self.model.predict(image[None])[0,0]
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

    for i, image_fn in enumerate(args.images):
        tic = time.time()
        confs = cls.predict(image_fn)
        duration = time.time() - tic
        
        classes, vals = zip(*confs.items())
        log.info(
            'Predicted %d/%d: %s (%2.3fs), output: %s: %04.1f%%, %s: %04.1f%%', i, len(args.images), 
            os.path.basename(image_fn), duration, 
            classes[0], vals[0]*100,
            classes[1], vals[1]*100,
        )

