"""Generate new vase or bowl images"""
import logging
import argparse
import sys
import json
from os.path import join

import cv2
import numpy as np
from . import models

log = logging.getLogger(__name__)


class Generator(object):
    """Vase / Bowl generator"""

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
        self.model = models.generator(
            model_description['num_fields'],
            model_description['image_size']
        )
        self.model.load_weights(
            model_description['gen_weights']
        )
        self.num_fields = model_description['num_fields']
        self.field_idx = model_description['field_idx']
        self.field_mean = model_description['field_mean']
        self.field_std = model_description['field_std']
        self.im_max = model_description['im_max']

    def generate_images(self, num, out_path, fields_from=None, fields_to=None):
        """Generate images with the model

        Parameters
        ----------
        num : int
            number of images to generate
        out_path : str
            folder to place generated images in
        fields_from : str
            path to .fields file to use as generator input
        fields_to : str
            path to .fields file to use as 2nd generator input.
            if given, interpolates between fields_from and fields_to.
        """
        # first determine the generator inputs
        if fields_from is None and fields_to is None:
            # random numbers
            fields = np.random.normal(size=(num, self.num_fields))
        elif fields_to is None:
            with open(fields_from) as in_file:
                src_fields = np.array(json.load(in_file))[self.field_idx]
            fields = (src_fields[None] - self.field_mean)/self.field_std
            if num != 1:
                log.warning(
                    "Only one fields file given, generating only 1 image"
                )
        else:
            with open(fields_from) as in_file:
                values = np.array(json.load(in_file))[self.field_idx]
            from_fields = (values[None] - self.field_mean) / self.field_std

            with open(fields_to) as in_file:
                values = np.array(json.load(in_file))[self.field_idx]
            to_fields = (values[None] - self.field_mean) / self.field_std

            fields = np.zeros((num, self.num_fields), np.float32)
            for i in range(num):
                alpha = (i/(num-1))
                print(alpha)
                fields[i, :] = (1-alpha) * from_fields + alpha * to_fields

        fields = fields.astype(np.float32)
        print(fields)

        for i in range(fields.shape[0]):
            log.info("Generating %d/%d", i+1, fields.shape[0])
            image = self.model.predict(fields[i][None])[0]
            image = ((image+1)/2) * self.im_max * 255
            out_fn = join(out_path, "generated_%d.png" % i)
            cv2.imwrite(out_fn, image)


if __name__ == '__main__':
    # pylint: disable=invalid-name
    parser = argparse.ArgumentParser(
        description='Predict image')
    parser.add_argument(
        'model_json', type=str,
        help='path to model json file to use',
    )
    parser.add_argument(
        'out_path', type=str,
        help='folder to place the generated images in',
    )
    parser.add_argument(
        '-d', '--debug', action='store_true',
        help='enable debug logging',
    )
    parser.add_argument(
        '-n', '--num', type=int, default=1,
        help='number of images to create',
    )
    parser.add_argument(
        '-f', '--from_fn', type=str,
        help='path to .fields file, use expression values for generator',
    )
    parser.add_argument(
        '-t', '--to_fn', type=str,
        help='path to 2nd .fields file, interpolate between from and to',
    )

    args = parser.parse_args()
    loglvl = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        stream=sys.stdout,
        level=loglvl,
        format='[%(asctime)s: %(levelname)s] %(message)s'
    )

    log.info('Loading model')
    with open(args.model_json) as in_file:
        model_desc = json.load(in_file)

    gen = Generator(model_desc)
    gen.generate_images(args.num, args.out_path,
                        fields_from=args.from_fn, fields_to=args.to_fn)
