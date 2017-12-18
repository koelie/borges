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

    def generate_images(self, num, out_path, prefix, 
                        fields_files=None, random_params=None):
        """Generate images with the model

        Parameters
        ----------
        num : int
            number of images to generate
        out_path : str
            folder to place generated images in
        fields_files : str
            path to .fields files to use as generator inputs
        random_params : tuple
            mean and std to use for random sampling
        """
        # first determine the generator inputs
        if fields_files is None:
            # random numbers
            if random_params is None:
                random_params = 0, 1
            fields = np.random.normal(
                size=(num, self.num_fields), 
                loc=random_params[0], 
                scale=random_params[1]
            )
        else:
            sources = []
            for fields_file in fields_files:
                with open(fields_file) as in_file:
                    fld = np.array(json.load(in_file))[self.field_idx]
                    fld = (fld - self.field_mean)/self.field_std
                    sources.append(fld)

            if num > 1:
                # do interpolation loop
                src_interp = []
                for ffrom, fto in zip(sources, sources[1:]+sources[:1]):
                    # interpolate between successive fields file
                    for i in range(num):
                        alpha = (i/(num))
                        interp = (1-alpha) * ffrom + alpha * fto
                        src_interp.append(interp)
                sources = src_interp
            fields = np.array(sources)

        fields = fields.astype(np.float32)

        for i in range(fields.shape[0]):
            log.info("Generating %d/%d", i+1, fields.shape[0])
            image = self.model.predict(fields[i][None])[0]
            image = ((image+1)/2) * self.im_max * 255
            out_fn = join(out_path, "%s_%05d.png" % (prefix, i))
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
        '-p', '--prefix', type=str, default='generated',
        help='name prefix for generated images',
    )
    parser.add_argument(
        '-d', '--debug', action='store_true',
        help='enable debug logging',
    )
    parser.add_argument(
        '-n', '--num', type=int, default=1,
        help='number of images to create. If fields files are given, this '
        'instead determines the number of interpolation steps between '
        'successive inputs',
    )
    parser.add_argument(
        '-f', '--fields_files', type=str, nargs='+',
        help='path to .fields files, use expression values for generator',
    )
    parser.add_argument(
        '-r', '--random_params', type=float, nargs=2,
        help='mean and stdev for random generation',
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
    gen.generate_images(args.num, args.out_path, args.prefix,
                        fields_files=args.fields_files,
                        random_params=args.random_params)
