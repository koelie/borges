"""Prepare images for training"""
import os
import argparse
import sys
import logging
import json

import cv2
import numpy as np
import pylab as pl
from os.path import join, isdir
from keras.preprocessing.image import Iterator
from keras_adversarial import gan_targets

log = logging.getLogger(__name__)


def make_iterator(data_path, class_name, image_size, batch_size):
    """Create image iterator
    """
    # find the class names
    classes = [fn for fn in os.listdir(data_path) if isdir(join(data_path, fn))]
    assert class_name in classes, "Given class name %s does not exist in datadir %s" % (class_name, data_path)
    
    # find all the image files
    img_path = join(data_path, class_name, 'png')
    image_fns = [join(img_path, fn) for fn in os.listdir(img_path) if fn.endswith('.png')]

    # find the field files and join them in a big matrix
    fields_path = join(data_path, class_name, 'values')
    fields_fns = [join(fields_path, fn) for fn in os.listdir(fields_path) if fn.endswith('.fields')]
    fields = np.array([json.load(open(f)) for f in fields_fns])
    
    # determine which field values actually change
    varying_cols = np.where([len(np.unique(fields[:,i]))>1 for i in range(fields.shape[1])])[0]
    varying_fields = fields[:,varying_cols]
    log.info("%d fields, of wich %d varying", fields.shape[1], varying_fields.shape[1])

    # normalize fields by subtracting mean and dividing by std
    field_mean = np.mean(varying_fields, axis=0)
    field_std = np.std(varying_fields, axis=0)

    varying_fields -= field_mean
    varying_fields /= field_std

    iterator = ImageIterator(image_fns, varying_fields, image_size, batch_size)

    return iterator, varying_cols, field_mean, field_std


class ImageIterator(Iterator):
    def __init__(self, filenames, fields, image_size=(256, 256), 
                 batch_size=32, shuffle=True, seed=None):
        """Initialize images iterator
        
        Parameters
        ----------
        directory : str
            path to original images (should end in .orig.png, and
            ground truth labelling should be in corresponding .true_bg.png)
        image_size : tuple
            resize image to this size (width, height)
        batch_size : int
            number of images to return per iteration
        shuffle : bool
            whether to shuffle images
        seed : int
            random seed to use
        """
        self.filenames = filenames
        self.fields = fields
        self.image_size = image_size
        self.num_samples = len(self.filenames)
        log.info('Found %d images.' % self.num_samples)
        super(ImageIterator, self).__init__(self.num_samples, batch_size, shuffle, seed)

    def _get_batches_of_transformed_samples(self, index_array):
        """Load next batch of images
        
        Returns
        -------
        imgs : ndarray
            images for current batch in single array
        labels : ndarray
            ground truth for current batch in single array
        """

        # The transformation of images is not under thread lock so it can be done in parallel
        sz = self.image_size
        imgs = np.zeros((len(index_array), sz, sz, 3), np.float32)
        fields = np.zeros((len(index_array), self.fields.shape[1]), np.float32)

        # build batch of image data
        for i, j in enumerate(index_array):
            imgs[i] = self.prepare_image(self.filenames[j])
            fields[i] = self.fields[j]
        inputs = {
            "generator_input": fields,
            "discriminator_input": imgs,
        }
        return inputs, gan_targets(len(index_array))

    def prepare_image(self, filename):
        image = cv2.imread(filename, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        sz = self.image_size
        image = cv2.resize(image, (sz, sz), cv2.INTER_LINEAR)
        return image

    def next(self):
        """Returns the next batch.
        """
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)

if __name__ == '__main__':
    # pylint: disable=invalid-name
    parser = argparse.ArgumentParser(
        description='Display ImageIterator samples')
    parser.add_argument(
        '-d', '--debug', action='store_true',
        help='enable debug logging',
    )
    parser.add_argument(
        '-s', '--image_size', type=int, default=128,
        help='image size',
    )
    parser.add_argument(
        'data_path', type=str,
        help='data directory from the assessment, with bowl and vase images',
    )
    parser.add_argument(
        'class_name', type=str, choices=('vase','bowl'),
        help='which class to learn to generate'
    )

    args = parser.parse_args()
    loglvl = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        stream=sys.stdout,
        level=loglvl,
        format='[%(asctime)s: %(levelname)s] %(message)s'
    )

    log.info("Make iterator")
    train_iter, fidx, fmean, fstd = make_iterator(args.data_path, args.class_name, args.image_size, 8)
    log.info("Number of fields: %d", len(fidx))
    log.info("Field indexes: %s", str(fidx))
    log.info("Field means: %s", str(fmean))
    log.info("Field stds: %s", str(fstd))

    for inputs, _ in train_iter:
        imgs = inputs['discriminator_input']
        fields = inputs['generator_input']
        pl.figure()
        for i in range(8):
            pl.subplot(2, 4, i+1)
            pl.imshow(imgs[i])
            pl.axis('off')
        pl.tight_layout()
        pl.show()
        pl.close('all')

