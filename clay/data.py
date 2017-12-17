"""Prepare images for training"""
import argparse
import sys
import logging
import json
import os
from os.path import join, isdir

import cv2
import numpy as np
import pylab as pl
from keras.preprocessing.image import Iterator
from scipy.stats import truncnorm

log = logging.getLogger(__name__)


def make_iterator(data_path, class_name, image_size, batch_size,
                  random_fields=False, num_random=100):
    """Create image iterators

    Assumes file structure under data_path as follows:

    data_path/<classname>/png/<samplename>.png

    where classname and samplename can be anything.

    Parameters
    ----------
    data_path : str
        path to dataset (structured as assessment dataset)
    image_size : tuple
        image size to resize to for training
    batch_size : int
        number of images in each training batch
    random_fields : bool
        use random distribution instead of values from fields files
        as inputs to the generator model
    num_random : int
        when random_fields is true, this defines the number
        of random numbers to use

    Returns
    -------
    iterator : ImageIterator
        iterator that produces images for training
    vary_cols : nparray
        indices of fields that are not fixed
    field_mean : nparray
        mean of each varying field
    field_std : nparray
        standard deviation of each varying field
    vary_fields : nparray
        values of the varying fields for each dataset element
    im_max : float
        maximum image brightness in training set
    """
    # find the class names
    classes = [fn for fn in os.listdir(data_path)
               if isdir(join(data_path, fn))]

    if class_name != 'all':
        assert class_name in classes, "Class %s not found" % class_name
        classes = [class_name]

    # find all the image files
    image_fns = []
    for class_name in classes:
        # find all the image files
        img_path = join(data_path, class_name, 'png')
        image_fns.extend([join(img_path, fn) for fn in os.listdir(img_path)
                          if fn.endswith('.png')])

    # find amount we can crop (assumes all images are same size)
    h, w = cv2.imread(image_fns[0], cv2.IMREAD_COLOR).shape[:2]
    counts = np.zeros((h, w))
    im_max = 0
    for filename in image_fns:
        image = cv2.imread(filename, cv2.IMREAD_COLOR)
        counts = np.maximum(counts, np.sum(image, axis=-1))
        im_max = np.maximum(im_max, np.max(image))
    im_max /= 255
    log.info("Image max: %f", im_max)

    nnz_h = np.where(np.sum(counts, axis=0) > 0)[0]
    nnz_v = np.where(np.sum(counts, axis=1) > 0)[0]
    crop = [
        int(nnz_v[0]),       # top
        int(nnz_v[-1]) + 1,  # bot
        int(nnz_h[0]),       # left
        int(nnz_h[-1]) + 1,  # right
    ]
    log.info("Crop found: %s", str(crop))

    if random_fields:
        vary_fields = np.random.random((len(image_fns), num_random))
        vary_cols = np.arange(num_random)
        field_mean = np.mean(vary_fields, axis=0)
        field_std = np.std(vary_fields, axis=0)

    else:
        # find the field files and join them in a big matrix
        fields_fns = []
        for class_name in classes:
            fields_path = join(data_path, class_name, 'values')
            fields_fns.extend(
                [join(fields_path, fn) for fn in os.listdir(fields_path)
                 if fn.endswith('.fields')]
            )
        fields = np.array([json.load(open(f)) for f in fields_fns])

        # determine which field values actually change continuously
        # (remove fixed and boolean fields)
        vary_cols = np.where(
            [len(np.unique(fields[:, i])) > 2 for i in range(fields.shape[1])]
        )[0]
        vary_fields = fields[:, vary_cols]
        log.info(
            "%d fields, of wich %d vary",
            fields.shape[1], vary_fields.shape[1]
        )

        # normalize fields by subtracting mean and dividing by std
        field_mean = np.mean(vary_fields, axis=0)
        field_std = np.std(vary_fields, axis=0)

        vary_fields -= field_mean
        vary_fields /= field_std

    iterator = ImageIterator(image_fns, vary_fields,
                             image_size, crop, im_max, batch_size)

    return (
        iterator, vary_cols, field_mean, field_std, vary_fields, im_max, crop
    )


def sample_trunc_normal(mean, std, minimum, maximum):
    """Generates truncated normal samples

    Parameters
    ----------
    mean : float
        mean of the normal distribution
    std : float
        standard deviation of the normal distribution
    minimum : float
        truncate below this value
    maximum : float
        truncate above this value

    Returns
    -------
    value : float
        sample from specified distribution
    """
    x = (minimum - mean) / (std + 1e-10)
    y = (maximum - mean) / (std + 1e-10)
    return truncnorm.rvs(x, y, loc=mean, scale=std)


def gan_targets_smooth(number):
    """Standard training targets for crossentropy loss

    Parameters
    ----------
    number: int
        number of samples

    Returns
    -------
    targets : ndarray
        array of targets
    """
    generator_fake = np.random.uniform(0.8, 1.0, size=((number, 1)))
    generator_real = np.zeros((number, 1))
    discriminator_fake = np.zeros((number, 1))
    discriminator_real = np.random.uniform(0.8, 1.0, size=((number, 1)))
    return [generator_fake, generator_real,
            discriminator_fake, discriminator_real]


class ImageIterator(Iterator):
    """Iterator that generates batches of images and labels for training"""

    def __init__(self, filenames, fields, image_size=(256, 256),
                 crop=None, im_max=1, batch_size=32, shuffle=True,
                 seed=None, transform=False):
        """Initialize image iterator

        Parameters
        ----------
        filenames : list
            paths to image filenames
        fields : list
            expression values used as generator input
        image_size : tuple
            resize images to this size (width, height)
        crop : list
            crop values to apply to images
        im_max : float
            maximum image brightness (for normalization)
        batch_size : int
            number of images to return per iteration
        shuffle : bool
            whether to shuffle images
        seed : int
            random seed to use
        transform : bool
            whether to apply random rotation and scaling to each image
        """
        self.filenames = filenames
        self.fields = fields
        self.crop = crop
        self.im_max = im_max
        self.transform = transform
        self.image_size = image_size
        self.num_samples = len(self.filenames)
        log.info('Found %d images.', self.num_samples)
        super(ImageIterator, self).__init__(
            self.num_samples, batch_size, shuffle, seed
        )

    def _get_batches_of_transformed_samples(self, index_array):
        """Load next batch of images

        Parameters
        ----------
        index_array : list
            dataset indices to include in batch

        Returns
        -------
        imgs : ndarray
            images for current batch in single array
        labels : ndarray
            ground truth for current batch in single array
        """

        size = self.image_size
        imgs = np.zeros((len(index_array), size, size, 3), np.float32)
        fields = np.zeros(
            (len(index_array), self.fields.shape[1]), np.float32
        )

        # build batch of image data
        for i, j in enumerate(index_array):
            imgs[i] = self.prepare_image(self.filenames[j])
            fields[i] = self.fields[j]
        inputs = {
            "generator_input": fields,
            "discriminator_input": imgs,
        }
        return inputs, gan_targets_smooth(len(index_array))

    def prepare_image(self, filename):
        """Prepare image for training

        Crop/resize image and randomly scale and rotate

        Parameters
        ----------
        filename : str
            image to prepare

        Returns
        -------
        image : nparray
            prepared image
        """
        image = cv2.imread(filename, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.
        # crop image
        image = image[self.crop[0]:self.crop[1], self.crop[2]:self.crop[3]]
        # normalize to [-1, 1]
        image = (image / self.im_max) * 2 - 1
        # determine scale and transform params
        scale = self.image_size / np.min(image.shape[:2])
        if self.transform:
            angle = sample_trunc_normal(0, 5, -15, 15)
            scale *= sample_trunc_normal(1, 0.05, .9, 1.1)
            x = sample_trunc_normal(0.5, 0.05, 0.4, 0.6)
            y = sample_trunc_normal(0.5, 0.05, 0.4, 0.6)
        else:
            angle = 0
            y = .5
            x = .5
        y, x = self.image_size * y, self.image_size * x
        log.debug(
            "Sampled vars: angle %2.2f, scale %2.2f, center %2.3f, %2.3f",
            angle, scale, x, y
        )
        # determine rotation matrices
        rot = cv2.getRotationMatrix2D((x, y), angle, scale)
        # apply shift to rotation center
        rot[0, 2] += (self.image_size / 2) * scale - x
        rot[1, 2] += (self.image_size / 2) * scale - y
        # warp image
        image = cv2.warpAffine(
            image, rot, (self.image_size, self.image_size), None,
            cv2.INTER_LINEAR, cv2.BORDER_REFLECT
        )
        return image

    def next(self):
        """Returns the next batch.
        """
        with self.lock:
            index_array = next(self.index_generator)
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
        'class_name', type=str, choices=('vase', 'bowl'),
        help='which class to display'
    )

    args = parser.parse_args()
    loglvl = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        stream=sys.stdout,
        level=loglvl,
        format='[%(asctime)s: %(levelname)s] %(message)s'
    )

    log.info("Make iterator")
    train_iter, fidx, fmean, fstd, _, _, _ = make_iterator(
        args.data_path, args.class_name, args.image_size, 8
    )
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
            pl.imshow((imgs[i]+1)/2, cmap=None)
            pl.axis('off')
        pl.tight_layout()
        pl.show()
        pl.close('all')
