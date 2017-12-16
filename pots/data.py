"""Prepare images for training"""
import argparse
import sys
import logging
import os
from os.path import join, isdir

import cv2
import numpy as np
import pylab as pl
from keras.preprocessing.image import Iterator
from sklearn.model_selection import train_test_split
from scipy.stats import truncnorm

log = logging.getLogger(__name__)


def make_iterators(data_path, image_size, batch_size,
                   test_size=0.25, transform=True):
    """Create training and validation image iterators

    Collects all samples, splits them into training and validation sets,
    and creates an iterator for each.

    Assumes file structure under data_path as follows:

    data_path/<classname>/png/<samplename>.png

    where classname and samplename can be anything.
    This trains a binary classifier, so currently only supports two classes.

    Parameters
    ----------
    data_path : str
        path to dataset (structured as assessment dataset)
    image_size : tuple
        image size to resize to for training
    batch_size : int
        number of images in each training batch
    test_size : float
        proportion of the data to set aside for testing
    transform : bool
        whether to randomly scale and rotate the images

    Returns
    -------
    train_iter : ImageIterator
        trainingset iterator
    val_iter : ImageIterator
        validationset iterator
    class_map : dict
        maps 0/1 model outputs to class names
    """
    # find the class names
    clses = [fn for fn in os.listdir(data_path) if isdir(join(data_path, fn))]
    assert len(clses) == 2, "Only supports 2 classes, got %d" % len(clses)

    # find all the files
    filenames = []
    labels = []
    for i, class_name in enumerate(clses):
        img_path = join(data_path, class_name, 'png')
        fns = [join(img_path, fn) for fn in os.listdir(img_path)
               if fn.endswith('.png') or fn.endswith('.jpg')]
        filenames.extend(fns)
        labels.extend([i]*len(fns))

    # split data and make iterators
    train_fns, val_fns, train_labels, val_labels = train_test_split(
        filenames, labels, test_size=test_size, stratify=labels,
    )
    train_iter = ImageIterator(
        train_fns, train_labels, image_size, batch_size, transform=transform
    )
    val_iter = ImageIterator(
        val_fns, val_labels, image_size, batch_size, transform=transform
    )

    # store the class names for during prediction
    class_map = {i: name for i, name in enumerate(clses)}
    return train_iter, val_iter, class_map


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


class ImageIterator(Iterator):
    """Iterator that generates batches of images and labels for training"""

    def __init__(self, filenames, labels, image_size=(256, 256),
                 batch_size=32, shuffle=True, seed=None, transform=True):
        """Initialize image iterator

        Parameters
        ----------
        filenames : list
            paths to image filenames
        labels : list
            list of class labels for each image
        image_size : tuple
            resize images to this size (width, height)
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
        self.labels = labels
        self.image_size = tuple(image_size)
        self.num_samples = len(self.filenames)
        self.transform = transform
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
        # prepare arrays to hold batch data
        imgs = np.zeros(
            (len(index_array),) + self.image_size + (3,), np.float32
        )
        labels = np.zeros((len(index_array), 1), np.float32)

        # build batch of image/label data
        for i, j in enumerate(index_array):
            imgs[i] = self.prepare_image(self.filenames[j])
            labels[i] = self.labels[j]
        return imgs, labels

    def prepare_image(self, filename):
        """Prepare image for training

        Resize image and randomly scale and rotate

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
        image = image.astype(np.float32) / 255.

        w, h = self.image_size
        scale = h / np.min(image.shape[:2])

        if self.transform:
            # sample transform parameters
            angle = sample_trunc_normal(0, 15, -45, 45)
            scale *= sample_trunc_normal(1, 0.1, .8, 1.2)
            x = sample_trunc_normal(0.5, 0.15, 0.3, 0.7)
            y = sample_trunc_normal(0.5, 0.15, 0.3, 0.7)
        else:
            # default no-op params
            angle = 0
            y = .5
            x = .5

        y, x = h*y, w*x

        log.debug(
            "Sampled vars: angle %2.2f, scale %2.2f, center %2.3f, %2.3f",
            angle, scale, x, y
        )

        # determine rotation matrices
        rot = cv2.getRotationMatrix2D((x, y), angle, scale)
        # apply shift to rotation center
        rot[0, 2] += (w/2)*scale - x
        rot[1, 2] += (h/2)*scale - y
        # warp image
        image = cv2.warpAffine(
            image, rot, self.image_size, None,
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
        '-s', '--image_size', type=int, nargs=2, default=(256, 256),
        help='training image size (w*h)',
    )
    parser.add_argument(
        '-t', '--transform', action='store_true',
        help='randomly rotate and scale input',
    )
    parser.add_argument(
        'data_path', type=str,
        help='data directory from the assessment, with bowl and vase images',
    )

    args = parser.parse_args()
    loglvl = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        stream=sys.stdout,
        level=loglvl,
        format='[%(asctime)s: %(levelname)s] %(message)s'
    )

    # display some data samples
    log.info("Make iterators")
    train_iter, _, class_map = make_iterators(
        args.data_path, args.image_size, 8,
        test_size=2, transform=args.transform
    )
    for imgs, labels in train_iter:
        pl.figure()
        for i in range(8):
            pl.subplot(2, 4, i+1)
            pl.imshow(cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB))
            pl.title("class: %s" % class_map[int(labels[i])])
            pl.axis('off')
        pl.tight_layout()
        pl.show()
        pl.close('all')
