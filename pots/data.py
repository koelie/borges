"""Prepare receipts and ground truth masks for training"""
import os
import argparse
import sys
import logging

import cv2
import numpy as np
import pylab as pl
from os.path import join, isdir
from keras.preprocessing.image import Iterator
from sklearn.model_selection import train_test_split

log = logging.getLogger(__name__)



def make_iterators(data_path, image_size, batch_size, split_seed=42, test_size=0.25):
    """Create training and validation image iterators
    
    Collects all samples, splits them into training and validation sets, and creates an iterator for each.
    
    Assumes file structure under data_path as follows:
    
    data_path/<classname>/png/<samplename>.png

    where classname and samplename can be anything.
    This trains a binary classifier, so currently only supports two classes.
    
    """
    # find the class names
    classes = [fn for fn in os.listdir(data_path) if isdir(join(data_path, fn))]
    assert len(classes) == 2, "Only supports binary classification, found %d classes" % len(classes)
    
    # find all the files
    filenames = []
    labels = []
    for i, class_name in enumerate(classes):
        img_path = join(data_path, class_name, 'png')
        fns = [join(img_path, fn) for fn in os.listdir(img_path) if fn.endswith('.png')]
        filenames.extend(fns)
        labels.extend([i]*len(fns))
    
    train_fns, val_fns, train_labels, val_labels =  train_test_split(
        filenames, labels, test_size=test_size, 
        stratify=labels, random_state=split_seed
    )

    train_iter = ImageIterator(train_fns, train_labels, image_size, batch_size)
    val_iter = ImageIterator(val_fns, val_labels, image_size, batch_size)

    return train_iter, val_iter


class ImageIterator(Iterator):
    def __init__(self, filenames, labels, image_size=(256, 256), 
                 batch_size=32, shuffle=True, seed=None):
        """Initialize receipt iterator
        
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
        self.labels = labels
        self.image_size = tuple(image_size)
        self.num_samples = len(self.filenames)
        log.info('Found %d images.' % self.num_samples)
        super(ImageIterator, self).__init__(self.num_samples, batch_size, shuffle, seed)

    def next(self):
        """Load next batch of images
        
        Returns
        -------
        imgs : ndarray
            images for current batch in single array
        labels : ndarray
            ground truth for current batch in single array
        """
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)

        # The transformation of images is not under thread lock so it can be done in parallel
        imgs = np.zeros((current_batch_size,) + self.image_size + (3,), np.float32)
        labels = np.zeros((current_batch_size, 1), np.float32)

        # build batch of image data
        for i, j in enumerate(index_array):
            image = cv2.imread(self.filenames[j], cv2.IMREAD_COLOR).astype(np.float32) / 255.
            if image.shape[:2] != self.image_size:
                log.debug(
                    "Resizing image from %dx%d to %dx%d", 
                    image.shape[0], image.shape[1], 
                    self.image_size[0], self.image_size[1]
                )
                image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)

            imgs[i] = image
            labels[i] = self.labels[j]
        return imgs, labels



if __name__ == '__main__':
    # pylint: disable=invalid-name
    parser = argparse.ArgumentParser(
        description='Display ImageIterator samples')
    parser.add_argument(
        '-d', '--debug', action='store_true',
        help='enable debug logging',
    )
    parser.add_argument(
        '-s', '--image_size', type=int, nargs=2, default=(256,256),
        help='training image size (w*h)',
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

    log.info("Make iterators")
    train_iter, _ = make_iterators(args.data_path, args.image_size, 8, split_seed=42, test_size=2)

    for imgs, labels in train_iter:
        pl.figure()
        for i in range(8):
            pl.subplot(2, 4, i+1)
            pl.imshow(imgs[i])
            pl.title("class %d" % labels[i])
            pl.axis('off')
        pl.tight_layout()
        pl.show()
        pl.close('all')

