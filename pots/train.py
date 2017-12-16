"""Trains model for bowl/vase segmentation"""
import logging
import argparse
import sys
import json

import numpy as np
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler
from keras import backend as K

from . import data, models

log = logging.getLogger(__name__)


def train(data_path, name, image_size, batch_size=24,
          num_epochs=100, num_workers=3):
    """Trains model for bowl/vase segmentation

    Will produce .hdf5 model file in current directory for final model.

    Parameters
    ----------
    data_path : str
        data directory with folder layout as described in the assessment.
    name : str
        name to save model under
    image_size : tuple
        image size to resize to for training
    batch_size : int
        number of images in each training batch
    num_epochs : int
        number of epochs to train for
    num_workers : int
        number of cpu processes to use for batch preparation
    """

    log.info("Create data iterators")
    train_iter, val_iter, class_map = data.make_iterators(
        data_path, image_size, batch_size, test_size=0.25
    )

    log.info("Set up model")
    model = models.cnn(image_size)
    log.debug(model.summary())

    log.info("Set up learning rate scheduler and callbacks")

    def scheduler(epoch):
        """Updates learning at the end of each epoch"""
        learning_rate = K.get_value(model.optimizer.lr)
        if epoch % 2 == 0 and epoch > 5:
            learning_rate *= .97
        return float(learning_rate)

    weights_fn = "%s.hdf5" % name
    callbacks = [
        ModelCheckpoint(
            weights_fn, monitor='val_loss',
            save_best_only=True, save_weights_only=True
        ),
        LearningRateScheduler(scheduler),
        CSVLogger('%s.log' % name, append=True)
    ]

    log.debug("Save model description")
    with open("%s.json" % name, 'w') as out_file:
        meta = {
            'image_size': image_size,
            'weights_fn': weights_fn,
            'class_map': class_map,
        }
        json.dump(meta, out_file)

    log.info("Starting training")
    model.fit_generator(
        train_iter,
        steps_per_epoch=int(np.floor(train_iter.num_samples/batch_size)),
        epochs=num_epochs,
        verbose=1,
        callbacks=callbacks,
        validation_data=val_iter,
        validation_steps=int(np.floor(val_iter.num_samples/batch_size)),
        initial_epoch=0,
        workers=num_workers,
        use_multiprocessing=True,
    )
    log.info("Training complete")


if __name__ == '__main__':
    # pylint: disable=invalid-name
    parser = argparse.ArgumentParser(
        description='Train bowl/vase classifier')
    parser.add_argument(
        'data_path', type=str,
        help='data directory from the assessment, with bowl and vase images',
    )
    parser.add_argument(
        'name', type=str,
        help='name to save the model in (will append .json/.hdf5)',
    )
    parser.add_argument(
        '-s', '--image_size', type=int, nargs=2, default=(128, 128),
        help='image size used for training (h*w)',
    )
    parser.add_argument(
        '-d', '--debug', action='store_true',
        help='enable debug logging',
    )
    parser.add_argument(
        '-b', '--batch_size', type=int, default=32,
        help='batch size for training',
    )
    parser.add_argument(
        '-w', '--num_workers', type=int, default=3,
        help='number of CPU workers to use for preparing batches',
    )
    parser.add_argument(
        '-e', '--epochs', type=int, default=100,
        help='number of epochs to train for',
    )

    args = parser.parse_args()
    loglvl = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        stream=sys.stdout,
        level=loglvl,
        format='[%(asctime)s: %(levelname)s] %(message)s'
    )

    train(args.data_path, args.name, args.image_size, args.batch_size,
          args.epochs, args.num_workers)
