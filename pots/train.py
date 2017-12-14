"""Trains model for bowl/vase segmentation"""
import logging
import argparse
import sys
import numpy as np
import json

from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler
from keras import backend as K
from keras.models import load_model

log = logging.getLogger(__name__)


def train(data_path, name, batch_size=24, num_epochs=100, num_workers=3, samples_per_epoch=1000):
    """Trains model for bowl/vase segmentation
    
    Will produce .hdf5 model file in current directory for final model.
    
    Parameters
    ----------
    data_path : str
        data directory with folder layout as described in the assessment.
    name : str
        name to save model under
    """

    log.info("Create data iterators")
    train_iter, val_iter = data.make_iterators(data_path)

    log.info("Set up model")
    model = models.classification_model()
    log.debug(model.summary())
    weights_fn = "%s.hdf5" % name

    log.info("Set up learning rate scheduler and callbacks")
    def scheduler(epoch):
        lr = K.get_value(model.optimizer.lr)
        if epoch%2==0 and epoch > 5:
            lr *= .97
        return float(lr)
    callbacks = [
        ModelCheckpoint(
            weights_fn, monitor='val_loss', 
            save_best_only=True, save_weights_only=True
        ),
        LearningRateScheduler(scheduler),
        CSVLogger('training.%s.log' % name, append=True)
    ]

    log.info("Starting training")
    history = model.fit_generator(
        train_iter,
        steps_per_epoch=int(np.floor(train_iter.num_samples/batch_size)),
        epochs=num_epochs,
        verbose=1,
        validation_data=val_iter,
        validation_steps=int(np.floor(validation_iter.num_samples/batch_size)),
        callbacks=callbacks,
        initial_epoch=0,
        workers=num_workers,
        pickle_safe=True,
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
        help='name to save the model in (.hdf5 will be appended)',
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
    loglvl = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        stream=sys.stdout,
        level=loglvl,
        format='[%(asctime)s: %(levelname)s] %(message)s'
    )

    train(args.data_path, args.name, args.batch_size, args.epochs, args.num_workers)
