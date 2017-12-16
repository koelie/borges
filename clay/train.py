"""Trains model for bowl/vase segmentation"""
import logging
import argparse
import sys
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import pylab as pl

from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler
from keras import backend as K
from keras.models import load_model
from keras_adversarial.image_grid_callback import ImageGridCallback

from . import data, models

log = logging.getLogger(__name__)


def train(data_path, name, image_size, batch_size=24, num_epochs=100, num_workers=3, samples_per_epoch=1000):
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
    train_iter, field_idx, field_mean, field_std, fields = data.make_iterator(data_path, image_size, batch_size)
    num_fields = len(field_idx)

    log.info("Set up model")
    gan, generator, discriminator = models.make_models(num_fields, image_size)


    log.info("Set up callbacks")

    # create callback to generate images
    zsamples = np.random.normal(size=(7*7, num_fields))
    zsamples[:7] = [x*fields[0] + (1-x)*fields[-1] for x in np.linspace(0, 1, 7)]
    def generator_sampler():
        xpred = generator.predict(zsamples)
        return (xpred.reshape((7, 7) + xpred.shape[1:])+1)/2
    pl.rcParams['savefig.dpi'] = 300

    callbacks = [
        ImageGridCallback("generator_%s_epoch-{:03d}.png" % name, generator_sampler, cmap=None),
        CSVLogger('%s.log' % name, append=True)
    ]

    # save model description
    gen_fn = "%s.generator.hdf5" % name
    disc_fn = "%s.discriminator.hdf5" % name
    with open("%s.json" % name, 'w') as f:
        meta = {
            'image_size': image_size,
            'gen_weights': gen_fn,
            'disc_weights': disc_fn,
            'num_fields': num_fields,
            'field_idx': field_idx.tolist(),
            'field_mean': field_mean.tolist(),
            'field_std': field_std.tolist(),
        }
        json.dump(meta, f)

    log.info("Starting training")
    history = gan.fit_generator(
        train_iter,
        steps_per_epoch=int(np.floor(train_iter.num_samples/batch_size)),
        epochs=num_epochs,
        verbose=1,
        callbacks=callbacks,
        initial_epoch=0,
        workers=num_workers,
        use_multiprocessing=True,
    )

    # save model weights
    generator.save_weights(gen_fn)
    discriminator.save_weights(disc_fn)

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
        '-s', '--image_size', type=int, default=128,
        help='image size used for training/generation',
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

    train(args.data_path, args.name, args.image_size, args.batch_size, args.epochs, args.num_workers)
