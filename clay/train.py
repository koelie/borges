"""Trains model for bowl/vase segmentation"""
import logging
import argparse
import sys
import json

import matplotlib
matplotlib.use('Agg')

from keras.callbacks import ModelCheckpoint, CSVLogger
from keras_adversarial.image_grid_callback import ImageGridCallback
import numpy as np

from . import data, models

log = logging.getLogger(__name__)


def train(data_path, name, class_name, image_size, batch_size=24,
          num_epochs=100, num_workers=3, resume=False):
    """Trains model for bowl/vase generation

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
    resume : int
        if > 0 then resume training at this epoch
    """

    log.info("Create data iterators")
    x = data.make_iterator(data_path, class_name, image_size, batch_size)
    train_iter, field_idx, field_mean, field_std, fields, im_max, crop = x
    num_fields = len(field_idx)

    log.info("Set up model")
    gan, generator, discriminator = models.make_models(num_fields, image_size)

    log.info("Set up callbacks")

    # create callback to generate images
    zsamples = np.random.normal(size=(7*7, num_fields))
    zsamples[:7] = [x*fields[0] + (1-x)*fields[-1]
                    for x in np.linspace(0, 1, 7)]

    def generator_sampler():
        """Predict samples for visualization"""
        xpred = generator.predict(zsamples)
        return (xpred.reshape((7, 7) + xpred.shape[1:])+1) * im_max/2

    matplotlib.rcParams['savefig.dpi'] = 300
    weights_fn = "%s.weights.hdf5" % name
    callbacks = [
        ImageGridCallback(
            "generator_%s_epoch-{:03d}.png" % name,
            generator_sampler, cmap=None
        ),
        CSVLogger('%s.log' % name, append=True),
        ModelCheckpoint(
            weights_fn, monitor='loss',
            save_best_only=True, save_weights_only=True
        ),

    ]

    # save model description
    gen_fn = "%s.generator.hdf5" % name
    disc_fn = "%s.discriminator.hdf5" % name
    with open("%s.json" % name, 'w') as out_file:
        meta = {
            'image_size': image_size,
            'gen_weights': gen_fn,
            'disc_weights': disc_fn,
            'num_fields': num_fields,
            'field_idx': field_idx.tolist(),
            'field_mean': field_mean.tolist(),
            'field_std': field_std.tolist(),
            'im_max': im_max,
            'crop': crop
        }
        json.dump(meta, out_file)

    initial_epoch = 0
    if resume > 0:
        log.info("Resuming training at epoch %d", resume)
        gan.load_weights(weights_fn)
        initial_epoch = int(resume)

    log.info("Starting training")
    gan.fit_generator(
        train_iter,
        steps_per_epoch=int(np.floor(train_iter.num_samples/batch_size)),
        epochs=num_epochs,
        verbose=1,
        callbacks=callbacks,
        initial_epoch=initial_epoch,
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
        'class_name', type=str, choices=['bowl', 'vase', 'all'],
        help='class to train model for',
    )
    parser.add_argument(
        '-s', '--image_size', type=int, default=64, choices=[32,64,128,256],
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
    parser.add_argument(
        '-r', '--resume', type=int, default=0,
        help='attempt to resume earlier training from given epoch',
    )

    args = parser.parse_args()
    loglvl = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        stream=sys.stdout,
        level=loglvl,
        format='[%(asctime)s: %(levelname)s] %(message)s'
    )

    train(args.data_path, args.name, args.class_name, args.image_size,
          args.batch_size, args.epochs, args.num_workers, args.resume)
