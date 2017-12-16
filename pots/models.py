"""Keras model and metrics definition"""
import logging

from keras.models import Model
from keras.layers import (Conv2D, BatchNormalization, Activation,
                          MaxPooling2D, Dropout, Reshape, Dense, Input)
from keras.optimizers import Adam
from keras import losses, metrics

log = logging.getLogger(__name__)


def cnn(image_size):
    """A basic CNN classification model

    Parameters
    ----------
    image_size : tuple
        input image size (h*w)

    Returns
    -------
    model : keras Model
        compiled CNN Model
    """
    mdl_input = Input(tuple(image_size) + (3,), name="input")
    x = mdl_input

    for i, filt in enumerate((32, 64, 128, 256)):
        x = Conv2D(filt, (3, 3), name='conv_{}'.format(i))(x)
        x = BatchNormalization(scale=False, name='bn_{}'.format(i))(x)
        x = Activation('relu', name='act_{}'.format(i))(x)
        x = MaxPooling2D(pool_size=2, strides=2, name='pool_{}'.format(i))(x)
        x = Dropout(0.2, name='drop_{}'.format(i))(x)
    x = Reshape((-1,), name="flatten")(x)

    for i, filt in enumerate((256, 128)):
        x = Dense(filt, activation='relu', name='dense_{}'.format(i))(x)
        x = Dropout(0.2, name='dense_drop_{}'.format(i))(x)

    mdl_output = Dense(1, activation='sigmoid', name='output')(x)

    model = Model(inputs=[mdl_input], outputs=[mdl_output])
    model.compile(
        optimizer=Adam(lr=0.00001),
        loss=losses.binary_crossentropy,
        metrics=[metrics.binary_accuracy],
    )
    return model
