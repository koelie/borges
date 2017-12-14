"""Keras model and metrics definition"""
from keras.models import Model
from keras.layers import *
from keras.optimizers import Adam
from keras import losses, metrics

import logging

log = logging.getLogger(__name__)

def classification_model(image_size):
    mdl_input = Input(tuple(image_size) + (3,), name="input")
    x = mdl_input
    
    for i in range(3):
        x = Conv2D(32, (3, 3), name='conv_{}'.format(i))(x)
        x = BatchNormalization(scale=False, name='bn_{}'.format(i))(x)
        x = Activation('relu', name='act_{}'.format(i))(x)
        x = MaxPooling2D(pool_size=2, strides=2, name='pool_{}'.format(i))(x)
        x = Dropout(0.2, name='drop_{}'.format(i))(x)
    x = Reshape((-1,), name="flatten")(x)
    x = Dense(64, activation='relu', name='dense1')(x)
    x = Dropout(0.2, name='dense_drop'.format(i))(x)

    mdl_output = Dense(1, activation='sigmoid', name='output')(x)

    model = Model(inputs=[mdl_input], outputs=[mdl_output])
    model.compile(
        optimizer=Adam(lr=0.00001),
        loss=losses.binary_crossentropy,
        metrics=[metrics.binary_accuracy],
    )
    return model

