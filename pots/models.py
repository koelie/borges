"""Keras model and metrics definition"""
from keras.models import Model
from keras.layers import *
from keras.optimizers import Adam
from keras import losses, metrics
from keras import applications

import logging

log = logging.getLogger(__name__)

def classification_model(image_size):
    mdl_input = Input(tuple(image_size) + (3,), name="input")
    x = mdl_input
    
    for i, sz in enumerate((32,64,128,256)):
        x = Conv2D(sz, (3, 3), name='conv_{}'.format(i))(x)
        x = BatchNormalization(scale=False, name='bn_{}'.format(i))(x)
        x = Activation('relu', name='act_{}'.format(i))(x)
        x = MaxPooling2D(pool_size=2, strides=2, name='pool_{}'.format(i))(x)
        x = Dropout(0.2, name='drop_{}'.format(i))(x)
    x = Reshape((-1,), name="flatten")(x)
    
    for i, sz in enumerate((256, 128)):
        x = Dense(sz, activation='relu', name='dense_{}'.format(i))(x)
        x = Dropout(0.2, name='dense_drop_{}'.format(i))(x)

    mdl_output = Dense(1, activation='sigmoid', name='output')(x)

    model = Model(inputs=[mdl_input], outputs=[mdl_output])
    model.compile(
        optimizer=Adam(lr=0.00001),
        loss=losses.binary_crossentropy,
        metrics=[metrics.binary_accuracy],
    )
    return model


def vgg_model(image_size):
    input_tensor = Input(shape=(image_size[0], image_size[1],3))
    vgg = applications.VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)
    for layer in vgg.layers[:15]:
        layer.trainable = False

    x = Flatten(input_shape=vgg.output_shape[1:])(vgg.output)
    x = Dense(512, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    mdl_output = Dense(1, activation='sigmoid', name='output')(x)

    model = Model(inputs=[vgg.input], outputs=[mdl_output])
    model.compile(
        optimizer=Adam(lr=0.00001),
        loss=losses.binary_crossentropy,
        metrics=[metrics.binary_accuracy],
    )
    return model
