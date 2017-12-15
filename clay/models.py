"""Keras model and metrics definition"""
from keras.models import Model, Sequential
from keras.layers import *

from keras.optimizers import Adam
from keras.regularizers import L1L2
from keras_adversarial import AdversarialModel, simple_gan, gan_targets
from keras_adversarial import AdversarialOptimizerSimultaneous, normal_latent_sampling


def reg():
    return L1L2(l1=1e-7, l2=1e-7)


def generator(fields_size, image_size):
    model = Sequential()
    model.add(Dense(4096, input_dim=fields_size, kernel_regularizer=reg(), name="generator"))
    model.add(BatchNormalization())
    sz = int(image_size / 2**4)
    model.add(Reshape((sz, sz, 256)))

    for i, sz in enumerate((128, 128, 64, 32)):
        model.add(Conv2D(sz, (5, 5), padding='same', kernel_regularizer=reg()))
        model.add(BatchNormalization(axis=1))
        model.add(LeakyReLU(0.2))
        model.add(UpSampling2D(size=(2, 2)))

    model.add(Conv2D(3, (5, 5), padding='same', kernel_regularizer=reg()))
    model.add(Activation('sigmoid'))
    return model


def discriminator(image_size):
    model = Sequential()
    in_size = (image_size, image_size, 3)
    model.add(Conv2D(32, (5, 5), padding='same', input_shape=in_size, kernel_regularizer=reg(), name="discriminator"))

    for i, sz in enumerate((64, 128, 256, 1)):
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(LeakyReLU(0.2))
        model.add(Conv2D(sz, (5, 5), padding='same', kernel_regularizer=reg()))

    model.add(AveragePooling2D(pool_size=(4, 4), padding='valid'))
    model.add(Flatten())
    model.add(Activation('sigmoid'))
    return model


def make_models(fields_size, image_size):
    depth = 4
    assert image_size%(2**depth) == 0, "image size must be divisible by %d" % 2**depth

    gen = generator(fields_size, image_size)
    gen.summary()

    disc = discriminator(image_size)
    disc.summary()

    gan = simple_gan(gen, disc, None)

    model = AdversarialModel(
        base_model=gan,
        player_params=[gen.trainable_weights, disc.trainable_weights],
        player_names=["generator", "discriminator"]
    )
    model.adversarial_compile(
        adversarial_optimizer=AdversarialOptimizerSimultaneous(),
        player_optimizers=[Adam(1e-4, decay=1e-5), Adam(1e-3, decay=1e-5)],
        loss='binary_crossentropy'
    )
    
    return model, gen, disc

