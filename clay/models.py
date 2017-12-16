"""Keras model and metrics definition"""
from keras.models import Model, Sequential
from keras.layers import *

from keras.optimizers import Adam
from keras.regularizers import L1L2
from keras_adversarial import AdversarialModel, simple_gan, gan_targets
from keras_adversarial import AdversarialOptimizerSimultaneous, normal_latent_sampling


def generator(fields_size, image_size):
    reg = lambda: L1L2(l1=1e-7, l2=1e-7)
    model = Sequential()
    model.add(Dense(2048, input_dim=fields_size, kernel_regularizer=reg(), name="generator"))
    model.add(BatchNormalization())
    sz = int(image_size / 2**5)
    model.add(Reshape((sz, sz, -1)))

    for i, sz in enumerate((512, 256, 128, 64, 32)):
        model.add(Conv2D(sz, (5, 5), padding='same', kernel_regularizer=reg()))
        model.add(BatchNormalization())
        model.add(LeakyReLU(0.2))
        model.add(UpSampling2D(size=(2, 2)))

    model.add(Conv2D(3, (5, 5), padding='same', kernel_regularizer=reg()))
    model.add(Activation('tanh'))
    return model


def discriminator(image_size):
    reg = lambda: L1L2(l1=1e-7, l2=1e-7)
    model = Sequential()
    in_size = (image_size, image_size, 3)
    model.add(Conv2D(32, (5, 5), padding='same', input_shape=in_size, kernel_regularizer=reg(), name="discriminator"))
    model.add(BatchNormalization())
    
    for i, sz in enumerate((64, 128, 256, 1)):
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(LeakyReLU(0.2))
        model.add(Conv2D(sz, (5, 5), padding='same', kernel_regularizer=reg()))
        model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model


def make_models(fields_size, image_size):
    assert image_size%(2**4) == 0, "image size must be divisible by %d" % 2**4

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
        player_optimizers=[Adam(lr=1e-4, beta_1=0.2, decay=1e-6), Adam(lr=1e-4, beta_1=0.2, decay=1e-5)],
        loss='squared_hinge'
    )
    
    return model, gen, disc

