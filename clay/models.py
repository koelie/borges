"""Keras model and metrics definition"""
from keras.models import Sequential
from keras.layers import (
    Conv2D, BatchNormalization, Activation, LeakyReLU,
    AveragePooling2D, Reshape, Dense,
    GaussianNoise, Flatten, Conv2DTranspose,
)
from keras.optimizers import Adam
from keras.regularizers import L1L2
from keras_adversarial import (
    AdversarialModel, simple_gan,
    AdversarialOptimizerSimultaneous,
)


def generator(fields_size, image_size):
    """Generator model for GAN

    Parameters
    ----------
    fields_size : int
        number of input parameters
    image_size : int
        output image size

    Returns
    -------
    model : keras model
        generator model
    """
    reg = lambda: L1L2(l1=1e-7, l2=1e-7)
    model = Sequential()
    model.add(Dense(512*4*4, input_dim=fields_size,
                    kernel_regularizer=reg(), name="generator"))
    model.add(BatchNormalization())
    model.add(Reshape((4, 4, 512)))

    if image_size == 32:
        filters = (512, 256)
    elif image_size == 64:
        filters = (512, 256, 128)
    elif image_size == 128:
        filters = (512, 256, 128, 64)
    elif image_size == 256:
        filters = (512, 256, 128, 64, 32)

    for filt in filters:
        model.add(Conv2DTranspose(filt, (5, 5), strides=2, padding='same',
                                  kernel_regularizer=reg()))
        model.add(BatchNormalization())
        model.add(LeakyReLU(0.2))

    model.add(Conv2DTranspose(3, (5, 5), strides=2, padding='same',
                              kernel_regularizer=reg()))
    model.add(Activation('tanh'))
    return model


def discriminator(image_size):
    """Discriminator model for GAN

    Parameters
    ----------
    image_size : int
        output image size

    Returns
    -------
    model : keras model
        discriminator model
    """
    reg = lambda: L1L2(l1=1e-7, l2=1e-7)
    model = Sequential()
    in_size = (image_size, image_size, 3)
    model.add(GaussianNoise(0.0, input_shape=in_size, name="discriminator"))
    model.add(Conv2D(32, (5, 5), padding='same', kernel_regularizer=reg()))
    model.add(BatchNormalization())

    for filt in (64, 128, 256, 1):
        model.add(AveragePooling2D(pool_size=(2, 2)))
        model.add(LeakyReLU(0.2))
        model.add(Conv2D(filt, (5, 5), padding='same',
                         kernel_regularizer=reg()))
        model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model


def make_models(fields_size, image_size):
    """Create GAN model

    Parameters
    ----------
    fields_size : int
        number of generator input parameters
    image_size : int
        training image size

    Returns
    -------
    model : keras model
        GAN model
    gen : keras model
        generator model
    disc : keras model
        discriminator model
    """
    assert image_size % (2**4) == 0, "Size must be divisible by %d" % 2**4

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
        player_optimizers=[Adam(lr=1e-4, beta_1=0.2, decay=1e-5),
                           Adam(lr=1e-4, beta_1=0.2, decay=1e-5)],
        loss='binary_crossentropy'
    )

    return model, gen, disc
