"""
Module containing unet module architecture (recipe).
Inspired by U-Net: Convolutional Networks for Biomedical Image Segmentation
by Olaf Ronneberger, Philipp Fischer, Thomas Brox
https://arxiv.org/abs/1505.04597
"""

from typing import Tuple
from keras.models import Model
from keras.layers import Conv2D, Conv2DTranspose, concatenate, Input
from keras.layers import MaxPooling2D, BatchNormalization, ReLU


def double_conv(inputs, filters: int = 32):
    """
    Create double convolutional layer to reuse later.
        conv => batch_norm => ReLU => conv => batch_norm => ReLU
    """
    conv_settings = {
        "filters": filters,
        "kernel_size": 3,
        "padding": "same",
        "kernel_initializer": "he_normal",
        "use_bias": False
    }

    conv_layer = Conv2D(**conv_settings)(inputs)
    conv_layer = BatchNormalization()(conv_layer)
    conv_layer = ReLU()(conv_layer)
    conv_layer = Conv2D(**conv_settings)(conv_layer)
    conv_layer = BatchNormalization()(conv_layer)
    conv_layer = ReLU()(conv_layer)

    return conv_layer


def encoder_block(inputs, filters: int = 32):
    """
    One block of encoding:
        double_conv => max_pooling -> return as layer output
                    => return as skip_connection

    Return:
        layer_output: usual output of layer to pass for next
        skip_connection: special output for unets, return output before max_pooling
    """
    conv_layer = double_conv(inputs, filters)
    output = MaxPooling2D(pool_size=(2, 2))(conv_layer)

    return output, conv_layer


def decoder_block(inputs, skip_layer_input, filters: int = 32):
    """
    One Block of decoding

    Args:
        inputs: input from last layer
        skip_layer_input: special input for unet which gonna get concatenated
    """
    upscale = Conv2DTranspose(
        filters=filters,
        kernel_size=(2, 2),
        strides=2,
        padding="same"
    )(inputs)
    merge = concatenate([upscale, skip_layer_input], axis=3)
    return double_conv(merge)


def unet_model(input_size: Tuple[int, int, int] = (224, 224, 3), starting_filters=32):
    """
    Whole unet model architecture.
    """
    inputs = Input(input_size)

    # Encoding path
    encoder_1 = encoder_block(inputs, starting_filters)
    encoder_2 = encoder_block(encoder_1[0], starting_filters * 2)
    encoder_3 = encoder_block(encoder_2[0], starting_filters * 4)
    encoder_4 = encoder_block(encoder_3[0], starting_filters * 8)

    # Bridge
    bridge1 = double_conv(encoder_4[0], starting_filters * 16)

    # Decoding path
    decoder_1 = decoder_block(bridge1, encoder_4[1], starting_filters * 8)
    decoder_2 = decoder_block(decoder_1, encoder_3[1], starting_filters * 4)
    decoder_3 = decoder_block(decoder_2, encoder_2[1], starting_filters * 2)
    decoder_4 = decoder_block(decoder_3, encoder_1[1], starting_filters)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(decoder_4)

    model = Model(inputs=[inputs], outputs=[outputs])

    return model
