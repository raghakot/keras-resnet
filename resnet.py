from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    merge,
    Dense,
    Flatten
)
from keras.layers.convolutional import (
    Convolution2D,
    MaxPooling2D,
    AveragePooling2D
)
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K


def _bn_relu(input):
    """Helper to build a BN -> relu block
    """
    norm = BatchNormalization(mode=0, axis=CHANNEL_AXIS)(input)
    return Activation("relu")(norm)


def _conv_bn_relu(**conv_params):
    """Helper to build a conv -> BN -> relu block
    """
    nb_filter = conv_params["nb_filter"]
    nb_row = conv_params["nb_row"]
    nb_col = conv_params["nb_col"]
    subsample = conv_params.setdefault("subsample", (1, 1))
    init = conv_params.setdefault("init", "he_normal")
    border_mode = conv_params.setdefault("border_mode", "same")
    W_regularizer = conv_params.setdefault("W_regularizer", l2(1.e-4))

    def f(input):
        conv = Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, subsample=subsample,
                             init=init, border_mode=border_mode, W_regularizer=W_regularizer)(input)
        return _bn_relu(conv)

    return f


def _bn_relu_conv(**conv_params):
    """Helper to build a BN -> relu -> conv block.
    This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    nb_filter = conv_params["nb_filter"]
    nb_row = conv_params["nb_row"]
    nb_col = conv_params["nb_col"]
    subsample = conv_params.setdefault("subsample", (1,1))
    init = conv_params.setdefault("init", "he_normal")
    border_mode = conv_params.setdefault("border_mode", "same")
    W_regularizer = conv_params.setdefault("W_regularizer", l2(1.e-4))

    def f(input):
        activation = _bn_relu(input)
        return Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, subsample=subsample,
                             init=init, border_mode=border_mode, W_regularizer=W_regularizer)(activation)

    return f


def _shortcut(input, residual):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    stride_width = input._keras_shape[ROW_AXIS] // residual._keras_shape[ROW_AXIS]
    stride_height = input._keras_shape[COL_AXIS] // residual._keras_shape[COL_AXIS]
    equal_channels = residual._keras_shape[CHANNEL_AXIS] == input._keras_shape[CHANNEL_AXIS]

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Convolution2D(nb_filter=residual._keras_shape[CHANNEL_AXIS],
                                 nb_row=1, nb_col=1,
                                 subsample=(stride_width, stride_height),
                                 init="he_normal", border_mode="valid",
                                 W_regularizer=l2(0.0001))(input)

    return merge([shortcut, residual], mode="sum")


def _residual_block(block_function, nb_filter, repetitions, is_first_layer=False):
    """Builds a residual block with repeating bottleneck blocks.
    """
    def f(input):
        for i in range(repetitions):
            init_subsample = (1, 1)
            if i == 0 and not is_first_layer:
                init_subsample = (2, 2)
            input = block_function(
                    nb_filter=nb_filter,
                    init_subsample=init_subsample,
                    is_first_block_of_first_layer=(is_first_layer and i == 0)
                )(input)
        return input

    return f


def basic_block(nb_filter, init_subsample=(1, 1), is_first_block_of_first_layer=False):
    """Basic 3 X 3 convolution blocks for use on resnets with layers <= 34.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv1 = Convolution2D(nb_filter=nb_filter,
                                 nb_row=3, nb_col=3,
                                 subsample=init_subsample,
                                 init="he_normal", border_mode="same",
                                 W_regularizer=l2(0.0001))(input)
        else:
            conv1 = _bn_relu_conv(nb_filter=nb_filter, nb_row=3, nb_col=3, subsample=init_subsample)(input)

        residual = _bn_relu_conv(nb_filter=nb_filter, nb_row=3, nb_col=3)(conv1)
        return _shortcut(input, residual)

    return f


def bottleneck(nb_filter, init_subsample=(1, 1), is_first_block_of_first_layer=False):
    """Bottleneck architecture for > 34 layer resnet.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf

    :return: A final conv layer of nb_filter * 4
    """
    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv_1_1 = Convolution2D(nb_filter=nb_filter,
                                 nb_row=1, nb_col=1,
                                 subsample=init_subsample,
                                 init="he_normal", border_mode="same",
                                 W_regularizer=l2(0.0001))(input)
        else:
            conv_1_1 = _bn_relu_conv(nb_filter=nb_filter, nb_row=1, nb_col=1, subsample=init_subsample)(input)

        conv_3_3 = _bn_relu_conv(nb_filter=nb_filter, nb_row=3, nb_col=3)(conv_1_1)
        residual = _bn_relu_conv(nb_filter=nb_filter * 4, nb_row=1, nb_col=1)(conv_3_3)
        return _shortcut(input, residual)

    return f


def handle_dim_ordering():
    global ROW_AXIS
    global COL_AXIS
    global CHANNEL_AXIS
    if K.image_dim_ordering() == 'tf':
        ROW_AXIS = 1
        COL_AXIS = 2
        CHANNEL_AXIS = 3
    else:
        CHANNEL_AXIS = 1
        ROW_AXIS = 2
        COL_AXIS = 3


class ResnetBuilder(object):
    @staticmethod
    def build(input_shape, num_outputs, block_fn, repetitions):
        """Builds a custom ResNet like architecture.
        :param input_shape: The input shape in the form (nb_channels, nb_rows, nb_cols)

        :param num_outputs: The number of outputs at final softmax layer

        :param block_fn: The block function to use. This is either :func:`basic_block` or :func:`bottleneck`.
        The original paper used basic_block for layers < 50

        :param repetitions: Number of repetitions of various block units.
        At each block unit, the number of filters are doubled and the input size is halved

        :return: The keras model.
        """
        handle_dim_ordering()
        if len(input_shape) != 3:
            raise Exception("Input shape should be a tuple (nb_channels, nb_rows, nb_cols)")

        # Permute dimension order if necessary
        if K.image_dim_ordering() == 'tf':
            input_shape = (input_shape[1], input_shape[2], input_shape[0])

        input = Input(shape=input_shape)
        conv1 = _conv_bn_relu(nb_filter=64, nb_row=7, nb_col=7, subsample=(2, 2))(input)
        pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode="same")(conv1)

        block = pool1
        nb_filter = 64
        for i, r in enumerate(repetitions):
            block = _residual_block(block_fn, nb_filter=nb_filter, repetitions=r, is_first_layer=(i == 0))(block)
            nb_filter *= 2

        # Last activation
        block = _bn_relu(block)

        # Classifier block
        pool2 = AveragePooling2D(pool_size=(block._keras_shape[ROW_AXIS],
                                            block._keras_shape[COL_AXIS]),
                                 strides=(1, 1))(block)
        flatten1 = Flatten()(pool2)
        dense = Dense(output_dim=num_outputs, init="he_normal", activation="softmax")(flatten1)

        model = Model(input=input, output=dense)
        return model

    @staticmethod
    def build_resnet_18(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, basic_block, [2, 2, 2, 2])

    @staticmethod
    def build_resnet_34(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, basic_block, [3, 4, 6, 3])

    @staticmethod
    def build_resnet_50(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 4, 6, 3])

    @staticmethod
    def build_resnet_101(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 4, 23, 3])

    @staticmethod
    def build_resnet_152(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 8, 36, 3])


def main():
    model = ResnetBuilder.build_resnet_18((3, 224, 224), 1000)
    model.compile(loss="categorical_crossentropy", optimizer="sgd")
    model.summary()


if __name__ == '__main__':
    main()

