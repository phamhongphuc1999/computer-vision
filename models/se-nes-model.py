from keras import backend as K, Model
from keras.src.layers import (
    Conv2D,
    BatchNormalization,
    Activation,
    concatenate,
    GlobalAveragePooling2D,
    Dense,
    Reshape,
    multiply,
    add,
    Lambda,
    AveragePooling2D,
    Input,
)
import numpy as np


class SeResModel(object):
    def __init__(
        self,
        size=96,
        num_classes=10,
        depth=64,
        reduction_ratio=4,
        num_split=8,
        num_block=3,
    ):
        self.depth = depth
        self.ratio = reduction_ratio
        self.num_split = num_split
        self.num_block = num_block
        if K.image_data_format() == "channel_first":
            self.channel_axis = 1
        else:
            self.channel_axis = 3
        self.model = self.build_model(Input(shape=(size, size, 3)), num_classes)

    @staticmethod
    def conv_bn(x, filters, kernel_size, stride, padding="same"):
        x = Conv2D(
            filters=filters,
            kernel_size=[kernel_size, kernel_size],
            strides=[stride, stride],
            padding=padding,
        )(x)
        x = BatchNormalization()(x)

        return x

    @staticmethod
    def activation(x, func="relu"):
        return Activation(func)(x)

    def channel_zeropad(self, x):
        shape = list(x.shape)
        y = K.zeros_like(x)

        if self.channel_axis == 3:
            y = y[:, :, :, : shape[self.channel_axis] // 2]
        else:
            y = y[:, : shape[self.channel_axis] // 2, :, :]
        return concatenate([y, x, y], self.channel_axis)

    def channel_zeropad_output(self, input_shape):
        shape = list(input_shape)
        shape[self.channel_axis] *= 2
        return tuple(shape)

    def initial_layer(self, inputs):
        x = self.conv_bn(inputs, self.depth, 3, 1)
        x = self.activation(x)

        return x

    def transform_layer(self, x, stride):
        x = self.conv_bn(x, self.depth, 1, 1)
        x = self.activation(x)

        x = self.conv_bn(x, self.depth, 3, stride)
        x = self.activation(x)

        return x

    def split_layer(self, x, stride):
        splitted_branches = list()
        for i in range(self.num_split):
            branch = self.transform_layer(x, stride)
            splitted_branches.append(branch)

        return concatenate(splitted_branches, axis=self.channel_axis)

    def squeeze_excitation_layer(self, x, out_dim):
        squeeze = GlobalAveragePooling2D()(x)

        excitation = Dense(units=out_dim // self.ratio)(squeeze)
        excitation = self.activation(excitation)
        excitation = Dense(units=out_dim)(excitation)
        excitation = self.activation(excitation, "sigmoid")
        excitation = Reshape((1, 1, out_dim))(excitation)

        scale = multiply([x, excitation])

        return scale

    def residual_layer(self, x, out_dim):
        for i in range(self.num_block):
            input_dim = int(np.shape(x)[-1])

            if input_dim * 2 == out_dim:
                flag = True
                stride = 2
            else:
                flag = False
                stride = 1

            subway_x = self.split_layer(x, stride)
            subway_x = self.conv_bn(subway_x, out_dim, 1, 1)
            subway_x = self.squeeze_excitation_layer(subway_x, out_dim)

            if flag:
                pad_x = AveragePooling2D(
                    pool_size=(2, 2), strides=(2, 2), padding="same"
                )(x)
                pad_x = Lambda(
                    self.channel_zeropad, output_shape=self.channel_zeropad_output
                )(pad_x)
            else:
                pad_x = x

            x = self.activation(add([pad_x, subway_x]))

        return x

    def build_model(self, inputs, num_classes):
        x = self.initial_layer(inputs)

        x = self.residual_layer(x, out_dim=64)
        x = self.residual_layer(x, out_dim=128)
        x = self.residual_layer(x, out_dim=256)

        x = GlobalAveragePooling2D()(x)
        x = Dense(units=num_classes, activation="softmax")(x)

        return Model(inputs, x)
