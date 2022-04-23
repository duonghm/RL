from typing import List, Tuple
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D
from tensorflow.keras import Model


class ModelFactory(object):
    @staticmethod
    def create_mlp(input_shape: int,
                   output_shape: int,
                   hidden_layers_shapes: List[int],
                   hidden_layers_activation: str
                   ):
        inputs = Input(shape=input_shape, dtype=tf.float32)
        x = Dense(units=hidden_layers_shapes[0], activation=hidden_layers_activation)(inputs)
        for shape in hidden_layers_shapes[1:]:
            x = Dense(units=shape, activation=hidden_layers_activation)(x)
        outputs = Dense(units=output_shape, activation="linear")(x)
        return Model(inputs=inputs, outputs=outputs)

    @staticmethod
    def create_cnn(input_shape,
                   output_shape: int,
                   hidden_layers_shapes: List[tuple],
                   hidden_layers_activation: str,
                   hidden_layers_mask_shape: List[tuple],
                   flatten_hidden_layers: List[int]):
        inputs = Input(shape=input_shape, dtype=tf.float32)
        x = Conv2D()
