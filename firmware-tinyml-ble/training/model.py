"""CNN with the same op set the firmware's TFLite-Micro resolver registers:
SUB, CONV_2D, MAX_POOL_2D, MEAN, FULLY_CONNECTED, SOFTMAX  (main.cpp init_model).

Layout mirrors the extracted v3.2 graph:
  log_mel_input [64,126,1]
    -> feature_normalization (Sub: x - mean)
    -> conv2d  + relu -> max_pool
    -> conv2d  + relu -> max_pool
    -> conv2d  + relu
    -> global_average_pooling2d (Mean)
    -> dense + relu
    -> dense(3) + softmax

Keep it small: the on-device tensor arena is 300 KB and the shipped model is
~12 KB INT8.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from constants import N_MELS, TIME_FRAMES


@tf.keras.utils.register_keras_serializable()
class FeatureCentering(layers.Layer):
    """x - mean. Converts to a single TFLite SUB op (the firmware's resolver
    registers SUB but not ADD/MUL), unlike keras Normalization which can emit
    an extra MUL and, in Keras 3, crashes with axis=None here."""

    def __init__(self, mean: float = 0.0, **kw):
        super().__init__(**kw)
        self.mean = float(mean)

    def call(self, x):
        return x - self.mean

    def get_config(self):
        return {**super().get_config(), "mean": self.mean}


def build_model(feature_mean: float, n_classes: int = 3,
                filters=(8, 16, 24), dense_units: int = 24) -> tf.keras.Model:
    inp = tf.keras.Input(shape=(N_MELS, TIME_FRAMES, 1), name="log_mel_input")
    x = FeatureCentering(feature_mean, name="feature_normalization")(inp)

    for i, f in enumerate(filters):
        x = layers.Conv2D(f, 3, padding="same", use_bias=True,
                          name=f"conv2d_{i}")(x)
        x = layers.ReLU(name=f"relu_{i}")(x)
        if i < len(filters) - 1:
            x = layers.MaxPooling2D(2, name=f"max_pooling2d_{i}")(x)

    x = layers.GlobalAveragePooling2D(name="global_average_pooling2d")(x)
    x = layers.Dense(dense_units, name="dense")(x)
    x = layers.ReLU(name="dense_relu")(x)
    out = layers.Dense(n_classes, activation="softmax", name="logits")(x)

    return tf.keras.Model(inp, out, name="horn_noise_siren")


def representative_dataset_factory(x_float: np.ndarray, n: int = 400):
    idx = np.random.default_rng(0).choice(len(x_float), size=min(n, len(x_float)),
                                          replace=False)

    def gen():
        for i in idx:
            yield [x_float[i:i + 1].astype(np.float32)]

    return gen


def convert_int8(model: tf.keras.Model, x_float_repr: np.ndarray) -> bytes:
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = representative_dataset_factory(x_float_repr)
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    return conv.convert()
