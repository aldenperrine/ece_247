# Mixed precision for running on Nvidia GPU

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.mixed_precision import experimental as mixed_precision

# inputs: a keras.Input
# layers: set of layers built on top of the input
def make_mixed_model(inputs, layers):
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)
    outputs = layers.Activation('softmax', dtype='float32', name='predictions')(layers)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(loss='sparse_categorical_crossentropy',
              optimizer=keras.optimizers.RMSprop(),
              metrics=['accuracy'])
    return model
