# Mixed precision for running on Nvidia GPU

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.mixed_precision import experimental as mixed_precision

# inputs: a keras.Input
# layers: set of layers built on top of the input
def make_fc_model(x_train, y_train, x_test, y_test):
    x_train = x_train.reshape(2115, -1)
    x_test = x_test.reshape(443, -1)
    y_train -= 769
    y_test -= 769

    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)
    num_units = 4096
    inputs = keras.Input(shape=(22000, ), name='eeg_data')
    dense1 = layers.Dense(num_units, activation='relu', name='dense_1')
    x = dense1(inputs)
    dense2 = layers.Dense(num_units, activation='relu', name='dense_2')
    x = dense2(x)
    # 'kernel' is dense1's variable
    x = layers.Dense(4, name='dense_logits')(x)
    outputs = layers.Activation('softmax', dtype='float32', name='predictions')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=keras.optimizers.RMSprop(),
                  metrics=['accuracy'])
    history = model.fit(x_train, y_train,
                        batch_size=10,
                        epochs=10,
                        validation_split=0.1)
    test_scores = model.evaluate(x_test, y_test, verbose=2)
    print('Test loss:', test_scores[0])
    print('Test accuracy:', test_scores[1])

    return model
