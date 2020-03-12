# Mixed precision for running on Nvidia GPU

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.mixed_precision import experimental as mixed_precision

import load
import argparse
import matplotlib.pyplot as plt

def init():
    gpus = tf.config.experimental.list_physical_devices('GPU')

    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

def plot(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

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
    dense3 = layers.Dense(num_units, activation='relu', name='dense_3')
    x = dense3(x)


    # 'kernel' is dense1's variable
    x = layers.Dense(4, name='dense_logits')(x)
    outputs = layers.Activation('softmax', dtype='float32', name='predictions')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
    history = model.fit(x_train, y_train,
                        batch_size=20,
                        epochs=10,
                        validation_split=0.1)
    test_scores = model.evaluate(x_test, y_test, verbose=2)
    print('Test loss:', test_scores[0])
    print('Test accuracy:', test_scores[1])

    return model


def make_cnn_model(x_train, y_train, x_test, y_test,  reg=0.001, alpha=.7, learning_rate=0.001, dropout=0.5, epochs=100, relative_size=1.0, optim='SGD'):
    #policy = mixed_precision.Policy('mixed_float16')
    #mixed_precision.set_policy(policy)
    x_train = x_train.transpose((0, 2, 1))[:, :, :, None]
    x_test = x_test.transpose((0, 2, 1))[:, :, :, None]
    y_train -= 769
    y_test  -= 769

    model = keras.models.Sequential()
    size = int(25 * relative_size)
    conv1 = layers.Conv2D(size, kernel_size=(10, 1), strides=1,  kernel_regularizer=regularizers.l2(reg))
    conv2 = layers.Conv2D(size, kernel_size=(1, 22), kernel_regularizer=regularizers.l2(reg))
    perm1 = layers.Permute((1, 3, 2))
    pool1 = layers.AveragePooling2D(pool_size=(3, 1))
    drop1 = layers.Dropout(dropout)

    model.add(conv1)
    model.add(layers.ELU(alpha))
    model.add(layers.BatchNormalization())
    model.add(conv2)
    model.add(layers.ELU(alpha))
    model.add(layers.BatchNormalization())
    model.add(perm1)
    model.add(pool1)
    model.add(drop1)

    conv3 = layers.Conv2D(2*size, kernel_size=(10, size), kernel_regularizer=regularizers.l2(reg))
    model.add(layers.ELU(alpha))
    perm2 = layers.Permute((1, 3, 2))
    pool2 = layers.AveragePooling2D(pool_size=(3, 1))
    drop2 = layers.Dropout(dropout)

    model.add(conv3)
    model.add(layers.ELU(alpha))
    model.add(layers.BatchNormalization())
    model.add(perm2)
    model.add(pool2)
    model.add(drop2)

    conv4 = layers.Conv2D(4*size, kernel_size=(10, 2*size), kernel_regularizer=regularizers.l2(reg))
    perm3 = layers.Permute((1, 3, 2))
    pool3 = layers.AveragePooling2D(pool_size=(3, 1))
    drop3 = layers.Dropout(dropout)

    model.add(conv4)
    model.add(layers.ELU(alpha))
    model.add(layers.BatchNormalization())
    model.add(perm3)
    model.add(pool3)
    model.add(drop3)

    conv5 = layers.Conv2D(8*size, kernel_size=(10, 4*size), kernel_regularizer=regularizers.l2(reg))
    perm4 = layers.Permute((1, 3, 2))
    pool4 = layers.AveragePooling2D(pool_size=(3, 1))
    drop4 = layers.Dropout(dropout)

    model.add(conv5)
    model.add(layers.ELU(alpha))
    model.add(layers.BatchNormalization())
    model.add(perm4)
    model.add(pool4)
    model.add(drop4)


    model.add(layers.Flatten())

    model.add(layers.Dense(4, name='dense_logits'))
    model.add(layers.Activation('softmax', dtype='float32', name='predictions'))

    if optim == 'Adam':
        optimizer=keras.optimizers.Adam(learning_rate, beta_1=0.85, beta_2=0.92, amsgrad=True)
    elif optim == 'RMSprop':
        optimizer=keras.optimizers.RMSprop(learning_rate)
    else:
        optimizer=keras.optimizers.SGD(learning_rate, nesterov=True)

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    history = model.fit(x_train, y_train,
                        batch_size=20,
                        epochs=epochs,
                        validation_split=0.2,
                        verbose=1)
    test_scores = model.evaluate(x_test, y_test, verbose=2)
    print('Test loss:', test_scores[0])
    print('Test accuracy:', test_scores[1])

    plot(history)

    return model


def make_lstm_model(x_train, y_train, x_test, y_test, reg=0.001):
    #policy = mixed_precision.Policy('mixed_float16')
    #mixed_precision.set_policy(policy)
    x_train = x_train.transpose((0, 2, 1))[:, :, :, None]
    x_test = x_test.transpose((0, 2, 1))[:, :, :, None]
    y_train -= 769
    y_test -= 769

    model = keras.models.Sequential()


    conv1 = layers.Conv2D(30, kernel_size=(10, 1), input_shape=(1000, 22, 1), strides=1, activation='elu', kernel_regularizer=regularizers.l2(reg))
    conv2 = layers.Conv2D(30, kernel_size=(1, 22), activation='elu', kernel_regularizer=regularizers.l2(reg))
    perm1 = layers.Permute((1, 3, 2))
    pool1 = layers.MaxPool2D(pool_size=(3, 1))
    drop1 = layers.Dropout(.8)

    model.add(conv1)
    model.add(layers.BatchNormalization())
    model.add(conv2)
    model.add(layers.BatchNormalization())
    model.add(perm1)
    model.add(pool1)
    model.add(drop1)

    model.add(layers.Reshape((330, 30)))

    model.add(layers.LSTM(20, return_sequences=True, kernel_regularizer=regularizers.l2(reg)))
    model.add(layers.BatchNormalization())
    drop1 = layers.Dropout(.8)
    model.add(drop1)

    model.add(layers.LSTM(20, return_sequences=True, kernel_regularizer=regularizers.l2(reg)))
    model.add(layers.BatchNormalization())
    drop1 = layers.Dropout(.8)
    model.add(drop1)

    print(model.output_shape)
    model.add(layers.Reshape((330, 20, 1)))
    conv5 = layers.Conv2D(120, kernel_size=(10, 1), activation='elu', kernel_regularizer=regularizers.l2(reg))
    perm4 = layers.Permute((1, 3, 2))
    pool4 = layers.MaxPool2D(pool_size=(3, 1))
    drop4 = layers.Dropout(.8)

    model.add(conv5)
    model.add(layers.BatchNormalization())
    model.add(perm4)
    model.add(pool4)
    model.add(drop4)


    #dense1 = layers.Dense(1024, name='dense_1', kernel_regularizer=regularizers.l2(0.001))
    #model.add(layers.ELU(alpha=0.05))
    #model.add(layers.TimeDistributed(dense1))
    #model.add(layers.BatchNormalization())

    dense2 = layers.Dense(1024, activation='elu', name='dense_2')
    model.add(layers.TimeDistributed(dense2))
    model.add(layers.BatchNormalization())
    drop2 = layers.Dropout(.5)
    model.add(drop2)
    model.add(layers.Flatten())
    model.add(layers.Dense(4, name='dense_logits', kernel_regularizer=regularizers.l2(reg)))
    model.add(layers.Activation('softmax', dtype='float32', name='predictions'))

    model.compile(loss='sparse_categorical_crossentropy',
                          optimizer=keras.optimizers.Adam(),
                          metrics=['accuracy'])
    history = model.fit(x_train, y_train,
                        batch_size=20,
                        epochs=40,
                        validation_split=0.2)
    test_scores = model.evaluate(x_test, y_test, verbose=2)
    print('Test loss:', test_scores[0])
    print('Test accuracy:', test_scores[1])
    print(history)

    return model

def make_vae_model(x_train, y_train, x_test, y_test, reg=0.001, dropout=0.5, learning_rate=0.001, alpha=0.75):
    x_train = x_train.transpose((0, 2, 1))[:, :, :, None]
    x_test = x_test.transpose((0, 2, 1))[:, :, :, None]
    y_train -= 769
    y_test -= 769

    model = keras.models.Sequential()
    pass

if __name__ == "__main__":
    init()
    x_test, y_test, _, x_train, y_train, _ = load.load_data()
    make_cnn_model(x_train, y_train, x_test, y_test, reg=0.005, dropout=0.6, learning_rate=0.00075, alpha=0.8, epochs=500)
    #make_lstm_model(x_train, y_train, x_test, y_test)
