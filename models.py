# Mixed precision for running on Nvidia GPU

from __future__ import absolute_import, division, print_function, unicode_literals

# keras is distinct from tf.keras
from keras import layers as klayers
from keras import models as kmodels
from tensorflow.keras import backend as K

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.mixed_precision import experimental as mixed_precision

import numpy as np

import load
import matplotlib.pyplot as plt


def init():
    gpus = tf.config.experimental.list_physical_devices('GPU')

    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(
                logical_gpus), "Logical GPUs")
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
    outputs = layers.Activation(
        'softmax', dtype='float32', name='predictions')(x)
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
    y_test -= 769

    print(x_train.shape)

    model = keras.models.Sequential()
    size = int(25 * relative_size)
    conv1 = layers.Conv2D(size, kernel_size=(
        10, 1), strides=1,  kernel_regularizer=regularizers.l2(reg))
    conv2 = layers.Conv2D(size, kernel_size=(
        1, 22), kernel_regularizer=regularizers.l2(reg))
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

    conv3 = layers.Conv2D(2*size, kernel_size=(10, size),
                          kernel_regularizer=regularizers.l2(reg))
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

    conv4 = layers.Conv2D(4*size, kernel_size=(10, 2*size),
                          kernel_regularizer=regularizers.l2(reg))
    perm3 = layers.Permute((1, 3, 2))
    pool3 = layers.AveragePooling2D(pool_size=(3, 1))
    drop3 = layers.Dropout(dropout)

    model.add(conv4)
    model.add(layers.ELU(alpha))
    model.add(layers.BatchNormalization())
    model.add(perm3)
    model.add(pool3)
    model.add(drop3)

    conv5 = layers.Conv2D(8*size, kernel_size=(10, 4*size),
                          kernel_regularizer=regularizers.l2(reg))
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
        optimizer = keras.optimizers.Adam(
            learning_rate, beta_1=0.85, beta_2=0.92, amsgrad=True)
    elif optim == 'RMSprop':
        optimizer = keras.optimizers.RMSprop(learning_rate)
    else:
        optimizer = keras.optimizers.SGD(learning_rate, nesterov=True)

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


def make_lstm_model(x_train, y_train, x_test, y_test, reg=0.001, alpha=.7, learning_rate=0.001, dropout=0.5, epochs=100, relative_size=1.0, optim='SGD'):
    #policy = mixed_precision.Policy('mixed_float16')
    #mixed_precision.set_policy(policy)
    x_train = x_train.transpose((0, 2, 1))[:, :, :, None]
    x_test = x_test.transpose((0, 2, 1))[:, :, :, None]
    y_train -= 769
    y_test -= 769

    model = keras.models.Sequential()

    conv1 = layers.Conv2D(30, kernel_size=(10, 1), strides=1,
                          kernel_regularizer=regularizers.l2(reg))
    conv2 = layers.Conv2D(30, kernel_size=(
        1, 22), kernel_regularizer=regularizers.l2(reg))
    perm1 = layers.Permute((1, 3, 2))
    pool1 = layers.MaxPool2D(pool_size=(3, 1))
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

    model.add(layers.Reshape((330, 30)))

    model.add(layers.LSTM(20, return_sequences=True,
                          kernel_regularizer=regularizers.l2(reg)))
    model.add(layers.BatchNormalization())
    drop1 = layers.Dropout(dropout)
    model.add(drop1)

    model.add(layers.LSTM(20, return_sequences=True,
                          kernel_regularizer=regularizers.l2(reg)))
    model.add(layers.BatchNormalization())
    drop1 = layers.Dropout(dropout)
    model.add(drop1)

    model.add(layers.Reshape((330, 20, 1)))
    conv5 = layers.Conv2D(60, kernel_size=(
        10, 1), kernel_regularizer=regularizers.l2(reg))
    perm4 = layers.Permute((1, 3, 2))
    pool4 = layers.MaxPool2D(pool_size=(3, 1))
    drop4 = layers.Dropout(dropout)

    model.add(conv5)
    model.add(layers.ELU(alpha))
    model.add(layers.BatchNormalization())
    model.add(perm4)
    model.add(pool4)
    model.add(drop4)

    dense2 = layers.Dense(128, name='dense_2')
    model.add(layers.TimeDistributed(dense2))
    model.add(layers.ELU(alpha))
    model.add(layers.BatchNormalization())
    drop2 = layers.Dropout(dropout)
    model.add(drop2)
    model.add(layers.Flatten())
    model.add(layers.Dense(4, name='dense_logits',
                           kernel_regularizer=regularizers.l2(reg)))
    model.add(layers.Activation('softmax', dtype='float32', name='predictions'))

    if optim == 'Adam':
        optimizer = keras.optimizers.Adam(
            learning_rate, beta_1=0.85, beta_2=0.92, amsgrad=True)
    elif optim == 'RMSprop':
        optimizer = keras.optimizers.RMSprop(learning_rate)
    else:
        optimizer = keras.optimizers.SGD(learning_rate, nesterov=True)

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    history = model.fit(x_train, y_train,
                        batch_size=20,
                        epochs=epochs,
                        validation_split=0.2)
    test_scores = model.evaluate(x_test, y_test, verbose=2)
    print('Test loss:', test_scores[0])
    print('Test accuracy:', test_scores[1])
    plot(history)

    return model


def nll(y_true, y_pred):
    """ Negative log likelihood (Bernoulli). """

    # keras.losses.binary_crossentropy gives the mean
    # over the last axis. we require the sum
    return K.sum(keras.losses.mean_squared_error(y_true, y_pred))


class KLDivergenceLayer(klayers.Layer):

    """ Identity transform layer that adds KL divergence
    to the final model loss.
    """

    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):

        mu, log_var = inputs

        kl_batch = - .5 * K.sum(1 + log_var -
                                K.square(mu) -
                                K.exp(log_var), axis=-1)

        self.add_loss(K.mean(kl_batch), inputs=inputs)

        return inputs


def make_vae_model(x_train, y_train, x_test, y_test, reg=0.001, alpha=.7, learning_rate=0.001, dropout=0.5, epochs=100, relative_size=1.0, optim='SGD'):
    y_train -= 769
    y_test -= 769

    # latent_dim should be much smaller, but right now its equal to the original cnn input size
    latent_dim = 1000 * 22  # 2
    original_dim = 22000
    intermediate_dim = 256
    batch_size = 100
    epochs = 50
    epsilon_std = 1.0

    x_train = x_train.reshape(-1, original_dim)
    train_mean = np.mean(x_train)
    train_std = np.std(x_train)
    norm_x_train = (x_train -train_mean ) / train_std


    x_test = x_test.reshape(-1, original_dim)
    test_mean = np.mean(x_test)
    test_std = np.std(x_test)
    norm_x_test = (x_train -test_mean ) / test_std



    decoder = kmodels.Sequential([
        klayers.Dense(intermediate_dim, input_dim=latent_dim,
                      activation='relu'),
        klayers.Dense(original_dim, activation='sigmoid')
    ])

    x = klayers.Input(shape=(original_dim,))
    h = klayers.Dense(intermediate_dim, activation='relu')(x)

    z_mu = klayers.Dense(latent_dim)(h)
    z_log_var = klayers.Dense(latent_dim)(h)

    z_mu, z_log_var = KLDivergenceLayer()([z_mu, z_log_var])
    z_sigma = klayers.Lambda(lambda t: K.exp(.5 * t))(z_log_var)

    eps = klayers.Input(tensor=K.random_normal(stddev=epsilon_std,
                                               shape=(K.shape(x)[0], latent_dim)))
    z_eps = klayers.Multiply()([z_sigma, eps])
    z = klayers.Add()([z_mu, z_eps])

    x_pred = decoder(z)
    vae = kmodels.Model(inputs=[x, eps], outputs=x_pred)
    vae.compile(optimizer='adam', loss=nll)

    history = vae.fit(norm_x_train,
                      norm_x_train,
                      shuffle=True,
                      epochs=epochs,
                      batch_size=batch_size,
                      validation_split=.2)

    encoder = kmodels.Model(x, z_mu)
    z_train = encoder.predict(norm_x_train, batch_size=batch_size)
    z_test = encoder.predict(norm_x_test, batch_size=batch_size)

    z_train = z_train.reshape(-1, 1000, 22, 1)
    z_test = z_test.reshape(-1, 1000, 22, 1)

    # now pass encoded input into cnn

    # expected: (2115, 1000, 22, 1)

    size = int(25 * relative_size)
    conv1 = layers.Conv2D(size, kernel_size=(
        10, 1), strides=1,  kernel_regularizer=regularizers.l2(reg))
    conv2 = layers.Conv2D(size, kernel_size=(
        1, 22), kernel_regularizer=regularizers.l2(reg))
    perm1 = layers.Permute((1, 3, 2))
    pool1 = layers.AveragePooling2D(pool_size=(3, 1))
    drop1 = layers.Dropout(dropout)

    model = keras.models.Sequential()

    model.add(conv1)
    model.add(layers.ELU(alpha))
    model.add(layers.BatchNormalization())
    model.add(conv2)
    model.add(layers.ELU(alpha))
    model.add(layers.BatchNormalization())
    model.add(perm1)
    model.add(pool1)
    model.add(drop1)

    conv3 = layers.Conv2D(2*size, kernel_size=(10, size),
                          kernel_regularizer=regularizers.l2(reg))
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

    conv4 = layers.Conv2D(4*size, kernel_size=(10, 2*size),
                          kernel_regularizer=regularizers.l2(reg))
    perm3 = layers.Permute((1, 3, 2))
    pool3 = layers.AveragePooling2D(pool_size=(3, 1))
    drop3 = layers.Dropout(dropout)

    model.add(conv4)
    model.add(layers.ELU(alpha))
    model.add(layers.BatchNormalization())
    model.add(perm3)
    model.add(pool3)
    model.add(drop3)

    conv5 = layers.Conv2D(8*size, kernel_size=(10, 4*size),
                          kernel_regularizer=regularizers.l2(reg))
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
        optimizer = keras.optimizers.Adam(
            learning_rate, beta_1=0.85, beta_2=0.92, amsgrad=True)
    elif optim == 'RMSprop':
        optimizer = keras.optimizers.RMSprop(learning_rate)
    else:
        optimizer = keras.optimizers.SGD(learning_rate, nesterov=True)

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    history = model.fit(z_train, y_train,
                        batch_size=20,
                        epochs=epochs,
                        validation_split=0.2,
                        verbose=1)
    test_scores = model.evaluate(z_test, y_test, verbose=2)
    print('Test loss:', test_scores[0])
    print('Test accuracy:', test_scores[1])

    plot(history)

    return model


if __name__ == "__main__":
    init()
    x_test, y_test, _, x_train, y_train, _ = load.load_data()
    #make_cnn_model(x_train, y_train, x_test, y_test, reg=0.005, dropout=0.6, learning_rate=0.00075, alpha=0.8, epochs=100)
    # make_lstm_model(x_train, y_train, x_test, y_test,
    # reg=0.002, dropout=0.45, alpha=.8)
    make_vae_model(x_train, y_train, x_test, y_test)
