"""
Image Classification Network.
"""

from __future__ import print_function

import numpy as np
from keras.layers import Conv2D, MaxPooling2D, Dropout
from keras.layers import Dense, Flatten
from keras.models import Sequential

NUM_CLASSES = 10
IMG_COLS = 150
IMG_ROWS = 150
RGB_CHANNELS = 3
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, RGB_CHANNELS)


def compile_model(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES, loss="sparse_categorical_crossentropy", optimizer="RMSprop", metrics=None):
    """
    Constructs and compiles a sequential model.
    """

    if metrics is None:
        metrics = ["accuracy"]

    model = Sequential()

    # Apply convolutional layers
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten the 2D data
    model.add(Flatten())

    # Apply dense layers
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=metrics)
    return model
