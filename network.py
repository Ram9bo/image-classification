"""
Image Classification Network.
"""

import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Dropout
from keras.layers import Dense, Flatten
from keras.models import Sequential
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.xception import Xception

NUM_CLASSES = 6
IMG_COLS = 512
IMG_ROWS = 512
RGB_CHANNELS = 3
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, RGB_CHANNELS)


# TODO: try different resolutions, analyse prediction performance, runtime speed, and minimal model size

def xception(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES):
    base = Xception(include_top=False, input_shape=input_shape, weights='imagenet', pooling='max')

    for layer in base.layers:
        layer.trainable = False

    model = Sequential()
    model.add(base)
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['accuracy']
    )

    model.summary()

    return model


def efficient_net(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES):
    base = EfficientNetB0(include_top=False, input_shape=input_shape, weights='imagenet', pooling='max')

    for layer in base.layers:
        layer.trainable = False

    model = Sequential()
    model.add(base)
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-3),
        metrics=['accuracy']
    )

    model.summary()

    return model


def vgg16():
    base = ResNet50(include_top=False, input_shape=INPUT_SHAPE, weights='imagenet', pooling='max')

    for layer in base.layers:
        layer.trainable = False

    model = Sequential()
    model.add(base)
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='Adam',
        metrics=['accuracy']
    )

    model.summary()

    return model


def compile_model(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES, loss="sparse_categorical_crossentropy",
                  optimizer="Adam", metrics=None):
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
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten the 2D data
    model.add(Flatten())

    # Apply dense layers
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=metrics)
    return model


# Define the ResNet block
def resnet_block(x, filters, kernel_size=3, stride=1, conv_shortcut=False):
    shortcut = x

    if conv_shortcut:
        shortcut = layers.Conv2D(filters, 1, strides=stride)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)

    x = layers.add([x, shortcut])
    x = layers.ReLU()(x)
    return x


# Define the ResNet model
def resnet(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES):
    # Create the ResNet model
    input_tensor = layers.Input(shape=input_shape)
    x = layers.Conv2D(16, 7, strides=2, padding='same')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(3, strides=2)(x)

    num_blocks_list = [2, 2]

    for stage, num_blocks in enumerate(num_blocks_list):
        for block in range(num_blocks):
            stride = 1
            if stage > 0 and block == 0:
                stride = 2
            x = resnet_block(x, 64 * 2 ** stage, stride=stride, conv_shortcut=True)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=input_tensor, outputs=x)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# TODO: try a network that does binary (>3, <=3) possibly do a network with once such head in addition to the
#  classification or regression
#  TODO: try a regression approach TODO: refactor the separate networks into classes so
#   we can share some functions and add identifiers to be used in the data storage and visualisation
