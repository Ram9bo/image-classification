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


# TODO: evaluate pretrained models with unfrozen layers, has a huge memory requirement, maybe I can do it at home, maybe not, could (un)freeze just a few layers

def obo_accuracy(y_true, y_pred):
    # Calculate the argmax of predicted values to get the predicted class labels
    predicted_labels = tf.argmax(y_pred, axis=-1)

    # Cast y_true to the data type of predicted_labels
    y_true = tf.cast(y_true, predicted_labels.dtype)

    # Calculate the absolute difference between true and predicted class labels
    absolute_difference = tf.abs(y_true - predicted_labels)

    # Check if the absolute difference is less than or equal to 1
    correct_predictions = tf.cast(tf.less_equal(absolute_difference, 1), tf.float32)

    # Calculate the mean accuracy across all predictions
    accuracy = tf.reduce_mean(correct_predictions)

    return accuracy


NUM_CLASSES = 6
IMG_COLS = 512
IMG_ROWS = 512
RGB_CHANNELS = 3
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, RGB_CHANNELS)
CLASSIFICATION_METRICS = ["accuracy", obo_accuracy]
REGRESSION_METRICS = ["mean_squared_error"]


# TODO: try different resolutions, analyse prediction performance, runtime speed, and minimal model size

def xception(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES, freeze=True, task_mode="classification"):
    base = Xception(include_top=False, input_shape=input_shape, weights='imagenet', pooling='max')

    for layer in base.layers:
        layer.trainable = False
    if not freeze:
        # Unfreeze specific layers
        for layer in base.layers[-5:]:  # Unfreeze the last 10 layers
            layer.trainable = True

    model = Sequential()
    model.add(base)
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(num_classes, activation='softmax'))

    if task_mode == "classification":
        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=tf.keras.optimizers.Adam(),
            metrics=CLASSIFICATION_METRICS
        )
    elif task_mode == "regression":
        model.compile(
            loss='mean_squared_error',
            optimizer=tf.keras.optimizers.Adam(),
            metrics=REGRESSION_METRICS
        )

    return model


def efficient_net(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES, freeze=True, task_mode="classification"):
    base = EfficientNetB0(include_top=False, input_shape=input_shape, weights='imagenet', pooling='max')

    for layer in base.layers:
        layer.trainable = False
    if not freeze:
        # Unfreeze specific layers
        for layer in base.layers[-10:]:  # Unfreeze the last 10 layers
            layer.trainable = True

    model = Sequential()
    model.add(base)
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(num_classes, activation='softmax'))

    if task_mode == "classification":
        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=tf.keras.optimizers.Adam(),
            metrics=CLASSIFICATION_METRICS
        )
    elif task_mode == "regression":
        model.compile(
            loss='mean_squared_error',
            optimizer=tf.keras.optimizers.Adam(),
            metrics=REGRESSION_METRICS
        )

    return model


def vgg16(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES, freeze=True, task_mode="classification"):
    base = ResNet50(include_top=False, input_shape=input_shape, weights='imagenet', pooling='max')

    for layer in base.layers:
        layer.trainable = False
    if not freeze:
        # Unfreeze specific layers
        for layer in base.layers[-10:]:  # Unfreeze the last 10 layers
            layer.trainable = True

    model = Sequential()
    model.add(base)
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(num_classes, activation='softmax'))

    if task_mode == "classification":
        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=tf.keras.optimizers.Adam(),
            metrics=CLASSIFICATION_METRICS
        )
    elif task_mode == "regression":
        model.compile(
            loss='mean_squared_error',
            optimizer=tf.keras.optimizers.Adam(),
            metrics=REGRESSION_METRICS
        )

    return model


def compile_model(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES, task_mode="classification"):
    """
    Constructs and compiles a sequential model.
    """
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

    if task_mode == "classification":
        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=tf.keras.optimizers.Adam(),
            metrics=CLASSIFICATION_METRICS
        )
    elif task_mode == "regression":
        model.compile(
            loss='mean_squared_error',
            optimizer=tf.keras.optimizers.Adam(),
            metrics=REGRESSION_METRICS
        )
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
def resnet(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES, task_mode="classification"):
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

    if task_mode == "classification":
        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=tf.keras.optimizers.Adam(),
            metrics=CLASSIFICATION_METRICS
        )
    elif task_mode == "regression":
        model.compile(
            loss='mean_squared_error',
            optimizer=tf.keras.optimizers.Adam(),
            metrics=REGRESSION_METRICS
        )

    return model
