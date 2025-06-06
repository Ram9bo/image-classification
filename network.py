"""
Image Classification Network.
"""

from abc import ABC, abstractmethod

import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Dropout
from keras.layers import Dense, Flatten
from keras.models import Sequential
from tensorflow.keras import layers, Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.xception import Xception

# Default Constants
NUM_CLASSES = 6
IMG_COLS = 512
IMG_ROWS = 512
RGB_CHANNELS = 3
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, RGB_CHANNELS)


class Network(ABC):
    """
    Network base class. Determines some variables based on the given parameters, then calls on the subclass to provide
    the base of the model, before adding the dense layers for classification/regression.

    Model can be accessed through the 'model' property, or its summary printed through the 'summary' method.
    """

    model = None

    def __init__(self, input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES, freeze=False, dense_layers=6, dense_size=128,
                 lr=0.001, dropout=0.0, unfreeze=0):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.final_activation = "softmax"
        self.loss = "sparse_categorical_crossentropy"
        self.metrics = ["accuracy"]
        self.freeze = freeze
        self.dense_layers = dense_layers
        self.dense_size = dense_size
        self.lr = lr
        self.dropout = dropout
        self.unfreeze = unfreeze

        self.create_base()
        self.add_dense_layers()

    @abstractmethod
    def create_base(self):
        pass

    def summary(self):
        self.model.summary()

    def add_dense_layers(self):
        for i in range(self.dense_layers):
            self.model.add(Dense(self.dense_size, activation='relu'))
            self.model.add(Dropout(self.dropout))
        self.model.add(Dense(self.num_classes, activation=self.final_activation))

        self.model.compile(
            loss=self.loss,
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
            metrics=self.metrics
        )

    def reset_dense_layers(self):
        for layer in reversed(self.model.layers):
            if isinstance(layer, tf.keras.layers.Dense):
                # Create Glorot-initialized weights and zero-initialized biases
                glorot_weights = tf.keras.initializers.glorot_uniform()(layer.kernel.shape)
                zero_biases = tf.zeros(layer.bias.shape)

                # Set the layer's weights to the Glorot-initialized weights and zero-initialized biases
                layer.set_weights([glorot_weights, zero_biases])
            else:
                break

        self.model.pop()

        self.model.add(Dense(self.num_classes, activation=self.final_activation))

        self.model.compile(
            loss=self.loss,
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
            metrics=self.metrics
        )


class XceptionNetwork(Network):

    def create_base(self):
        base = Xception(include_top=False, input_shape=self.input_shape, weights='imagenet', pooling='max')

        for layer in base.layers:
            layer.trainable = False
        if not self.unfreeze == 0:
            # Unfreeze specific layers
            for layer in base.layers[-self.unfreeze:]:  # Unfreeze the last x layers
                layer.trainable = True

        self.model = Sequential()
        self.model.add(base)


class VGG16Network(Network):

    def create_base(self):
        base = ResNet50(include_top=False, input_shape=self.input_shape, weights='imagenet', pooling='max')

        for layer in base.layers:
            layer.trainable = False
        if not self.unfreeze == 0:
            # Unfreeze specific layers
            for layer in base.layers[-self.unfreeze:]:  # Unfreeze the last x layers
                layer.trainable = True

        self.model = Sequential()
        self.model.add(base)


class CustomCNNNetwork(Network):

    def create_base(self):
        self.model = Sequential()

        # Apply convolutional layers
        self.model.add(Conv2D(32, kernel_size=(3, 3),
                              activation='relu',
                              input_shape=self.input_shape))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(32, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(32, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        # Flatten the 2D data
        self.model.add(Flatten())


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


class CustomResNetNetwork(Network):

    def create_base(self):

        input_tensor = layers.Input(shape=self.input_shape)
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

        self.model = Sequential(Model(inputs=input_tensor, outputs=x))
