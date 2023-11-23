"""
Image Classification Network.
"""

from abc import ABC, abstractmethod

import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten
from keras.models import Sequential
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.models import load_model

from enums import TaskMode
from metrics import obo_accuracy, obo_accuracy_r, obh_accuracy_r, obt_accuracy_r, accuracy

# Default Constants
NUM_CLASSES = 6
IMG_COLS = 512
IMG_ROWS = 512
RGB_CHANNELS = 3
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, RGB_CHANNELS)

# Classmode variable dictionaries
activations = {
    TaskMode.CLASSIFICATION: "softmax",
    TaskMode.REGRESSION: None
}

metrics = {
    TaskMode.CLASSIFICATION: [accuracy, obo_accuracy],
    # Built-in accuracy is acting up, replacing it with a custom implementation for investigation
    TaskMode.REGRESSION: ["mean_absolute_error", obo_accuracy_r, obh_accuracy_r, obt_accuracy_r]
}

losses = {
    TaskMode.CLASSIFICATION: 'sparse_categorical_crossentropy',
    TaskMode.REGRESSION: 'mean_squared_error'
}


class Network(ABC):
    """
    Network base class. Determines some variables based on the given parameters, then calls on the subclass to provide
    the base of the model, before adding the dense layers for classification/regression.

    Model can be accessed through the 'model' property, or its summary printed through the 'summary' method.
    """

    model = None

    def __init__(self, input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES, task_mode=TaskMode.CLASSIFICATION,
                 freeze=False, dense_layers=6, dense_size=128, lr=0.001):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.task_mode = task_mode
        self.final_activation = activations[task_mode]
        self.loss = losses[task_mode]
        self.metrics = metrics[task_mode]
        self.freeze = freeze
        self.dense_layers = dense_layers
        self.dense_size = dense_size
        self.lr = lr

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


# TODO: try to unfreeze more layers

# TODO: make the networks able to switch to grayscale data (at least the custom ones)

# TODO: a model that predicts the three constituent features of the classification
#  (either alone or as an additional head next to the class)

# TODO: try different resolutions, analyse prediction performance, runtime speed, and minimal model size

class PathonetNetwork(Network):

    def create_base(self):
        base = load_model("pretrained_models/PathoNet.hdf5")

        # Find the index where the encoder layers end
        encoder_end = None
        for i, layer in enumerate(base.layers):
            if layer.name == 'concatenate_6':
                encoder_end = i + 1  # Add 1 to include the identified layer
                break

        if encoder_end:
            # Separate the encoder layers
            encoder_layers = base.layers[:encoder_end]

            # Create a new model containing only the encoder layers
            base = tf.keras.Model(inputs=base.input, outputs=encoder_layers[-1].output)
        else:
            print("Specified layer marking the end of encoder not found.")

        for layer in base.layers:
            layer.trainable = False
        if not self.freeze:
            # Unfreeze specific layers
            for layer in base.layers[-5:]:  # Unfreeze the last x layers
                layer.trainable = True

        self.model = Sequential()
        self.model.add(base)
        self.model.add(Flatten())


class XceptionNetwork(Network):

    def create_base(self):
        base = Xception(include_top=False, input_shape=self.input_shape, weights='imagenet', pooling='max')

        for layer in base.layers:
            layer.trainable = False
        if not self.freeze:
            # Unfreeze specific layers
            for layer in base.layers[-5:]:  # Unfreeze the last x layers
                layer.trainable = True

        # TODO: turn freeze into an integer, also make it part of the parent class

        self.model = Sequential()
        self.model.add(base)


class EfficientNetNetwork(Network):

    def create_base(self):
        base = EfficientNetB0(include_top=False, input_shape=self.input_shape, weights='imagenet', pooling='max')

        for layer in base.layers:
            layer.trainable = False
        if not self.freeze:
            # Unfreeze specific layers
            for layer in base.layers[-5:]:  # Unfreeze the last 10 layers
                layer.trainable = True

        self.model = Sequential()
        self.model.add(base)


class VGG16Network(Network):

    def create_base(self):
        base = ResNet50(include_top=False, input_shape=self.input_shape, weights='imagenet', pooling='max')

        for layer in base.layers:
            layer.trainable = False
        if not self.freeze:
            # Unfreeze specific layers
            for layer in base.layers[-5:]:  # Unfreeze the last 10 layers
                layer.trainable = True

        self.model = Sequential()
        self.model.add(base)

    # TODO: efficient pretrained model does not learn at all right now. Might want to manually figure out
    #  how to get it to work and/or include them in the HPO


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
