"""
Model training.
"""
import os

import keras_tuner as kt
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

import dataloader
import network
from enums import ClassMode, TaskMode

print('Available GPUs', tf.config.list_physical_devices('GPU'))


# Define a function to extract labels from dataset elements
def extract_labels(features, labels):
    return labels


class CustomTuner(kt.BayesianOptimization):

    def run_trial(self, trial, *args, **kwargs):
        hp = trial.hyperparameters

        # TODO: once we have a crossfold validation implementation, apply that here as well (either instead of or
        #   in addition to executions per trial

        resize = hp.Choice("resize", [512, 256, 128, 72])
        epochs = hp.Int("epochs", min_value=1, max_value=30, step=1)
        augment = hp.Choice("augment", [True, False])
        transfer = hp.Choice("transfer", [True, False])
        freeze = hp.Choice("freeze", [True, False])
        transfer_source = hp.Choice("transfer_source", ["xception", "efficient", "vgg16"])
        balance = hp.Choice("balance", [True, False])
        class_weights = hp.Choice("class_weights", [False, True])
        recombination_ratio = hp.Float("recombination_ratio", min_value=0, max_value=10)
        dense_layers = hp.Int("dense_layers", min_value=1, max_value=10)
        dense_size = hp.Choice("dense_size", [8, 16, 32, 64, 128, 256, 512])

        results = []

        for _ in range(self.executions_per_trial):
            try:
                results.append(train_network(epochs=epochs,
                                             augment=augment,
                                             transfer=transfer,
                                             freeze=freeze,
                                             transfer_source=transfer_source,
                                             balance=balance,
                                             class_weights=class_weights,
                                             recombination_ratio=recombination_ratio,
                                             resize=(resize, resize),
                                             dense_layers=dense_layers,
                                             dense_size=dense_size))
            except Exception as e:
                print(e)
                return 1.0

        if len(results) > 1:
            return sum(results) / len(results)


def train_network(epochs=10, augment=True, transfer=False,
                  classmode=ClassMode.STANDARD,
                  freeze=True,
                  task_mode=TaskMode.CLASSIFICATION, transfer_source="xception", colour="rgb", pretrain=False,
                  feature=None, balance=True, class_weights=False, recombination_ratio=1.0, resize=(256, 256),
                  dense_layers=6, dense_size=128):
    num_classes = 6
    if classmode == ClassMode.COMPRESSED_START or classmode == ClassMode.COMPRESSED_END:
        num_classes = 5
    elif classmode == ClassMode.COMPRESSED_BOTH:
        num_classes = 4

    if task_mode == task_mode.REGRESSION:
        num_classes = 1
    print(f"Predicting {num_classes} classes")

    input_shape = network.INPUT_SHAPE
    if resize is not None:
        width, height = resize
        channels = 3 if colour == 'rgb' else 1
        input_shape = width, height, channels

    if pretrain:
        model = pretrain_model(transfer=transfer, num_classes=num_classes, freeze=freeze,
                               transfer_source=transfer_source)
    else:
        if transfer:
            if transfer_source == "xception":
                net = network.XceptionNetwork(num_classes=num_classes, freeze=freeze, task_mode=task_mode,
                                              input_shape=input_shape, dense_layers=dense_layers, dense_size=dense_size)
                model = net.model
            elif transfer_source == "efficient":
                net = network.EfficientNetNetwork(num_classes=num_classes, freeze=freeze, task_mode=task_mode,
                                                  input_shape=input_shape, dense_layers=dense_layers,
                                                  dense_size=dense_size)
                model = net.model
            elif transfer_source == "vgg16":
                net = network.VGG16Network(num_classes=num_classes, freeze=freeze, task_mode=task_mode,
                                           input_shape=input_shape, dense_layers=dense_layers, dense_size=dense_size)
                model = net.model
            else:
                raise Exception("Transfer source not recognised")
        else:
            net = network.CustomResNetNetwork(num_classes=num_classes, task_mode=task_mode, input_shape=input_shape,
                                              dense_layers=dense_layers, dense_size=dense_size)
            model = net.model

    if feature is not None:
        train, val = dataloader.feature_data(feature=feature, augment=augment)
    else:
        train, val = dataloader.all_data(augment=augment, classmode=classmode,
                                         colour=colour, balance=balance, recombination_ratio=recombination_ratio,
                                         resize=resize)

    # TODO: include both the presence and the strength of dropout in the HPO, also consider batch normalization, also switch input normalization (to 0-1) on/off

    class_weights_dict = None
    if class_weights:
        # Calculate class weights
        y_train = np.concatenate([y for x, y in train], axis=0)
        class_weights_list = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weights_dict = dict(enumerate(class_weights_list))
        print(class_weights_dict)

    # Define an early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_accuracy' if task_mode == TaskMode.CLASSIFICATION else 'val_mean_absolute_error',
        # Metric to monitor for improvement
        patience=epochs,  # Number of epochs with no improvement before stopping
        restore_best_weights=True  # Restore the best weights when stopping
    )

    hist = model.fit(train, epochs=epochs, verbose=1, validation_data=val, callbacks=[early_stopping],
                     class_weight=class_weights_dict)

    if task_mode == TaskMode.CLASSIFICATION:
        preds = np.argmax(model.predict(val), axis=1)
    elif task_mode == TaskMode.REGRESSION:
        preds = np.round(model.predict(val), decimals=1).flatten()
    else:
        raise KeyError(f"Task mode {task_mode} not recognised.")

    true_labels = np.concatenate([y for x, y in val], axis=0)

    print(preds)
    print(true_labels)

    correct, obo, incorrect = 0, 0, 0
    for i in range(len(preds)):
        t = true_labels[i]
        p = preds[i]
        if t == p:
            correct += 1
        elif abs(t - p) <= 1:
            obo += 1
        else:
            incorrect += 1
    print(correct, obo, incorrect)
    print(correct / len(preds), obo / len(preds), incorrect / len(preds))

    return 1 - (correct / len(preds))


def pretrain_model(transfer=False, num_classes=6, freeze=True, task_mode=TaskMode.CLASSIFICATION,
                   transfer_source="xception", data="nombacter", save_dir="pretrained_models"):
    pretrain_classes = 15
    model_name = f"pretrained_{transfer_source}_model.h5"

    custom_objects = {
        'obo_accuracy': network.obo_accuracy,
        'obo_accuracy_r': network.obo_accuracy_r,
        'obh_accuracy_r': network.obh_accuracy_r,
        'obt_accuracy_r': network.obt_accuracy_r
    }

    # Check if a saved model exists
    if os.path.exists(os.path.join(save_dir, model_name)):
        print("Loading pre-trained model...")
        model = load_model(os.path.join(save_dir, model_name), custom_objects=custom_objects)

        net = network.XceptionNetwork(num_classes=num_classes, freeze=freeze, task_mode=task_mode)
        # TODO: temporary workaround for mismatch in training, make the saving loading more robust in future
        net.model = model
        net.num_classes = num_classes
        net.reset_dense_layers()
        return net.model

    if transfer:
        if transfer_source == "xception":
            net = network.XceptionNetwork(num_classes=pretrain_classes, freeze=freeze, task_mode=task_mode)
            model = net.model
        elif transfer_source == "efficient":
            net = network.EfficientNetNetwork(num_classes=pretrain_classes, freeze=freeze, task_mode=task_mode)
            model = net.model
        elif transfer_source == "vgg16":
            net = network.VGG16Network(num_classes=pretrain_classes, freeze=freeze, task_mode=task_mode)
            model = net.model
        else:
            raise Exception("Transfer source not recognized")
    else:
        net = network.CustomResNetNetwork(num_classes=pretrain_classes, task_mode=task_mode)
        model = net.model

    train, val = dataloader.ssnombacter_data()
    pretrain_epochs = 30
    # Define an early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_accuracy',  # Metric to monitor for improvement
        patience=pretrain_epochs,  # Number of epochs with no improvement before stopping
        restore_best_weights=True  # Restore the best weights when stopping
    )

    hist = model.fit(train, epochs=pretrain_epochs, verbose=1, validation_data=val, callbacks=[early_stopping])

    # Save the pre-trained model
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model.save(os.path.join(save_dir, model_name))
    print(f"Pre-trained model saved as {model_name}.")

    net.num_classes = num_classes
    net.reset_dense_layers()

    # TODO: Check the effect of grayscaling our own image in combination with this pretraining
    # TODO: Consider cutting off multiple/all dense layers after pretraining
    # TODO: Tune the pretraining length (and architecture) a bit
    # TODO: Consider unfreezing a larger part of the base model (either just during pretraining or during both stages)

    return model


# TODO: set up a BO-HPO experiment to optimize the architecture and hyperparameters


tuner = CustomTuner(
    max_trials=100,
    overwrite=False,
    directory="tuning",
    project_name="biofilm",
    executions_per_trial=3
)

tuner.search()

print(tuner.get_best_hyperparameters())
