"""
Model training.
"""
import json
import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model

import dataloader
import network
import tuner
import util
from enums import ClassMode, TaskMode
import random
from collections import Counter
print('Available GPUs', tf.config.list_physical_devices('GPU'))


# Define a function to extract labels from dataset elements
def extract_labels(features, labels):
    return labels


def train_network(fold, epochs=10, augment=True, transfer=True,
                  classmode=ClassMode.STANDARD,
                  freeze=True,
                  task_mode=TaskMode.CLASSIFICATION, transfer_source="xception", colour="rgb", pretrain=False,
                  feature=None, class_weights=False, recombination_ratio=1.0, resize=(256, 256),
                  dense_layers=6, dense_size=128, lr=0.001, rotate=True, flip=True, brightness_delta=0, batch_size=2, dropout=0.0):
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
        channels = 3
        input_shape = width, height, channels

    if pretrain:
        model = pretrain_model(transfer=transfer, num_classes=num_classes, freeze=freeze,
                               transfer_source=transfer_source)
    else:
        if transfer:
            if transfer_source == "xception":
                net = network.XceptionNetwork(num_classes=num_classes, freeze=freeze, task_mode=task_mode,
                                              input_shape=input_shape, dense_layers=dense_layers, dense_size=dense_size,
                                              lr=lr, dropout=dropout)
                model = net.model
            elif transfer_source == "efficient":
                net = network.EfficientNetNetwork(num_classes=num_classes, freeze=freeze, task_mode=task_mode,
                                                  input_shape=input_shape, dense_layers=dense_layers,
                                                  dense_size=dense_size, lr=lr, dropout=dropout)
                model = net.model
            elif transfer_source == "vgg16":
                net = network.VGG16Network(num_classes=num_classes, freeze=freeze, task_mode=task_mode,
                                           input_shape=input_shape, dense_layers=dense_layers, dense_size=dense_size,
                                           lr=lr, dropout=dropout)
                model = net.model
            elif transfer_source == "pathonet":
                net = network.PathonetNetwork(num_classes=num_classes, freeze=freeze, task_mode=task_mode,
                                              input_shape=input_shape, dense_layers=dense_layers, dense_size=dense_size,
                                              lr=lr, dropout=dropout)
                model = net.model
                resize = (256, 256)
            else:
                raise Exception("Transfer source not recognised")
        else:
            net = network.CustomResNetNetwork(num_classes=num_classes, task_mode=task_mode, input_shape=input_shape,
                                              dense_layers=dense_layers, dense_size=dense_size, lr=lr, dropout=dropout)
            model = net.model

    if feature is not None:
        train, val = dataloader.feature_data(feature=feature, augment=augment)
        test = None
    else:
        train, val, test = dataloader.fold_to_data(fold, color=colour, batch_size=batch_size, resize=resize,
                                             recombination_ratio=recombination_ratio, rotate=rotate, flip=flip,
                                             brightness_delta=brightness_delta)

    class_weights_dict = None
    if class_weights:
        # Calculate class weights
        y_train = np.concatenate([y for x, y in train], axis=0)
        class_weights_list = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weights_dict = dict(enumerate(class_weights_list))
        print(class_weights_dict)

    # Define ModelCheckpoint callback
    checkpoint_filepath = 'best_model.h5'  # Specify the path to save the best model
    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,  # Save only the model weights, not the entire model
        monitor='val_accuracy',  # Monitor validation accuracy
        mode='max',  # 'max' means save the model when the monitored quantity is maximized
        save_best_only=True,  # Save only the best model
        options = tf.train.CheckpointOptions(compression=tf.train.CheckpointOptions.GZIP)
    )

    hist = model.fit(train, epochs=epochs, verbose=1, validation_data=val,
                     callbacks=[model_checkpoint_callback],
                     class_weight=class_weights_dict)

    # Load the best model weights
    model.load_weights(checkpoint_filepath)

    if task_mode == TaskMode.CLASSIFICATION:
        preds = np.argmax(model.predict(val), axis=1)
    elif task_mode == TaskMode.REGRESSION:
        preds = np.round(model.predict(val), decimals=1).flatten()
    else:
        raise KeyError(f"Task mode {task_mode} not recognised.")

    true_labels = np.concatenate([y for x, y in test], axis=0)

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

    return hist, correct / len(preds), preds, true_labels


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

    hist = model.fit(train, epochs=pretrain_epochs, verbose=0, validation_data=val, callbacks=[early_stopping])

    # Save the pre-trained model
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model.save(os.path.join(save_dir, model_name))
    print(f"Pre-trained model saved as {model_name}.")

    net.num_classes = num_classes
    net.reset_dense_layers()

    return model


def run_cifar():
    """
    Finetune an Xcpetion model on differently sized subsets of the cifar dataset.
    """
    merged_df = pd.DataFrame(columns=['Epochs', 'Validation Accuracy', 'Setting'])
    for size in [50, 500, 5000, 50000]:
        for i in range(5):
            model = network.XceptionNetwork(input_shape=(32, 32, 3), num_classes=10).model
            (cifar_train_x, cifar_train_y), (cifar_test_x, cifar_test_y) = dataloader.cifar_data()
            cifar_train_x = cifar_train_x[:size]
            cifar_train_y = cifar_train_y[:size]

            hist = model.fit(x=cifar_train_x, y=cifar_train_y, epochs=30, verbose=0,
                             validation_data=(cifar_test_x, cifar_test_y)).history
            epochs_range = range(1, len(hist["val_accuracy"]) + 1)
            val_accuracy = hist["val_accuracy"]

            # Create a DataFrame for the current run with a 'Setting' column
            run_df = pd.DataFrame({'Epochs': epochs_range, 'Validation Accuracy': val_accuracy})
            run_df['Setting'] = f"cifar_{size}"  # Add the 'Setting' column with the current setting name
            merged_df = pd.concat([merged_df, run_df], ignore_index=True)

    merged_df.to_csv("cifar_sizes.csv")
    print(merged_df)


def majority_vote(predictions, true_labels):
    voted_predictions = []

    for i in range(len(predictions[0])):
        votes = [prediction[i] for prediction in predictions]

        vote_counts = Counter(votes)
        max_count = max(vote_counts.values())

        # Find all votes that have the maximum count
        max_votes = [vote for vote, count in vote_counts.items() if count == max_count]

        # Randomly choose one of the tying votes
        chosen_vote = random.choice(max_votes)

        voted_predictions.append(chosen_vote)

    accuracy = np.mean(voted_predictions == true_labels)

    return voted_predictions, accuracy

def average_train(name, file, runs=5, epochs=20, augment=True, recombination_ratio=1.0, transfer=True,
                  classmode=ClassMode.STANDARD,
                  freeze=True, task_mode=TaskMode.CLASSIFICATION, transfer_source="xception", colour="rgb",
                  pretrain=False, feature=None, balance=True, class_weights=False, resize=512,
                  dense_layers=4, dense_size=64, lr=0.001, max_training=None, ensemble=False):
    """
    Perform training runs according to the given parameters and save the results.
    """
    start = time.time()
    # Initialize an empty DataFrame to store the merged data
    merged_df = pd.DataFrame(columns=['Epochs', 'Validation Accuracy', 'Setting'])

    full_accs = []
    for i in range(runs):
        folds = dataloader.folds(classmode=classmode, window_size=2, balance=balance, max_training=max_training)
        fold_accs = []
        true_labels = None
        all_preds = []
        solo_accs = []
        for fold_id, fold in folds.items():
            hist_object, accuracy, preds, true_labels = train_network(fold=fold, epochs=epochs, augment=augment,
                                                  transfer=transfer,
                                                  classmode=classmode, freeze=freeze, task_mode=task_mode,
                                                  transfer_source=transfer_source, colour=colour, pretrain=pretrain,
                                                  feature=feature,
                                                  class_weights=class_weights,
                                                  recombination_ratio=recombination_ratio, resize=(resize, resize),
                                                  dense_layers=dense_layers,
                                                  dense_size=dense_size, lr=lr)
            hist = hist_object.history
            all_preds.append(preds)
            solo_accs.append(accuracy)

            # single model accuracy
            if not ensemble:
                fold_accs.append(accuracy)

            if task_mode == TaskMode.CLASSIFICATION:

                # Extract the epoch and validation accuracy values
                epochs_range = list(range(1, len(hist["val_accuracy"]) + 1))
                epochs_range += epochs_range

                val_accuracy = hist["val_accuracy"]
                obo_val_accuracy = hist["val_obo_accuracy"]

                metrics = ["Validation Accuracy"] * len(val_accuracy) + ["Validation Off-By-One Accuracy"] * len(
                    obo_val_accuracy)
                values = val_accuracy + obo_val_accuracy

                # Create a DataFrame for the current run with a 'Setting' column
                run_df = pd.DataFrame({'Epochs': epochs_range, 'Value': values,
                                       'Metric': metrics})
                run_df['Setting'] = name  # Add the 'Setting' column with the current setting name
            elif task_mode == TaskMode.REGRESSION:
                # Extract the epoch and validation accuracy values
                epochs_range = list(range(1, len(hist["val_mean_absolute_error"]) + 1))
                epochs_range *= 4

                val_mae = hist["val_mean_absolute_error"]
                val_obo = hist["val_obo_accuracy_r"]
                val_obh = hist["val_obh_accuracy_r"]
                val_obt = hist["val_obt_accuracy_r"]

                values = val_mae + val_obo + val_obh + val_obt
                metrics = ["Validation MAE"] * len(val_mae) + ["Validation Off-By-One Accuracy"] * len(
                    val_obo) + len(val_obh) * ["Validation Off-By-Half Accuracy"] + len(val_obt) * [
                              "Validation Off-By-Tenth Accuracy"]

                # Create a DataFrame for the current run with a 'Setting' column
                run_df = pd.DataFrame(
                    {'Epochs': epochs_range, 'Value': values, 'Metric': metrics})
                run_df['Setting'] = name  # Add the 'Setting' column with the current setting name

            # Concatenate the current run's DataFrame to the merged DataFrame
            merged_df = pd.concat([merged_df, run_df], ignore_index=True)

            print(f"Completed fold {fold_id}")

        if ensemble:
            votes, ensemble_acc = majority_vote(all_preds, true_labels) # TODO: test this ensemble thing, if it works build a flag, otherwise remove it
            fold_accs.append(ensemble_acc)
            print(f"Ensemble accuracy: {ensemble_acc}, Average solo accuracy {np.mean(solo_accs)}")
        full_accs.extend(fold_accs)
        print(f"Average accuracy of folds {np.mean(fold_accs)}")

    add_runs(merged_df, file)
    print(f"Finished {runs}  \"{name}\" runs in {(time.time() - start) / 60} minutes")
    print(
        f"Evaluation of final models for setting {name}: Mean {np.mean(full_accs)}, Standard Deviation {np.std(full_accs)}")

    return np.mean(full_accs), np.std(full_accs), name


def add_runs(run_results, file):
    """
    Add the given run results to the given file.
    """
    if os.path.exists(file):
        existing_data = pd.read_csv(file)
    else:
        existing_data = pd.DataFrame()

    new_data = pd.concat([existing_data, run_results], ignore_index=True)
    new_data.to_csv(file, index=False)
