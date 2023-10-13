"""
Model training.
"""

import util
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

import dataloader
import network

print('Available GPUs', tf.config.list_physical_devices('GPU'))


# Define a function to extract labels from dataset elements
def extract_labels(features, labels):
    return labels


# TODO: ensemble model (based on different splits of the training data)

def train_network(conf_matrix_name, epochs=10, augment=True, recombinations=10, transfer=False, classmode="standard",
                  freeze=True,
                  task_mode="classification", transfer_source="xception", colour="rgb", pretrain=False):
    num_classes = 6
    if classmode == "halved":
        num_classes = num_classes // 2
    elif classmode == "compressed":
        num_classes = num_classes - 2

    if task_mode == "regression":
        num_classes = 1
    print(f"Predicting {num_classes} classes")

    if pretrain:
        model = pretrain_model(transfer=transfer, num_classes=num_classes, freeze=freeze, transfer_source=transfer_source)
    else:
        if transfer:
            if transfer_source == "xception":
                model = network.xception(num_classes=num_classes, freeze=freeze, task_mode=task_mode)
            elif transfer_source == "efficient":
                model = network.efficient_net(num_classes=num_classes, freeze=freeze, task_mode=task_mode)
            elif transfer_source == "vgg16":
                model = network.efficient_net(num_classes=num_classes, freeze=freeze, task_mode=task_mode)
            else:
                raise Exception("Transfer source not recognised")
        else:
            model = network.resnet(num_classes=num_classes, task_mode=task_mode)

    train, val = dataloader.all_data(augment=augment, recombinations=recombinations, classmode=classmode, colour=colour)

    # TODO: include both the presence and the strength of dropout in the HPO, also consider batch normalization, also switch input normalization (to 0-1) on/off

    # TODO: manually pretrain on a dataset other than imagenet (ideally the same sort of microscopy, could also be in
    #  combination with imagenet)

    hist = model.fit(train, epochs=epochs, verbose=1, validation_data=val)

    # TODO: after the fitting, the program for some reason initiates another training run with seemingly default parameters which then crashes due to undefined variables being used

    if task_mode == 'classification':
        preds = np.argmax(model.predict(val), axis=1)
    elif task_mode == 'regression':
        preds = np.round(model.predict(val), decimals=1).flatten()

    # Use the map function to apply the extract_labels function and convert to NumPy array
    true_labels = np.array(list(val.map(extract_labels))).flatten()
    print(preds)
    print(true_labels)

    if task_mode == "classification":
        conf_matrix = tf.math.confusion_matrix(true_labels, preds)  # / len(true_labels)

        plt.figure(figsize=(8, 6))
        sns.set(font_scale=1.2)  # Adjust font size
        sns.heatmap(conf_matrix, annot=True, cmap="Blues", cbar=False,
                    xticklabels=list(range(6)), yticklabels=list(range(6)))
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.savefig("confusion_matrix.png")
        plt.close()

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

    # Compute the confusion matrix
    confusion = tf.math.confusion_matrix(
        labels=true_labels,
        predictions=preds,
    )

    # Create a heatmap of the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', cbar=False)

    # Customize labels and title
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

    # Save the heatmap figure to an image file (e.g., PNG)
    plt.savefig(conf_matrix_name + ".png")
    plt.close()
    return hist


def pretrain_model(transfer=False, num_classes=6, freeze=True, task_mode="classification",
                   transfer_source="xception", data="nombacter"):
    pretrain_classes = 15

    if transfer:
        if transfer_source == "xception":
            model = network.xception(num_classes=pretrain_classes, freeze=freeze)
        elif transfer_source == "efficient":
            model = network.efficient_net(num_classes=pretrain_classes, freeze=freeze)
        elif transfer_source == "vgg16":
            model = network.efficient_net(num_classes=pretrain_classes, freeze=freeze)
        else:
            raise Exception("Transfer source not recognised")
    else:
        model = network.resnet(num_classes=pretrain_classes, task_mode=task_mode)

    train, val = dataloader.ssnombacter_data()
    hist = model.fit(train, epochs=10, verbose=1, validation_data=val)

    model.layers.pop()
    activation = 'softmax' if task_mode == 'classification' else None
    model.layers.append(tf.keras.layers.Dense(num_classes, activation=activation))

    # TODO: consider cutting off multiple/all dense layers after pretraining
    # TODO: tune the pretraining length (and architecture) a bit
    # TODO: consider unfreezing a larger part of the base model (either just during pretraining or during both stages)

    if task_mode == "classification":
        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=tf.keras.optimizers.Adam(),
            metrics=network.CLASSIFICATION_METRICS
        )
    elif task_mode == "regression":
        model.compile(
            loss='mean_squared_error',
            optimizer=tf.keras.optimizers.Adam(),
            metrics=network.REGRESSION_METRICS
        )

    return model


# TODO: set up a BO-HPO experiment to optimize the architecture and hyperparameters

def run_cifar():
    merged_df = pd.DataFrame(columns=['Epochs', 'Validation Accuracy', 'Setting'])
    for size in [50, 500, 5000, 50000]:
        for i in range(5):
            model = network.efficient_net(input_shape=(32, 32, 3), num_classes=10)
            (cifar_train_x, cifar_train_y), (cifar_test_x, cifar_test_y) = dataloader.cifar_data()
            cifar_train_x = cifar_train_x[:size]
            cifar_train_y = cifar_train_y[:size]

            hist = model.fit(x=cifar_train_x, y=cifar_train_y, epochs=30, verbose=1,
                             validation_data=(cifar_test_x, cifar_test_y)).history
            epochs_range = range(1, len(hist["val_accuracy"]) + 1)
            val_accuracy = hist["val_accuracy"]

            # Create a DataFrame for the current run with a 'Setting' column
            run_df = pd.DataFrame({'Epochs': epochs_range, 'Validation Accuracy': val_accuracy})
            run_df['Setting'] = f"cifar_{size}"  # Add the 'Setting' column with the current setting name
            merged_df = pd.concat([merged_df, run_df], ignore_index=True)

    merged_df.to_csv("cifar_sizes.csv")
    print(merged_df)


def average_train(name, file, runs=5, epochs=10, augment=True, recombinations=10, transfer=False, classmode="standard",
                  freeze=True, task_mode='classification', transfer_source="xception", colour="rgb", pretrain=False):
    start = time.time()
    # Initialize an empty DataFrame to store the merged data
    merged_df = pd.DataFrame(columns=['Epochs', 'Validation Accuracy', 'Setting'])

    for i in range(runs):
        hist = train_network(conf_matrix_name=name, epochs=epochs, augment=augment, recombinations=recombinations,
                             transfer=transfer,
                             classmode=classmode, freeze=freeze, task_mode=task_mode,
                             transfer_source=transfer_source, colour=colour, pretrain=pretrain).history

        if task_mode == "classification":

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
        elif task_mode == "regression":
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

    add_runs(merged_df, file)
    print(f"Finished {runs}  \"{name}\" runs in {(time.time() - start) / 60} minutes")


def ablation():
    # Create DataFrames for different settings
    file = util.data_path("classmode.csv")

    if not os.path.exists(file):
        pd.DataFrame().to_csv(file)
    runs = 1
    epochs = 5

    average_train("Standard", file, runs=runs, epochs=epochs, augment=True, recombinations=5, transfer=False,
                  freeze=False, classmode="standard", transfer_source="xception", task_mode='classification')

    average_train("Compress End", file, runs=runs, epochs=epochs, augment=True, recombinations=5, transfer=False,
                  freeze=False, classmode="compressed-end", transfer_source="xception", task_mode='classification')

    average_train("Compress Start", file, runs=runs, epochs=epochs, augment=True, recombinations=5, transfer=False,
                  freeze=False, classmode="compressed-start", transfer_source="xception", task_mode='classification')

    average_train("Compress Both", file, runs=runs, epochs=epochs, augment=True, recombinations=5, transfer=False,
                  freeze=False, classmode="compressed-both", transfer_source="xception", task_mode='classification')


def add_runs(run_results, file):
    if os.path.exists(file):
        existing_data = pd.read_csv(file)
    else:
        existing_data = pd.DataFrame()

    new_data = pd.concat([existing_data, run_results], ignore_index=True)
    new_data.to_csv(file, index=False)

    # TODO: something isn't entirely right, one unnamed column still shows up, might just want to manually take the desired columns here


ablation()
