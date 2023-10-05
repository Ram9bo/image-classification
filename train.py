"""
Model training.
"""

import time

import numpy as np
import pandas as pd
import tensorflow as tf

import dataloader
import network
import os

print('Available GPUs', tf.config.list_physical_devices('GPU'))


# TODO: try regression, consider MAE and RMSE metrics
# TODO: do an in-depth confusion analysis
# TODO: class-wise precision, recall, F1

def train_network(epochs=10, augment=True, recombinations=10, transfer=False, classmode="standard", freeze=True):
    num_classes = 6
    if classmode == "halve":
        num_classes = num_classes // 2
    elif classmode == "compress":
        num_classes = num_classes - 2

    if transfer:
        model = network.xception(num_classes=num_classes, freeze=freeze)
    else:
        model = network.resnet(num_classes=num_classes)

    train, val = dataloader.all_data(augment=augment, recombinations=recombinations, classmode=classmode)

    # TODO: manually pretrain on a dataset other than imagenet (ideally the same sort of microscopy could also be in
    #  combination with imagenet)

    hist = model.fit(train, epochs=epochs, verbose=1, validation_data=val)
    preds = np.argmax(model.predict(val), axis=1)

    # Define a function to extract labels from dataset elements
    def extract_labels(features, labels):
        return labels

    # Use the map function to apply the extract_labels function and convert to NumPy array
    true_labels = np.array(list(val.map(extract_labels))).flatten()
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

    return hist


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


def average_train(name, runs=5, epochs=10, augment=True, recombinations=10, transfer=False, classmode="compress",
                  freeze=True):
    # Initialize an empty DataFrame to store the merged data
    merged_df = pd.DataFrame(columns=['Epochs', 'Validation Accuracy', 'Setting'])

    for i in range(runs):
        hist = train_network(epochs=epochs, augment=augment, recombinations=recombinations, transfer=transfer,
                             classmode=classmode, freeze=freeze).history
        # Extract the epoch and validation accuracy values
        epochs_range = range(1, len(hist["val_accuracy"]) + 1)
        val_accuracy = hist["val_accuracy"]
        obo_val_accuracy = hist["val_obo_accuracy"]

        # Create a DataFrame for the current run with a 'Setting' column
        run_df = pd.DataFrame({'Epochs': epochs_range, 'Validation Accuracy': val_accuracy,
                               'Validation Off-By-One Accuracy': obo_val_accuracy})
        run_df['Setting'] = name  # Add the 'Setting' column with the current setting name

        # Concatenate the current run's DataFrame to the merged DataFrame
        merged_df = pd.concat([merged_df, run_df], ignore_index=True)

    return merged_df


def ablation():
    # TODO: incorporate the add_runs function in this  so that intermediate results are saved (after each setting)
    # Create DataFrames for different settings
    runs = 5
    epochs = 30
    latest = time.time()
    t1 = average_train("No Recombination", runs=runs, epochs=epochs, augment=True, recombinations=0, transfer=True,
                       freeze=False, classmode="standard")
    print(f"Setting completed in {np.round(time.time() - latest, decimals=0)}s")
    latest = time.time()
    t2 = average_train("5 Recombinations", runs=runs, epochs=epochs, augment=True, recombinations=5,
                       transfer=True, freeze=False, classmode="standard")
    print(f"Setting completed in {np.round(time.time() - latest, decimals=0)}s")
    latest = time.time()
    t3 = average_train("10 Recombinations", runs=runs, epochs=epochs, augment=True, recombinations=10, transfer=True,
                       freeze=False, classmode="standard")
    print(f"Setting completed in {np.round(time.time() - latest, decimals=0)}s")
    latest = time.time()
    t4 = average_train("20 Recombinations", runs=runs, epochs=epochs, augment=True, recombinations=20,
                       transfer=True,
                       freeze=False, classmode="standard")
    print(f"Setting completed in {np.round(time.time() - latest, decimals=0)}s")
    # Merge the DataFrames into one
    merged_all = pd.concat([t1, t2, t3, t4], ignore_index=True)
    # Print the merged DataFrame
    print(merged_all)
    merged_all.to_csv("recombinations.csv")


def add_runs(run_results, file):
    if os.path.exists(file):
        existing_data = pd.read_csv(file)
    else:
        existing_data = pd.DataFrame()

    new_data = pd.concat([existing_data, run_results], ignore_index=True)
    new_data.to_csv(file)


h = average_train("100 recombinations", runs=5, epochs=30, augment=True, recombinations=100, transfer=True,
                  freeze=False, classmode='standard')
add_runs(h, "recombinations.csv")
# ablation()
# run_cifar()
