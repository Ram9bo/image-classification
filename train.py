"""
Model training.
"""

import time

import numpy as np
import pandas as pd
import tensorflow as tf

import dataloader
import network

print('Available GPUs', tf.config.list_physical_devices('GPU'))


def train_network(epochs=10, augment=True, recombinations=10, transfer=False):
    # model = network.compile_model()
    if transfer:
        model = network.efficient_net()
    else:
        model = network.resnet()
    # model = network.vgg16()

    train, val = dataloader.all_data(augment=augment, recombinations=recombinations)

    # TODO: manually pretrain on a dataset other than imagenet (ideally the same sort of microscopy could also be in
    #  combination with imagenet)

    # history = model.fit(train_data, epochs=20, verbose=1, validation_data=test_data)
    # model.fit(x=cifar_train_x, y=cifar_train_y, epochs=10, verbose=1, validation_data=(cifar_test_x, cifar_test_y))
    return model.fit(train, epochs=epochs, verbose=0, validation_data=val)


# TODO: create averaged plots for cifar with different subset sizes of the training data (full test set can be used)
#  to illustrate accuracy gains on larger datasets and make estimates regarding the amount of extra data needed

# TODO: set up a BO-HPO experiment to optimize the architecture and hyperparameters

def run_cifar(name, subsize=50):
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


def average_train(name, runs=5, epochs=10, augment=True, recombinations=10, transfer=False):
    # Initialize an empty DataFrame to store the merged data
    merged_df = pd.DataFrame(columns=['Epochs', 'Validation Accuracy', 'Setting'])

    for i in range(runs):
        hist = train_network(epochs=epochs, augment=augment, recombinations=recombinations, transfer=transfer).history
        # Extract the epoch and validation accuracy values
        epochs_range = range(1, len(hist["val_accuracy"]) + 1)
        val_accuracy = hist["val_accuracy"]

        # Create a DataFrame for the current run with a 'Setting' column
        run_df = pd.DataFrame({'Epochs': epochs_range, 'Validation Accuracy': val_accuracy})
        run_df['Setting'] = name  # Add the 'Setting' column with the current setting name

        # Concatenate the current run's DataFrame to the merged DataFrame
        merged_df = pd.concat([merged_df, run_df], ignore_index=True)

    return merged_df


def ablation():
    global runs
    # Create DataFrames for different settings
    runs = 5
    epochs = 30
    latest = time.time()
    t1 = average_train("Initial Dataset", runs=runs, epochs=epochs, augment=False, recombinations=0, transfer=False)
    print(f"Setting completed in {np.round(time.time() - latest, decimals=0)}s")
    latest = time.time()
    t2 = average_train("Augmented Dataset", runs=runs, epochs=epochs, augment=True, recombinations=0, transfer=False)
    print(f"Setting completed in {np.round(time.time() - latest, decimals=0)}s")
    latest = time.time()
    t3 = average_train("Synthesized Dataset", runs=runs, epochs=epochs, augment=False, recombinations=10,
                       transfer=False)
    print(f"Setting completed in {np.round(time.time() - latest, decimals=0)}s")
    latest = time.time()
    t4 = average_train("Transfer Learning", runs=runs, epochs=epochs, augment=False, recombinations=0, transfer=True)
    print(f"Setting completed in {np.round(time.time() - latest, decimals=0)}s")
    latest = time.time()
    t5 = average_train("A + S", runs=runs, epochs=epochs, augment=True, recombinations=10, transfer=False)
    print(f"Setting completed in {np.round(time.time() - latest, decimals=0)}s")
    latest = time.time()
    t6 = average_train("A + T", runs=runs, epochs=epochs, augment=True, recombinations=0, transfer=True)
    print(f"Setting completed in {np.round(time.time() - latest, decimals=0)}s")
    latest = time.time()
    t7 = average_train("S + T", runs=runs, epochs=epochs, augment=False, recombinations=10, transfer=True)
    print(f"Setting completed in {np.round(time.time() - latest, decimals=0)}s")
    latest = time.time()
    t8 = average_train("A + S + T", runs=runs, epochs=epochs, augment=True, recombinations=10, transfer=True)
    print(f"Setting completed in {np.round(time.time() - latest, decimals=0)}s")
    latest = time.time()
    # TODO, more variations (possibly all 8)
    # Merge the DataFrames into one
    merged_all = pd.concat([t1, t2, t3, t4, t5, t6, t7, t8], ignore_index=True)
    # Print the merged DataFrame
    print(merged_all)
    merged_all.to_csv("ablation-efficient.csv")
    # test_and_train()
    # run_cifar()
    # TODO: do a 5-run average (50-50 data split) for: no data augment, only rotation, only flipping, full augmentation,
    #  in combination with CNN, ResNet
    # TODO: try ensemble models
    # TODO: once we have more data we can also train those models on separate parts of the data
    # TODO: maybe it could even work on differently augmented data sets


ablation()
# run_cifar("cifar", 50 * 1000)
