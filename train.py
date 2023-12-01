"""
Model training.
"""
import os
import random
import time
from collections import Counter

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import ModelCheckpoint

import dataloader
import network
from enums import ClassMode

print('Available GPUs', tf.config.list_physical_devices('GPU'))


# Define a function to extract labels from dataset elements
def extract_labels(features, labels):
    return labels


def train_network(fold, epochs=10, transfer=True,
                  classmode=ClassMode.STANDARD,
                  freeze=True, transfer_source="xception", colour="rgb", pretrain=False,
                  class_weights=False, recombination_ratio=1.0, resize=(256, 256),
                  dense_layers=6, dense_size=128, lr=0.001, rotate=True, flip=True, brightness_delta=0, batch_size=2,
                  dropout=0.0):
    num_classes = 6
    if classmode == ClassMode.COMPRESSED_START or classmode == ClassMode.COMPRESSED_END:
        num_classes = 5
    elif classmode == ClassMode.COMPRESSED_BOTH:
        num_classes = 4

    print(f"Predicting {num_classes} classes")

    input_shape = network.INPUT_SHAPE
    if resize is not None:
        width, height = resize
        channels = 3
        input_shape = width, height, channels

    if transfer:
        if transfer_source == "xception":
            net = network.XceptionNetwork(num_classes=num_classes, freeze=freeze,
                                          input_shape=input_shape, dense_layers=dense_layers, dense_size=dense_size,
                                          lr=lr, dropout=dropout)
            model = net.model
        elif transfer_source == "efficient":
            net = network.EfficientNetNetwork(num_classes=num_classes, freeze=freeze,
                                              input_shape=input_shape, dense_layers=dense_layers,
                                              dense_size=dense_size, lr=lr, dropout=dropout)
            model = net.model
        elif transfer_source == "vgg16":
            net = network.VGG16Network(num_classes=num_classes, freeze=freeze,
                                       input_shape=input_shape, dense_layers=dense_layers, dense_size=dense_size,
                                       lr=lr, dropout=dropout)
            model = net.model
        else:
            raise Exception("Transfer source not recognised")
    else:
        net = network.CustomResNetNetwork(num_classes=num_classes, input_shape=input_shape,
                                          dense_layers=dense_layers, dense_size=dense_size, lr=lr, dropout=dropout)
        model = net.model

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
    )

    hist = model.fit(train, epochs=epochs, verbose=1, validation_data=val,
                     callbacks=[model_checkpoint_callback],
                     class_weight=class_weights_dict)

    # Load the best model weights
    model.load_weights(checkpoint_filepath)
    os.remove(checkpoint_filepath)

    preds = np.argmax(model.predict(val), axis=1)

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


def average_train(name, file, runs=5, epochs=20, recombination_ratio=1.0, transfer=True,
                  classmode=ClassMode.STANDARD,
                  freeze=True, transfer_source="xception", colour="rgb",
                  pretrain=False, balance=True, class_weights=False, resize=512,
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
            hist_object, accuracy, preds, true_labels = train_network(fold=fold, epochs=epochs, transfer=transfer,
                                                                      classmode=classmode, freeze=freeze,
                                                                      transfer_source=transfer_source, colour=colour,
                                                                      pretrain=pretrain,
                                                                      class_weights=class_weights,
                                                                      recombination_ratio=recombination_ratio,
                                                                      resize=(resize, resize),
                                                                      dense_layers=dense_layers,
                                                                      dense_size=dense_size, lr=lr)
            hist = hist_object.history
            all_preds.append(preds)
            solo_accs.append(accuracy)

            # single model accuracy
            if not ensemble:
                fold_accs.append(accuracy)

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

            # Concatenate the current run's DataFrame to the merged DataFrame
            merged_df = pd.concat([merged_df, run_df], ignore_index=True)

            print(f"Completed fold {fold_id}")

        if ensemble:
            votes, ensemble_acc = majority_vote(all_preds, true_labels)
            # TODO: since we don't need to save the whole models, we can simply always do ensemble eval
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
