"""
Model training.
"""
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import ModelCheckpoint

import dataloader
import network
from enums import ClassMode

print('Available GPUs', tf.config.list_physical_devices('GPU'))


def extract_labels(features, labels):
    return labels


def train_network(data_split, epochs=10, transfer=True,
                  classmode=ClassMode.STANDARD,
                  transfer_source="xception", colour="rgb", class_weights=False, recombination_ratio=1.0,
                  resize=(256, 256),
                  dense_layers=6, dense_size=128, lr=0.001, rotate=True, flip=True, brightness_delta=0.0, batch_size=2,
                  dropout=0.1, unfreeze=0, checkpoint_select="val_accuracy", verbose=1):
    num_classes = 6
    if classmode == ClassMode.COMPRESSED_START or classmode == ClassMode.COMPRESSED_END:
        num_classes = 5
    elif classmode == ClassMode.COMPRESSED_BOTH:
        num_classes = 4

    print(f"Predicting {num_classes} classes")

    width, height, channels = network.INPUT_SHAPE
    if resize is not None:
        width, height = resize
    input_shape = width, height, channels

    if transfer:
        if transfer_source == "xception":
            net = network.XceptionNetwork(num_classes=num_classes,
                                          input_shape=input_shape, dense_layers=dense_layers, dense_size=dense_size,
                                          lr=lr, dropout=dropout, unfreeze=unfreeze)
            model = net.model
        elif transfer_source == "vgg16":
            net = network.VGG16Network(num_classes=num_classes,
                                       input_shape=input_shape, dense_layers=dense_layers, dense_size=dense_size,
                                       lr=lr, dropout=dropout, unfreeze=unfreeze)
            model = net.model
        else:
            raise Exception("Transfer source not recognised")
    else:
        net = network.CustomResNetNetwork(num_classes=num_classes, input_shape=input_shape,
                                          dense_layers=dense_layers, dense_size=dense_size, lr=lr, dropout=dropout)
        model = net.model

    train, val, test = dataloader.split_to_data(data_split, color=colour, batch_size=batch_size, resize=resize,
                                                recombination_ratio=recombination_ratio, rotate=rotate, flip=flip,
                                                brightness_delta=brightness_delta, verbose=verbose)

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
        monitor=checkpoint_select,  # Monitor validation accuracy
        mode='max',  # 'max' means save the model when the monitored quantity is maximized
        save_best_only=True,  # Save only the best model
    )

    hist = model.fit(train, epochs=epochs, verbose=verbose, validation_data=val,
                     callbacks=[],
                     class_weight=class_weights_dict)

    # Load the best model weights
    # model.load_weights(checkpoint_filepath)

    return eval_test(hist, model, test, verbose, resize)


def eval_test(hist, model, test, verbose, resize):
    true_labels = np.concatenate([y for x, y in test], axis=0)

    preds = np.argmax(model.predict(test, verbose=verbose), axis=1)

    report = classification_report(true_labels, preds)
    report_dict = classification_report(true_labels, preds, output_dict=True)
    if verbose:
        print("Classification Report:\n", report)
    correct, obo, incorrect = evaluate(preds, true_labels)
    if verbose:
        print(f"Accuracy: {correct}, Off-By-One: {obo}, Error Rate: {incorrect}")
    return hist, correct, obo, preds, true_labels, report_dict["weighted avg"]["f1-score"]


def evaluate(preds, true_labels):
    """
    Evaluates the given predictions according to the given labels. Returns accuracy, off-by-one accuracy and error rate.
    """

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
    n = len(preds)
    return correct / n, obo / n, incorrect / n


def average_train(name, file, runs=5, epochs=20, recombination_ratio=1.0, transfer=True,
                  classmode=ClassMode.STANDARD, transfer_source="xception", colour="rgb", balance=True,
                  class_weights=False, resize=256, dense_layers=4, dense_size=64, lr=0.001,
                  max_training=None, rotate=True, flip=False, brightness_delta=0.0, batch_size=32,
                  dropout=0.0, unfreeze=0, checkpoint_select="val_accuracy"):
    """
    Perform training runs according to the given parameters and save the results.
    """

    with open('eval.json', "w") as json_file:
        json.dump({"acc": 0.0, "obo": 0.0}, json_file)

    start = time.time()
    # Initialize an empty DataFrame to store the merged data
    merged_df = pd.DataFrame(columns=['Epochs', 'Validation Accuracy', 'Setting'])

    full_accs = []
    full_obo = []
    full_f1 = []
    all_preds = []
    all_labels = []
    best_f1 = 0.0
    best_obo = 0.0
    best_accuracy = 0.0

    for i in range(runs):
        splits = dataloader.split(classmode=classmode, balance=balance,
                                  max_training=max_training)
        fold_accs = []
        fold_obo = []
        fold_f1 = []
        for fold_id, split in splits.items():
            hist_object, accuracy, obo, preds, true_labels, f1 = train_network(data_split=split, epochs=epochs,
                                                                               transfer=transfer,
                                                                               classmode=classmode,
                                                                               transfer_source=transfer_source,
                                                                               colour=colour,
                                                                               class_weights=class_weights,
                                                                               recombination_ratio=recombination_ratio,
                                                                               resize=(resize, resize),
                                                                               dense_layers=dense_layers,
                                                                               dense_size=dense_size, lr=lr,
                                                                               rotate=rotate,
                                                                               flip=flip,
                                                                               brightness_delta=brightness_delta,
                                                                               batch_size=batch_size,
                                                                               dropout=dropout, unfreeze=unfreeze,
                                                                               checkpoint_select=checkpoint_select)

            # Check if the current F1 score is the best so far
            if accuracy > best_accuracy:
                print("Achieved a better accuracy, saving new confusion matrix.")
                best_accuracy = accuracy

                # Generate and save confusion matrix
                cm = confusion_matrix(true_labels, preds)
                plt.figure(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[f"Class {i}" for i in range(len(cm))],
                            yticklabels=[f"Class {i}" for i in range(len(cm))])
                plt.title(f'Confusion Matrix - Single Model')
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.savefig(f'confusion_matrix_{name}.png')
                plt.close()

            hist = hist_object.history
            fold_accs.append(accuracy)
            fold_obo.append(obo + accuracy)
            fold_f1.append(f1)
            all_preds.extend(preds)
            all_labels.extend(true_labels)
        full_accs.extend(fold_accs)
        full_obo.extend(fold_obo)
        full_f1.extend(fold_f1)
        print(f"Average accuracy of folds {np.mean(fold_accs)}")
        print(f"Average off-by-one accuracy of folds {np.mean(fold_obo)}")
        print(f"Average F1 score of folds {np.mean(fold_f1)}")

    add_runs(merged_df, file)
    print(f"Finished {runs}  \"{name}\" runs in {(time.time() - start) / 60} minutes")
    print(
        f"Accuracy of final models for setting {name}: Mean {np.mean(full_accs)}, Standard Deviation {np.std(full_accs)}")
    print(
        f"Off-by-one accuracy of final models for setting {name}: Mean {np.mean(full_obo)}, Standard Deviation {np.std(full_obo)}")
    print(
        f"F1 score of final models for setting {name}: Mean {np.mean(full_f1)}, Standard Deviation {np.std(full_f1)}")

    full_report = classification_report(all_labels, all_preds)
    print("Classification Report:\n", full_report)

    # Generate and save normalized confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=[f"Class {i}" for i in range(len(cm))],
                yticklabels=[f"Class {i}" for i in range(len(cm))])
    plt.title(f'Normalized Confusion Matrix - Averaged')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f'normalized_confusion_matrix_{name}_full.png')
    plt.close()

    return np.mean(full_accs), np.std(full_accs), name, np.mean(full_obo), np.std(full_obo), np.mean(full_f1), np.std(
        full_f1)


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
