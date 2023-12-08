"""
Model training.
"""
import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import ModelCheckpoint

import dataloader
import network
from enums import ClassMode

print('Available GPUs', tf.config.list_physical_devices('GPU'))


def extract_labels(features, labels):
    return labels


def train_network(fold, epochs=10, transfer=True,
                  classmode=ClassMode.STANDARD,
                  transfer_source="xception", colour="rgb", class_weights=False, recombination_ratio=1.0,
                  resize=(256, 256),
                  dense_layers=6, dense_size=128, lr=0.001, rotate=True, flip=True, brightness_delta=0.0, batch_size=2,
                  dropout=0.0, unfreeze=0):
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

    preds = np.argmax(model.predict(test), axis=1)
    true_labels = np.concatenate([y for x, y in test], axis=0)

    report = classification_report(true_labels, preds)
    report_dict = classification_report(true_labels, preds, output_dict=True)
    print("Classification Report:\n", report)

    correct, obo, incorrect = evaluate(preds, true_labels)
    print(f"Accuracy: {correct}, Off-By-One: {obo}, Error Rate: {incorrect}")

    return hist, correct, preds, true_labels, report_dict["weighted avg"]["f1-score"]


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
                  classmode=ClassMode.STANDARD, transfer_source="xception", colour="rgb", pretrain=False,
                  balance=True, class_weights=False, resize=256, dense_layers=4, dense_size=64, lr=0.001,
                  max_training=None, window_size=2, rotate=True, flip=False, brightness_delta=0.0, batch_size=32,
                  dropout=0.0, unfreeze=0):
    """
    Perform training runs according to the given parameters and save the results.
    """
    start = time.time()
    # Initialize an empty DataFrame to store the merged data
    merged_df = pd.DataFrame(columns=['Epochs', 'Validation Accuracy', 'Setting'])

    full_accs = []
    for i in range(runs):
        folds = dataloader.folds(classmode=classmode, window_size=window_size, balance=balance,
                                 max_training=max_training)
        fold_accs = []
        for fold_id, fold in folds.items():
            hist_object, accuracy, preds, true_labels, f1 = train_network(fold=fold, epochs=epochs, transfer=transfer,
                                                                          classmode=classmode,
                                                                          transfer_source=transfer_source,
                                                                          colour=colour,
                                                                          class_weights=class_weights,
                                                                          recombination_ratio=recombination_ratio,
                                                                          resize=(resize, resize),
                                                                          dense_layers=dense_layers,
                                                                          dense_size=dense_size, lr=lr, rotate=rotate,
                                                                          flip=flip,
                                                                          brightness_delta=brightness_delta,
                                                                          batch_size=batch_size,
                                                                          dropout=dropout, unfreeze=unfreeze)
            hist = hist_object.history
            fold_accs.append(accuracy)

            # Extract the epoch and validation accuracy values
            epochs_range = list(range(1, epochs + 1))

            val_accuracy = hist["val_accuracy"]

            metrics = ["Validation Accuracy"] * len(val_accuracy)
            values = val_accuracy

            # Create a DataFrame for the current run with a 'Setting' column
            run_df = pd.DataFrame({'Epochs': epochs_range, 'Value': values,
                                   'Metric': metrics})
            run_df['Setting'] = name  # Add the 'Setting' column with the current setting name

            # Concatenate the current run's DataFrame to the merged DataFrame
            merged_df = pd.concat([merged_df, run_df], ignore_index=True)

            print(f"Completed fold {fold_id}")
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
