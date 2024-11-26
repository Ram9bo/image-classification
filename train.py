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

import dataloader
import network
from enums import ClassMode

print('Available GPUs', tf.config.list_physical_devices('GPU'))


def train_network(data_split, tested_dict, epochs=10, transfer=True,
                  classmode=ClassMode.STANDARD,
                  transfer_source="xception", colour="rgb", class_weights=False, recombination_ratio=1.0,
                  resize=(256, 256),
                  dense_layers=6, dense_size=128, lr=0.001, rotate=True, flip=True, brightness_delta=0.0, batch_size=2,
                  dropout=0.1, unfreeze=0, verbose=1):
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

    for l in data_split:
        for file in data_split[l]["test"]:
            if file not in tested_dict:
                tested_dict[file] = {"class": l, "n": 0, "f": 0, "obo": 0, "c": 0}

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

    hist = model.fit(train, epochs=epochs, verbose=verbose, validation_data=test,
                     callbacks=[],
                     class_weight=class_weights_dict)

    for l in data_split:
        for file in data_split[l]["test"]:
            img = dataloader.load_image(file, color_mode=colour, resize=resize)
            img = np.array(img)
            # print(img)
            # print(img[0].shape)
            pred = np.argmax(model.predict(img, verbose=0)[0])
            # print(pred)
            tested_dict[file]["n"] += 1
            diff = abs(pred - l)
            if diff > 1:
                tested_dict[file]["f"] += 1
            elif diff == 1:
                tested_dict[file]["obo"] += 1
            else:
                tested_dict[file]["c"] += 1

    return eval_test(hist, model, test, verbose)


def eval_test(hist, model, test, verbose):
    true_labels = np.concatenate([y for x, y in test], axis=0)

    preds = np.argmax(model.predict(test, verbose=verbose), axis=1)

    report = classification_report(true_labels, preds)
    report_dict = classification_report(true_labels, preds, output_dict=True)
    if verbose:
        print("Classification Report:\n", report)
    correct, obo, incorrect, obo_high, obo_low, high, low = evaluate(preds, true_labels)
    if verbose:
        print(f"Accuracy: {correct}, Off-By-One: {obo}, Error Rate: {incorrect}")
    return hist, correct, obo, preds, true_labels, report_dict["weighted avg"][
        "f1-score"], model, obo_high, obo_low, high, low


def evaluate(preds, true_labels):
    """
    Evaluates the given predictions according to the given labels. Returns accuracy, off-by-one accuracy and error rate.
    """

    correct, obo, incorrect, obo_high, obo_low, high, low = 0, 0, 0, 0, 0, 0, 0
    for i in range(len(preds)):
        t = true_labels[i]
        p = preds[i]
        if t == p:
            correct += 1
        elif abs(t - p) <= 1:
            obo += 1
            if p < t:
                obo_low += 1
            elif p > t:
                obo_high += 1
        else:
            incorrect += 1
            if p < t:
                low += 1
            elif p > t:
                high += 1
    n = len(preds)

    return protected_div(correct, n), \
           protected_div(obo, n), \
           protected_div(incorrect, n), \
           protected_div(obo_high, obo), \
           protected_div(obo_low, obo), \
           protected_div(high, incorrect), \
           protected_div(low, incorrect)


def protected_div(e, d):
    return e / d if d > 0 else 0


def average_train(name, file, runs=5, epochs=20, recombination_ratio=1.0, transfer=True,
                  classmode=ClassMode.STANDARD, transfer_source="xception", colour="rgb", balance=True,
                  class_weights=False, resize=256, dense_layers=4, dense_size=64, lr=0.001,
                  max_training=None, rotate=True, flip=False, brightness_delta=0.0, batch_size=32,
                  dropout=0.0, unfreeze=0, checkpoint_select=None, patches=None):
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
    full_obo_high = []
    full_obo_low = []
    full_high = []
    full_low = []
    all_preds = []
    all_labels = []
    best_f1 = 0.0
    best_obo = 0.0
    best_accuracy = 0.0

    history_list = []
    tested_dict = {}

    for i in range(runs):
        splits = dataloader.split(classmode=classmode, balance=balance,
                                  max_training=max_training)
        fold_accs = []
        fold_obo = []
        fold_f1 = []
        fold_obo_high = []
        fold_obo_low = []
        fold_low = []
        fold_high = []
        for fold_id, split in splits.items():
            print(f"Running {name} {i}")
            hist_object, accuracy, obo, preds, true_labels, f1, model, obo_high, obo_low, high, low = train_network(
                data_split=split, tested_dict=tested_dict, epochs=epochs,
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
                dropout=dropout,
                unfreeze=unfreeze, verbose=0)

            for j in range(epochs):
                history_list.append({
                    'run': i,
                    'epoch': j + 1,
                    'loss': hist_object.history['loss'][j],
                    'val_loss': hist_object.history['val_loss'][j],
                    'accuracy': hist_object.history['accuracy'][j],
                    'val_accuracy': hist_object.history['val_accuracy'][j]
                })

            # Check if the current F1 score is the best so far
            if accuracy > best_accuracy or (accuracy == best_accuracy and obo > best_obo) or (accuracy == best_accuracy and obo == best_obo and float(f1) > best_f1):
                print("Achieved a better accuracy, saving new confusion matrix.")
                best_accuracy = accuracy
                best_obo = obo
                best_f1 = float(f1)

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

                model.save("best_model.keras")

            hist = hist_object.history
            fold_accs.append(accuracy)
            fold_obo.append(obo + accuracy)
            fold_f1.append(f1)
            fold_obo_high.append(obo_high)
            fold_obo_low.append(obo_low)
            fold_high.append(high)
            fold_low.append(low)
            all_preds.extend(preds)
            all_labels.extend(true_labels)
        full_accs.extend(fold_accs)
        full_obo.extend(fold_obo)
        full_f1.extend(fold_f1)
        full_obo_high.extend(fold_obo_high)
        full_obo_low.extend(fold_obo_low)
        full_low.extend(fold_low)
        full_high.extend(fold_high)
        print(f"Average accuracy of folds {np.mean(fold_accs)}")
        print(f"Average off-by-one accuracy of folds {np.mean(fold_obo)}")
        print(f"Average F1 score of folds {np.mean(fold_f1)}")

    history_df = pd.DataFrame(history_list)
    history_df.to_csv(f"history-{name}.csv", index=False)

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
    np.save(f"all-labels-{name}", np.array(all_labels))
    np.save(f"all-preds-{name}", np.array(all_preds))

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

    tups = []
    for k, v in tested_dict.items():
        tups.append((protected_div(v["f"], v["n"]), protected_div(v["obo"], v["n"]), v["n"], k))
    tups.sort(key=lambda x: x[0], reverse=True)
    print(tups)

    with open('data/tested list.txt', "w") as f:
        f.write('\n'.join(str(i) for i in tups))

    with open('data/tested_dict.json', 'w') as f:
        json.dump(tested_dict, f)

    print("Metrics achieved by best model")
    print("Accuracy: ", best_accuracy)
    print("Obo: ", best_obo)
    print("F1: ", best_f1)

    return np.mean(full_accs), np.std(full_accs), name, np.mean(full_obo), np.std(full_obo), np.mean(full_f1), np.std(
        full_f1), np.mean(full_obo_high), np.std(full_obo_high), np.mean(full_obo_low), np.std(full_obo_low), np.mean(
        full_low), np.std(full_low), np.mean(full_high), np.std(full_high)


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
