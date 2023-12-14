"""
Model training.
"""

import keras_tuner as kt
import numpy as np
import tensorflow as tf

import dataloader
import train
from enums import ClassMode
from tqdm import tqdm
print('Available GPUs', tf.config.list_physical_devices('GPU'))


# Define a function to extract labels from dataset elements
def extract_labels(features, labels):
    return labels


def get_best():
    tuner = CustomTuner(
        max_trials=100,
        overwrite=False,
        directory="tuning",
        project_name="acc",
        executions_per_trial=3
    )

    # Get the best hyperparameters
    best_hyperparameters = tuner.get_best_hyperparameters()[0].values
    return best_hyperparameters


class CustomTuner(kt.BayesianOptimization):

    def run_trial(self, trial, *args, **kwargs):
        hp = trial.hyperparameters

        resize = hp.Choice("resize", [512, 256, 128], default=128)
        unfreeze = hp.Int("unfreeze", min_value=0, max_value=10, default=0)
        balance = hp.Choice("balance", [True, False], default=True)
        class_weights = hp.Choice("class_weights", [False, True], default=True)
        recombination_ratio = hp.Choice("recombination_ratio", [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], default=1.0)
        dense_layers = hp.Int("dense_layers", min_value=0, max_value=10, default=2)
        dense_size = hp.Choice("dense_size", [8, 16, 32, 64, 128, 256, 512], default=64)
        colour = hp.Choice("colour", ["gray_scale", "rgb"], default="gray_scale")
        learning_rate = hp.Choice("lr", [0.001, 0.0001, 0.00001, 0.005, 0.0005, 0.00005],
                                  default=0.001)
        rotate = hp.Choice("rotate", [True, False], default=True)
        flip = hp.Choice("flip", [True, False], default=True)
        brightness_delta = hp.Choice("brightness_delta", [0.0, 0.05, 0.1, 0.2, 0.5], default=0)
        batch_size = 16
        dropout = hp.Float("dropout", min_value=0.0, max_value=0.3, step=0.05)
        fold_size = hp.Choice("fold_size", [1, 2, 3, 4, 5], default=3)
        checkpoint_select = hp.Choice("checkpoint_select", ["val_accuracy", "val_loss"], default="val_accuracy")

        results = []

        epochs = 10

        for _ in tqdm(range(self.executions_per_trial)):
            folds = dataloader.folds(classmode=ClassMode.STANDARD, window_size=fold_size, balance=balance)
            for fold_id, fold in folds.items():
                try:
                    hist, acc, obo, preds, true_labels, f1 = train.train_network(fold=fold, epochs=epochs,
                                                                                 class_weights=class_weights,
                                                                                 recombination_ratio=recombination_ratio,
                                                                                 resize=(resize, resize),
                                                                                 dense_layers=dense_layers,
                                                                                 dense_size=dense_size,
                                                                                 colour=colour,
                                                                                 lr=learning_rate,
                                                                                 transfer_source="xception",
                                                                                 rotate=rotate,
                                                                                 flip=flip,
                                                                                 brightness_delta=brightness_delta,
                                                                                 batch_size=batch_size,
                                                                                 dropout=dropout,
                                                                                 unfreeze=unfreeze,
                                                                                 checkpoint_select=checkpoint_select, verbose=1)
                    results.append(acc)
                except Exception as e:
                    print(e)
                    return 10000

        if len(results) > 1:
            return 1 - np.mean(results)


if __name__ == "__main__":
    tuner = CustomTuner(
        max_trials=300,
        overwrite=False,
        directory="tuning",
        project_name="acc",
        executions_per_trial=3,
    )

    tuner.search()

    # Get the best hyperparameters
    best_hyperparameters = tuner.get_best_hyperparameters()[0].values
    print(best_hyperparameters)
    # Get the best trial and its performance
    best_trial = tuner.oracle.get_best_trials(1)[0]
    best_score = best_trial.score
    print(best_score, best_trial.trial_id)
