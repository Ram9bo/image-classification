"""
Model training.
"""

import keras_tuner as kt
import numpy as np
import tensorflow as tf

import dataloader
import train
from enums import ClassMode

print('Available GPUs', tf.config.list_physical_devices('GPU'))


# Define a function to extract labels from dataset elements
def extract_labels(features, labels):
    return labels


class CustomTuner(kt.BayesianOptimization):

    def run_trial(self, trial, *args, **kwargs):
        hp = trial.hyperparameters

        resize = hp.Choice("resize", [512, 256, 128, 72], default=128)
        epochs = hp.Int("epochs", min_value=1, max_value=20, step=1, default=5)
        augment = hp.Choice("augment", [True, False], default=True)
        freeze = hp.Choice("freeze", [True, False], default=True)
        balance = hp.Choice("balance", [True, False], default=True)
        class_weights = hp.Choice("class_weights", [False, True], default=True)
        recombination_ratio = hp.Float("recombination_ratio", min_value=0.0, max_value=5.0, default=5.0)
        dense_layers = hp.Int("dense_layers", min_value=1, max_value=10, default=2)
        dense_size = hp.Choice("dense_size", [8, 16, 32, 64, 128, 256, 512], default=64)
        colour = hp.Choice("colour", ["gray_scale", "rgb"], default="gray_scale")
        learning_rate = hp.Choice("lr", [0.01, 0.001, 0.0001, 0.00001, 0.005, 0.0005, 0.00005],
                                  default=0.001)
        rotate = hp.Choice("rotate", [True, False], default=True)
        flip = hp.Choice("flip", [True, False], default=True)
        brightness_delta = hp.Choice("brightness_delta", [0.0, 0.05, 0.1, 0.2, 0.5], default=0)
        batch_size = hp.Int("batch_size", min_value=1, max_value=64, default=2)

        results = []

        for _ in range(self.executions_per_trial):
            folds = dataloader.folds(classmode=ClassMode.STANDARD, window_size=5, balance=balance)
            for fold_id, fold in folds.items():
                try:
                    hist, acc = train.train_network(fold=fold, epochs=epochs,
                                                    augment=augment,
                                                    freeze=freeze,
                                                    class_weights=class_weights,
                                                    recombination_ratio=recombination_ratio,
                                                    resize=(resize, resize),
                                                    dense_layers=dense_layers,
                                                    dense_size=dense_size,
                                                    colour=colour,
                                                    lr=learning_rate,
                                                    transfer_source="xception", rotate=rotate, flip=flip,
                                                    brightness_delta=brightness_delta, batch_size=batch_size)
                    results.append(acc)
                except Exception as e:
                    print(e)
                    return 1000

        if len(results) > 1:
            return (1 - np.mean(results)) * 100 + np.std(results) + 0.1 / batch_size


if __name__ == "__main__":
    tuner = CustomTuner(
        max_trials=300,
        overwrite=False,
        directory="tuning",
        project_name="folded-std",
        executions_per_trial=5
    )

    tuner.search()

    # Get the best hyperparameters
    best_hyperparameters = tuner.get_best_hyperparameters()[0].values
    print(best_hyperparameters)
    # Get the best trial and its performance
    best_trial = tuner.oracle.get_best_trials(1)[0]
    best_score = best_trial.score
    print(best_score, best_trial.trial_id)
