"""
Model training.
"""
import json

import pandas as pd
import tensorflow as tf

import tuner
import util
from enums import ClassMode
from train import average_train

print('Available GPUs', tf.config.list_physical_devices('GPU'))


def ablation():
    # Create DataFrames for different settings
    name = "classmode"
    file = util.data_path(f"{name}.csv")

    pd.DataFrame().to_csv(file)
    runs = 5
    epochs = 20

    # Get the best hyperparameters
    best_hyperparameters = tuner.get_best()
    best_hyperparameters["epochs"] = epochs

    print(best_hyperparameters)

    accs = {}

    acc, std, setting = average_train("Standard", file, runs=runs, **best_hyperparameters, classmode=ClassMode.STANDARD)

    accs[setting] = {"mean": acc, "std": std}

    acc, std, setting = average_train("Compressed Start", file, runs=runs, **best_hyperparameters,
                                      classmode=ClassMode.COMPRESSED_START)

    accs[setting] = {"mean": acc, "std": std}

    acc, std, setting = average_train("Compressed End", file, runs=runs, **best_hyperparameters,
                                      classmode=ClassMode.COMPRESSED_END)

    accs[setting] = {"mean": acc, "std": std}

    acc, std, setting = average_train("Compressed Both", file, runs=runs, **best_hyperparameters,
                                      classmode=ClassMode.COMPRESSED_BOTH)

    accs[setting] = {"mean": acc, "std": std}

    with open(util.data_path(f"{name}.json"), "w") as json_file:
        json.dump(accs, json_file)


if __name__ == "__main__":
    ablation()
