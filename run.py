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
    name = "conf"
    file = util.data_path(f"{name}.csv")

    pd.DataFrame().to_csv(file)
    runs = 5
    epochs = 20

    try:
        # Get the best hyperparameters
        best_hyperparameters = tuner.get_best()
        best_hyperparameters.pop('patches', None)  # delete this obsolete parameter

        print(best_hyperparameters)
    except IndexError:
        best_hyperparameters = {}

    best_hyperparameters["epochs"] = epochs
    best_hyperparameters["batch_size"] = 64

    print(f"Doing {runs} runs per setting.")

    accs = {}

    acc, std, setting, obo, obosd, f1, f1sd = average_train("Standard", file, runs=runs, **best_hyperparameters,
                                                            classmode=ClassMode.STANDARD)

    accs[setting] = {"mean": acc, "std": std, "obo": obo, "obo_sd": obosd, "f1": f1, "f1_sd": f1sd}

    acc, std, setting, obo, obosd, f1, f1sd = average_train("Compressed Start", file, runs=runs, **best_hyperparameters,
                                                            classmode=ClassMode.COMPRESSED_START)

    accs[setting] = {"mean": acc, "std": std, "obo": obo, "obo_sd": obosd, "f1": f1, "f1_sd": f1sd}

    acc, std, setting, obo, obosd, f1, f1sd = average_train("Compressed End", file, runs=runs, **best_hyperparameters,
                                                            classmode=ClassMode.COMPRESSED_END)

    accs[setting] = {"mean": acc, "std": std, "obo": obo, "obo_sd": obosd, "f1": f1, "f1_sd": f1sd}

    acc, std, setting, obo, obosd, f1, f1sd = average_train("Compressed Both", file, runs=runs, **best_hyperparameters,
                                                            classmode=ClassMode.COMPRESSED_BOTH)

    accs[setting] = {"mean": acc, "std": std, "obo": obo, "obo_sd": obosd, "f1": f1, "f1_sd": f1sd}

    with open(util.data_path(f"{name}.json"), "w") as json_file:
        json.dump(accs, json_file)


if __name__ == "__main__":
    ablation()
# TODO: do a run to generate a final model for in the application (do x runs on standard classmode and save the model with the highest test f1/acc)
# TODO: make sure the final hyperparameters are included in the repo for default use without further tuning
# TODO: write a clickable script that performs inference on the images in some folder, include an instruction txt and a json config
#   config should include things like image directory, maybe model paths, maybe settings like ensemble/solo
