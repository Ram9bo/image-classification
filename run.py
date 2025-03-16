"""
Model training.
"""
import json
import os

import pandas as pd
import tensorflow as tf

import tuner
import util
from enums import ClassMode
from train import average_train

print('Available GPUs', tf.config.list_physical_devices('GPU'))


def ablation():
    group = "box-plots-alternate-settings"
    file = util.data_path(f"{group}.csv")

    pd.DataFrame().to_csv(file)
    runs = 50
    epochs = 20

    # Retrieve the stored training params or do a new HPO run
    if os.path.exists("params.json"):
        with open("params.json", "r") as json_file:
            best_hyperparameters = json.load(json_file)
    else:
        try:
            # Get the best hyperparameters
            best_hyperparameters = tuner.get_best()

            with open("params.json", "w") as json_file:
                json.dump(best_hyperparameters, json_file)

            print(best_hyperparameters)
        except IndexError:
            best_hyperparameters = {}

    best_hyperparameters["epochs"] = epochs
    best_hyperparameters["batch_size"] = 64

    print(f"Doing {runs} runs per setting.")

    accs = {}

    acc, std, setting, obo, obosd, f1, f1sd, obo_h, obo_hsd, obo_l, obo_lsd, low, low_sd, high, high_sd = average_train(
        group,"Standard", file, runs=runs, **best_hyperparameters,
        classmode=ClassMode.STANDARD)

    accs[setting] = {"mean": acc, "std": std, "obo": obo, "obo_sd": obosd, "f1": f1, "f1_sd": f1sd, "obo_h": obo_h,
                     "obo_h_sd": obo_hsd, "obo_l": obo_l, "obo_lsd": obo_lsd, "low": low, "low_sd": low_sd,
                     "high": high, "high_sd": high_sd}

    acc, std, setting, obo, obosd, f1, f1sd, obo_h, obo_hsd, obo_l, obo_lsd, low, low_sd, high, high_sd = average_train(
        group, "Compressed Start", file, runs=runs, **best_hyperparameters,
        classmode=ClassMode.COMPRESSED_START)

    accs[setting] = {"mean": acc, "std": std, "obo": obo, "obo_sd": obosd, "f1": f1, "f1_sd": f1sd, "obo_h": obo_h,
                     "obo_h_sd": obo_hsd, "obo_l": obo_l, "obo_lsd": obo_lsd, "low": low, "low_sd": low_sd,
                     "high": high, "high_sd": high_sd}
    acc, std, setting, obo, obosd, f1, f1sd, obo_h, obo_hsd, obo_l, obo_lsd, low, low_sd, high, high_sd = average_train(
        group, "Compressed End", file, runs=runs, **best_hyperparameters,
        classmode=ClassMode.COMPRESSED_END)

    accs[setting] = {"mean": acc, "std": std, "obo": obo, "obo_sd": obosd, "f1": f1, "f1_sd": f1sd, "obo_h": obo_h,
                     "obo_h_sd": obo_hsd, "obo_l": obo_l, "obo_lsd": obo_lsd, "low": low, "low_sd": low_sd,
                     "high": high, "high_sd": high_sd}
    acc, std, setting, obo, obosd, f1, f1sd, obo_h, obo_hsd, obo_l, obo_lsd, low, low_sd, high, high_sd = average_train(
        group, "Compressed Both", file, runs=runs, **best_hyperparameters,
        classmode=ClassMode.COMPRESSED_BOTH)

    accs[setting] = {"mean": acc, "std": std, "obo": obo, "obo_sd": obosd, "f1": f1, "f1_sd": f1sd, "obo_h": obo_h,
                     "obo_h_sd": obo_hsd, "obo_l": obo_l, "obo_lsd": obo_lsd, "low": low, "low_sd": low_sd,
                     "high": high, "high_sd": high_sd}
    with open(util.group_path(group, f"results.json"), "w") as json_file:
        json.dump(accs, json_file)

    print(accs)


if __name__ == "__main__":
    ablation()
