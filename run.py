"""
Model training.
"""
import json
from enums import ClassMode
import pandas as pd
import tensorflow as tf

import tuner
import util
from train import average_train

print('Available GPUs', tf.config.list_physical_devices('GPU'))


def ablation():
    # Create DataFrames for different settings
    name = "conf"
    file = util.data_path(f"{name}.csv")

    pd.DataFrame().to_csv(file)
    runs = 5
    epochs = 20

    try:
        # Get the best hyperparameters
        best_hyperparameters = tuner.get_best()

        print(best_hyperparameters)
    except IndexError:
        best_hyperparameters = {}

    best_hyperparameters["epochs"] = epochs
    best_hyperparameters["batch_size"] = 64

    print(f"Doing {runs} runs per setting.")

    accs = {}

    acc, std, setting, obo, obosd, f1, f1sd = average_train("Standard", file, runs=runs, **best_hyperparameters, classmode=ClassMode.STANDARD)

    accs[setting] = {"mean": acc, "std": std, "obo": obo, "obo_sd": obosd, "f1": f1, "f1_sd": f1sd}

    acc, std, setting, obo, obosd, f1, f1sd = average_train("Compressed Start", file, runs=runs, **best_hyperparameters, classmode=ClassMode.COMPRESSED_START)

    accs[setting] = {"mean": acc, "std": std, "obo": obo, "obo_sd": obosd, "f1": f1, "f1_sd": f1sd}

    acc, std, setting, obo, obosd, f1, f1sd = average_train("Compressed End", file, runs=runs, **best_hyperparameters, classmode=ClassMode.COMPRESSED_END)

    accs[setting] = {"mean": acc, "std": std, "obo": obo, "obo_sd": obosd, "f1": f1, "f1_sd": f1sd}

    acc, std, setting, obo, obosd, f1, f1sd = average_train("Compressed Both", file, runs=runs, **best_hyperparameters, classmode=ClassMode.COMPRESSED_BOTH)

    accs[setting] = {"mean": acc, "std": std, "obo": obo, "obo_sd": obosd, "f1": f1, "f1_sd": f1sd}

    with open(util.data_path(f"{name}.json"), "w") as json_file:
        json.dump(accs, json_file)


if __name__ == "__main__":
    ablation()
# TODO: do a run to generate a final model for in the application (do x runs on standard classmode and save the model with the highest test f1/acc)
# TODO: make sure the final hyperparameters are included in the repo for default use without further tuning
# TODO: big cleanup, get rid of everything that is not used in the training of the final model or the inference functionality
# TODO: write a clickable script that performs inference on the images in some folder, include an instruction txt and a json config
#   config should include things like image directory, maybe model paths, maybe settings like ensemble/solo
# TODO: provide a streamlined way to retrain the model(s) based on the data in a configurable directory
# TODO: provide a streamlined way to run additional tuning based on the data in a configurable directory
# TODO: provide a streamlined way to visualize the evaluation of models at a given/configurable path
# TODO: logging cleanup, at least get rid of irrelevant prints, make sure everything printed is intuitive and clear, also consider using
#   logging with different levels (logging module)
# TODO: comment/documentation cleanup, both adding where needed and removing where obsolete (or rudimentary chatgpt comments)
# TODO: do a thorough check on file paths and such
# TODO: if disk size or RAM consumption turns out to be an issue, we can look into pruning
#   https://www.dlology.com/blog/how-to-compress-your-keras-model-x5-smaller-with-tensorflow-model-optimization/
# TODO: If I can find the time, try a multi-resolution ensemble
