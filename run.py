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
    runs = 10
    epochs = 10

    try:
        # Get the best hyperparameters
        best_hyperparameters = tuner.get_best()

        print(best_hyperparameters)
    except IndexError:
        best_hyperparameters = {}

    best_hyperparameters["epochs"] = epochs
    best_hyperparameters["batch_size"] = 64
    best_hyperparameters["fold_size"] = 3

    accs = {}

    acc, std, setting, obo, obosd = average_train("Standard", file, runs=runs, **best_hyperparameters)

    accs[setting] = {"mean": acc, "std": std, "obo": obo, "obo_sd": obosd}

    acc, std, setting, obo, obosd = average_train("Compressed Both", file, runs=runs, **best_hyperparameters,
                                                  classmode=ClassMode.COMPRESSED_BOTH)

    accs[setting] = {"mean": acc, "std": std, "obo": obo, "obo_sd": obosd}

    acc, std, setting, obo, obosd = average_train("Compressed Start", file, runs=runs, **best_hyperparameters,
                                                  classmode=ClassMode.COMPRESSED_START)

    accs[setting] = {"mean": acc, "std": std, "obo": obo, "obo_sd": obosd}

    acc, std, setting, obo, obosd = average_train("Compressed End", file, runs=runs, **best_hyperparameters,
                                                  classmode=ClassMode.COMPRESSED_END)

    accs[setting] = {"mean": acc, "std": std, "obo": obo, "obo_sd": obosd}

    with open(util.data_path(f"{name}.json"), "w") as json_file:
        json.dump(accs, json_file)


if __name__ == "__main__":
    ablation()

# TODO: once all the data is in, do a random test-set splitoff, then do the tuning on the remaining data, and then do the
#   training/eval, to ensure that the test data is not used for tuning OR maybe wrap it all in another crossfold,
#   so the test set is also swapped and then do tuning for each run (that is of course an issue with runtime)
# TODO: try a patch-based approach after all (probably 64x64 or so)
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
