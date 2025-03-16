import os
import shutil

DATA_FOLDER = "data/generated/"


def data_path(file):
    """
    Convert given file to proper data folder location.
    """
    return DATA_FOLDER + file


def create_results_folder(name, setting):
    # Construct the path
    path = f"{DATA_FOLDER}/{name}/{setting}"

    # If the folder exists, clear it
    if os.path.exists(path):
        shutil.rmtree(path)  # Removes the folder and all its contents

    # Recreate the empty folder
    os.makedirs(path)
    print(f"Folder prepared: {path}")

    return path


def results_path(name, setting, file):
    return f"{DATA_FOLDER}/{name}/{setting}/{file}"


def group_path(name, file):
    return f"{DATA_FOLDER}/{name}/{file}"
