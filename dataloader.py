"""
Data loading and preparation.
"""
import os
import random
import re

import numpy as np
import tensorflow as tf
from PIL import Image

from enums import ClassMode

### Root data folder
BASE_DATA_DIR = "data/all_images"


def augment_data(train, batch_size, rotate=True, flip=True, brightness_delta=0.2, shuffle=True):
    """
    Augment the given dataset according to the given parameters and shuffle the resulting dataset.
    """
    final = train

    if rotate:
        # Apply rotations of 90, 180, and 270 degrees
        for r in [1, 2, 3]:
            final = final.concatenate(train.map(lambda x, y: (tf.image.rot90(x, k=r), y)))

    if flip:
        # Apply horizontal and vertical flipping
        final = final.concatenate(train.map(lambda x, y: (tf.image.flip_left_right(x), y)))
        final = final.concatenate(train.map(lambda x, y: (tf.image.random_flip_up_down(x), y)))

    if brightness_delta > 0:
        # Apply brightness delta
        final = final.concatenate(train.map(lambda x, y: (tf.image.adjust_brightness(x, brightness_delta), y)))

    if shuffle:
        return final.shuffle(final.cardinality() * (batch_size + 1), reshuffle_each_iteration=False)
    else:
        return final


def load_image(file_path, color_mode="rgb", resize=(256, 256)):
    img = Image.open(file_path)

    if color_mode == "gray_scale":
        img = img.convert('L')
        img = Image.merge('RGB', (img, img, img))

    if resize is not None:
        img = img.resize(resize)

    img_array = np.array(img)

    w, h, c = img_array.shape
    if c == 4:
        img_array = img_array[:, :, :3]  # Remove the last channel

    return [img_array / 255]


def file_path_dict(classmode=ClassMode.STANDARD, excluded_filename_substrings=None):
    """
    Loads lists of the complete filepaths of all images into a dictionary indexed by label.
    A list of strings can be passed to exclude files. Any file that matches one or more substrings will not be loaded.
    """

    base_dir = BASE_DATA_DIR
    file_paths = {}

    if excluded_filename_substrings is None:
        excluded_filename_substrings = []

    folder_names = os.listdir(base_dir)

    for folder_name in folder_names:
        folder_path = os.path.join(base_dir, folder_name)
        label = get_label(classmode, folder_name)

        if label not in file_paths:
            file_paths[label] = []

        if os.path.isdir(folder_path):
            files = list(os.listdir(folder_path))
            image_filenames = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            random.shuffle(image_filenames)

            for i, filename in enumerate(image_filenames):
                for sub in excluded_filename_substrings:
                    if sub in filename:
                        print("Skipping", filename, "as it matched substring: ", sub)
                        continue
                # if "Saureus" in filename:
                #     print("skipping", filename)
                #     continue

                file_path = os.path.join(folder_path, filename)
                file_paths[label].append(file_path)

    return file_paths


def split(classmode=ClassMode.STANDARD, balance=False, max_training=None, test_size=5, excluded_filename_substrings=None):
    """
    Splits data into test and train sets (in the form of filenames)
    """

    file_paths = file_path_dict(classmode, excluded_filename_substrings)

    folds = {}

    least_class_count = min([len(value) for value in file_paths.values()])
    fold_count = 1

    for label, file_paths in file_paths.items():

        # test = test[:len(test) // 2]
        # if len(test) == 0:
        #     test = value[:5]
        # print(test)
        # file_paths = [v for v in file_paths if
        #          "MRSA" not in v and "Saureus" not in v and "(Epidermis)" not in v]  # exclude non-standard data
        test = file_paths[:test_size]
        # test = [v for v in value if "MRSA" in v] # Lisbon MRSA
        # test = [v for v in value if "Saureus" in v] # Lisbon s. Aureus
        # test = [v for v in value if "(Epidermis)" in v]  # LUMC Epidermis

        for i in range(fold_count):
            if i not in folds:
                folds[i] = {}

            if label not in folds[i]:
                folds[i][label] = {}

            folds[i][label]["val"] = []
            folds[i][label]["test"] = test

            total_train_set = [v for v in file_paths if v not in folds[i][label]["val"] and v not in folds[i][label]["test"]]

            overlap = [e for e in total_train_set if e in folds[i][label]["test"]]
            if len(overlap) > 0:
                print(overlap)

            if balance:
                k = least_class_count - test_size
                if max_training is not None and test_size <= max_training <= k:
                    k = max_training
                folds[i][label]["train"] = random.choices(total_train_set, k=k)
            else:
                folds[i][label]["train"] = total_train_set
    return folds


def split_to_data(data_split, color, resize=(128, 128), recombination_ratio=4.5, batch_size=2, rotate=True,
                  flip=True, brightness_delta=0, verbose=1):
    """
    Converts the given data split of file paths to training and test datasets.
    """

    train_images = []
    train_labels = []

    test_images = []
    test_labels = []

    for label, filepaths in data_split.items():

        for path in filepaths["test"]:
            test_labels.append(label)
            test_images.extend(load_image(path, color_mode=color, resize=resize))

        base_train_images = []
        class_train_images = []

        for path in filepaths["train"]:
            img = load_image(path, color_mode=color, resize=resize)
            class_train_images.extend(img)
            base_train_images.extend(img)

        recombinations = int(len(base_train_images) * recombination_ratio)
        if recombinations > 0:
            add_recombinations(base_train_images, class_train_images, recombinations)

        train_labels.extend([label] * len(class_train_images))
        train_images.extend(class_train_images)

    train_data = make_data_set(train_images, train_labels, name="train", batch_size=batch_size, rotate=rotate,
                               flip=flip,
                               brightness_delta=brightness_delta, shuffle=True)
    test_data = make_data_set(test_images, test_labels, name="test", batch_size=batch_size, rotate=True, flip=False,
                              brightness_delta=0, shuffle=False)

    return train_data, None, test_data


def make_data_set(images, labels, batch_size=2, name='', rotate=True, flip=True, brightness_delta=0,
                  shuffle=False):
    assert len(images) == len(labels)

    images = np.array(images)
    labels = np.array(labels)

    data_set = tf.data.Dataset.from_tensor_slices((images, labels))

    data_set = augment_data(data_set, batch_size=batch_size, rotate=rotate, flip=flip,
                            brightness_delta=brightness_delta, shuffle=shuffle)

    class_counts = count_images_per_class(data_set)

    print(f"Dataset {name} class counts: {class_counts}")

    return data_set.batch(batch_size=batch_size)


def determine_max_image_count(base_dir, folder_names):
    """
    Determine the (standard) class with the lowest number of images and return that number.
    """

    images_per_class = float('inf')  # Set to positive infinity initially
    for folder_name in folder_names:
        folder_path = os.path.join(base_dir, folder_name)

        if os.path.isdir(folder_path):
            files = list(os.listdir(folder_path))
            images = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            img_count = len(images)

            if img_count < images_per_class:
                images_per_class = img_count

    return images_per_class


def add_recombinations(class_images, permuted_images, recombinations):
    """
    Add recombinations of the original images to the permutated_images collection.
    """

    image_parts = []
    for image in class_images:
        row_split = np.array_split(image, 2, axis=0)
        for part_row in row_split:
            col_split = np.array_split(part_row, 2, axis=1)
            image_parts.extend(col_split)
    for _ in range(recombinations):
        try:
            random_combination = random.sample(image_parts, 4)
            combined_image = np.hstack(random_combination[:2])
            combined_image = np.vstack([combined_image, np.hstack(random_combination[2:])])
            permuted_images.append(combined_image)
        except ValueError as e:
            print(e)
            print(random_combination.shape)
            raise e


def count_images_per_class(dataset):
    class_counts = {}
    for _, labels in dataset:
        label = labels.numpy()

        if label in class_counts:
            class_counts[label] += 1
        else:
            class_counts[label] = 1
    return class_counts


def get_label(classmode, folder_name):
    match = re.search(r'\d+', folder_name)
    if match:
        label = int(match.group())
    else:
        raise KeyError(f"No number found in folder name: {folder_name}")

    label_mapping = {
        ClassMode.COMPRESSED_END: {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 4},
        ClassMode.COMPRESSED_START: {0: 0, 1: 1, 2: 1, 3: 2, 4: 3, 5: 4},
        ClassMode.COMPRESSED_BOTH: {0: 0, 1: 1, 2: 1, 3: 2, 4: 3, 5: 3}
    }

    return label_mapping.get(classmode, {}).get(label, label)
