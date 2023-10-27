"""
Data loading and preparation.
"""

import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from tensorflow.keras import datasets

from enums import ClassMode


def cifar_data():
    """
    Load the cifar10 dataset, in 4 parts, train_in, train_out, test_in, test_out
    """
    return datasets.cifar10.load_data()


def augment_data(train, batch_size, rotate=True, flip=True, brightness_delta=0.2, translate=True):
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

    return final.shuffle(final.cardinality() * (batch_size + 1), reshuffle_each_iteration=False)


def load_image(file_path, color_mode="rgb"):
    img = Image.open(file_path)
    if color_mode == "gray_scale":
        img = img.convert('L')
    img_array = np.array(img)[:, :, :3] if img.mode == 'RGBA' else np.array(img)
    return img_array / 255


def images(val_split=0.5, recombinations=5, augment=True):
    base_dir = "data/all_images"

    train_images = []
    val_images = []

    folder_names = os.listdir(base_dir)

    images_per_class = 999

    # For now, we only equalize between the original classes, not the modified ones.

    for folder_name in folder_names:
        folder_path = os.path.join(base_dir, folder_name)

        if os.path.isdir(folder_path):
            class_images = []
            permuted_images = []

            files = list(os.listdir(folder_path))
            images = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]

            for i, filename in enumerate(images):
                if i >= images_per_class:
                    break
                file_path = os.path.join(folder_path, filename)

                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    img_array = load_image(file_path, color_mode="rgb")

                    class_images.append(img_array)
                    permuted_images.append(img_array)

            if recombinations > 0:
                add_recombinations(class_images, permuted_images, recombinations)

            train_images.extend(permuted_images)

    print(f"Loaded {len(train_images)} autoencoder images")

    return np.array(train_images)


def test_data(batch_size=2, classmode="standard", colour="rgb"):
    base_dir = "data/test_set"
    test_images = []
    test_labels = []

    folder_names = os.listdir(base_dir)

    images_per_class = determine_max_image_count(base_dir, folder_names)

    # For now, we only equalize between the original classes, not the modified ones.

    for folder_name in folder_names:
        folder_path = os.path.join(base_dir, folder_name)

        if os.path.isdir(folder_path):
            label = get_label(classmode, folder_name, folder_names)

            class_images = []
            permuted_images = []

            files = list(os.listdir(folder_path))
            images = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]

            for i, filename in enumerate(images):
                if i >= images_per_class:
                    break
                file_path = os.path.join(folder_path, filename)

                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    img_array = load_image(file_path, color_mode=colour)
                    class_images.append(img_array)
                    permuted_images.append(img_array)

            test_images.extend(permuted_images)
            test_labels.extend([label] * len(permuted_images))

    train_dataset = make_data_set(test_images, test_labels, batch_size=batch_size, name="Test")

    return train_dataset


def training_data(batch_size=2, recombination_ratio=1, augment=True, classmode="standard", colour="rgb", balance=True):
    base_dir = "data/train_set"
    train_images = []
    train_labels = []

    folder_names = os.listdir(base_dir)

    images_per_class = determine_max_image_count(base_dir, folder_names) if balance else int(1e7)

    # For now, we only equalize between the original classes, not the modified ones.

    for folder_name in folder_names:
        folder_path = os.path.join(base_dir, folder_name)

        if os.path.isdir(folder_path):
            label = get_label(classmode, folder_name, folder_names)

            class_images = []
            permuted_images = []

            files = list(os.listdir(folder_path))
            images = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]

            for i, filename in enumerate(images):
                if i >= images_per_class:
                    break
                file_path = os.path.join(folder_path, filename)

                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    img_array = load_image(file_path, color_mode=colour)
                    class_images.append(img_array)
                    permuted_images.append(img_array)

            recombinations = len(class_images) * recombination_ratio
            if recombinations > 0:
                add_recombinations(class_images, permuted_images, recombinations)

            train_images.extend(permuted_images)
            train_labels.extend([label] * len(permuted_images))

    train_dataset = make_data_set(train_images, train_labels, batch_size=batch_size, augment=augment, name="Training")

    return train_dataset


def all_data(val_split=0.5, batch_size=2, recombinations=5, augment=True, classmode="standard", colour="rgb",
             recombination_ratio=1, balance=True):
    train = training_data(batch_size=batch_size, recombination_ratio=recombination_ratio, augment=augment,
                               classmode=classmode, balance=balance, colour=colour)

    test = test_data(batch_size=batch_size, classmode=classmode, colour=colour)

    return train, test


def make_data_set(images, labels, batch_size=2, augment=False, name=''):
    assert len(images) == len(labels)

    images = np.array(images)
    labels = np.array(labels)

    data_set = tf.data.Dataset.from_tensor_slices((images, labels)).batch(batch_size)

    # TODO: perform augmentation before wrapping in dataset (gives more flexibility)
    if augment:
        data_set = augment_data(data_set, batch_size=batch_size)

    class_counts = count_images_per_class(data_set)

    print(f"Dataset {name} class counts: {class_counts}")

    return data_set


def make_data_sets(augment, batch_size, test_images, test_labels, train_images, train_labels):
    """
    Converts the given collections of images and labels to Tensorflow dataset objects. Applies augmentation on the
    training set if specified. Batches the dataset according to the given batch size.
    """

    assert len(train_labels) == len(train_images)
    assert len(test_labels) == len(test_images)
    train_images = np.array(train_images)
    test_images = np.array(test_images)
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(batch_size)
    if augment:
        train_dataset = augment_data(train_dataset, batch_size=batch_size)
    class_counts_train = count_images_per_class(train_dataset)
    class_counts_val = count_images_per_class(val_dataset)
    print("Train Dataset Class Counts:", class_counts_train)
    print("Validation Dataset Class Counts:", class_counts_val)
    return train_dataset, val_dataset


def determine_max_image_count(base_dir, folder_names):
    """
    Determine the (standard) class with the lowest number of images and return that number.
    """

    images_per_class = float('inf')  # Set to positive infinity initially
    for folder_name in folder_names:
        folder_path = os.path.join(base_dir, folder_name)

        if os.path.isdir(folder_path):
            files = list(os.listdir(folder_path))
            images = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
            img_count = len(images)

            print(f"{folder_name} has {img_count} images.")

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
        random_combination = random.sample(image_parts, 4)
        combined_image = np.hstack(random_combination[:2])
        combined_image = np.vstack([combined_image, np.hstack(random_combination[2:])])
        permuted_images.append(combined_image)


def count_images_per_class(dataset):
    class_counts = {}
    for _, labels in dataset.unbatch():
        label = labels.numpy()

        if label in class_counts:
            class_counts[label] += 1
        else:
            class_counts[label] = 1
    return class_counts


def get_label(classmode, folder_name, folder_names):
    label = folder_names.index(folder_name)  # Use folder name as the label

    label_mapping = {
        ClassMode.COMPRESSED_END: {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 4},
        ClassMode.COMPRESSED_START: {0: 0, 1: 1, 2: 1, 3: 2, 4: 3, 5: 4},
        ClassMode.COMPRESSED_BOTH: {0: 0, 1: 1, 2: 1, 3: 2, 4: 3, 5: 3}
    }

    return label_mapping.get(classmode, {}).get(label, label)


def ssnombacter_data(val_split=0.1, batch_size=8):
    """
    Load the SSNOMBACTER datasets
    """

    # Define the root directory containing your data
    data_directory = "data/transfer_data/nombacter"

    # Use the image_dataset_from_directory function with custom label mapping
    dataset = tf.keras.utils.image_dataset_from_directory(
        data_directory,
        labels="inferred",
        label_mode="int",
        batch_size=batch_size,
        image_size=(512, 512),
        shuffle=True,
        validation_split=val_split,
        subset="both",
        seed=42
    )

    print(dataset[0].class_names)

    return dataset


def reaarange_nombacter():
    # Define the source directory with the original structure
    source_directory = "data/transfer_data/SSNOMBACTER/Dataset of TIFF files"

    # Define the target directory where the reorganized data will be saved
    target_directory = "data/transfer_data/nombacter"

    # Create the target directory if it doesn't exist
    os.makedirs(target_directory, exist_ok=True)

    class_folders = list(os.listdir(source_directory))
    # Walk through the source directory and its subdirectories
    for folder in class_folders:
        source_folder = os.path.join(source_directory, folder)

        # Define the destination directory for the current file
        destination_directory = os.path.join(target_directory, folder)
        os.makedirs(destination_directory, exist_ok=True)

        class_files = []

        for sub_folder in os.listdir(source_folder):
            sub_path = os.path.join(source_folder, sub_folder)
            files = [f for f in list(os.listdir(sub_path)) if f.endswith('.tiff')]
            class_files.extend(files)

            for file in files:
                tiff_image_path = os.path.join(sub_path, file)
                # Load the TIFF image
                tiff_image = Image.open(tiff_image_path)

                # Define the target PNG file path
                png_image_path = os.path.join(destination_directory, file)
                png_image_path = os.path.splitext(png_image_path)[0] + ".png"

                # Ensure the target directory exists
                os.makedirs(os.path.dirname(png_image_path), exist_ok=True)

                # Convert and save the image as PNG
                tiff_image.save(png_image_path)

    print("Data reorganization completed.")


def feature_data(feature="ECM", val_split=0.5, batch_size=2, augment=True):
    base_dir = "data/all_images"
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []

    folder_names = os.listdir(base_dir)
    for folder_name in folder_names:

        folder_path = os.path.join(base_dir, folder_name)

        feature_data = pd.read_csv(f"{base_dir}/{folder_name}/features.csv")

        for i, row in feature_data.iterrows():
            i = int(i)
            filename = row["Filename"] + ".png"
            label = row[feature] / 100

            file_path = os.path.join(folder_path, filename)

            # Ensure the file is an image (e.g., PNG or JPG)
            if filename.endswith(('.png', '.jpg', '.jpeg', '.gif')):
                img_array = load_image(file_path)

                # Append the image and label to the lists
                if i < 10 * val_split:
                    train_images.append(img_array)  # Always include the originals as well
                    train_labels.append(label)
                else:
                    test_images.append(img_array)
                    test_labels.append(label)

    assert len(train_labels) == len(train_images)
    assert len(test_labels) == len(test_images)
    print(f"Length train {len(train_images)}")

    # TODO: implement optional conversion to grayscale (maybe it helps classfication performance, otherwise it could
    #  at least help with model complexity/runtime performance)

    train, val = make_data_sets(augment, batch_size, test_images, test_labels, train_images, train_labels)

    return train, val
