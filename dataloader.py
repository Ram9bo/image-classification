"""
Data loading and preparation.
"""

import os
import random

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras import datasets
import shutil
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


def all_data(val_split=0.5, batch_size=2, recombinations=10, augment=True, classmode="standard", colour="rgb"):
    """
    Retrieve the dataset in two parts: the augmented training set and the unmodified test set, split according
    to the val_split parameter.
    """

    base_dir = "data/all_images"
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []

    folder_names = os.listdir(base_dir)
    for folder_name in folder_names:
        class_images = []
        permuted_images = []
        folder_path = os.path.join(base_dir, folder_name)

        if os.path.isdir(folder_path):
            label = folder_names.index(folder_name)  # Use folder name as the label
            # TODO: if we modify labels, we may need to rebalance class sampling around the new class counts
            if classmode == ClassMode.COMPRESSED_END:
                label = {
                    0: 0,
                    1: 1,
                    2: 2,
                    3: 3,
                    4: 4,
                    5: 4
                }[label]
            elif classmode == ClassMode.COMPRESSED_START:
                label = {
                    0: 0,
                    1: 1,
                    2: 1,
                    3: 2,
                    4: 3,
                    5: 4
                }[label]
            elif classmode == ClassMode.COMPRESSED_BOTH:
                label = {
                    0: 0,
                    1: 1,
                    2: 1,
                    3: 2,
                    4: 3,
                    5: 3
                }[label]

            files = list(os.listdir(folder_path))
            random.shuffle(files)
            num_files = len(files)
            max_imgs = 10  # limit the number of used images to ensure class balance (right now the percentual
            # differences are large)
            # TODO: make this dynamic based on class with the fewest examples
            for i, filename in enumerate(files):
                if i >= max_imgs:
                    break
                file_path = os.path.join(folder_path, filename)

                # Ensure the file is an image (e.g., PNG or JPG)
                if filename.endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    img = Image.open(file_path)

                    if colour == "gray_scale":
                        img = img.convert('L')

                    # Convert the image to a NumPy array
                    img_array = np.array(img) / 255
                    # There is (currently 1) image with 4 channels instead 3, but the 4th value is always 255,
                    # so we get rid of it
                    if img_array.shape == (512, 512, 4):
                        img_array = img_array[:, :, :3]

                    # Append the image and label to the lists
                    if i < max_imgs * val_split:
                        class_images.append(img_array)
                        permuted_images.append(img_array)  # Always include the originals as well
                    else:
                        test_images.append(img_array)
                        test_labels.append(label)

            # TODO: implement variable tile sizes and test their impact
            # Split images into four parts and create some recombinations.
            if recombinations > 0:
                image_parts = []
                # Split each image into four parts and append them to the image_parts list
                for image in class_images:
                    row_split = np.array_split(image, 2, axis=0)
                    for part_row in row_split:
                        col_split = np.array_split(part_row, 2, axis=1)
                        image_parts.extend(col_split)

                for _ in range(recombinations):
                    # Randomly sample a combination of four parts
                    random_combination = random.sample(image_parts, 4)

                    # Create a new image by combining the sampled parts
                    combined_image = np.hstack(random_combination[:2])
                    combined_image = np.vstack([combined_image, np.hstack(random_combination[2:])])
                    permuted_images.append(combined_image)

            train_images.extend(permuted_images)
            train_labels.extend([label] * len(permuted_images))

    assert len(train_labels) == len(train_images)
    assert len(test_labels) == len(test_images)

    # TODO: implement optional conversion to grayscale (maybe it helps classfication performance, otherwise it could
    #  at least help with model complexity/runtime performance)

    train_images = np.array(train_images)
    test_images = np.array(test_images)
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    train = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(batch_size)
    val = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(batch_size)

    if augment:
        train = augment_data(train, batch_size=batch_size)
        # print(f"Augmented training set to {train.cardinality() * batch_size} images")
    # TODO: check if more shuffling is required/helpful (is already done after augmentation, but maybe do it again here)
    return train, val


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
