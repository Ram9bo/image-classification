import os

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

from dataloader import augment_data

val_split = 0.5
recombinations = 0
augment = True
batch_size = 1

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

    feature_data = pd.read_csv(f"{base_dir}/{folder_name}/features.csv")

    for i, row in feature_data.iterrows():
        i = int(i)
        filename = row["Filename"] + ".png"
        label = (row["Material"] / 100,
                 row["Bacterial cells"] / 100,
                 row["ECM"] / 100)

        file_path = os.path.join(folder_path, filename)

        # Ensure the file is an image (e.g., PNG or JPG)
        if filename.endswith(('.png', '.jpg', '.jpeg', '.gif')):
            img = Image.open(file_path)

            # Convert the image to a NumPy array
            img_array = np.array(img) / 255
            # There is (currently 1) image with 4 channels instead 3, but the 4th value is always 255,
            # so we get rid of it
            if img_array.shape == (512, 512, 4):
                img_array = img_array[:, :, :3]

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

# Create a permutation using numpy
permutation = np.random.permutation(len(train_images))

# Use the same permutation to shuffle both lists
train_images = [train_images[i] for i in permutation]
train_labels = [train_labels[i] for i in permutation]

# Create a permutation using numpy
permutation = np.random.permutation(len(test_images))

# Use the same permutation to shuffle both lists
test_images = [test_images[i] for i in permutation]
test_labels = [test_labels[i] for i in permutation]

train_images = np.array(train_images)
test_images = np.array(test_images)
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

train = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(batch_size)
val = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(batch_size)

if augment:
    train = augment_data(train, batch_size=batch_size)

train.shuffle(buffer_size=10)

from network import XceptionNetwork
from enums import TaskMode

net = XceptionNetwork(task_mode=TaskMode.REGRESSION, num_classes=1)
model = net.model
model.fit(train, epochs=10, validation_data=val)
