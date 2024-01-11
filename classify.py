import os
import csv
from pathlib import Path
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

print("Starting classification.")
start = time.time()

# Set the path to the model and image folder
model_path = 'model/model.keras'
print(f"Using model at {model_path}.")
image_folder = 'images'
print(f"Using images in {image_folder} folder.")

# Load the model
model = load_model(model_path)

# Initialize lists to store results
file_names = []
class_labels = []

# Initialize lists to store images
images = []

# Loop through images in the folder
print("Loading images.")
for image_path in Path(image_folder).rglob('*.png'):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(128, 128))  # Adjust target_size as needed
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    images.append(img_array)

    # Store file name for later
    file_names.append(image_path.name)

# Convert the list of images to a numpy array
images = np.array(images)

print("Classifying images.")
# Make predictions for the entire batch
predictions = model.predict(images, verbose=1)

# Extract predicted labels for each image
predicted_labels = np.argmax(predictions, axis=1)

# Store results
class_labels.extend(predicted_labels)

# Write results to a CSV file
output_csv_path = 'results.csv'
print(f"Writing results to {output_csv_path}")
with open(output_csv_path, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['File Name', 'Class Label'])

    for file_name, label in zip(file_names, class_labels):
        csv_writer.writerow([file_name, label])

print(f"Classification finished and saved. Runtime: {round(time.time() - start)} seconds")
print("Shutting down.")

# TODO: save (some of) the parameters used to build the model and use them to make this script more dynamic
#  (e.g. read the target resize from the parameters)


# TODO: add a config file for things like the image folder, model path, and output path
