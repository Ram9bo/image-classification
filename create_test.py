import os
import random
import shutil
import pandas as pd

# Define the source directory where the subfolders and CSV files are located
source_directory = 'data/all_images'

# Define the destination directory where the "Test Set" folder will be created
destination_directory = 'data/test_set'

# Create the "Test Set" folder if it doesn't exist
os.makedirs(destination_directory, exist_ok=True)

# Iterate through the subfolders in the source directory
for subfolder in os.listdir(source_directory):
    subfolder_path = os.path.join(source_directory, subfolder)

    # Check if the subfolder is a directory
    if os.path.isdir(subfolder_path):
        # Load the CSV file within the subfolder
        csv_path = os.path.join(subfolder_path, 'features.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)

            # Select 5 random filenames from the CSV
            random_files = random.sample(df['Filename'].tolist(), 5)

            # Create a corresponding subfolder in the "Test Set" directory
            test_set_subfolder_path = os.path.join(destination_directory, subfolder)
            os.makedirs(test_set_subfolder_path, exist_ok=True)

            # Move the selected files to the "Test Set" subfolder
            for filename in random_files:
                filename += ".png"
                source_file = os.path.join(subfolder_path, filename)
                destination_file = os.path.join(test_set_subfolder_path, filename)
                shutil.move(source_file, destination_file)

print("Files moved to the 'Test Set' directory.")