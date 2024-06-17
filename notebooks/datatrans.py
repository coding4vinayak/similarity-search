import os
import csv
import random
from shutil import move

# Path to your dataset directory
dataset_dir = 'path/to/celebrity_dataset/'

# Output directories for train and validation
train_dir = 'path/to/celebrity_dataset/train'
validation_dir = 'path/to/celebrity_dataset/validation'

# Output metadata files
train_metadata_file = 'train_metadata.csv'
validation_metadata_file = 'validation_metadata.csv'

# Function to split data and update metadata
def split_data_and_metadata(dataset_dir, train_dir, validation_dir, train_metadata_file, validation_metadata_file):
    # Create directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(validation_dir, exist_ok=True)

    # Open metadata files for writing
    train_metadata = open(train_metadata_file, 'w', newline='')
    train_metadata_writer = csv.writer(train_metadata)
    train_metadata_writer.writerow(['filename', 'celebrity_name'])  # Header

    validation_metadata = open(validation_metadata_file, 'w', newline='')
    validation_metadata_writer = csv.writer(validation_metadata)
    validation_metadata_writer.writerow(['filename', 'celebrity_name'])  # Header

    # Walk through each subdirectory
    for root, dirs, files in os.walk(dataset_dir):
        if files:  # If there are files in the directory
            celebrity_name = os.path.basename(root)  # Get the celebrity name from the directory name
            image_files = [file for file in files if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png')]
            random.shuffle(image_files)  # Shuffle the list of images

            # Determine split sizes
            num_images = len(image_files)
            num_train = int(0.7 * num_images)
            num_validation = num_images - num_train

            # Split images
            train_images = image_files[:num_train]
            validation_images = image_files[num_train:]

            # Move images to respective directories and update metadata
            for img in train_images:
                src = os.path.join(root, img)
                dst = os.path.join(train_dir, celebrity_name, img)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                move(src, dst)
                train_metadata_writer.writerow([dst, celebrity_name])

            for img in validation_images:
                src = os.path.join(root, img)
                dst = os.path.join(validation_dir, celebrity_name, img)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                move(src, dst)
                validation_metadata_writer.writerow([dst, celebrity_name])

    # Close metadata files
    train_metadata.close()
    validation_metadata.close()

# Call the function to split data and update metadata
split_data_and_metadata(dataset_dir, train_dir, validation_dir, train_metadata_file, validation_metadata_file)
