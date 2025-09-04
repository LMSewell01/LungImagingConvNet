import tensorflow as tf
import os
import numpy as np
import random
from sklearn.model_selection import train_test_split
import glob

# --- Configuration Parameters (adjust as needed) ---
IMG_HEIGHT = 256
IMG_WIDTH = 256
NUM_CHANNELS = 3
NUM_CLASSES = 3  # Updated for classification: COVID-19, Normal, Non-COVID

# Define class names and a mapping from name to label
CLASS_LABELS = {
    'Normal': 0,
    'Non-COVID': 1,
    'COVID-19': 2,
}
# Reverse mapping for easier lookup
LABEL_TO_CLASS = {v: k for k, v in CLASS_LABELS.items()}

# --- Data Loading and Preprocessing Functions for Classification ---

def load_image_and_label(image_path, label):
    """
    Loads and preprocesses a single image for classification.
    """
    # Load image
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=NUM_CHANNELS)
    image = tf.image.convert_image_dtype(image, tf.float32)  # Normalize to [0, 1]
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    
    # Convert integer label to one-hot encoded vector
    label_one_hot = tf.one_hot(label, depth=NUM_CLASSES)
    
    return image, label_one_hot

def get_classification_paths(data_dir):
    """
    Collects paths to images and their corresponding classification labels.
    """
    all_image_paths = []
    all_labels = []

    for class_name, class_label in CLASS_LABELS.items():
        # Adjust the path to look for images within the class directories
        class_path = os.path.join(data_dir, 'Train', class_name, 'images')
        
        # Use glob to find all images in the directory
        image_paths = glob.glob(os.path.join(class_path, '*.png'))
        
        # Append all found paths and their corresponding labels
        all_image_paths.append(image_paths)
        all_labels.append([class_label] * len(image_paths))
    
    # Flatten the lists
    all_image_paths = [item for sublist in all_image_paths for item in sublist]
    all_labels = [item for sublist in all_labels for item in sublist]

    # Split data into training, validation, and test sets
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        all_image_paths, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        test_paths, test_labels, test_size=0.5, random_state=42, stratify=test_labels
    )
    
    print(f"\nTotal images found for processing: {len(all_image_paths)}")
    print(f"Training samples after custom split: {len(train_paths)}")
    print(f"Validation samples after custom split: {len(val_paths)}")
    print(f"Test samples after custom split: {len(test_paths)}")
    
    return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels

def get_classification_dataset(image_paths, labels, batch_size, shuffle=True):
    """
    Creates a TensorFlow dataset for classification from image paths and labels.
    """
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(image_paths), reshuffle_each_iteration=True)
    
    # Use the new loading function
    dataset = dataset.map(load_image_and_label, num_parallel_calls=tf.data.AUTOTUNE)
    
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return dataset

# For standalone testing
if __name__ == '__main__':
    DATA_DIR = '../data/raw/COVID-QU-Ex_Dataset'
    train_paths, train_labels, _, _, test_paths, test_labels = get_classification_paths(DATA_DIR)
    
    train_dataset = get_classification_dataset(train_paths, train_labels, batch_size=128)
    test_dataset = get_classification_dataset(test_paths, test_labels, batch_size=128)
    
    # Print a sample to verify the data structure
    for images, labels in train_dataset.take(1):
        print("Training Image Batch Shape:", images.shape)
        print("Training Label Batch Shape (one-hot):", labels.shape)
        print("Sample Labels:", tf.argmax(labels, axis=1).numpy())
