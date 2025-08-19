import tensorflow as tf
import os
import numpy as np
import random
from sklearn.model_selection import train_test_split
import glob

# --- Configuration Parameters (adjust as needed) ---
IMG_HEIGHT = 256
IMG_WIDTH = 256
NUM_CHANNELS = 3 # RGB for X-rays (even if grayscale, often loaded as 3 channels for backbone compatibility)
NUM_CLASSES = 1 # Binary segmentation: abnormality vs. background

# --- Data Loading and Preprocessing Functions ---

def load_image(image_path, mask_path):
    """
    Loads and preprocesses a single image and its corresponding mask.
    Handles both actual mask files and the 'PLACEHOLDER_HEALTHY_MASK' for normal images.
    """
    # Load image
    image = tf.io.read_file(image_path)
    # Decode as PNG since your directory structure shows .png files
    image = tf.image.decode_png(image, channels=NUM_CHANNELS)
    image = tf.image.convert_image_dtype(image, tf.float32) # Normalize to [0, 1]
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])

    # Load or create mask
    if mask_path == 'PLACEHOLDER_HEALTHY_MASK':
        # Create an all-black mask (0s) for healthy images.
        mask = tf.zeros((IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES), dtype=tf.float32)
    else:
        mask = tf.io.read_file(mask_path)
        # Masks are typically grayscale (single channel), and your structure shows .png
        mask = tf.image.decode_png(mask, channels=1)
        mask = tf.image.convert_image_dtype(mask, tf.float32)
        # Resize masks using NEAREST_NEIGHBOR to preserve crisp boundaries
        mask = tf.image.resize(mask, [IMG_HEIGHT, IMG_WIDTH], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # Binarize the mask: anything > 0.5 (or any non-zero value in original mask) becomes 1.0, else 0.0
        mask = tf.where(mask > 0.5, 1.0, 0.0)

    return image, mask

def augment_data(image, mask):
    """
    Performs data augmentation on image and mask synchronously.
    (Add more augmentation techniques here as needed)
    """
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)
    
    # You can add other augmentations like random rotations, brightness changes, etc.
    # Ensure they are applied consistently to both image and mask where appropriate.

    return image, mask

def get_dataset(image_paths, mask_paths, batch_size, augment=True, shuffle=True):
    """
    Creates a TensorFlow Dataset from image and mask paths.
    """
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    
    if shuffle:
        # Shuffle a large buffer of data to ensure randomness
        dataset = dataset.shuffle(buffer_size=len(image_paths))

    # Use map to apply load_image function
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        # Apply augmentation if enabled
        dataset = dataset.map(augment_data, num_parallel_calls=tf.data.AUTOTUNE)

    # Batch and prefetch for performance
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset

def get_covid_qu_ex_paths(base_data_dir):
    """
    Gathers image and specific infection mask paths for the COVID-QU-Ex Dataset,
    based on the provided directory structure.
    Focuses on COVID-19 cases with specific masks and Normal cases from main splits.
    """
    print(f"Loading COVID-QU-Ex dataset paths from: {base_data_dir}")

    all_image_paths = []
    all_mask_paths = []

    # --- Define base paths based on your provided directory structure ---
    # Note: Using 'COVID-19' instead of 'COVID' for image folders as per your screenshot
    # and 'COVID-QU-Ex_Dataset' with underscore.

    # --- Train Split Paths ---
    train_covid_images_dir = os.path.join(base_data_dir, 'Train', 'COVID-19', 'images')
    train_normal_images_dir = os.path.join(base_data_dir, 'Train', 'Normal', 'images')
    # Specific infection masks are in a separate top-level folder
    train_infection_masks_dir = os.path.join(base_data_dir, 'Infection_Segmentation_Data', 'Train', 'COVID-19', 'infection masks')

    # --- Val Split Paths ---
    val_covid_images_dir = os.path.join(base_data_dir, 'Val', 'COVID-19', 'images')
    val_normal_images_dir = os.path.join(base_data_dir, 'Val', 'Normal', 'images')
    val_infection_masks_dir = os.path.join(base_data_dir, 'Infection_Segmentation_Data', 'Val', 'COVID-19', 'infection masks')

    # --- Test Split Paths ---
    test_covid_images_dir = os.path.join(base_data_dir, 'Test', 'COVID-19', 'images')
    test_normal_images_dir = os.path.join(base_data_dir, 'Test', 'Normal', 'images')
    test_infection_masks_dir = os.path.join(base_data_dir, 'Infection_Segmentation_Data', 'Test', 'COVID-19', 'infection masks')

    # --- Helper to collect paths for a given split ---
    def collect_paths_for_split(images_dir, masks_dir, normal_dir):
        split_image_paths = []
        split_mask_paths = []

        # Collect COVID-19 images and their *specific infection masks*
        covid_images = sorted(glob.glob(os.path.join(images_dir, '*.png')))
        print(f"  Found {len(covid_images)} images in {images_dir}")

        for img_path in covid_images:
            base_filename = os.path.basename(img_path)
            # Mask filename is assumed to be the same as image filename
            mask_path = os.path.join(masks_dir, base_filename)
            
            if os.path.exists(mask_path):
                split_image_paths.append(img_path)
                split_mask_paths.append(mask_path)
            else:
                # This could happen if not all COVID-19 images have specific infection masks,
                # or if the naming convention is slightly off for the infection masks.
                # print(f"Warning: Infection mask not found for {img_path} at expected path: {mask_path}. Skipping.")
                pass # We will only add images for which a specific infection mask exists.

        # Collect Normal images (masks will be all black / 'healthy' - synthetically made)
        normal_images = sorted(glob.glob(os.path.join(normal_dir, '*.png')))
        print(f"  Found {len(normal_images)} images in {normal_dir}")
        for img_path in normal_images:
            split_image_paths.append(img_path)
            split_mask_paths.append('PLACEHOLDER_HEALTHY_MASK')
        
        return split_image_paths, split_mask_paths

    print("\nCollecting Train split paths:")
    train_images, train_masks = collect_paths_for_split(
        train_covid_images_dir, train_infection_masks_dir, train_normal_images_dir
    )
    all_image_paths.extend(train_images)
    all_mask_paths.extend(train_masks)

    print("\nCollecting Val split paths:")
    val_images, val_masks = collect_paths_for_split(
        val_covid_images_dir, val_infection_masks_dir, val_normal_images_dir
    )
    all_image_paths.extend(val_images)
    all_mask_paths.extend(val_masks)

    print("\nCollecting Test split paths:")
    test_images, test_masks = collect_paths_for_split(
        test_covid_images_dir, test_infection_masks_dir, test_normal_images_dir
    )
    all_image_paths.extend(test_images)
    all_mask_paths.extend(test_masks)


    if not all_image_paths:
        raise ValueError("No paired image and mask files or normal images found. Check dataset paths and naming conventions as per your directory structure.")

    # Combine all collected paths for splitting
    combined_paths = list(zip(all_image_paths, all_mask_paths))
    random.shuffle(combined_paths) # Shuffle before splitting to ensure good distribution

    images, masks = zip(*combined_paths)
    
    # Perform standard 80/10/10 train/val/test split
    # Stratify by mask type to ensure balanced distribution of pathological/normal cases
    stratify_labels = ['pathology' if m != 'PLACEHOLDER_HEALTHY_MASK' else 'normal' for m in masks]

    # Ensure consistent splitting behavior by always performing the full split here
    train_val_images, test_images, train_val_masks, test_masks = train_test_split(
        images, masks, test_size=0.1, random_state=42, stratify=stratify_labels
    )
    
    train_images, val_images, train_masks, val_masks = train_test_split(
        train_val_images, train_val_masks, test_size=(0.1 / 0.9), random_state=42, 
        stratify=['pathology' if m != 'PLACEHOLDER_HEALTHY_MASK' else 'normal' for m in train_val_masks]
    )

    print(f"\nTotal images found for processing: {len(images)}")
    print(f"Training samples after custom split: {len(train_images)}")
    print(f"Validation samples after custom split: {len(val_images)}")
    print(f"Test samples after custom split: {len(test_images)}")

    return (list(train_images), list(train_masks),
            list(val_images), list(val_masks),
            list(test_images), list(test_masks))

