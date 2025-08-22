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
NUM_CLASSES = 4 # NEW: 4 classes (0:Background, 1:Healthy Lung, 2:COVID Infection, 3:Non-COVID Infection)

# --- Data Loading and Preprocessing Functions ---

def load_image_and_multi_class_mask(image_path, covid_infection_mask_path, lung_mask_path, image_type):
    """
    Loads and preprocesses a single image and creates its multi-class mask.
    Handles different image types (Normal, COVID, Non-COVID) to create appropriate masks.
    """
    # Load image
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=NUM_CHANNELS)
    image = tf.image.convert_image_dtype(image, tf.float32) # Normalize to [0, 1]
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])

    # Initialize an empty mask with NUM_CLASSES channels (one-hot encoding)
    # The last dimension for one-hot encoded masks should be NUM_CLASSES
    multi_class_mask = tf.zeros((IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES), dtype=tf.float32)

    # Load general lung mask for all cases
    lung_mask = tf.io.read_file(lung_mask_path)
    lung_mask = tf.image.decode_png(lung_mask, channels=1)
    lung_mask = tf.image.convert_image_dtype(lung_mask, tf.float32)
    lung_mask = tf.image.resize(lung_mask, [IMG_HEIGHT, IMG_WIDTH], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    lung_mask = tf.where(lung_mask > 0.5, 1.0, 0.0) # Binarize

    # Create the multi-class mask based on image_type
    if image_type == 'Normal':
        # Class 1: Healthy Lung (where lung_mask is 1)
        # Class 0: Background (where lung_mask is 0)
        healthy_lung_pixels = tf.cast(lung_mask, tf.int32) # Pixels in lung get class 1
        multi_class_mask = tf.one_hot(tf.squeeze(healthy_lung_pixels, axis=-1), NUM_CLASSES) # Squeeze to remove channel dim before one_hot
        # The background (0) is handled by tf.one_hot default (all zeros except class 0 if that's what you want, but here, it's implicit)

    elif image_type == 'COVID':
        # Load specific COVID-19 infection mask
        covid_infection_mask = tf.io.read_file(covid_infection_mask_path)
        covid_infection_mask = tf.image.decode_png(covid_infection_mask, channels=1)
        covid_infection_mask = tf.image.convert_image_dtype(covid_infection_mask, tf.float32)
        covid_infection_mask = tf.image.resize(covid_infection_mask, [IMG_HEIGHT, IMG_WIDTH], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        covid_infection_mask = tf.where(covid_infection_mask > 0.5, 1.0, 0.0) # Binarize

        # Start with background (0)
        # Then, fill with Healthy Lung (1) where lung is present and no infection
        # Then, fill with COVID Infection (2) where infection is present
        
        # Create an integer mask first
        int_mask = tf.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=tf.int32)
        
        # Where lung_mask is 1 (inside lung), set to Class 1 (Healthy Lung)
        int_mask = tf.where(lung_mask[:,:,0] == 1, 1, int_mask)
        
        # Where covid_infection_mask is 1, set to Class 2 (COVID Infection)
        # This will overwrite Class 1 in the infected regions
        int_mask = tf.where(covid_infection_mask[:,:,0] == 1, 2, int_mask)

        multi_class_mask = tf.one_hot(int_mask, NUM_CLASSES)

    elif image_type == 'Non-COVID':
        # For Non-COVID, we define the *entire lung region* as Class 3 (Non-COVID Infection)
        # since specific infection masks are not provided for non-COVID in this dataset.
        # Start with background (0)
        int_mask = tf.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=tf.int32)
        
        # Where lung_mask is 1 (inside lung), set to Class 3 (Non-COVID Infection)
        int_mask = tf.where(lung_mask[:,:,0] == 1, 3, int_mask)

        multi_class_mask = tf.one_hot(int_mask, NUM_CLASSES)
    
    # Ensure the final mask has the correct shape [H, W, NUM_CLASSES]
    multi_class_mask = tf.reshape(multi_class_mask, (IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES))

    return image, multi_class_mask


def augment_data(image, mask):
    """
    Performs data augmentation on image and mask synchronously.
    (Add more augmentation techniques here as needed)
    """
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)
    
    return image, mask

def get_dataset(image_paths_list, mask_paths_list, image_types_list, batch_size, augment=True, shuffle=True):
    """
    Creates a TensorFlow Dataset from image, mask paths, and image types.
    """
    dataset = tf.data.Dataset.from_tensor_slices((image_paths_list, mask_paths_list[0], mask_paths_list[1], image_types_list))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(image_paths_list))

    # Use map to apply load_image_and_multi_class_mask function
    dataset = dataset.map(lambda img_p, covid_m_p, lung_m_p, img_t: load_image_and_multi_class_mask(img_p, covid_m_p, lung_m_p, img_t), num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        dataset = dataset.map(augment_data, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset


def get_covid_qu_ex_paths(base_data_dir):
    """
    Gathers image paths, specific infection mask paths, general lung mask paths, and image types
    for the COVID-QU-Ex Dataset for multi-class segmentation.
    """
    print(f"Loading COVID-QU-Ex dataset paths from: {base_data_dir}")

    all_image_paths = []
    all_covid_infection_mask_paths = [] # Specific infection masks for COVID cases
    all_lung_mask_paths = []            # General lung masks for all cases
    all_image_types = []                # To differentiate Normal, COVID, Non-COVID

    # --- Define base paths based on your provided directory structure ---
    def collect_paths_for_split(split_name):
        _image_paths = []
        _covid_infection_mask_paths = []
        _lung_mask_paths = []
        _image_types = []

        # Paths for images
        covid_images_dir = os.path.join(base_data_dir, split_name, 'COVID-19', 'images')
        non_covid_images_dir = os.path.join(base_data_dir, split_name, 'Non-COVID', 'images')
        normal_images_dir = os.path.join(base_data_dir, split_name, 'Normal', 'images') # Assuming normal also has an 'images' folder. Verify!

        # Paths for masks
        covid_specific_infection_masks_dir = os.path.join(base_data_dir, 'Infection_Segmentation_Data', split_name, 'COVID-19', 'infection masks')
        
        # General lung masks for all types are in 'lung masks' folder
        covid_lung_masks_dir = os.path.join(base_data_dir, split_name, 'COVID-19', 'lung masks')
        non_covid_lung_masks_dir = os.path.join(base_data_dir, split_name, 'Non-COVID', 'lung masks')
        normal_lung_masks_dir = os.path.join(base_data_dir, split_name, 'Normal', 'lung masks')


        # --- Collect COVID-19 cases ---
        covid_images = sorted(glob.glob(os.path.join(covid_images_dir, '*.png')))
        print(f"  Found {len(covid_images)} images in {covid_images_dir}")
        for img_path in covid_images:
            base_filename = os.path.basename(img_path)
            infection_mask_path = os.path.join(covid_specific_infection_masks_dir, base_filename)
            general_lung_mask_path = os.path.join(covid_lung_masks_dir, base_filename)

            # Only include COVID images that have both a specific infection mask and a general lung mask
            if os.path.exists(infection_mask_path) and os.path.exists(general_lung_mask_path):
                _image_paths.append(img_path)
                _covid_infection_mask_paths.append(infection_mask_path)
                _lung_mask_paths.append(general_lung_mask_path)
                _image_types.append('COVID')
            else:
                # print(f"Warning: Missing infection or lung mask for COVID image: {img_path}")
                pass # Skip if critical masks are missing

        # --- Collect Non-COVID cases ---
        non_covid_images = sorted(glob.glob(os.path.join(non_covid_images_dir, '*.png')))
        print(f"  Found {len(non_covid_images)} images in {non_covid_images_dir}")
        for img_path in non_covid_images:
            base_filename = os.path.basename(img_path)
            general_lung_mask_path = os.path.join(non_covid_lung_masks_dir, base_filename) # Only general lung mask expected

            if os.path.exists(general_lung_mask_path):
                _image_paths.append(img_path)
                _covid_infection_mask_paths.append('NO_COVID_INFECTION_MASK') # Placeholder for COVID infection mask
                _lung_mask_paths.append(general_lung_mask_path)
                _image_types.append('Non-COVID')
            else:
                # print(f"Warning: Missing general lung mask for Non-COVID image: {img_path}")
                pass # Skip if lung mask is missing

        # --- Collect Normal cases ---
        normal_images = sorted(glob.glob(os.path.join(normal_images_dir, '*.png')))
        print(f"  Found {len(normal_images)} images in {normal_images_dir}")
        for img_path in normal_images:
            base_filename = os.path.basename(img_path)
            general_lung_mask_path = os.path.join(normal_lung_masks_dir, base_filename) # Only general lung mask expected

            if os.path.exists(general_lung_mask_path):
                _image_paths.append(img_path)
                _covid_infection_mask_paths.append('NO_COVID_INFECTION_MASK') # Placeholder for COVID infection mask
                _lung_mask_paths.append(general_lung_mask_path)
                _image_types.append('Normal')
            else:
                # print(f"Warning: Missing general lung mask for Normal image: {img_path}")
                pass # Skip if lung mask is missing
        
        return _image_paths, _covid_infection_mask_paths, _lung_mask_paths, _image_types

    print("\nCollecting Train split paths:")
    train_paths = collect_paths_for_split('Train')
    all_image_paths.extend(train_paths[0])
    all_covid_infection_mask_paths.extend(train_paths[1])
    all_lung_mask_paths.extend(train_paths[2])
    all_image_types.extend(train_paths[3])

    print("\nCollecting Val split paths:")
    val_paths = collect_paths_for_split('Val')
    all_image_paths.extend(val_paths[0])
    all_covid_infection_mask_paths.extend(val_paths[1])
    all_lung_mask_paths.extend(val_paths[2])
    all_image_types.extend(val_paths[3])

    print("\nCollecting Test split paths:")
    test_paths = collect_paths_for_split('Test')
    all_image_paths.extend(test_paths[0])
    all_covid_infection_mask_paths.extend(test_paths[1])
    all_lung_mask_paths.extend(test_paths[2])
    all_image_types.extend(test_paths[3])

    if not all_image_paths:
        raise ValueError("No images found. Check dataset paths and naming conventions as per your directory structure.")

    # Combine all collected paths for splitting - ensures all lists are shuffled together
    combined_data = list(zip(all_image_paths, all_covid_infection_mask_paths, all_lung_mask_paths, all_image_types))
    random.shuffle(combined_data)

    images, covid_masks, lung_masks, types = zip(*combined_data)
    
    # Stratify by image type to ensure balanced distribution of Normal, COVID, Non-COVID in splits
    stratify_labels = list(types) # Already strings like 'Normal', 'COVID', 'Non-COVID'

    train_val_indices, test_indices = train_test_split(
        range(len(images)), test_size=0.1, random_state=42, stratify=stratify_labels
    )
    train_indices, val_indices = train_test_split(
        train_val_indices, test_size=(0.1 / 0.9), random_state=42, stratify=[stratify_labels[i] for i in train_val_indices]
    )

    # Helper to create lists from indices
    def get_split_data(indices, data_list):
        return [data_list[i] for i in indices]

    train_image_paths = get_split_data(train_indices, images)
    train_covid_mask_paths = get_split_data(train_indices, covid_masks)
    train_lung_mask_paths = get_split_data(train_indices, lung_masks)
    train_image_types = get_split_data(train_indices, types)

    val_image_paths = get_split_data(val_indices, images)
    val_covid_mask_paths = get_split_data(val_indices, covid_masks)
    val_lung_mask_paths = get_split_data(val_indices, lung_masks)
    val_image_types = get_split_data(val_indices, types)

    test_image_paths = get_split_data(test_indices, images)
    test_covid_mask_paths = get_split_data(test_indices, covid_masks)
    test_lung_mask_paths = get_split_data(test_indices, lung_masks)
    test_image_types = get_split_data(test_indices, types)

    print(f"\nTotal images found for processing: {len(images)}")
    print(f"Training samples after custom split: {len(train_image_paths)}")
    print(f"Validation samples after custom split: {len(val_image_paths)}")
    print(f"Test samples after custom split: {len(test_image_paths)}")

    # Return lists of lists for masks and types, as get_dataset now expects this
    return (train_image_paths, [train_covid_mask_paths, train_lung_mask_paths], train_image_types,
            val_image_paths, [val_covid_mask_paths, val_lung_mask_paths], val_image_types,
            test_image_paths, [test_covid_mask_paths, test_lung_mask_paths], test_image_types)
