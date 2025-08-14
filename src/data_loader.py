import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess_input
# Define image dimensions (e.g., 256x256 or 512x512)
# Choose a size that balances detail preservation with computational resources.
# ResNet50 typically expects 224x224, but U-Net can handle larger inputs.
# For CXR, 256x256 or 512x512 are common. Let's start with 256x256.
IMG_HEIGHT = 256
IMG_WIDTH = 256
NUM_CHANNELS = 3 # CXR will be greyscale but 3 channels used for ResNet50 compat
NUM_CLASSES = 1 # Binary segmentation

def load_image(image_path, mask_path):
    """
    Loads image and related mask from file paths
    """
    # Load Image
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=NUM_CHANNELS) # Decode as RGB
    image = tf.image.convert_image_dtype(image, tf.float32) # Convert to float32

    # If image is grayscale (1 channel) but NUM_CHANNELS is 3, duplicate channels
    if image.shape[-1] == 1 and NUM_CHANNELS == 3: # Convert grayscale to RGB
        image = tf.image.grayscale_to_rgb(image)

    # Load Mask
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_jpeg(mask, channels=NUM_CLASSES) # Decode as grayscale
    mask = tf.image.convert_image_dtype(mask, tf.float32) # Convert to float32
    mask = tf.where(mask > 0, 1.0, 0.0) # Ensure mask is binary (0 or 1)

    # Resize images and masks to a consistent size
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH], method=tf.image.ResizeMethod.BILINEAR)
    mask = tf.image.resize(mask, [IMG_HEIGHT, IMG_WIDTH], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # Use NEAREST_NEIGHBOR for masks to prevent interpolation of class labels (only nearest pixels should be considered), and values like 0.5 aren't generated that break the binary nature of the mask
    # Bilinear interpolation This method calculates new pixel values based on a weighted average of the four nearest pixel values, resulting in a smoother, more natural-looking resized image.


    return image, mask

def augment_data(image, mask):
    """
    Random data augmentation applied to image and mask (identical for each set of these).
    This enables expansion of the data set for training.
    As abnormalities on the lung will be rare, it is useful to  artificially increase the data set to address class imbalance.
    """
    # Stack image and mask for simultaneous transformation
    stacked_image_mask = tf.concat([image, mask], axis=-1)

    # Random horizontal flip
    if tf.random.uniform(()) > 0.5: # generates a random number between 0 and 1
        stacked_image_mask = tf.image.flip_left_right(stacked_image_mask)
    # Random vertical flip
    if tf.random.uniform(()) > 0.5:
        stacked_image_mask = tf.image.flip_up_down(stacked_image_mask)
    # Random rotations
    k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    stacked_image_mask = tf.image.rot90(stacked_image_mask, k) # rot 90 applies k*90 rotation

    # Unstack image and mask according to channels defined previously
    image = stacked_image_mask[:, :, :NUM_CHANNELS]
    mask = stacked_image_mask[:, :, NUM_CHANNELS:]

    return image, mask

def preprocess_for_model(image, mask):
    """
    ResNet specific preprocessing. Normalizes the input around 0 [-1,1], and coverts from RGB to BGR.
    Mask included to retain image, mask pairing during processing.
    """
    image = resnet_preprocess_input(image*255.0) # ResNet expect 0-255 range and we originally scaled 0-1 as tf.float32
    return image, mask

def get_dataset(image_paths, mask_paths, batch_size, augment=True, shuffle=True):
    """
    Create a pipeline in tf for training or validation using tf.data API
    image_paths[i] must equal mask_paths[i]
    augment/shuffle often true for training and false for validation 
    """
    # Creates tuple for each pair of image/mask
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))

    if shuffle:
        dataset.shuffle(buffer_size=len(image_paths)) # maximum shuffling
    # Map loading and initial preprocessing. Does this on-the-fly and in parallel.
    dataset= dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE) # Automates performance optimization

    if augment:
        dataset = dataset.map(augment_data, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Apply Resnet specific preprocessing
    dataset =dataset.map(preprocess_for_model, num_parallel_calls=tf.data.AUTOTUNE)

    # Batch and prefetch for performance. E.g. if batch is 32, each element yielded will be 
    # a tuple of two tensors, an image tensor of shape (32, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS), and 
    # a mask tensor with last axis number of channels (here 1).
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset
    
# Example usage
if __name__ == "__main__":
    # Dummy paths for demonstration.
    dummy_image_paths = [f"data/raw/images/image_{i}.jpg" for i in range(10)]
    dummy_mask_paths = [f"data/raw/masks/mask_{i}.jpg" for i in range(10)]

    # Create dummy files for testing the loader
    os.makedirs("data/raw/images", exist_ok=True)
    os.makedirs("data/raw/masks", exist_ok=True)
    for i in range(10):
        # Create a simple white image
        dummy_img = np.ones((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8) * 255
        tf.io.write_file(dummy_image_paths[i], tf.image.encode_jpeg(dummy_img, quality=100))
        # Create a simple black mask with a white square in the middle
        dummy_mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.uint8)
        dummy_mask[IMG_HEIGHT//4:3*IMG_HEIGHT//4, IMG_WIDTH//4:3*IMG_WIDTH//4] = 255
        tf.io.write_file(dummy_mask_paths[i], tf.image.encode_jpeg(dummy_mask, quality=100))

    print(f"Created {len(dummy_image_paths)} dummy image and mask files for demonstration.")

    # Get a sample dataset
    sample_dataset = get_dataset(dummy_image_paths, dummy_mask_paths, batch_size=2, augment=True, shuffle=True)

    print("\nIterating through a sample batch:")
    for images, masks in sample_dataset.take(1):
        print(f"Image batch shape: {images.shape}, dtype: {images.dtype}")
        print(f"Mask batch shape: {masks.shape}, dtype: {masks.dtype}")
        print(f"Image pixel range (after ResNet preprocess): min={tf.reduce_min(images)}, max={tf.reduce_max(images)}")
        print(f"Mask pixel range (binary): min={tf.reduce_min(masks)}, max={tf.reduce_max(masks)}")

    print("\nData loading and preprocessing setup complete for `src/data_loader.py`.")

