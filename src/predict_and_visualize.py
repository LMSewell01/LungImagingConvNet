import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from data_loader import get_dataset, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS, NUM_CLASSES
from model import build_unet_resnet50 # Not strictly used for building, but good for context if needed


# --- Configuration Parameters ---
DATA_DIR = 'data/raw'
IMAGE_DIR = os.path.join(DATA_DIR, 'images')
MASK_DIR = os.path.join(DATA_DIR, 'masks')

MODEL_SAVE_DIR = 'saved_models'
CHECKPOINT_FILENAME = 'best_unet_resnet50.h5'
CHECKPOINT_FILEPATH = os.path.join(MODEL_SAVE_DIR, CHECKPOINT_FILENAME)

# Custom Dice Loss function (needed when loading model if it was used as a metric or loss)
def dice_coef(y_true, y_pred, smooth=1e-7):
    """
    Dice coefficient for segmentation.
    Args:
        y_true: Ground truth masks.
        y_pred: Predicted masks.
        smooth: Smoothing factor to prevent division by zero.
    Returns:
        float: Dice coefficient.
    """
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
    return (2. * intersection + smooth) / (union + smooth)

def display_images(display_list):
    """
    Helper function to display a list of images side-by-side.
    Used for showing original image, true mask, and predicted mask.
    """
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        # For masks, use 'gray' colormap to make 0/1 visually clear
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]), cmap='gray' if i > 0 else None)
        plt.axis('off')
    plt.show()

def main():
    # --- 1. Prepare Data Paths (using dummy data for initial setup) ---
    # We'll re-use dummy data creation logic for consistency.
    num_dummy_samples = 100
    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs(MASK_DIR, exist_ok=True)

    dummy_image_paths = []
    dummy_mask_paths = []

    if not os.path.exists(os.path.join(IMAGE_DIR, f"image_0.jpg")):
        print(f"Creating {num_dummy_samples} dummy image and mask files for demonstration...")
        for i in range(num_dummy_samples):
            img_path = os.path.join(IMAGE_DIR, f"image_{i}.jpg")
            mask_path = os.path.join(MASK_DIR, f"mask_{i}.jpg")
            dummy_image_paths.append(img_path)
            dummy_mask_paths.append(mask_path)

            dummy_img = np.ones((IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS), dtype=np.uint8) * 255
            tf.io.write_file(img_path, tf.image.encode_jpeg(dummy_img, quality=100))
            
            dummy_mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.uint8)
            dummy_mask[IMG_HEIGHT//4:3*IMG_HEIGHT//4, IMG_WIDTH//4:3*IMG_HEIGHT//4] = 255
            tf.io.write_file(mask_path, tf.image.encode_jpeg(dummy_mask, quality=100))
        print("Dummy data creation complete.")
    else:
        for i in range(num_dummy_samples):
            dummy_image_paths.append(os.path.join(IMAGE_DIR, f"image_{i}.jpg"))
            dummy_mask_paths.append(os.path.join(MASK_DIR, f"mask_{i}.jpg"))

    # --- 2. Create Test Dataset ---
    # Use a small subset of the dummy data to visualize
    num_test = 5 # Number of samples to predict and visualize
    test_image_paths = dummy_image_paths[-num_test:]
    test_mask_paths = dummy_mask_paths[-num_test:]

    print(f"\nPreparing {num_test} test samples for prediction and visualization.")

    # Get the test dataset (no augmentation, no shuffling)
    # Use a batch size of 1 for easier individual sample processing in loop
    test_dataset = get_dataset(test_image_paths, test_mask_paths, 1, augment=False, shuffle=False)
    
    # --- 3. Load the Best Trained Model ---
    if not os.path.exists(CHECKPOINT_FILEPATH):
        print(f"Error: Model checkpoint not found at {CHECKPOINT_FILEPATH}")
        print("Please ensure you have run 'python src/train.py' first to train and save the model.")
        return

    try:
        model = tf.keras.models.load_model(CHECKPOINT_FILEPATH, custom_objects={'dice_coef': dice_coef})
        print(f"\nSuccessfully loaded best model from {CHECKPOINT_FILEPATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Ensure the custom_objects dictionary correctly maps custom function names.")
        return

    # --- 4. Make Predictions and Visualize ---
    print("\nMaking predictions and visualizing results...")
    for i, (image, mask) in enumerate(test_dataset.take(num_test)):
        prediction_mask = model.predict(image)[0] # Get prediction for the first (and only) image in batch

        # Threshold the probability mask to get a binary mask (0 or 1)
        # We can use 0.5 as a simple threshold for sigmoid output
        predicted_binary_mask = (prediction_mask > 0.5).astype(np.float32)

        print(f"Visualizing sample {i+1}/{num_test}...")
        display_images([image[0], mask[0], predicted_binary_mask])

    print("\nPrediction and visualization complete.")

if __name__ == "__main__":
    main()