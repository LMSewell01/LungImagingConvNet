import tensorflow as tf
import os
import numpy as np

# Module and constant import
from data_loader import get_dataset, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS, NUM_CLASSES
from model import build_unet_resnet50 # Previously built, will load saved models

# Config Parameters
DATA_DIR = 'data/raw'
IMAGE_DIR = os.path.join(DATA_DIR, 'images')
MASK_DIR = os.path.join(DATA_DIR, 'masks')

# Define path for saving trained models
MODEL_SAVE_DIR = 'saved_models'
CHECKPOINT_FILENAME = 'best_unet_resnet50.h5'
CHECKPOINT_FILEPATH = os.path.join(MODEL_SAVE_DIR, CHECKPOINT_FILENAME)

# Eval Hyperparameters
BATCH_SIZE = 8

# Custom Dice Loss function (needed when loading model if it was used as a metric or loss)
def dice_coeff(y_true, y_pred, smooth=1e-7):
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

def main():
    # ---- 1. Prepare Data Paths (As in model.py, this is for dummy data atm) ----
    num_dummy_samples = 100  # For demonstration, adjust as needed
    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs(MASK_DIR, exist_ok=True)

    dummy_image_paths = []
    dummy_mask_paths = []

    # Ensure dummy files exist
    # Ensure dummy files exist, just in case train.py wasn't run first
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
        # If files exist, just get their paths
        for i in range(num_dummy_samples):
            dummy_image_paths.append(os.path.join(IMAGE_DIR, f"image_{i}.jpg"))
            dummy_mask_paths.append(os.path.join(MASK_DIR, f"mask_{i}.jpg"))

    # --- 2. Create Test Dataset ---
    # In a real scenario, you'd load your separate test set paths here.
    # For dummy data, let's use the last 20% as our 'test' set (assuming train.py used 80% for train/val)
    num_test = int(num_dummy_samples * 0.2) # Use 20% of dummy data for 'test' evaluation
    test_image_paths = dummy_image_paths[-num_test:]
    test_mask_paths = dummy_mask_paths[-num_test:]

    print(f"\nTest samples: {len(test_image_paths)}")

    # Get the test dataset (no augmentation, no shuffling for evaluation)
    test_dataset = get_dataset(test_image_paths, test_mask_paths, BATCH_SIZE, augment=False, shuffle=False)
    
    # --- 3. Load the Best Trained Model ---
    if not os.path.exists(CHECKPOINT_FILEPATH):
        print(f"Error: Model checkpoint not found at {CHECKPOINT_FILEPATH}")
        print("Please ensure you have run 'python src/train.py' first to train and save the model.")
        return

    # Load the model, specifying the custom_objects if any custom functions were used (like dice_coef as a metric)
    try:
        model = tf.keras.models.load_model(CHECKPOINT_FILEPATH, custom_objects={'dice_coeff': dice_coeff})
        print(f"\nSuccessfully loaded best model from {CHECKPOINT_FILEPATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Ensure the custom_objects dictionary correctly maps custom function names.")
        return

    # --- 4. Evaluate the Model ---
    print("\nEvaluating model on the test dataset...")
    # The evaluate method returns the loss value and metrics values for the model
    loss, binary_accuracy, dice_coefficient = model.evaluate(test_dataset, verbose=1)

    print("\n--- Evaluation Results ---")
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Binary Accuracy: {binary_accuracy:.4f}")
    print(f"Test Dice Coefficient: {dice_coefficient:.4f}")
    print("------------------------")

if __name__ == "__main__":
    main()
