import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# Import functions and constants from data_loader.py
from data_loader import get_covid_qu_ex_paths, get_dataset, NUM_CLASSES, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS
# Import the U-Net model from model.py
from model import build_unet_resnet50 # Changed from unet_model to build_unet_resnet50

# Import custom metrics and loss functions from train.py (now at top-level)
from train import dice_coeff, dice_loss, combined_loss

# --- Configuration Parameters for Evaluation ---
DATA_DIR = 'data/raw/COVID-QU-Ex_Dataset' # MAKE SURE THIS PATH IS CORRECT
BATCH_SIZE = 8 # Same batch size as training or chosen for evaluation
MODEL_SAVE_DIR = 'saved_models' # Must match the directory in train.py
MODEL_PATH = os.path.join(MODEL_SAVE_DIR, 'best_unet_resnet50.h5') # Path to the saved best model

# --- Main Evaluation Function ---
def main():
    print("Starting evaluation script...")

    # Load test dataset paths
    (_, _, _, _,
     test_image_paths, test_mask_paths) = get_covid_qu_ex_paths(DATA_DIR)

    # Create TensorFlow Dataset for testing
    test_dataset = get_dataset(test_image_paths, test_mask_paths, BATCH_SIZE, augment=False, shuffle=False)

    # Load the trained model, including custom objects (metrics/loss functions)
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}. Please train the model first.")
        return

    print(f"Loading model from {MODEL_PATH}...")
    # When loading a model saved with custom objects, they must be provided.
    model = load_model(
        MODEL_PATH,
        custom_objects={
            'dice_coeff': dice_coeff, # Now correctly importing dice_coeff
            'dice_loss': dice_loss,
            'combined_loss': combined_loss,
            'build_unet_resnet50': build_unet_resnet50 # Include the model building function for robustness
        }
    )
    print("Model loaded successfully.")

    # Evaluate the model on the test dataset
    print("\nEvaluating model on the test set...")
    # The metrics during evaluation will match those compiled in train.py: loss, binary_accuracy, dice_coeff
    loss, binary_accuracy, dice_coefficient_score = model.evaluate(test_dataset)

    print(f"\nTest Loss: {loss:.4f}")
    print(f"Test Binary Accuracy: {binary_accuracy:.4f}")
    print(f"Test Dice Coefficient: {dice_coefficient_score:.4f}")

    print("\nEvaluation complete.")

if __name__ == '__main__':
    main()
