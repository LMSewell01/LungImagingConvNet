import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# Import functions and constants from data_loader.py
# IMPORTANT: These imports now correctly point to the multi-class setup
from .data_loader import get_covid_qu_ex_paths, get_dataset, NUM_CLASSES, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS
# Import the U-Net model from model.py
from .model import build_unet_resnet50

# Import custom metrics and loss functions (now multi-class versions)
# These are defined in train.py and needed for loading the model.
from .train import dice_coeff_multi_class, dice_loss_multi_class, combined_loss_multi_class

# --- Configuration Parameters for Evaluation ---
# IMPORTANT: DATA_DIR should be relative to the PROJECT ROOT
DATA_DIR = 'data/raw/COVID-QU-Ex_Dataset'
BATCH_SIZE = 8 # Can be larger for evaluation as gradients are not computed. Adjust based on A100 VRAM.
MODEL_SAVE_DIR = 'saved_models'
MODEL_PATH = os.path.join(MODEL_SAVE_DIR, 'best_unet_resnet50_multi_class.h5') # Matches the name in train.py

# --- Main Evaluation Function ---
def main():
    print("Starting multi-class evaluation script...")

    # Load test dataset paths (get_covid_qu_ex_paths now returns 9 items for multi-class)
    # We use underscores (_) for the train and validation paths as we only need the test paths here.
    _, _, _, \
    _, _, _, \
    test_image_paths, test_mask_paths_list, test_image_types = get_covid_qu_ex_paths(DATA_DIR)

    # Create TensorFlow Dataset for testing
    # IMPORTANT: Pass the test_mask_paths_list (which is a list of two lists) and test_image_types
    test_dataset = get_dataset(test_image_paths, test_mask_paths_list, test_image_types,
                               batch_size=BATCH_SIZE, augment=False, shuffle=False)

    # Load the trained model, including custom objects (metrics/loss functions)
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Please ensure you have run 'python -m src.main --train' first to train and save the model.")
        return

    print(f"Loading model from {MODEL_PATH}...")
    try:
        model = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={
                'dice_coeff_multi_class': dice_coeff_multi_class,
                'dice_loss_multi_class': dice_loss_multi_class,
                'combined_loss_multi_class': combined_loss_multi_class,
                'build_unet_resnet50': build_unet_resnet50 # Include model builder for robustness
            }
        )
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Ensure the custom_objects dictionary correctly maps custom function names and the model structure.")
        return

    # Evaluate the model on the test dataset
    print("\nEvaluating model on the test set...")
    # The metrics during evaluation will match those compiled in train.py
    loss, categorical_accuracy, dice_coefficient_score = model.evaluate(test_dataset, verbose=1)

    print("\n--- Evaluation Results ---")
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Categorical Accuracy: {categorical_accuracy:.4f}")
    print(f"Test Dice Coefficient (Mean across classes): {dice_coefficient_score:.4f}")
    print("------------------------")

    print("\nEvaluation complete.")

if __name__ == '__main__':
    main()