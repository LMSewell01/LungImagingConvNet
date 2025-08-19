import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# Import functions and constants from data_loader.py
from data_loader import get_covid_qu_ex_paths, get_dataset, NUM_CLASSES, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS
# Import the U-Net model from model.py
from model import build_unet_resnet50

# Import custom metrics and loss functions (now multi-class versions)
from train import dice_coeff_multi_class, dice_loss_multi_class, combined_loss_multi_class

# --- Configuration Parameters for Evaluation ---
DATA_DIR = '../data/raw/COVID-QU-Ex_Dataset'
BATCH_SIZE = 8
MODEL_SAVE_DIR = 'saved_models'
MODEL_PATH = os.path.join(MODEL_SAVE_DIR, 'best_unet_resnet50_multi_class.h5') # New model name

# --- Main Evaluation Function ---
def main():
    print("Starting multi-class evaluation script...")

    # Load test dataset paths (get_covid_qu_ex_paths now returns extra lists)
    _, _, _, _, \
    _, _, _, \
    test_image_paths, test_mask_paths_list, test_image_types = get_covid_qu_ex_paths(DATA_DIR)

    # Create TensorFlow Dataset for testing
    test_dataset = get_dataset(test_image_paths, test_mask_paths_list, test_image_types,
                               batch_size=BATCH_SIZE, augment=False, shuffle=False)

    # Load the trained model, including custom objects
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}. Please train the model first.")
        return

    print(f"Loading model from {MODEL_PATH}...")
    model = load_model(
        MODEL_PATH,
        custom_objects={
            'dice_coeff_multi_class': dice_coeff_multi_class,
            'dice_loss_multi_class': dice_loss_multi_class,
            'combined_loss_multi_class': combined_loss_multi_class,
            'build_unet_resnet50': build_unet_resnet50
        }
    )
    print("Model loaded successfully.")

    # Evaluate the model on the test dataset
    print("\nEvaluating model on the test set...")
    # The metrics during evaluation will match those compiled in train.py
    loss, categorical_accuracy, dice_coefficient_score = model.evaluate(test_dataset)

    print(f"\nTest Loss: {loss:.4f}")
    print(f"Test Categorical Accuracy: {categorical_accuracy:.4f}")
    print(f"Test Dice Coefficient (Mean across classes): {dice_coefficient_score:.4f}")

    print("\nEvaluation complete.")

if __name__ == '__main__':
    main()
