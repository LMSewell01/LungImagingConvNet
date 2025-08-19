import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import numpy as np
import matplotlib.pyplot as plt
import random

# Import functions and constants from data_loader.py
from data_loader import get_covid_qu_ex_paths, load_image, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS, NUM_CLASSES
# Import the U-Net model from model.py
from model import build_unet_resnet50 # Changed from unet_model to build_unet_resnet50

# Import custom metrics and loss functions from train.py (now at top-level)
from train import dice_coeff, dice_loss, combined_loss

# --- Configuration Parameters for Prediction and Visualization ---
DATA_DIR = 'data/raw/COVID-QU-Ex_Dataset' # MAKE SURE THIS PATH IS CORRECT
MODEL_SAVE_DIR = 'saved_models' # Must match the directory in train.py
MODEL_PATH = os.path.join(MODEL_SAVE_DIR, 'best_unet_resnet50.h5') # Path to the saved best model
NUM_PREDICT_SAMPLES = 5 # Number of random samples to predict and visualize

# --- Main Prediction and Visualization Function ---
def main():
    print("Starting prediction and visualization script...")

    # Load test dataset paths
    (_, _, _, _,
     test_image_paths, test_mask_paths) = get_covid_qu_ex_paths(DATA_DIR)

    # Load the trained model
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}. Please train the model first.")
        return

    print(f"Loading model from {MODEL_PATH}...")
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

    print(f"\nPredicting and visualizing {NUM_PREDICT_SAMPLES} random samples from the test set...")

    # Select random samples for prediction
    selected_indices = random.sample(range(len(test_image_paths)), min(NUM_PREDICT_SAMPLES, len(test_image_paths)))

    plt.figure(figsize=(15, NUM_PREDICT_SAMPLES * 3)) # Adjust figure size dynamically

    for i, idx in enumerate(selected_indices):
        img_path = test_image_paths[idx]
        mask_path = test_mask_paths[idx]

        # Load and preprocess a single image and mask
        image, true_mask = load_image(img_path, mask_path)
        
        # Add batch dimension for prediction
        input_image = tf.expand_dims(image, 0) 

        # Make prediction
        predicted_mask = model.predict(input_image)[0] # Remove batch dimension
        
        # Threshold the predicted mask to get binary segmentation (0 or 1)
        predicted_mask_binary = (predicted_mask > 0.5).astype(np.float32)

        # Plot original image
        plt.subplot(NUM_PREDICT_SAMPLES, 3, i * 3 + 1)
        plt.imshow(image.numpy())
        plt.title(f"Original Image ({os.path.basename(img_path)})")
        plt.axis('off')

        # Plot true mask
        plt.subplot(NUM_PREDICT_SAMPLES, 3, i * 3 + 2)
        plt.imshow(true_mask.numpy().squeeze(), cmap='gray') # squeeze for 2D mask
        plt.title("True Mask")
        plt.axis('off')

        # Plot predicted mask
        plt.subplot(NUM_PREDICT_SAMPLES, 3, i * 3 + 3)
        plt.imshow(predicted_mask_binary.squeeze(), cmap='gray') # squeeze for 2D mask
        plt.title("Predicted Mask (Binary)")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

    print("\nVisualization complete.")

if __name__ == '__main__':
    main()
