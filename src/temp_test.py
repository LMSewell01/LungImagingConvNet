import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Adjust the Python path to allow importing modules from the 'src' directory
# This assumes temp_test.py is located in the 'src' directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__)))) # Add src directory itself

# Import functions and constants from data_loader.py
from data_loader import get_covid_qu_ex_paths, load_image, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS, NUM_CLASSES, get_dataset
# Import the model building function from model.py
from model import build_unet_resnet50

# --- Configuration for Testing ---
# DATA_DIR should be relative to where your 'data' folder is located
# If temp_test.py is in 'src/' and 'data' is in the project root:
DATA_DIR = './data/raw/COVID-QU-Ex_Dataset' # Note the underscore to match your dataset folder name

def main():
    print("--- Starting Data Loader & Model Test ---")

    # --- 1. Test Data Loading and Preprocessing ---
    print("\nAttempting to load data paths from get_covid_qu_ex_paths...")
    try:
        train_image_paths, train_mask_paths, val_image_paths, val_mask_paths, test_image_paths, test_mask_paths = get_covid_qu_ex_paths(DATA_DIR)
        print(f"Successfully loaded {len(train_image_paths)} training samples.")
        print(f"Successfully loaded {len(val_image_paths)} validation samples.")
        print(f"Successfully loaded {len(test_image_paths)} test samples.")

        if not train_image_paths:
            print("WARNING: No training images found. Please check DATA_DIR and dataset structure closely.")
        
        print("\nVisualizing a few samples from the training set:")
        # Take a subset to visualize to avoid too many plots
        samples_to_visualize = 5
        indices = np.random.choice(len(train_image_paths), min(samples_to_visualize, len(train_image_paths)), replace=False)

        plt.figure(figsize=(15, samples_to_visualize * 3))
        plt.suptitle("Sample Images and Masks (Pathology & Healthy)", fontsize=16)

        for i, idx in enumerate(indices):
            image, mask = load_image(train_image_paths[idx], train_mask_paths[idx])
            
            # Convert tensors to numpy arrays for plotting
            image_np = image.numpy()
            mask_np = mask.numpy().squeeze() # Remove singleton dimension for grayscale mask

            # Plot Original Image
            plt.subplot(samples_to_visualize, 3, i * 3 + 1)
            plt.imshow(image_np)
            plt.title(f"Original Image {os.path.basename(train_image_paths[idx])}")
            plt.axis('off')

            # Plot True Mask
            plt.subplot(samples_to_visualize, 3, i * 3 + 2)
            plt.imshow(mask_np, cmap='gray')
            mask_type = "Pathology" if np.any(mask_np > 0.5) else "Healthy (Blank Mask)"
            plt.title(f"True Mask ({mask_type})")
            plt.axis('off')

            # Placeholder for predicted mask (not applicable in this test)
            plt.subplot(samples_to_visualize, 3, i * 3 + 3)
            plt.axis('off')
            plt.text(0.5, 0.5, "Prediction Placeholder", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=10)


        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent suptitle overlap
        plt.show()

    except Exception as e:
        print(f"\nERROR during data loading/visualization test: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for detailed error info
        sys.exit(1) # Exit if data loading fails

    print("\n--- Data Loading Test Complete ---")

    # --- 2. Test Model Instantiation ---
    print("\nAttempting to build the U-Net ResNet50 model...")
    try:
        model = build_unet_resnet50(input_shape=(IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS), num_classes=NUM_CLASSES)
        print("\nModel Built Successfully.")
        model.summary() # Print model summary to verify architecture
        # Optional: Test with a dummy input to ensure output shape is correct
        dummy_input = tf.random.uniform((1, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS))
        dummy_output = model(dummy_input)
        print(f"\nDummy model output shape: {dummy_output.shape}")
        if dummy_output.shape == (1, IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES):
            print("Model output shape is correct.")
        else:
            print("WARNING: Model output shape is unexpected.")

    except Exception as e:
        print(f"\nERROR during model building test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) # Exit if model building fails

    print("\n--- Model Instantiation Test Complete ---")
    print("\nAll pre-training checks passed. You can now proceed with training!")

if __name__ == "__main__":
    main()