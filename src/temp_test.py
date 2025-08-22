import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import random # Needed for selecting random samples for visualization

# Adjust the Python path to allow importing modules from the 'src' directory
# This assumes temp_test.py is located in the 'src' directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__)))) # Add src directory itself

# Import functions and constants from data_loader.py
# Note: load_image is replaced by load_image_and_multi_class_mask
from .data_loader import get_covid_qu_ex_paths, load_image_and_multi_class_mask, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS, NUM_CLASSES
# Import the model building function from model.py
from .model import build_unet_resnet50

# --- Configuration for Testing ---
# DATA_DIR should be relative to where your 'data' folder is located
# If temp_test.py is in 'src/' and 'data' is in the project root:
DATA_DIR = './data/raw/COVID-QU-Ex_Dataset' # Note the underscore to match your dataset folder name

# Define a colormap for multi-class visualization (copied from predict_and_visualize.py)
# Class 0: Background (Black)
# Class 1: Healthy Lung (Green)
# Class 2: COVID-19 Infection (Red)
# Class 3: Non-COVID Infection (Blue)
COLORMAP = np.array([
    [0, 0, 0],         # 0: Background (Black)
    [0, 128, 0],       # 1: Healthy Lung (Green)
    [255, 0, 0],       # 2: COVID-19 Infection (Red)
    [0, 0, 255]        # 3: Non-COVID Infection (Blue)
], dtype=np.uint8)

def visualize_multi_class_mask_for_test(mask_one_hot):
    """
    Converts a one-hot encoded mask to an RGB image for visualization.
    This is a local helper function for temp_test.py.
    """
    # Get the class index for each pixel
    mask_class_indices = tf.argmax(mask_one_hot, axis=-1)
    
    # Map class indices to colors
    colored_mask = tf.gather(COLORMAP, mask_class_indices)
    return colored_mask.numpy()

def main():
    print("--- Starting Multi-Class Data Loader & Model Test ---")

    # --- 1. Test Data Loading and Preprocessing ---
    print("\nAttempting to load data paths from get_covid_qu_ex_paths...")
    try:
        # get_covid_qu_ex_paths now returns nested lists for masks and types
        train_image_paths, train_mask_paths_list, train_image_types, \
        val_image_paths, val_mask_paths_list, val_image_types, \
        test_image_paths, test_mask_paths_list, test_image_types = get_covid_qu_ex_paths(DATA_DIR)

        print(f"Successfully loaded {len(train_image_paths)} training samples.")
        print(f"Successfully loaded {len(val_image_paths)} validation samples.")
        print(f"Successfully loaded {len(test_image_paths)} test samples.")

        if not train_image_paths:
            print("WARNING: No training images found. Please check DATA_DIR and dataset structure closely.")
        
        print("\nVisualizing a few samples from the training set:")
        samples_to_visualize = 5
        indices = random.sample(range(len(train_image_paths)), min(samples_to_visualize, len(train_image_paths))) # Use random.sample for unique indices

        plt.figure(figsize=(15, samples_to_visualize * 3))
        plt.suptitle("Sample Images and Multi-Class Masks", fontsize=16)

        for i, idx in enumerate(indices):
            img_path = train_image_paths[idx]
            # Accessing the specific masks from the lists returned by get_covid_qu_ex_paths
            covid_mask_path = train_mask_paths_list[0][idx]
            lung_mask_path = train_mask_paths_list[1][idx]
            image_type = train_image_types[idx]

            # Use the updated load_image_and_multi_class_mask
            image, true_mask_one_hot = load_image_and_multi_class_mask(img_path, covid_mask_path, lung_mask_path, image_type)
            
            # Convert one-hot encoded mask to colored image for visualization
            true_colored_mask = visualize_multi_class_mask_for_test(true_mask_one_hot)

            # Plot Original Image
            plt.subplot(samples_to_visualize, 3, i * 3 + 1)
            plt.imshow(image.numpy())
            plt.title(f"Original ({image_type}): {os.path.basename(img_path)}")
            plt.axis('off')

            # Plot True Mask (multi-class colored)
            plt.subplot(samples_to_visualize, 3, i * 3 + 2)
            plt.imshow(true_colored_mask)
            plt.title("True Mask (Multi-Class)")
            plt.axis('off')

            # Placeholder for predicted mask (not applicable in this test)
            plt.subplot(samples_to_visualize, 3, i * 3 + 3)
            plt.axis('off')
            plt.text(0.5, 0.5, "Prediction Placeholder", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=10)

        # Add a legend for the colormap (copied from predict_and_visualize.py)
        legend_elements = [
            plt.Line2D([0], [0], marker='s', color='w', label='Class 0: Background (Black)', markerfacecolor='black', markersize=10),
            plt.Line2D([0], [0], marker='s', color='w', label='Class 1: Healthy Lung (Green)', markerfacecolor='green', markersize=10),
            plt.Line2D([0], [0], marker='s', color='w', label='Class 2: COVID-19 Infection (Red)', markerfacecolor='red', markersize=10),
            plt.Line2D([0], [0], marker='s', color='w', label='Class 3: Non-COVID Infection (Blue)', markerfacecolor='blue', markersize=10)
        ]
        plt.figlegend(handles=legend_elements, loc='lower center', ncol=2, bbox_to_anchor=(0.5, 0.0), fontsize=10) # Adjusted bbox_to_anchor for test script

        plt.tight_layout(rect=[0, 0.08, 1, 0.95]) # Adjust layout to prevent suptitle/legend overlap
        plt.show()

    except Exception as e:
        print(f"\nERROR during data loading/visualization test: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for detailed error info
        sys.exit(1) # Exit if data loading fails

    print("\n--- Data Loading Test Complete ---")

    # --- 2. Test Model Instantiation ---
    print("\nAttempting to build the U-Net ResNet50 model (Multi-Class)...")
    try:
        model = build_unet_resnet50(input_shape=(IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS), num_classes=NUM_CLASSES)
        print("\nModel Built Successfully.")
        model.summary() # Print model summary to verify architecture
        # Optional: Test with a dummy input to ensure output shape is correct
        dummy_input = tf.random.uniform((1, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS))
        dummy_output = model(dummy_input)
        print(f"\nDummy model output shape: {dummy_output.shape}")
        if dummy_output.shape == (1, IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES):
            print("Model output shape is correct for multi-class segmentation.")
        else:
            print("WARNING: Model output shape is unexpected for multi-class segmentation.")

    except Exception as e:
        print(f"\nERROR during model building test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) # Exit if model building fails

    print("\n--- Model Instantiation Test Complete ---")
    print("\nAll pre-training checks passed. You can now proceed with training your multi-class model!")

if __name__ == "__main__":
    main()