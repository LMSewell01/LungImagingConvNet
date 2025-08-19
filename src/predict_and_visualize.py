import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import numpy as np
import matplotlib.pyplot as plt
import random

# Import functions and constants from data_loader.py
from data_loader import get_covid_qu_ex_paths, load_image_and_multi_class_mask, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS, NUM_CLASSES
# Import the U-Net model from model.py
from model import build_unet_resnet50

# Import custom metrics and loss functions (now multi-class versions)
from train import dice_coeff_multi_class, dice_loss_multi_class, combined_loss_multi_class

# --- Configuration Parameters for Prediction and Visualization ---
DATA_DIR = '../data/raw/COVID-QU-Ex_Dataset'
MODEL_SAVE_DIR = 'saved_models'
MODEL_PATH = os.path.join(MODEL_SAVE_DIR, 'best_unet_resnet50_multi_class.h5') # New model name
NUM_PREDICT_SAMPLES = 5 # Number of random samples to predict and visualize

# Define a colormap for multi-class visualization
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

def visualize_multi_class_mask(mask_one_hot):
    """
    Converts a one-hot encoded mask to an RGB image for visualization.
    """
    # Get the class index for each pixel
    mask_class_indices = tf.argmax(mask_one_hot, axis=-1)
    
    # Map class indices to colors
    colored_mask = tf.gather(COLORMAP, mask_class_indices)
    return colored_mask.numpy()

# --- Main Prediction and Visualization Function ---
def main():
    print("Starting multi-class prediction and visualization script...")

    # Load test dataset paths
    _, _, _, _, \
    _, _, _, \
    test_image_paths, test_mask_paths_list, test_image_types = get_covid_qu_ex_paths(DATA_DIR)

    # Load the trained model
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

    print(f"\nPredicting and visualizing {NUM_PREDICT_SAMPLES} random samples from the test set...")

    # Select random samples for prediction
    selected_indices = random.sample(range(len(test_image_paths)), min(NUM_PREDICT_SAMPLES, len(test_image_paths)))

    plt.figure(figsize=(18, NUM_PREDICT_SAMPLES * 4)) # Adjust figure size

    for i, idx in enumerate(selected_indices):
        img_path = test_image_paths[idx]
        covid_mask_path = test_mask_paths_list[0][idx]
        lung_mask_path = test_mask_paths_list[1][idx]
        image_type = test_image_types[idx]

        # Load and preprocess a single image and mask using the multi-class loader
        image, true_mask_one_hot = load_image_and_multi_class_mask(img_path, covid_mask_path, lung_mask_path, image_type)
        
        # Add batch dimension for prediction
        input_image = tf.expand_dims(image, 0) 

        # Make prediction (output will be probabilities for each class)
        predicted_mask_probs = model.predict(input_image)[0] # Remove batch dimension
        
        # Convert predicted probabilities to class indices (highest probability wins)
        predicted_mask_one_hot = tf.one_hot(tf.argmax(predicted_mask_probs, axis=-1), NUM_CLASSES)
        
        # Visualize the masks
        true_colored_mask = visualize_multi_class_mask(true_mask_one_hot)
        predicted_colored_mask = visualize_multi_class_mask(predicted_mask_one_hot)

        # Plot original image
        plt.subplot(NUM_PREDICT_SAMPLES, 3, i * 3 + 1)
        plt.imshow(image.numpy())
        plt.title(f"Original ({image_type}): {os.path.basename(img_path)}")
        plt.axis('off')

        # Plot true mask
        plt.subplot(NUM_PREDICT_SAMPLES, 3, i * 3 + 2)
        plt.imshow(true_colored_mask)
        plt.title("True Mask")
        plt.axis('off')

        # Plot predicted mask
        plt.subplot(NUM_PREDICT_SAMPLES, 3, i * 3 + 3)
        plt.imshow(predicted_colored_mask)
        plt.title("Predicted Mask")
        plt.axis('off')

    # Add a legend for the colormap
    legend_elements = [
        plt.Line2D([0], [0], marker='s', color='w', label='Class 0: Background', markerfacecolor='black', markersize=10),
        plt.Line2D([0], [0], marker='s', color='w', label='Class 1: Healthy Lung', markerfacecolor='green', markersize=10),
        plt.Line2D([0], [0], marker='s', color='w', label='Class 2: COVID-19 Infection', markerfacecolor='red', markersize=10),
        plt.Line2D([0], [0], marker='s', color='w', label='Class 3: Non-COVID Infection', markerfacecolor='blue', markersize=10)
    ]
    plt.figlegend(handles=legend_elements, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.05), fontsize=12)
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()

    print("\nVisualization complete.")

if __name__ == '__main__':
    main()
