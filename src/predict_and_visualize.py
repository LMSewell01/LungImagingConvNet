import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import glob

# Import functions and constants from data_loader.py
from data_loader import get_covid_qu_ex_paths, load_image_and_multi_class_mask, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS, NUM_CLASSES
# Import the U-Net model from model.py
from model import build_unet_resnet50

# Import custom metrics and loss functions (now multi-class versions)
from train import dice_coeff_multi_class, dice_loss_multi_class, combined_loss_multi_class, iou_metric_per_class

# --- Configuration Parameters for Prediction and Visualization ---
DATA_DIR = '../data/raw/COVID-QU-Ex_Dataset'
print(os.path.exists('../data/raw/COVID-QU-Ex_Dataset'))
MODEL_SAVE_DIR = '../saved_models'
MODEL_PATH = os.path.join(MODEL_SAVE_DIR, 'best_unet_resnet50_multi_class_iter2.h5')
NUM_PREDICT_SAMPLES = 5 # Number of random samples to predict and visualize

# Define a colormap for multi-class visualization
# Class 0: Background (Black)
# Class 1: Healthy Lung (Green)
# Class 2: COVID-19 Infection (Red)
# Class 3: Non-COVID Infection (Blue)
COLORMAP = np.array([
    [0, 0, 0],       # Black for Background
    [0, 255, 0],     # Green for Healthy Lung
    [255, 0, 0],     # Red for COVID-19 Infection
    [0, 0, 255]      # Blue for Non-COVID Infection
])

def visualize_multi_class_mask(one_hot_mask):
    """
    Converts a one-hot encoded mask to a colored RGB mask for visualization.
    """
    # Get the class indices for each pixel
    class_indices = tf.argmax(one_hot_mask, axis=-1)
    
    # Map the class indices to the predefined colormap
    colored_mask = tf.gather(COLORMAP, class_indices)
    
    return colored_mask

def main():
    """
    Main function to predict on a few random samples and visualize the results.
    """
    print("Starting prediction and visualization script...")

    # ---- 1. Prepare Data Paths ----
    _, _, _, _, _, _, \
    test_image_paths, test_mask_paths_list, test_image_types = get_covid_qu_ex_paths(DATA_DIR)
    
    if not test_image_paths:
        print("No test images found. Please check your data directory and paths.")
        return

    # ---- 2. Load the Model ----
    print(f"Loading model from {MODEL_PATH}...")
    
    custom_objects = {
        'combined_loss_multi_class': combined_loss_multi_class,
        'dice_coeff_multi_class': dice_coeff_multi_class,
        'iou_healthy_lung': iou_metric_per_class(1, 'iou_healthy_lung'),
        'iou_covid': iou_metric_per_class(2, 'iou_covid'),
        'iou_non_covid': iou_metric_per_class(3, 'iou_non_covid'),
        'iou_background': iou_metric_per_class(0, 'iou_background'),
    }

    try:
        model = load_model(MODEL_PATH, custom_objects=custom_objects)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading the model: {e}")
        print("Please ensure the model file exists and the custom objects are correctly defined.")
        return

    # ---- 3. Select Random Samples ----
    # Ensure we don't try to select more samples than available
    num_samples = min(NUM_PREDICT_SAMPLES, len(test_image_paths))
    selected_indices = random.sample(range(len(test_image_paths)), num_samples)

    print(f"\nPredicting and visualizing {num_samples} random samples from the test set...")

    # ---- 4. Plot Predictions ----
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, num_samples * 5))
    if num_samples == 1:
        axes = [axes] # Ensure axes is iterable for a single sample

    for i, idx in enumerate(selected_indices):
        img_path = test_image_paths[idx]
        covid_mask_path = test_mask_paths_list[0][idx]
        lung_mask_path = test_mask_paths_list[1][idx]
        image_type = test_image_types[idx]

        # Load and preprocess the single image and its mask
        image, true_mask = load_image_and_multi_class_mask(img_path, covid_mask_path, lung_mask_path, image_type)
        
        # Expand dimensions to create a batch of 1
        image_batch = tf.expand_dims(image, axis=0)
        
        # Predict the mask
        predicted_mask = model.predict(image_batch, verbose=0)
        
        # Squeeze the batch dimension and get the one-hot encoded mask
        predicted_mask = tf.squeeze(predicted_mask, axis=0)

        # Convert the one-hot encoded masks to colored masks
        true_colored_mask = visualize_multi_class_mask(true_mask)
        predicted_colored_mask = visualize_multi_class_mask(predicted_mask)

        # Plot original image
        axes[i][0].imshow(image.numpy())
        axes[i][0].set_title(f"Original ({image_type}):\n{os.path.basename(img_path)}")
        axes[i][0].axis('off')

        # Plot true mask
        axes[i][1].imshow(true_colored_mask)
        axes[i][1].set_title("True Mask")
        axes[i][1].axis('off')

        # Plot predicted mask
        axes[i][2].imshow(predicted_colored_mask)
        axes[i][2].set_title("Predicted Mask")
        axes[i][2].axis('off')
    
    # Add a legend for the colormap
    legend_elements = [
        plt.Line2D([0], [0], marker='s', color='w', label='Class 0: Background', markerfacecolor='black', markersize=10),
        plt.Line2D([0], [0], marker='s', color='w', label='Class 1: Healthy Lung', markerfacecolor='green', markersize=10),
        plt.Line2D([0], [0], marker='s', color='w', label='Class 2: COVID-19 Infection', markerfacecolor='red', markersize=10),
        plt.Line2D([0], [0], marker='s', color='w', label='Class 3: Non-COVID Infection', markerfacecolor='blue', markersize=10),
    ]
    
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=4)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # ---- 5. Additional Visualization for COVID-19 Test Images ----
    NUM_SAMPLES = 10  # Set how many samples to show

    covid_img_dir = os.path.join('../data/raw/Infection_Segmentation_Data', 'Test', 'COVID-19', 'images')
    covid_infection_mask_dir = os.path.join('../data/raw/Infection_Segmentation_Data', 'Test', 'COVID-19', 'infection masks')

    covid_image_paths = sorted(glob.glob(os.path.join(covid_img_dir, '*.png')) + glob.glob(os.path.join(covid_img_dir, '*.jpg')))
    covid_infection_mask_paths = []
    for img_path in covid_image_paths:
        img_name = os.path.basename(img_path)  # No suffix, just the filename
        mask_path = os.path.join(covid_infection_mask_dir, img_name)
        covid_infection_mask_paths.append(mask_path)

    # Only keep pairs where infection mask exists
    covid_pairs = [(img, mask) for img, mask in zip(covid_image_paths, covid_infection_mask_paths) if os.path.exists(mask)]
    if not covid_pairs:
        print("No COVID-19 test images with infection masks found.")
        return

    # Select random samples
    num_samples = min(NUM_SAMPLES, len(covid_pairs))
    selected_pairs = random.sample(covid_pairs, num_samples)

    fig, axes = plt.subplots(num_samples, 3, figsize=(15, num_samples * 5))
    if num_samples == 1:
        axes = [axes]

    for i, (img_path, infection_mask_path) in enumerate(selected_pairs):
        image_type = 'COVID-19'
        # Load image
        image = tf.io.read_file(img_path)
        image = tf.image.decode_png(image, channels=NUM_CHANNELS)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])

        # Load infection mask
        infection_mask = tf.io.read_file(infection_mask_path)
        infection_mask = tf.image.decode_png(infection_mask, channels=1)
        infection_mask = tf.image.convert_image_dtype(infection_mask, tf.float32)
        infection_mask = tf.image.resize(infection_mask, [IMG_HEIGHT, IMG_WIDTH], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        infection_mask_bin = tf.where(infection_mask > 0.5, 1, 0)  # Binarize

        # Convert binary infection mask to one-hot multi-class mask (COVID class = 2)
        int_mask = tf.squeeze(infection_mask_bin, axis=-1) * 2  # 0 for background, 2 for COVID
        one_hot_mask = tf.one_hot(int_mask, NUM_CLASSES)
        colored_infection_mask = visualize_multi_class_mask(one_hot_mask)

        # Predict mask
        image_batch = tf.expand_dims(image, axis=0)
        predicted_mask = model.predict(image_batch, verbose=0)
        predicted_mask = tf.squeeze(predicted_mask, axis=0)
        pred_colored_mask = visualize_multi_class_mask(predicted_mask)

        # Visualize
        axes[i][0].imshow(image.numpy())
        axes[i][0].set_title(f"Original (COVID-19):\n{os.path.basename(img_path)}")
        axes[i][0].axis('off')

        axes[i][1].imshow(colored_infection_mask)
        axes[i][1].set_title("Infection Segmentation Mask (Colorized)")
        axes[i][1].axis('off')

        axes[i][2].imshow(pred_colored_mask)
        axes[i][2].set_title("Predicted Mask")
        axes[i][2].axis('off')

    legend_elements = [
        plt.Line2D([0], [0], marker='s', color='w', label='Class 0: Background', markerfacecolor='black', markersize=10),
        plt.Line2D([0], [0], marker='s', color='w', label='Class 1: Healthy Lung', markerfacecolor='green', markersize=10),
        plt.Line2D([0], [0], marker='s', color='w', label='Class 2: COVID-19 Infection', markerfacecolor='red', markersize=10),
        plt.Line2D([0], [0], marker='s', color='w', label='Class 3: Non-COVID Infection', markerfacecolor='blue', markersize=10),
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=4)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == '__main__':
    main()
