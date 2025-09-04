import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import numpy as np
import matplotlib.pyplot as plt
import random

# Import constants and functions from data_loader.py
from data_loader import get_classification_paths, load_image_and_label, LABEL_TO_CLASS
from model import build_classification_model

# --- Configuration Parameters for Prediction and Visualization ---
DATA_DIR = '../data/raw/COVID-QU-Ex_Dataset'
MODEL_SAVE_DIR = '../saved_models/saved_models_classification_iter2'
MODEL_PATH = os.path.join(MODEL_SAVE_DIR, 'best_classification_model.h5')
NUM_PREDICT_SAMPLES = 5  # Number of random samples to predict and visualize

def main():
    """
    Main function to run predictions and visualize results for the classification model.
    Shows a 4x4 grid of images and a pie chart of correct/incorrect predictions.
    """
    NUM_PREDICT_SAMPLES = 16
    GRID_ROWS, GRID_COLS = 4, 4

    print("Starting prediction and visualization script...")

    # ---- 1. Prepare Data Paths and Labels ----
    _, _, _, _, test_image_paths, test_labels = get_classification_paths(DATA_DIR)

    if not test_image_paths:
        print("Error: No test images found. Cannot run prediction.")
        return

    # ---- 2. Load the Trained Model ----
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}. Please train the model first.")
        return

    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")

    # ---- 3. Select Random Samples and Predict ----
    selected_indices = random.sample(range(len(test_image_paths)), min(NUM_PREDICT_SAMPLES, len(test_image_paths)))
    correct_count = 0
    incorrect_count = 0
    results = []

    fig, axes = plt.subplots(GRID_ROWS, GRID_COLS, figsize=(18, 18))
    plt.suptitle("Classification Predictions (4x4 Grid)", fontsize=18)

    for i, idx in enumerate(selected_indices):
        if i == NUM_PREDICT_SAMPLES - 1:
            # Pie chart will go here, skip image
            continue

        image_path = test_image_paths[idx]
        true_label_idx = test_labels[idx]
        true_label = LABEL_TO_CLASS[true_label_idx]

        # Load and preprocess a single image
        image, _ = load_image_and_label(image_path, true_label_idx)
        image_batch = tf.expand_dims(image, axis=0)
        prediction = model.predict(image_batch, verbose=0)
        predicted_label_idx = np.argmax(prediction, axis=1)[0]
        predicted_label = LABEL_TO_CLASS[predicted_label_idx]

        is_correct = (true_label == predicted_label)
        if is_correct:
            correct_count += 1
        else:
            incorrect_count += 1
        results.append(is_correct)

        row, col = divmod(i, GRID_COLS)
        ax = axes[row][col]
        ax.imshow(image.numpy())
        title = f"True: {true_label}\nPred: {predicted_label}"
        title_color = 'green' if is_correct else 'red'
        ax.set_title(title, color=title_color, fontweight='bold', fontsize=12)
        ax.axis('off')

    # ---- 4. Pie Chart of Correct/Incorrect Predictions ----
    # Place pie chart in the last subplot (bottom right)
    pie_ax = axes[GRID_ROWS - 1][GRID_COLS - 1]
    pie_ax.clear()
    pie_ax.pie([correct_count, incorrect_count],
               labels=['Correct', 'Incorrect'],
               colors=['#4CAF50', '#F44336'],
               autopct='%1.0f%%',
               startangle=90,
               textprops={'fontsize': 12})
    pie_ax.set_title("Sample Accuracy", fontsize=14)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# For standalone execution
if __name__ == '__main__':
    main()
