import tensorflow as tf
import os
import numpy as np

# Import constants and functions from other modules
from .data_loader import get_classification_paths, get_classification_dataset, NUM_CLASSES
from .model import build_classification_model
from .train import categorical_accuracy_per_class

# ---- Config Params ----
DATA_DIR = 'data/raw/COVID-QU-Ex_Dataset' 
MODEL_SAVE_DIR = 'saved_models_classification_iter2'
MODEL_NAME = 'best_classification_model.h5'
BATCH_SIZE = 64

def main():
    """
    Main function to evaluate the trained classification model on the test dataset.
    """
    print("Starting evaluation script...")

    # ---- 1. Prepare Data Paths ----
    _, _, _, _, test_image_paths, test_labels = get_classification_paths(DATA_DIR)

    # ---- 2. Create Tensorflow Test Dataset ----
    test_dataset = get_classification_dataset(test_image_paths, test_labels, BATCH_SIZE, shuffle=False)

     # ---- 3. Create a dictionary of custom objects ----

    class_names = ['Normal', 'Non-COVID', 'COVID-19']
    custom_objects = {}
    for i in range(NUM_CLASSES):
        metric_name = f'class_{i}_accuracy_{class_names[i]}'
        custom_objects[metric_name] = categorical_accuracy_per_class(i, name=metric_name)

    # ---- 4. Load the Saved Model ----
    model_path = os.path.join(MODEL_SAVE_DIR, MODEL_NAME)
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}. Please train the model first.")
        return

    # Load the model, passing the custom objects dictionary.
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    print("Model loaded successfully.")
    # ---- 4. Evaluate the Model ----
    print("\nStarting model evaluation on the test set...")
    results = model.evaluate(test_dataset, verbose=1)

    # ---- 5. Display results ---
    print("\nEvaluation Results:")
    for metric_name, result in zip(model.metrics_names, results):
        print(f"{metric_name}: {result:.4f}")

# For standalone execution
if __name__ == '__main__':
    main()
