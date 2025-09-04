import tensorflow as tf
import os
import numpy as np

# Import constants and functions from other modules
from .data_loader import get_dataset, get_covid_qu_ex_paths, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS
from .model import build_unet_resnet50

# Import the custom metrics and loss functions directly from the training script
from .train import combined_loss_multi_class, dice_coeff_multi_class, iou_metric_per_class 

# ---- Config Params ----
DATA_DIR = '../data/raw/COVID-QU-Ex_Dataset' 
MODEL_SAVE_DIR = 'saved_models' 
MODEL_NAME = 'best_unet_resnet50_multi_class.h5'
BATCH_SIZE = 128

def main():
    """
    Main function to evaluate the trained model on the test dataset.
    """
    print("Starting evaluation script...")

    # ---- 1. Prepare Data Paths ----
    train_image_paths, train_mask_paths_list, train_image_types, \
    val_image_paths, val_mask_paths_list, val_image_types, \
    test_image_paths, test_mask_paths_list, test_image_types = get_covid_qu_ex_paths(DATA_DIR)

    # ---- 2. Create Tensorflow Test Dataset ----
    test_dataset = get_dataset(test_image_paths, test_mask_paths_list, test_image_types, BATCH_SIZE, shuffle=False)
    print("\nTest dataset created.")

    # ---- 3. Load the Model ----
    model_path = os.path.join(MODEL_SAVE_DIR, MODEL_NAME)
    print(f"Loading model from {model_path}...")
    
    # Custom objects required to load the model correctly
    custom_objects = {
        'combined_loss_multi_class': combined_loss_multi_class,
        'dice_coeff_multi_class': dice_coeff_multi_class,
        'iou_healthy_lung': iou_metric_per_class(1, 'iou_healthy_lung'),
        'iou_covid': iou_metric_per_class(2, 'iou_covid'),
        'iou_non_covid': iou_metric_per_class(3, 'iou_non_covid'),
        'iou_background': iou_metric_per_class(0, 'iou_background'),
    }
    
    try:
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading the model: {e}")
        print("Please ensure the model file exists and the custom objects are correctly defined.")
        return
    
    # ---- 4. Re-compile the model with the same metrics used during training ----
    metrics = [
        dice_coeff_multi_class,
        iou_metric_per_class(0, 'iou_background'),
        iou_metric_per_class(1, 'iou_healthy_lung'),
        iou_metric_per_class(2, 'iou_covid'),
        iou_metric_per_class(3, 'iou_non_covid'),
    ]
    model.compile(optimizer='adam', loss=combined_loss_multi_class, metrics=metrics)
    

    # ---- 5. Evaluate the Model ----
    print("\nStarting model evaluation on the test set...")
    # The output of evaluate() will be in the same order as the metrics defined in .compile()
    results = model.evaluate(test_dataset, verbose=1)

    # ---- 6. Display results using a manually defined list of names ----
    metric_names = [
        'loss', 
        'dice_coeff_multi_class',
        'iou_background',
        'iou_healthy_lung',
        'iou_covid',
        'iou_non_covid'
    ]
    
    print("\n--- Evaluation Results ---")
    for name, value in zip(metric_names, results):
        print(f"{name}: {value:.4f}")
    
    print("\nEvaluation finished.")

if __name__ == '__main__':
    main()
