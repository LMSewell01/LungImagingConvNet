# Importing modules
import tensorflow as tf
import os
import datetime
import numpy as np

from .data_loader import get_classification_paths, get_classification_dataset, NUM_CLASSES
from .model import build_classification_model

# ---- Config Params ----
DATA_DIR = 'data/raw/COVID-QU-Ex_Dataset' 
MODEL_SAVE_DIR = 'saved_models_classification_iter4'
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# Hyperparams
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-4

CLASS_WEIGHTS = {
    0: 1.0,   # Normal
    1: 1.2,   # Non-COVID
    2: 1.0    # COVID-19
}
# --- Custom Metrics ---

def categorical_accuracy_per_class(class_id, name):
    """
    Creates a custom metric function to compute categorical accuracy for a specific class.
    
    Args:
        class_id (int): The index of the class to monitor.
        name (str): The name for the metric to be displayed in training logs.

    Returns:
        A callable function that computes the metric.
    """
    def metric_fn(y_true, y_pred, **kwargs):
        """
        Custom metric function to be used by Keras.
        
        Args:
            y_true (tf.Tensor): The true labels.
            y_pred (tf.Tensor): The predicted labels.
            **kwargs: Catches any extra arguments passed by Keras, such as 'sample_weight'.
        """
        # Cast y_true and y_pred to a common float type for computation
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)

        # Get the one-hot encoded true and predicted labels for the target class
        class_true = y_true[:, class_id]
        class_pred = tf.cast(tf.argmax(y_pred, axis=1) == class_id, dtype=tf.float32)

        # Compute the accuracy for the specified class
        return tf.reduce_mean(tf.cast(tf.equal(class_true, class_pred), dtype=tf.float32))

    # This is the key change. We now return the callable function itself, not a Metric object.
    metric_fn.__name__ = name
    return metric_fn

# --- Main Training Function ---

def main():
    print("Starting training script for classification model...")
    
    # --- 1. Prepare Data ---
    train_image_paths, train_labels, val_image_paths, val_labels, _, _ = get_classification_paths(DATA_DIR)
    
    train_dataset = get_classification_dataset(train_image_paths, train_labels, BATCH_SIZE)
    val_dataset = get_classification_dataset(val_image_paths, val_labels, BATCH_SIZE)

    # --- 2. Build and Compile Model ---
    model = build_classification_model(num_classes=NUM_CLASSES, fine_tune=True)
    
    # Use a classification loss and metric
    loss_function = tf.keras.losses.CategoricalCrossentropy()
    
    # Define a list of metrics, including the new per-class accuracies
    metrics = [tf.keras.metrics.CategoricalAccuracy(name='categorical_accuracy')]
    class_names = ['Normal', 'Non-COVID', 'COVID-19']
    for i in range(NUM_CLASSES):
        metrics.append(categorical_accuracy_per_class(i, name=f'class_{i}_accuracy_{class_names[i]}'))

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    
    model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)
    print("Model successfully Compiled")

    # --- 3. Define Callbacks ---
    checkpoint_filepath = os.path.join(MODEL_SAVE_DIR, 'best_classification_model.h5')
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath = checkpoint_filepath,
        monitor = 'val_loss',
        mode = 'min',
        save_best_only = True,
        verbose=1
    )
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        mode='min',
        restore_best_weights=True,
        verbose=1
    )
    
    callback_list =[model_checkpoint_callback, early_stopping_callback]

    log_dir = "logs/fit_classification/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    callback_list.append(tensorboard_callback)

    # --- 4. Train the Model ---
    print("\nStarting model training...")
    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=val_dataset,
        class_weight=CLASS_WEIGHTS,
        callbacks=callback_list,
        verbose=1
    )
    print("\nTraining complete.")

# For standalone execution
if __name__ == '__main__':
    main()


