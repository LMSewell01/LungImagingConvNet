import tensorflow as tf
import os
import numpy as np
import datetime

# Importing modules
from data_loader import get_dataset, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS, NUM_CLASSES, get_covid_qu_ex_paths
from model import build_unet_resnet50

# ---- Config Params ----
DATA_DIR = '../data/raw/COVID-QU-Ex_Dataset' 

MODEL_SAVE_DIR = 'saved_models' 
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# Hyperparams
BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 1e-4

# Manually defined class weights based on the imbalance observed.
CLASS_WEIGHTS = {
    0: 0.1,  # Background
    1: 3.0,  # Healthy Lung
    2: 10.0,  # COVID-19 Infection
    3: 3.0,  # Non-COVID Infection
}

# --- Custom Metrics and Loss Functions (Multi-Class versions) ---

def dice_coeff_multi_class(y_true, y_pred, smooth=1e-7):
    """
    Computes the Dice Coefficient for multi-class segmentation.
    y_true: One-hot encoded true masks.
    y_pred: Softmax predicted masks.
    """
    # Sum over spatial dimensions, then average over classes
    y_true_f = tf.cast(tf.reshape(y_true, [-1, NUM_CLASSES]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1, NUM_CLASSES]), tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=0)
    union = tf.reduce_sum(y_true_f, axis=0) + tf.reduce_sum(y_pred_f, axis=0)
    dice = (2. * intersection + smooth) / (union + smooth)
    return tf.reduce_mean(dice)

def dice_loss_multi_class(y_true, y_pred):
    return 1 - dice_coeff_multi_class(y_true, y_pred)

def combined_loss_multi_class(y_true, y_pred):
    """
    Combines Weighted Categorical Cross-Entropy and Dice Loss.
    """
    y_true_f = tf.cast(y_true, tf.float32)
    y_pred_f = tf.cast(y_pred, tf.float32)

    # Dice Loss
    dice_loss = dice_loss_multi_class(y_true_f, y_pred_f)

    # Weighted Categorical Cross-Entropy
    cross_entropy_loss = tf.keras.losses.categorical_crossentropy(y_true_f, y_pred_f)
    
    # Apply class weights
    weights = np.array(list(CLASS_WEIGHTS.values()))
    weights_tensor = tf.constant(weights, dtype=tf.float32)
    
    # y_true is one-hot encoded, so we can use tf.argmax to get the class index
    class_indices = tf.argmax(y_true_f, axis=-1)
    # Gather the weights for each pixel based on its true class
    per_pixel_weights = tf.gather(weights_tensor, class_indices)
    
    # Reshape the weights to match the loss tensor shape and apply element-wise
    weighted_cross_entropy = cross_entropy_loss * per_pixel_weights
    weighted_cross_entropy = tf.reduce_mean(weighted_cross_entropy)

    return weighted_cross_entropy + dice_loss

def iou_metric_per_class(class_id, name):
    """
    Computes the Intersection over Union (IoU) for a specific class.
    """
    def iou_for_class(y_true, y_pred):
        y_true_class = tf.cast(y_true[..., class_id], tf.float32)
        y_pred_class = tf.cast(tf.round(y_pred[..., class_id]), tf.float32)
        intersection = tf.reduce_sum(y_true_class * y_pred_class)
        union = tf.reduce_sum(y_true_class) + tf.reduce_sum(y_pred_class) - intersection
        # Add a small epsilon to avoid division by zero
        return (intersection + tf.keras.backend.epsilon()) / (union + tf.keras.backend.epsilon())
    iou_for_class.__name__ = name
    return iou_for_class

# --- Main Training Function ---
def main():
    print("Starting training script...")

    # ---- 1. Prepare Data Paths ----
    train_image_paths, train_mask_paths_list, train_image_types, \
    val_image_paths, val_mask_paths_list, val_image_types, \
    test_image_paths, test_mask_paths_list, test_image_types = get_covid_qu_ex_paths(DATA_DIR)

    # ---- 2. Create TensorFlow Datasets ----
    train_dataset = get_dataset(train_image_paths, train_mask_paths_list, train_image_types, BATCH_SIZE)
    val_dataset = get_dataset(val_image_paths, val_mask_paths_list, val_image_types, BATCH_SIZE)
    
    print("\nTraining and validation datasets created.")
    
    # ---- 3. Build the Model ----
    model = build_unet_resnet50(input_shape=(IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS), num_classes=NUM_CLASSES)
    
    print("\nU-Net model with ResNet50 backbone built.")
    
    # ---- 4. Compile the Model ----
    # Define the custom metrics to be used during training
    metrics = [
        dice_coeff_multi_class,
        iou_metric_per_class(0, 'iou_background'),
        iou_metric_per_class(1, 'iou_healthy_lung'),
        iou_metric_per_class(2, 'iou_covid'),
        iou_metric_per_class(3, 'iou_non_covid'),
    ]
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=combined_loss_multi_class,
        metrics=metrics,
    )
    print("Model successfully Compiled")

    # ---- 5. Define Callbacks ----
    checkpoint_filepath = os.path.join(MODEL_SAVE_DIR, 'best_unet_resnet50_multi_class.h5') # New model name
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath = checkpoint_filepath,
        monitor = 'val_loss', # Still monitor validation loss for saving best model
        mode = 'min',
        save_best_only = True,
        verbose=1
    )
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min',
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr_on_plateau_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        mode='min',
        verbose=1
    )
    callback_list =[model_checkpoint_callback, early_stopping_callback, reduce_lr_on_plateau_callback]

    log_dir = "logs/fit_multi_class/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # New log directory
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    callback_list.append(tensorboard_callback)

    # ---- 6. Train the Model ----
    print("\nStarting model training...")
    try:
        history = model.fit(
            train_dataset,
            epochs=EPOCHS,
            validation_data=val_dataset,
            callbacks=callback_list,
            verbose=1
        )
        print("\nModel training finished.")
        return history
    except Exception as e:
        print(f"An error occurred during training: {e}")
        return None

if __name__ == '__main__':
    main()
