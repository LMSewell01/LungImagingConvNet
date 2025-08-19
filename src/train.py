import tensorflow as tf
import os
import numpy as np  
import datetime # For TensorBoard logs

# Importing modules
from data_loader import get_dataset, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS, NUM_CLASSES, get_covid_qu_ex_paths
from model import build_unet_resnet50

# ---- Config Params ----
#DATA_DIR = 'data/raw'
DATA_DIR = 'data/raw/COVID-QU-Ex_Dataset'

MODEL_SAVE_DIR = 'saved_models'
os.makedirs(MODEL_SAVE_DIR, exist_ok=True) # Ensure the directory exists

# Hyperparams
BATCH_SIZE = 8
EPOCHS = 20 
LEARNING_RATE = 1e-4
# --- Custom Metrics and Loss Functions (Moved to top-level for importability) ---
def dice_coeff(y_true, y_pred, smooth = 1e-7):
    """ Dice Coefficeints for segmentation
    Args:
        y_true: Ground truth masks
        y_pred: Predicted masks
        smooth: Smoothing factor to avoid division by zero
    Returns:
        float: Dice coefficient
    """
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
    return (2. * intersection + smooth) / (union + smooth)

def dice_loss(y_true, y_pred):
    """
    Computes the Dice Loss, often used in conjunction with Binary Cross-Entropy.
    """
    return 1 - dice_coeff(y_true, y_pred)

def combined_loss(y_true, y_pred):
    """
    Combines Binary Cross-Entropy and Dice Loss.
    BCE often helps with overall pixel classification, Dice helps with class imbalance.
    """
    bce_loss = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return bce_loss + dice

# --- Main Training Function ---
def main():
    print("Starting training script...")

    # ---- 1. Prepare Data Paths ----
    train_image_paths, train_mask_paths, val_image_paths, val_mask_paths, test_image_paths, test_mask_paths = get_covid_qu_ex_paths(DATA_DIR)

    # ---- 3. Create Tensorflow Datasets ----
    train_dataset = get_dataset(train_image_paths, train_mask_paths, batch_size=BATCH_SIZE, augment=True, shuffle=True)
    val_dataset = get_dataset(val_image_paths, val_mask_paths, batch_size=BATCH_SIZE, augment=False, shuffle=False)

    # ---- 4. Build the Model ----
    model = build_unet_resnet50(input_shape=(IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS), num_classes=NUM_CLASSES)
    print("\nModel Built Successfully")
    model.summary() # Added model summary for inspection

    # ---- 5. Compile the Model ----
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss=combined_loss, metrics=[tf.keras.metrics.BinaryAccuracy(), dice_coeff])

    print("\nModel Successfully Compiled")

    # ---- 6. Define Callbacks ----
    checkpoint_filepath = os.path.join(MODEL_SAVE_DIR, 'best_unet_resnet50.h5')
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath = checkpoint_filepath,
        monitor = 'val_loss',
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

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    callback_list.append(tensorboard_callback) # Add TensorBoard to the list

    # ---- 7. Training the Model ----
    print("\nStarting training...")
    history = model.fit(train_dataset, epochs=EPOCHS, validation_data=val_dataset, verbose=1, callbacks=callback_list)
    print("\nModel training complete.")


if __name__ == "__main__":
    main()
