import tensorflow as tf
import os
import numpy as np
import datetime

# Importing modules
from .data_loader import get_dataset, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS, NUM_CLASSES, get_covid_qu_ex_paths
from .model import build_unet_resnet50

# ---- Config Params ----
DATA_DIR = 'data/raw/COVID-QU-Ex_Dataset' # Path relative to src/

MODEL_SAVE_DIR = 'saved_models' # This path is relative to the current working directory (src/)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# Hyperparams
BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 1e-4

# --- Custom Metrics and Loss Functions (Multi-Class versions) ---

def dice_coeff_multi_class(y_true, y_pred, smooth=1e-7):
    """
    Computes the Dice Coefficient for multi-class segmentation.
    y_true and y_pred are expected to be one-hot encoded.
    """
    # Flatten spatial dimensions
    y_true_f = tf.cast(tf.reshape(y_true, [-1, NUM_CLASSES]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1, NUM_CLASSES]), tf.float32)

    # Calculate Dice for each class
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=0)
    union = tf.reduce_sum(y_true_f + y_pred_f, axis=0)
    dice = (2. * intersection + smooth) / (union + smooth)
    
    # Exclude background class (class 0) from mean calculation if desired,
    # or include all for overall metric. Here, we'll average over all classes.
    return tf.reduce_mean(dice)

def dice_loss_multi_class(y_true, y_pred):
    """
    Computes the Dice Loss for multi-class segmentation.
    """
    return 1 - dice_coeff_multi_class(y_true, y_pred)

def combined_loss_multi_class(y_true, y_pred):
    """
    Combines Categorical Cross-Entropy and Multi-Class Dice Loss.
    """
    # Categorical Cross-Entropy for one-hot encoded masks
    cce_loss = tf.keras.losses.CategoricalCrossentropy()(y_true, y_pred)
    dice = dice_loss_multi_class(y_true, y_pred)
    return cce_loss + dice

# --- Main Training Function ---
def main():
    print("Starting multi-class training script...")
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print('Mixed precision enabled')
    # ---- 1. Prepare Data Paths ----
    # Updated get_covid_qu_ex_paths to return (image_paths, [covid_mask_paths, lung_mask_paths], image_types)
    train_image_paths, train_mask_paths_list, train_image_types, \
    val_image_paths, val_mask_paths_list, val_image_types, \
    test_image_paths, test_mask_paths_list, test_image_types = get_covid_qu_ex_paths(DATA_DIR)

    # ---- 3. Create Tensorflow Datasets ----
    # Pass the image types to the dataset as well, as load_image_and_multi_class_mask needs it
    train_dataset = get_dataset(train_image_paths, train_mask_paths_list, train_image_types,
                                batch_size=BATCH_SIZE, augment=True, shuffle=True)
    val_dataset = get_dataset(val_image_paths, val_mask_paths_list, val_image_types,
                              batch_size=BATCH_SIZE, augment=False, shuffle=False)

    # ---- 4. Build the Model ----
    tf.config.optimizer.set_jit(True)
    print('XLA JIT compiler enabled')
    model = build_unet_resnet50(input_shape=(IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS), num_classes=NUM_CLASSES)
    print("\nModel Built Successfully")
    model.summary()

    # ---- 5. Compile the Model ----
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    # Use multi-class loss and metric
    model.compile(optimizer=optimizer, loss=combined_loss_multi_class, metrics=[tf.keras.metrics.CategoricalAccuracy(), dice_coeff_multi_class])

    print("\nModel Successfully Compiled")

    # ---- 6. Define Callbacks ----
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

    # ---- 7. Training the Model ----
    print("\nStarting training...")
    history = model.fit(train_dataset, epochs=EPOCHS, validation_data=val_dataset, verbose=1, callbacks=callback_list)
    print("\nModel training complete.")


if __name__ == "__main__":
    main()