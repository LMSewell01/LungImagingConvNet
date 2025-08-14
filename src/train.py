import tensorflow as tf
import os
import numpy as np  

# Importing modules
from data_loader import get_dataset, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS, NUM_CLASSES
from model import build_unet_resnet50

# ---- Config Params ----
DATA_DIR = 'data/raw'
IMAGE_DIR = os.path.join(DATA_DIR, 'images')
MASK_DIR = os.path.join(DATA_DIR, 'masks')

# Define path for saving trained models
MODEL_SAVE_DIR = 'saved_models'
os.makedirs(MODEL_SAVE_DIR, exist_ok=True) # Ensure the directory exists

# Hyperparams
BATCH_SIZE = 8
EPOCHS = 10 
LEARNING_RATE = 1e-4

def main():
    # ---- 1. Prepare Data Paths (Initially Dummy Data) ----
    num_dummy_samples = 100
    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs(MASK_DIR, exist_ok=True)

    dummy_image_paths = []
    dummy_mask_paths = []

    print(f"Creating {num_dummy_samples} dummy images and masks for demo..")
    for i in range(num_dummy_samples):
        img_path = os.path.join(IMAGE_DIR, f"image_{i}.jpg")
        mask_path = os.path.join(MASK_DIR, f"mask_{i}.jpg")
        dummy_image_paths.append(img_path)
        dummy_mask_paths.append(mask_path)

        # Creating simple all white image:
        dummy_img = np.ones((IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS), dtype=np.uint8) * 255
        tf.io.write_file(img_path, tf.image.encode_jpeg(dummy_img, quality=100))

        # Create a black mask with a white square to simulate an abnormality
        dummy_mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.uint8)
        dummy_mask[IMG_HEIGHT//4:3*IMG_HEIGHT//4, IMG_WIDTH//4:3*IMG_WIDTH//4] = 255
        tf.io.write_file(mask_path, tf.image.encode_jpeg(dummy_mask, quality=100))
    print("Dummy Image Creation Complete")


    # ---- 2. Split Data (For Dummy Data) ----
    # For actual data would use a train, train/dev, val, test splits
    split_ratio = 0.8 # Again, just for dummy tests
    num_train = int(num_dummy_samples * split_ratio)

    train_image_paths = dummy_image_paths[:num_train]
    train_mask_paths = dummy_mask_paths[:num_train]

    val_image_paths = dummy_image_paths[num_train:]
    val_mask_paths = dummy_mask_paths[num_train:]

    print(f"Training samples: {len(train_image_paths)}")
    print(f"Validation samples: {len(val_image_paths)}")

    # ---- 3. Create Tensorflow Datasets ----
    train_dataset = get_dataset(train_image_paths, train_mask_paths, batch_size=BATCH_SIZE, augment=True, shuffle=True)
    val_dataset = get_dataset(val_image_paths, val_mask_paths, batch_size=BATCH_SIZE, augment=False, shuffle=False)

    # ---- 4. Build the Model ----
    model = build_unet_resnet50(input_shape=(IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS), num_classes=NUM_CLASSES)
    print("\nModel Built Successfully")

    # ---- 5. Compile the Model ----
    # Adam Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    # Dice Loss Function for use in loss function or as a metric 
    # https://cvinvolution.medium.com/dice-loss-in-medical-image-segmentation-d0e476eb486
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

    # Loss function
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[tf.keras.metrics.BinaryAccuracy(), dice_coeff])

    print("\n Model Successfully Compiled")

    # ---- 6. Define Callbacks ----
    # Save best model based on validation loss
    checkpoint_filepath = os.path.join(MODEL_SAVE_DIR, 'best_unet_resnet50.h5')
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath = checkpoint_filepath,
        monitor = 'val_loss',
        mode = 'min',
        save_best_only = True,
        verbose=1
    )
    # Early stopping if validation loss doesn't improve after 10 epochs
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min',
        restore_best_weights=True,
        verbose=1
    )

    # Reduce Learning Rate on Plateau
    reduce_lr_on_plateau_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        mode='min',
        verbose=1
    )

    # List of callbacks to pass to model.fit
    callback_list =[model_checkpoint_callback, early_stopping_callback, reduce_lr_on_plateau_callback]


    # ---- 7. Training the Model ----
    print("\nStarting training...")
    history = model.fit(train_dataset, epochs=EPOCHS, validation_data=val_dataset, verbose=1, callbacks=callback_list)
    print("\nModel training complete.")



if __name__ == "__main__":
    main()
