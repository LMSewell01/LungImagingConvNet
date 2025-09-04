import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout

# Import constants from data_loader.py
from data_loader import IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS, NUM_CLASSES

def build_classification_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS), num_classes=NUM_CLASSES, fine_tune=True):
    """
    Builds a classification model using a ResNet50 backbone with optional fine-tuning.
    
    Args:
        input_shape (tuple): The shape of the input images (height, width, channels).
        num_classes (int): The number of output classes.
        fine_tune (bool): If True, unfreezes the last layers of the backbone for fine-tuning.
        
    Returns:
        tf.keras.Model: The classification model.
    """
    # Load ResNet50 as the backbone, pre-trained on ImageNet, without the top classification layer
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

    # --- Fine-Tuning Logic ---
    if fine_tune:
        print("Fine-tuning: Unfreezing the last few layers of the ResNet50 backbone...")
        # Unfreeze all layers from 'conv3_block4_out' onwards.
        set_trainable = False
        for layer in base_model.layers:
            if layer.name == 'conv3_block4_out':
                set_trainable = True
            if set_trainable:
                layer.trainable = True
            else:
                layer.trainable = False
    else:
        print("Freezing the entire ResNet50 backbone.")
        base_model.trainable = False

    # Create the classification head on top of the backbone
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    
    # Adding an additional Dense layer for better feature combination and to increase model capacity.
    x = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = Dropout(0.5)(x)
    
    # The final dense layer for multi-class classification
    predictions = Dense(num_classes, activation='softmax')(x)

    # Combine the backbone and the new classification head into a single model
    model = Model(inputs=base_model.input, outputs=predictions)

    return model
