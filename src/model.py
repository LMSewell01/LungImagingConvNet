import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import ResNet50

# Import constants from data_loader.py
from data_loader import IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS, NUM_CLASSES # NUM_CLASSES is 4

def build_unet_resnet50(input_shape=(IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS), num_classes=NUM_CLASSES):
    """
    Builds a U-Net model using a ResNet50 backbone for the encoder,
    adapted for multi-class semantic segmentation, ensuring output matches input spatial dimensions.
    Includes regularization to combat overfitting.

    Args:
        input_shape (tuple): The shape of the input images (height, width, channels).
        num_classes (int): The number of output classes for segmentation (e.g., 4 for our case).

    Returns:
        tf.keras.Model: The compiled U-Net model with ResNet50 encoder.
    """
    # Load ResNet50 as the encoder, pre-trained on ImageNet, without the top classification layer
    # Its input is `input_shape` (256, 256, 3)
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freeze the layers of the pre-trained ResNet50 backbone
    base_model.trainable = False
    
    # Define skip connections from the ResNet50 backbone. These are the outputs
    # of certain intermediate layers that we will concatenate in the decoder.
    encoder_outputs = [
        base_model.get_layer('conv1_relu').output,   # 128x128, 64 filters
        base_model.get_layer('conv2_block3_out').output, # 64x64, 256 filters
        base_model.get_layer('conv3_block4_out').output, # 32x32, 512 filters
        base_model.get_layer('conv4_block6_out').output # 16x16, 1024 filters
    ]
    
    # Decoder starts from the end of the encoder
    # This is the bottom-most part of the U-Net
    decoder_input = base_model.get_layer('conv5_block3_out').output # 8x8, 2048 filters

    # Decoder
    # 1st Up-sampling block (from 8x8 to 16x16)
    up1 = layers.UpSampling2D(size=(2, 2))(decoder_input)
    conv_trans1 = layers.Conv2D(512, 2, activation='relu', padding='same',
                                kernel_regularizer=regularizers.l2(1e-4))(up1)
    merge1 = layers.concatenate([encoder_outputs[3], conv_trans1], axis=3) # Skip from conv4_block6_out (16x16, 1024)
    conv1_dec = layers.Conv2D(1024, 3, activation='relu', padding='same',
                              kernel_regularizer=regularizers.l2(1e-4))(merge1)
    conv1_dec = layers.Conv2D(1024, 3, activation='relu', padding='same',
                              kernel_regularizer=regularizers.l2(1e-4))(conv1_dec)
    conv1_dec = layers.Dropout(0.5)(conv1_dec) # Add Dropout

    # 2nd Up-sampling block (from 16x16 to 32x32)
    up2 = layers.UpSampling2D(size=(2, 2))(conv1_dec)
    conv_trans2 = layers.Conv2D(256, 2, activation='relu', padding='same',
                                kernel_regularizer=regularizers.l2(1e-4))(up2)
    merge2 = layers.concatenate([encoder_outputs[2], conv_trans2], axis=3) # Skip from conv3_block4_out (32x32, 512)
    conv2_dec = layers.Conv2D(512, 3, activation='relu', padding='same',
                              kernel_regularizer=regularizers.l2(1e-4))(merge2)
    conv2_dec = layers.Conv2D(512, 3, activation='relu', padding='same',
                              kernel_regularizer=regularizers.l2(1e-4))(conv2_dec)
    conv2_dec = layers.Dropout(0.25)(conv2_dec) # Add Dropout

    # 3rd Up-sampling block (from 32x32 to 64x64)
    up3 = layers.UpSampling2D(size=(2, 2))(conv2_dec)
    conv_trans3 = layers.Conv2D(128, 2, activation='relu', padding='same',
                               kernel_regularizer=regularizers.l2(1e-4))(up3)
    merge3 = layers.concatenate([encoder_outputs[1], conv_trans3], axis=3) # Skip from conv2_block3_out (64x64, 256)
    conv3_dec = layers.Conv2D(256, 3, activation='relu', padding='same',
                              kernel_regularizer=regularizers.l2(1e-4))(merge3)
    conv3_dec = layers.Conv2D(256, 3, activation='relu', padding='same',
                              kernel_regularizer=regularizers.l2(1e-4))(conv3_dec)
    conv3_dec = layers.Dropout(0.25)(conv3_dec) # Add Dropout

    # 4th Up-sampling block (from 64x64 to 128x128)
    up4 = layers.UpSampling2D(size=(2, 2))(conv3_dec)
    conv_trans4 = layers.Conv2D(64, 2, activation='relu', padding='same',
                               kernel_regularizer=regularizers.l2(1e-4))(up4)
    merge4 = layers.concatenate([encoder_outputs[0], conv_trans4], axis=3) # Skip from conv1_relu (128x128, 64)
    conv4_dec = layers.Conv2D(128, 3, activation='relu', padding='same',
                              kernel_regularizer=regularizers.l2(1e-4))(merge4)
    conv4_dec = layers.Conv2D(128, 3, activation='relu', padding='same',
                              kernel_regularizer=regularizers.l2(1e-4))(conv4_dec)
    conv4_dec = layers.Dropout(0.25)(conv4_dec) # Add Dropout

    # 5th Up-sampling block (NEW: from 128x128 to 256x256)
    up5 = layers.UpSampling2D(size=(2, 2))(conv4_dec) # -> (256, 256, 128)
    conv_trans5 = layers.Conv2D(64, 2, activation='relu', padding='same',
                               kernel_regularizer=regularizers.l2(1e-4))(up5)
    merge5 = layers.concatenate([base_model.input, conv_trans5], axis=3) # Skip from base_model.input (256x256, 3)
    conv5_dec = layers.Conv2D(64, 3, activation='relu', padding='same',
                              kernel_regularizer=regularizers.l2(1e-4))(merge5)
    conv5_dec = layers.Conv2D(64, 3, activation='relu', padding='same',
                              kernel_regularizer=regularizers.l2(1e-4))(conv5_dec)
    
    # Output layer
    output_layer = layers.Conv2D(num_classes, 1, activation='softmax')(conv5_dec)

    # Create the full model
    model = models.Model(inputs=base_model.input, outputs=output_layer)

    return model
