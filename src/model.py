import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50

# Import constants from data_loader.py
from .data_loader import IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS, NUM_CLASSES # NUM_CLASSES is 4

def build_unet_resnet50(input_shape=(IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS), num_classes=NUM_CLASSES):
    """
    Builds a U-Net model using a ResNet50 backbone for the encoder,
    adapted for multi-class semantic segmentation, ensuring output matches input spatial dimensions.

    Args:
        input_shape (tuple): The shape of the input images (height, width, channels).
        num_classes (int): The number of output classes for segmentation (e.g., 4 for our case).

    Returns:
        tf.keras.Model: The compiled U-Net model with ResNet50 encoder.
    """
    # Load ResNet50 as the encoder, pre-trained on ImageNet, without the top classification layer
    # Its input is `input_shape` (256, 256, 3)
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

    # --- FIX: Freeze the layers of the pre-trained ResNet50 backbone ---
    base_model.trainable = False 
    # This line sets all layers within the base_model to non-trainable.
    # Only the newly added decoder layers and any layers you explicitly unfreeze will be trained.

    # Define skip connections from the ResNet50 backbone.
    # These are layers from the encoder path that will be concatenated with the decoder's upsampled outputs.
    # The output shapes (Height, Width, Channels) for a 256x256 input are:
    # conv1_relu: (128, 128, 64) - After initial 7x7 conv and max pooling
    # conv2_block3_out: (64, 64, 256)
    # conv3_block4_out: (32, 32, 512)
    # conv4_block6_out: (16, 16, 1024)
    # conv5_block3_out (bottleneck): (8, 8, 2048)

    encoder_outputs = [
        # New skip connection for the final upsampling stage to 256x256
        # This will connect to the output of the very first block of the encoder
        # that roughly corresponds to the 256x256 scale (before any significant downsampling)
        # For ResNet50, the input itself is often used as the highest resolution skip.
        base_model.input, # (256, 256, 3) - This is the highest resolution "skip"
        
        base_model.get_layer('conv1_relu').output,      # (128, 128, 64)
        base_model.get_layer('conv2_block3_out').output, # (64, 64, 256)
        base_model.get_layer('conv3_block4_out').output, # (32, 32, 512)
        base_model.get_layer('conv4_block6_out').output  # (16, 16, 1024)
    ]
    
    # Bottleneck is the final output of the ResNet50 encoder
    bottleneck = base_model.output # (8, 8, 2048) -> this is 'conv5_block3_out'

    # Decoder (Upsampling Path) - Now with 5 upsampling blocks to match 256x256

    # 1st Up-sampling block (from bottleneck 8x8 to 16x16)
    up1 = layers.UpSampling2D(size=(2, 2))(bottleneck) # -> (16, 16, 2048)
    conv_trans1 = layers.Conv2D(1024, 2, activation='relu', padding='same')(up1)
    merge1 = layers.concatenate([encoder_outputs[4], conv_trans1], axis=3) # Skip from conv4_block6_out (16x16, 1024)
    conv1_dec = layers.Conv2D(1024, 3, activation='relu', padding='same')(merge1)
    conv1_dec = layers.Conv2D(1024, 3, activation='relu', padding='same')(conv1_dec) # -> (16, 16, 1024)

    # 2nd Up-sampling block (from 16x16 to 32x32)
    up2 = layers.UpSampling2D(size=(2, 2))(conv1_dec) # -> (32, 32, 1024)
    conv_trans2 = layers.Conv2D(512, 2, activation='relu', padding='same')(up2)
    merge2 = layers.concatenate([encoder_outputs[3], conv_trans2], axis=3) # Skip from conv3_block4_out (32x32, 512)
    conv2_dec = layers.Conv2D(512, 3, activation='relu', padding='same')(merge2)
    conv2_dec = layers.Conv2D(512, 3, activation='relu', padding='same')(conv2_dec) # -> (32, 32, 512)

    # 3rd Up-sampling block (from 32x32 to 64x64)
    up3 = layers.UpSampling2D(size=(2, 2))(conv2_dec) # -> (64, 64, 512)
    conv_trans3 = layers.Conv2D(256, 2, activation='relu', padding='same')(up3)
    merge3 = layers.concatenate([encoder_outputs[2], conv_trans3], axis=3) # Skip from conv2_block3_out (64x64, 256)
    conv3_dec = layers.Conv2D(256, 3, activation='relu', padding='same')(merge3)
    conv3_dec = layers.Conv2D(256, 3, activation='relu', padding='same')(conv3_dec) # -> (64, 64, 256)

    # 4th Up-sampling block (from 64x64 to 128x128)
    up4 = layers.UpSampling2D(size=(2, 2))(conv3_dec) # -> (128, 128, 256)
    conv_trans4 = layers.Conv2D(128, 2, activation='relu', padding='same')(up4)
    merge4 = layers.concatenate([encoder_outputs[1], conv_trans4], axis=3) # Skip from conv1_relu (128x128, 64)
    conv4_dec = layers.Conv2D(128, 3, activation='relu', padding='same')(merge4)
    conv4_dec = layers.Conv2D(128, 3, activation='relu', padding='same')(conv4_dec) # -> (128, 128, 128)

    # 5th Up-sampling block (NEW: from 128x128 to 256x256)
    up5 = layers.UpSampling2D(size=(2, 2))(conv4_dec) # -> (256, 256, 128)
    # The last skip connection should ideally be from the initial input, which is 256x256.
    # We pass the input directly as encoder_outputs[0]
    conv_trans5 = layers.Conv2D(64, 2, activation='relu', padding='same')(up5)
    merge5 = layers.concatenate([encoder_outputs[0], conv_trans5], axis=3) # Skip from base_model.input (256x256, 3)
    conv5_dec = layers.Conv2D(64, 3, activation='relu', padding='same')(merge5)
    conv5_dec = layers.Conv2D(64, 3, activation='relu', padding='same')(conv5_dec) # -> (256, 256, 64)
    
    # Final output layer - changed to NUM_CLASSES and 'softmax' for multi-class segmentation
    outputs = layers.Conv2D(num_classes, 1, activation='softmax')(conv5_dec) # -> (256, 256, NUM_CLASSES)

    model = models.Model(inputs=base_model.input, outputs=outputs)

    return model
