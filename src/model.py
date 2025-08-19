import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50

# Import constants from data_loader.py (these will be imported into train.py, etc.)
from data_loader import IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS, NUM_CLASSES # NUM_CLASSES is now 4

def build_unet_resnet50(input_shape=(IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS), num_classes=NUM_CLASSES):
    """
    Builds a U-Net model using a ResNet50 backbone for the encoder,
    adapted for multi-class semantic segmentation.

    Args:
        input_shape (tuple): The shape of the input images (height, width, channels).
        num_classes (int): The number of output classes for segmentation (e.g., 4 for our case).

    Returns:
        tf.keras.Model: The compiled U-Net model with ResNet50 encoder.
    """
    # Load ResNet50 as the encoder, pre-trained on ImageNet, without the top classification layer
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

    # Define skip connections from the ResNet50 backbone
    # These are layers from the encoder path that will be concatenated with the decoder's upsampled outputs
    # Ensure these layer names are correct for ResNet50 in Keras
    encoder_outputs = [
        base_model.get_layer('conv1_relu').output,      # Output of first conv block (after activation)
        base_model.get_layer('conv2_block3_out').output, # Output of res2 block
        base_model.get_layer('conv3_block4_out').output, # Output of res3 block
        base_model.get_layer('conv4_block6_out').output  # Output of res4 block
    ]
    
    # Bottleneck is the final output of the ResNet50 encoder
    bottleneck = base_model.output # conv5_block3_out

    # Decoder (Upsampling Path)
    # The number of filters in Conv2DTranspose and Conv2D layers typically matches the encoder's corresponding layer
    # and then halves as we go up.

    # Decode from bottleneck (2048 filters)
    # 1st Up-sampling block (from conv5_block3_out to match conv4_block6_out)
    up1 = layers.UpSampling2D(size=(2, 2))(bottleneck)
    conv_trans1 = layers.Conv2D(1024, 2, activation='relu', padding='same')(up1) # Matches output filters of conv4_block6_out
    merge1 = layers.concatenate([encoder_outputs[3], conv_trans1], axis=3) # Skip from conv4_block6_out
    conv1_dec = layers.Conv2D(1024, 3, activation='relu', padding='same')(merge1)
    conv1_dec = layers.Conv2D(1024, 3, activation='relu', padding='same')(conv1_dec)

    # 2nd Up-sampling block (from conv1_dec to match conv3_block4_out)
    up2 = layers.UpSampling2D(size=(2, 2))(conv1_dec)
    conv_trans2 = layers.Conv2D(512, 2, activation='relu', padding='same')(up2) # Matches output filters of conv3_block4_out
    merge2 = layers.concatenate([encoder_outputs[2], conv_trans2], axis=3) # Skip from conv3_block4_out
    conv2_dec = layers.Conv2D(512, 3, activation='relu', padding='same')(merge2)
    conv2_dec = layers.Conv2D(512, 3, activation='relu', padding='same')(conv2_dec)

    # 3rd Up-sampling block (from conv2_dec to match conv2_block3_out)
    up3 = layers.UpSampling2D(size=(2, 2))(conv2_dec)
    conv_trans3 = layers.Conv2D(256, 2, activation='relu', padding='same')(up3) # Matches output filters of conv2_block3_out
    merge3 = layers.concatenate([encoder_outputs[1], conv_trans3], axis=3) # Skip from conv2_block3_out
    conv3_dec = layers.Conv2D(256, 3, activation='relu', padding='same')(merge3)
    conv3_dec = layers.Conv2D(256, 3, activation='relu', padding='same')(conv3_dec)

    # 4th Up-sampling block (from conv3_dec to match conv1_relu)
    up4 = layers.UpSampling2D(size=(2, 2))(conv3_dec)
    conv_trans4 = layers.Conv2D(64, 2, activation='relu', padding='same')(up4) # Matches output filters of conv1_relu
    merge4 = layers.concatenate([encoder_outputs[0], conv_trans4], axis=3) # Skip from conv1_relu
    conv4_dec = layers.Conv2D(64, 3, activation='relu', padding='same')(merge4)
    conv4_dec = layers.Conv2D(64, 3, activation='relu', padding='same')(conv4_dec)
    
    # Final output layer - changed to NUM_CLASSES and 'softmax' for multi-class segmentation
    outputs = layers.Conv2D(num_classes, 1, activation='softmax')(conv4_dec)

    model = models.Model(inputs=base_model.input, outputs=outputs)

    return model

