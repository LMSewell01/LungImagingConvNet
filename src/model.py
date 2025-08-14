import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, BatchNormalization, Activation, Conv2DTranspose

# Image dimensions, same as data_loader.py
IMG_HEIGHT = 256
IMG_WIDTH = 256
NUM_CHANNELS = 3
NUM_CLASSES = 1

def build_unet_resnet50(input_shape=(IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS), num_classes=NUM_CLASSES):
    """
    Builds a U-Net model with a pre-trained ResNet50 encoder. 
    ResNet Info: https://arxiv.org/abs/1512.03385, https://towardsdatascience.com/the-annotated-resnet-50-a6c536034758/
    U-Net Info: https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/

    Args:
        input_shape (tuple): The shape of the input images (height, width, channels).
        num_classes (int): The number of output classes for segmentation (1 for binary segmentation for lung abnormalities).

    Returns:
        tf.keras.Model: The compiled U-Net model.
    """
    inputs = Input(shape=input_shape)
    # Here we load the resnet model. Weights are from the ImageNet database (https://en.wikipedia.org/wiki/ImageNet)
    # Top layer is excluded because we want to train that for our task
    resnet_base = ResNet50(weights='imagenet', include_top=False, input_tensor=inputs)

    # Freeze all layers to prevent them from being retrained
    for layer in resnet_base.layers:
        layer.trainable = False
    # Extracting outputs from initial conv layers for use in skipped connections
    # Gives feature maps at different spatial resolutions.
    s1 = resnet_base.get_layer('input_layer').output # 256x256
    s2 = resnet_base.get_layer('conv1_relu').output # 128x128
    s3 = resnet_base.get_layer('conv2_block3_out').output # 64x64
    s4 = resnet_base.get_layer('conv3_block4_out').output # 32x32

    # Bridge layer, the final output
    b1 = resnet_base.get_layer('conv4_block6_out').output

    # ---- DECODER (Expanding Path of U-Net) ---- 
    # Features are upsampled using transpose convolutions

    def conv_block(input_tensor, num_filters):
        """ Standard Conv block for filter"""
        x = Conv2D(num_filters, (3, 3), padding='same', kernel_initializer='he_normal')(input_tensor)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(num_filters, (3, 3), padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x
    
    def decoder_block(input_tensor, skip_features, filters):
        """ Decoder block performing upsampling. """
        x = Conv2DTranspose(filters, (2,2), strides=(2,2), padding='same')(input_tensor)
        # Concatenate this with the skipped connections
        x = concatenate([x, skip_features])
        # Apply conv_block after concat
        x = conv_block(x, filters)
        return x
    # The different decoder stages during upscaling
    d1 = decoder_block(b1, s4, 512) # I.e. Upsample 16x16 to 32x32, concat with s4
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64) # Upsample 128x128 to 256x256, concat with s1 (input_1)


    #### ---- Final Output layer of model ----
    # Rather than more complex segmentation problems, here we only need 1 final filter to do a pixel-wise binary classification
    # The 1x1 conv with 1 channel allows dimensionality reduction from the number of filters in the d4 block (64).
    # The dimensionality reduction, combined with an activation function allows non-linearity to be learned.
    outputs = Conv2D(num_classes, (1,1), activation='sigmoid')(d4)

    model = Model(inputs=inputs, outputs=outputs, name='U_Net_ResNet50')

    return model

# Example Usage (for demonstration, not part of the main script execution)
if __name__ == "__main__":
    # Build the model with the defined input shape
    model = build_unet_resnet50()
    print("U-Net with ResNet50 encoder model built successfully.")
    
    # Print a summary of the model architecture
    model.summary()
