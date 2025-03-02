import os
# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import time
import math

def create_directories(directories):
    """Create directories if they don't exist."""
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

def load_and_preprocess_image(image_path, target_size=(256, 256)):
    """Load and preprocess an image for the model."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert from BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize image
    img = cv2.resize(img, target_size)
    
    # Normalize to [-1, 1]
    img = (img.astype(np.float32) - 127.5) / 127.5
    
    return img

def save_image(img, path):
    """Save a normalized image to disk."""
    # Denormalize from [-1, 1] to [0, 255]
    img = ((img + 1) * 127.5).astype(np.uint8)
    
    # Convert from RGB to BGR for OpenCV
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(path, img)

def compute_saliency_map(image, method='combined'):
    """
    Compute saliency map using OpenCV's saliency detection with improved combined method.
    Supported methods: 'spectral_residual', 'fine_grained', 'bing', 'bg_prior', 'combined'
    """
    # Convert the normalized image back to [0, 255] range for OpenCV
    if image.dtype == np.float32 and np.max(image) <= 1.0:
        image_cv = ((image + 1) * 127.5).astype(np.uint8)
    else:
        image_cv = image.astype(np.uint8)
    
    # Convert to BGR if it's in RGB
    if len(image_cv.shape) == 3 and image_cv.shape[2] == 3:
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
    
    if method == 'combined':
        # Compute spectral residual saliency
        spectral_saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        (success1, spectral_map) = spectral_saliency.computeSaliency(image_cv)
        
        # Compute fine-grained saliency
        fine_grained_saliency = cv2.saliency.StaticSaliencyFineGrained_create()
        (success2, fine_grained_map) = fine_grained_saliency.computeSaliency(image_cv)
        
        # Check if either method failed
        if not (success1 and success2):
            print("Failed to compute one of the saliency methods, falling back to spectral residual")
            # Try to use whichever one succeeded
            if success1:
                saliency_map = spectral_map
            elif success2:
                saliency_map = fine_grained_map
            else:
                # Return a uniform saliency map as fallback
                return np.ones(image_cv.shape[:2], dtype=np.float32)
        else:
            # Weight spectral residual higher as it's often more reliable
            saliency_map = 0.7 * spectral_map + 0.3 * fine_grained_map
    else:
        # Create saliency object based on method (original implementation)
        if method == 'spectral_residual':
            saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        elif method == 'fine_grained':
            saliency = cv2.saliency.StaticSaliencyFineGrained_create()
        elif method == 'bing':
            saliency = cv2.saliency.ObjectnessBING_create()
        elif method == 'bg_prior':
            saliency = cv2.saliency.MotionSaliencyBinWangApr2014_create()
        else:
            raise ValueError(f"Unsupported saliency method: {method}")
        
        # Compute saliency
        (success, saliency_map) = saliency.computeSaliency(image_cv)
        
        if not success:
            print(f"Failed to compute saliency using {method} method.")
            # Return a uniform saliency map as fallback
            return np.ones(image_cv.shape[:2], dtype=np.float32)
    
    # Normalize to [0, 1]
    if saliency_map.max() > 0:
        saliency_map = saliency_map / saliency_map.max()
    
    # Apply Gaussian blur to smooth the saliency map
    saliency_map = cv2.GaussianBlur(saliency_map, (5, 5), 0)
    
    return saliency_map

# In your utils.py or wherever create_saliency_mask is defined
def create_saliency_mask(saliency_map, threshold=0.1, smooth=True):
    if smooth:
        # More refined blurring with edge preservation
        mask = cv2.bilateralFilter(saliency_map, 9, 75, 75)
        mask = cv2.GaussianBlur(mask, (31, 31), 0)
        # Ensure values are in [0, 1]
        if mask.max() > 0:
            mask = mask / mask.max()
    else:
        # Create binary mask
        mask = (saliency_map > threshold).astype(np.float32)
    
    return mask

def build_generator(latent_dim, img_shape, name="generator"):
    """Build generator with skip connections."""
    # Latent input
    latent_input = keras.Input(shape=(latent_dim,))
    
    # Skip connection inputs - match exactly with encoder outputs
    skip1 = keras.Input(shape=(128, 128, 64))  # First skip connection
    skip2 = keras.Input(shape=(64, 64, 128))   # Second skip connection
    skip3 = keras.Input(shape=(32, 32, 256))   # Third skip connection
    
    # Initial dense layer
    x = layers.Dense(16 * 16 * 512)(latent_input)
    x = layers.Reshape((16, 16, 512))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    
    # Upsampling path with skip connections
    x = layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding='same')(x)  # 32x32
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Concatenate()([x, skip3])  # Add skip connection
    
    x = layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding='same')(x)  # 64x64
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Concatenate()([x, skip2])  # Add skip connection
    
    x = layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding='same')(x)  # 128x128
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Concatenate()([x, skip1])  # Add skip connection
    
    x = layers.Conv2DTranspose(32, kernel_size=4, strides=2, padding='same')(x)  # 256x256
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    
    # Output layer
    outputs = layers.Conv2D(img_shape[2], kernel_size=4, padding='same', activation='tanh')(x)
    
    # Create model
    generator_model = keras.Model(inputs=[latent_input, skip1, skip2, skip3], outputs=outputs, name=name)
    
    return generator_model

def build_encoder(img_shape, latent_dim, name="encoder"):
    """Build encoder with skip connections."""
    inputs = keras.Input(shape=img_shape)
    
    # Start with fewer filters and increase
    x = inputs
    skip_outputs = []
    
    # Downsampling path
    x1 = layers.Conv2D(64, kernel_size=4, strides=2, padding='same')(x)  # 128x128
    x1 = layers.LeakyReLU(0.2)(x1)
    skip_outputs.append(x1)
    
    x2 = layers.Conv2D(128, kernel_size=4, strides=2, padding='same')(x1)  # 64x64
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.LeakyReLU(0.2)(x2)
    skip_outputs.append(x2)
    
    x3 = layers.Conv2D(256, kernel_size=4, strides=2, padding='same')(x2)  # 32x32
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.LeakyReLU(0.2)(x3)
    skip_outputs.append(x3)
    
    x4 = layers.Conv2D(512, kernel_size=4, strides=2, padding='same')(x3)  # 16x16
    x4 = layers.BatchNormalization()(x4)
    x4 = layers.LeakyReLU(0.2)(x4)
    
    # Flatten and map to latent space
    x5 = layers.Flatten()(x4)
    latent_output = layers.Dense(latent_dim)(x5)
    
    # Create model
    encoder_model = keras.Model(inputs=inputs, outputs=[latent_output] + skip_outputs, name=name)
    
    return encoder_model

def build_discriminator(img_shape):
    """
    Build the discriminator model for the GAN.
    
    Args:
        img_shape: Shape of the input images (height, width, channels)
    """
    # Create a functional model instead of Sequential for more flexibility
    inputs = keras.Input(shape=img_shape)
    
    # Calculate the number of downsampling layers needed
    initial_size = img_shape[0]
    target_size = 4
    num_downsampling = max(1, int(np.log2(initial_size / target_size)))
    
    # If input size is not a power of 2, first resize to nearest power of 2
    x = inputs
    if not (initial_size & (initial_size - 1) == 0):
        power_of_two = 2 ** int(np.log2(initial_size))
        x = layers.Resizing(power_of_two, power_of_two)(x)
    
    # Start with fewer filters and increase
    filters = 16
    
    # Convolutional blocks
    for i in range(min(num_downsampling, 4)):  # Limit to 4 downsampling layers
        x = layers.Conv2D(filters, kernel_size=4, strides=2, padding='same')(x)
        x = layers.LeakyReLU(negative_slope=0.2)(x)
        
        # Add batch normalization after the first layer
        if i > 0:
            x = layers.BatchNormalization()(x)
            
        # Double the filters after each layer, up to 128
        filters = min(filters * 2, 128)
    
    # Output
    x = layers.Flatten()(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs, name="discriminator")
    
    return model

def build_adaptive_compression_model(img_shape, base_latent_dim):
    """
    Build the adaptive compression model that uses different latent dimensions
    based on saliency.
    
    Args:
        img_shape: Shape of the input images (height, width, channels)
        base_latent_dim: Base dimension of the latent space for non-salient regions
    """
    # Input layers
    img_input = keras.Input(shape=img_shape, name="image_input")
    saliency_input = keras.Input(shape=(img_shape[0], img_shape[1], 1), name="saliency_input")
    
    # High-quality encoder-decoder for salient regions (2x base_latent_dim)
    high_quality_encoder = build_encoder(img_shape, base_latent_dim * 2, name="hq_encoder")
    high_quality_latent = high_quality_encoder(img_input)
    high_quality_generator = build_generator(base_latent_dim * 2, img_shape, name="hq_generator")
    high_quality_output = high_quality_generator(high_quality_latent)
    
    # Low-quality encoder-decoder for non-salient regions
    low_quality_encoder = build_encoder(img_shape, base_latent_dim, name="lq_encoder")
    low_quality_latent = low_quality_encoder(img_input)
    low_quality_generator = build_generator(base_latent_dim, img_shape, name="lq_generator")
    low_quality_output = low_quality_generator(low_quality_latent)
    
    # Ensure saliency input is properly broadcast to match the output shapes
    # Reshape the saliency mask if needed to match the output dimensions
    if high_quality_output.shape[1:3] != img_shape[0:2]:
        # Resize saliency mask to match the output shape
        resized_saliency = layers.Resizing(
            high_quality_output.shape[1], 
            high_quality_output.shape[2]
        )(saliency_input)
        expanded_saliency = resized_saliency
    else:
        expanded_saliency = saliency_input
    
    # Create binary masks for blending
    hq_mask = expanded_saliency
    lq_mask = layers.Lambda(lambda x: 1.0 - x)(expanded_saliency)
    
    # Blend outputs based on saliency
    weighted_hq = layers.Multiply()([high_quality_output, hq_mask])
    weighted_lq = layers.Multiply()([low_quality_output, lq_mask])
    blended_output = layers.Add()([weighted_hq, weighted_lq])
    
    # Create model
    model = keras.Model(
        inputs=[img_input, saliency_input],
        outputs=[blended_output, high_quality_latent, low_quality_latent],
        name="adaptive_compression_model"
    )
    
    return model, high_quality_encoder, high_quality_generator, low_quality_encoder, low_quality_generator

def compute_metrics(original_img, compressed_img):
    """
    Compute image quality metrics between original and compressed images.
    
    Args:
        original_img: Original image (normalized to [-1, 1])
        compressed_img: Compressed image (normalized to [-1, 1])
    
    Returns:
        Dictionary with PSNR, SSIM, and MSE values
    """
    # Convert from [-1, 1] to [0, 1] for metric computation
    orig_0_1 = (original_img + 1) / 2
    comp_0_1 = (compressed_img + 1) / 2
    
    # Compute PSNR
    psnr_value = psnr(orig_0_1, comp_0_1, data_range=1.0)
    
    # Compute SSIM
    if len(original_img.shape) == 3 and original_img.shape[2] == 3:
        # For RGB images, compute SSIM for each channel and average
        ssim_value = np.mean([
            ssim(orig_0_1[:,:,i], comp_0_1[:,:,i], data_range=1.0)
            for i in range(3)
        ])
    else:
        ssim_value = ssim(orig_0_1, comp_0_1, data_range=1.0)
    
    # Compute MSE
    mse_value = np.mean((orig_0_1 - comp_0_1) ** 2)
    
    return {
        'psnr': psnr_value,
        'ssim': ssim_value,
        'mse': mse_value
    }

def estimate_compression_ratio(original_size, latent_size):
    """
    Estimate compression ratio based on original image size and latent representation size.
    
    Args:
        original_size: Size of the original image in bytes
        latent_size: Size of the latent representation in bytes
    
    Returns:
        Compression ratio and percentage reduction
    """
    compression_ratio = original_size / latent_size
    percentage_reduction = (1 - (latent_size / original_size)) * 100
    
    return compression_ratio, percentage_reduction

def visualize_results(original, saliency_map, compressed, save_path=None):
    """
    Visualize original image, saliency map, and compressed image.
    
    Args:
        original: Original image (normalized to [-1, 1])
        saliency_map: Saliency map with values in [0, 1]
        compressed: Compressed image (normalized to [-1, 1])
        save_path: If provided, save the visualization to this path
    """
    # Convert images from [-1, 1] to [0, 1] for display
    orig_display = (original + 1) / 2
    comp_display = (compressed + 1) / 2
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(orig_display)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(saliency_map, cmap='hot')
    axes[1].set_title('Saliency Map')
    axes[1].axis('off')
    
    axes[2].imshow(comp_display)
    axes[2].set_title('Compressed')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()