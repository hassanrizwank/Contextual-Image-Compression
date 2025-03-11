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
from skimage.filters import sobel
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

def compute_saliency_map(image, method='spectral_residual'):
    """
    Compute saliency map using different methods.
    
    Args:
        image: Input image (normalized to [-1, 1])
        method: Saliency method to use ('spectral_residual', 'fine_grained', 'combined')
    
    Returns:
        Saliency map with values in range [0, 1]
    """
    # Convert the normalized image to [0, 255] range for OpenCV
    if image.dtype == np.float32 and np.max(image) <= 1.0:
        image_cv = ((image + 1) * 127.5).astype(np.uint8)
    else:
        image_cv = image.astype(np.uint8)
    
    # Convert to BGR if it's in RGB
    if len(image_cv.shape) == 3 and image_cv.shape[2] == 3:
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
    
    if method == 'combined':
        # Compute both spectral residual and fine-grained saliency
        spectral_saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        (success1, spectral_map) = spectral_saliency.computeSaliency(image_cv)
        
        fine_grained_saliency = cv2.saliency.StaticSaliencyFineGrained_create()
        (success2, fine_grained_map) = fine_grained_saliency.computeSaliency(image_cv)
        
        if not (success1 and success2):
            print(f"Warning: One or more saliency methods failed. Using available method.")
            if success1:
                return spectral_map
            elif success2:
                return fine_grained_map
            else:
                print("All saliency methods failed. Returning uniform saliency.")
                return np.ones(image_cv.shape[:2], dtype=np.float32)
        
        # Combine the two saliency maps with weighted average
        # Spectral residual is good at capturing large salient objects
        # Fine-grained is better at capturing detailed structures
        combined_map = 0.6 * spectral_map + 0.4 * fine_grained_map
        
        # Normalize to [0, 1]
        if combined_map.max() > 0:
            combined_map = combined_map / combined_map.max()
            
        return combined_map
    
    elif method == 'spectral_residual':
        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    elif method == 'fine_grained':
        saliency = cv2.saliency.StaticSaliencyFineGrained_create()
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
    
    return saliency_map

def enhance_saliency_map(saliency_map):
    """
    Enhance saliency map with multi-scale processing for better detail preservation.
    
    Args:
        saliency_map: Original saliency map with values in range [0, 1]
    
    Returns:
        Enhanced saliency map with values in range [0, 1]
    """
    # Apply bilateral filter to preserve edges
    filtered_map = cv2.bilateralFilter(saliency_map.astype(np.float32), 9, 75, 75)
    
    # Create multi-scale representation with different Gaussian blurs
    scales = [3, 9, 15]
    multi_scale_maps = []
    
    for scale in scales:
        blurred = cv2.GaussianBlur(filtered_map, (scale, scale), 0)
        multi_scale_maps.append(blurred)
    
    # Weight smaller scales (finer details) more heavily
    weights = [0.5, 0.3, 0.2]  # Weights sum to 1
    enhanced_map = np.zeros_like(saliency_map)
    
    for i, scale_map in enumerate(multi_scale_maps):
        enhanced_map += weights[i] * scale_map
    
    # Apply contrast enhancement
    enhanced_map = np.power(enhanced_map, 0.8)  # Gamma correction to enhance mid-level saliency
    
    # Ensure values are in [0, 1]
    enhanced_map = np.clip(enhanced_map, 0, 1)
    
    return enhanced_map

def create_saliency_mask(saliency_map, threshold=None, smooth=True):
    """
    Create an adaptive saliency mask from the saliency map with improved edge preservation.
    
    Args:
        saliency_map: Saliency map with values in range [0, 1]
        threshold: Optional threshold value. If None, automatically determined
        smooth: Whether to apply smoothing to the mask
        
    Returns:
        Binary or smooth mask with values in range [0, 1]
    """
    # If threshold is not provided, determine it adaptively
    if threshold is None:
        # Use Otsu's method for adaptive thresholding
        if saliency_map.max() <= 1.0:
            # Convert to uint8 for Otsu
            saliency_uint8 = (saliency_map * 255).astype(np.uint8)
        else:
            saliency_uint8 = saliency_map.astype(np.uint8)
        
        # Get threshold using Otsu's method
        threshold, _ = cv2.threshold(saliency_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        threshold = threshold / 255.0  # Normalize back to [0, 1]
        
        # Adjust threshold based on the distribution of saliency values
        hist, bins = np.histogram(saliency_map.flatten(), 50, range=(0, 1))
        cumsum = np.cumsum(hist)
        cumsum = cumsum / cumsum[-1]  # Normalize to [0, 1]
        
        # Find the bin where cumsum exceeds 0.7 (keeping top ~30% as salient)
        salient_threshold = bins[np.argmax(cumsum > 0.7)]
        
        # Take the minimum of Otsu and distribution-based threshold
        final_threshold = min(threshold, salient_threshold)
        final_threshold = max(0.05, min(0.5, final_threshold))
    else:
        final_threshold = threshold
    
    if smooth:
        # Create continuous mask with edge preservation
        mask = cv2.bilateralFilter(saliency_map.astype(np.float32), 9, 75, 75)
        mask = cv2.GaussianBlur(mask, (31, 31), 0)
        if mask.max() > 0:
            mask = mask / mask.max()
    else:
        # Create binary mask
        mask = (saliency_map > final_threshold).astype(np.float32)
    
    return mask

def build_latent_saliency_model(latent_dim, name="latent_saliency_module"):
    """
    Build a model that learns to predict saliency from encoder latent representation.
    
    Args:
        latent_dim: Dimension of the latent space
        name: Name of the model
        
    Returns:
        Keras model for latent saliency prediction
    """
    # Input is the latent vector
    latent_input = keras.Input(shape=(latent_dim,))
    
    # Process through dense layers
    x = layers.Dense(512, activation='relu')(latent_input)
    x = layers.Dense(256, activation='relu')(x)
    
    # Output a saliency score - single value representing importance of this latent vector
    saliency_score = layers.Dense(1, activation='sigmoid')(x)
    
    # Create model
    model = keras.Model(inputs=latent_input, outputs=saliency_score, name=name)
    
    return model

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

def build_encoder(img_shape, latent_dim, name="encoder", add_attention=True):
    """
    Build encoder with skip connections and optional attention mechanism.
    
    Args:
        img_shape: Input image shape (H, W, C)
        latent_dim: Dimension of the latent space
        name: Name of the model
        add_attention: Whether to add self-attention layers
        
    Returns:
        Encoder model
    """
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
    
    # Add self-attention for better feature learning if requested
    if add_attention:
        # Add self-attention layer at 32x32 resolution (adapt this as needed)
        attention_layer = SelfAttention(256)
        x3 = attention_layer(x3)
    
    x4 = layers.Conv2D(512, kernel_size=4, strides=2, padding='same')(x3)  # 16x16
    x4 = layers.BatchNormalization()(x4)
    x4 = layers.LeakyReLU(0.2)(x4)
    
    # Flatten and map to latent space
    x5 = layers.Flatten()(x4)
    latent_output = layers.Dense(latent_dim)(x5)
    
    # Create model
    encoder_model = keras.Model(inputs=inputs, outputs=[latent_output] + skip_outputs, name=name)
    
    return encoder_model

class SelfAttention(layers.Layer):
    """Self-attention layer for capturing long-range dependencies."""
    
    def __init__(self, channels, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.channels = channels
        self.query_conv = layers.Conv2D(channels // 8, kernel_size=1, strides=1, padding='same')
        self.key_conv = layers.Conv2D(channels // 8, kernel_size=1, strides=1, padding='same')
        self.value_conv = layers.Conv2D(channels, kernel_size=1, strides=1, padding='same')
        self.gamma = self.add_weight(name='gamma', shape=[1], initializer='zeros', trainable=True)
    
    def call(self, inputs):
        batch_size, height, width, num_channels = inputs.shape
        
        # Project queries, keys, values
        query = self.query_conv(inputs)  # B x H x W x C'
        key = self.key_conv(inputs)  # B x H x W x C'
        value = self.value_conv(inputs)  # B x H x W x C
        
        # Reshape for matrix multiplication
        query_reshape = tf.reshape(query, [-1, height * width, self.channels // 8])
        key_reshape = tf.reshape(key, [-1, height * width, self.channels // 8])
        key_transpose = tf.transpose(key_reshape, [0, 2, 1])
        
        # Calculate attention map
        attention_map = tf.matmul(query_reshape, key_transpose)  # B x (H*W) x (H*W)
        attention_map = tf.nn.softmax(attention_map, axis=-1)
        
        # Apply attention to values
        value_reshape = tf.reshape(value, [-1, height * width, self.channels])
        context = tf.matmul(attention_map, value_reshape)
        context = tf.reshape(context, [-1, height, width, self.channels])
        
        # Apply learned gamma scaling and add residual connection
        output = self.gamma * context + inputs
        
        return output
    
    def get_config(self):
        config = super(SelfAttention, self).get_config()
        config.update({'channels': self.channels})
        return config
    
class SpectralNormalization(layers.Layer):
    """Spectral normalization layer for GAN stabilization."""
    
    def __init__(self, units, **kwargs):
        super(SpectralNormalization, self).__init__(**kwargs)
        self.units = units
    
    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="glorot_uniform",
            trainable=True,
            name='kernel'
        )
        self.bias = self.add_weight(
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
            name='bias'
        )
        self.u = self.add_weight(
            shape=(1, self.units),
            initializer=tf.initializers.RandomNormal(0, 1),
            trainable=False,
            name='sn_u'
        )
        super(SpectralNormalization, self).build(input_shape)
    
    def call(self, inputs, training=None):
        # Power iteration
        u_hat = self.u
        v_hat = None
        
        # One step of power iteration
        v_ = tf.matmul(u_hat, tf.transpose(self.w))
        v_hat = tf.nn.l2_normalize(v_, axis=1)
        
        u_ = tf.matmul(v_hat, self.w)
        u_hat = tf.nn.l2_normalize(u_, axis=1)
        
        # Update u value
        if training:
            self.u.assign(u_hat)
        
        # Calculate sigma (spectral norm)
        sigma = tf.matmul(tf.matmul(v_hat, self.w), tf.transpose(u_hat))
        
        # Apply normalization with bias
        output = tf.matmul(inputs, self.w / sigma) + self.bias
        
        return output


class AdaptiveQuantizationLayer(layers.Layer):
    """Layer for adaptive quantization of latent vectors based on saliency."""
    
    def __init__(self, **kwargs):
        super(AdaptiveQuantizationLayer, self).__init__(**kwargs)
    
    def call(self, inputs):
        latent, saliency_score, quant_strength = inputs
        # Scale quantization based on saliency and RD params
        effective_quant = quant_strength * (1.0 - saliency_score)
        
        # Implement soft quantization (differentiable approximation)
        scale = tf.exp(effective_quant * 3.0)  # Scale factor based on quantization strength
        
        # Soft quantization - round to a grid with spacing determined by scale
        quantized = tf.round(latent * scale) / scale
        
        return quantized
    
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
    
    # Convolutional blocks WITHOUT spectral normalization for now
    for i in range(min(num_downsampling, 4)):  # Limit to 4 downsampling layers
        x = layers.Conv2D(filters, kernel_size=4, strides=2, padding='same')(x)
        x = layers.LeakyReLU(negative_slope=0.2)(x)
        
        # Add batch normalization after the first layer
        if i > 0:
            x = layers.BatchNormalization()(x)
            
        # Double the filters after each layer, up to 128
        filters = min(filters * 2, 128)
    
    # Output layer
    x = layers.Flatten()(x)
    x = layers.Dense(128)(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs, name="discriminator")
    
    return model

def build_rate_distortion_optimizer(img_shape, latent_dims, name="rd_optimizer"):
    """
    Build a rate-distortion optimization module with direct bitrate control.
    """
    # Inputs
    img_input = keras.Input(shape=img_shape)
    saliency_input = keras.Input(shape=(img_shape[0], img_shape[1], 1))
    target_bpp_input = keras.Input(shape=(1,))  # Target bits per pixel
    
    # Normalize target BPP to [0,1] range for better stability
    target_bpp_normalized = layers.Lambda(
    lambda x: tf.clip_by_value(x / 5.0, 0.0, 1.0)  # Increased from 3.0 to 5.0
    )(target_bpp_input)

    
    # Process saliency map 
    x = layers.Conv2D(32, kernel_size=3, strides=2, padding='same')(saliency_input)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(64, kernel_size=3, strides=2, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.GlobalAveragePooling2D()(x)
    
    # Concatenate with normalized target BPP
    x = layers.Concatenate()([x, target_bpp_normalized])
    
    # Process through dense layers
    x = layers.Dense(128)(x)
    x = layers.LeakyReLU(0.2)(x)
    
    # Create a base parameter layer
    base_params = layers.Dense(3)(x)
    
    # For more direct control, make the final parameters directly dependent on target BPP
    # 1. Overall compression strength (inversely proportional to target BPP)
    overall_compression = layers.Lambda(
        lambda x: tf.sigmoid(x[0][:,0:1] + 1.0 - 2.0 * x[1])  # Higher BPP = lower compression
    )([base_params, target_bpp_normalized])
    
    # 2. HQ/LQ region boundary threshold (inversely proportional to target BPP)
    hq_lq_threshold = layers.Lambda(
        lambda x: tf.sigmoid(x[0][:,1:2] + 1.0 - 2.0 * x[1])  # Higher BPP = more HQ regions
    )([base_params, target_bpp_normalized])
    
    # 3. Quantization strength (inversely proportional to target BPP)
    quant_strength = layers.Lambda(
        lambda x: tf.sigmoid(x[0][:,2:3] + 1.0 - 1.5 * x[1])  # Higher BPP = finer quantization
    )([base_params, target_bpp_normalized])
    
    # Combine parameters
    bit_allocation_params = layers.Concatenate()([
        overall_compression, 
        hq_lq_threshold,
        quant_strength
    ])
    
    # Create model
    model = keras.Model(
        inputs=[img_input, saliency_input, target_bpp_input],
        outputs=bit_allocation_params,
        name=name
    )
    
    return model

def build_adaptive_compression_model(img_shape, base_latent_dim, target_bpp=None):
    """
    Build the adaptive compression model with rate control.
    
    Args:
        img_shape: Shape of the input images (height, width, channels)
        base_latent_dim: Base dimension of the latent space for non-salient regions
        target_bpp: Target bits per pixel (if None, use automatic rate control)
    
    Returns:
        Dictionary with all model components
    """
    # Input layers
    img_input = keras.Input(shape=img_shape, name="image_input")
    saliency_input = keras.Input(shape=(img_shape[0], img_shape[1], 1), name="saliency_input")
    
    if target_bpp is not None:
        target_bpp_input = keras.Input(shape=(1,), name="target_bpp_input")
        has_target_bpp = True
    else:
        # Default target BPP if not provided
        target_bpp_input = keras.Input(shape=(1,), name="target_bpp_input")
        default_bpp = tf.constant([[1.0]], dtype=tf.float32)  # Default 1.0 bpp
        has_target_bpp = False
    
    # Build the main components
    hq_encoder = build_encoder(img_shape, base_latent_dim * 2, name="hq_encoder", add_attention=True)
    hq_generator = build_generator(base_latent_dim * 2, img_shape, name="hq_generator")
    
    lq_encoder = build_encoder(img_shape, base_latent_dim, name="lq_encoder", add_attention=False)
    lq_generator = build_generator(base_latent_dim, img_shape, name="lq_generator")
    
    # Add latent saliency module to learn saliency in latent space
    latent_saliency_hq = build_latent_saliency_model(base_latent_dim * 2, name="hq_latent_saliency")
    latent_saliency_lq = build_latent_saliency_model(base_latent_dim, name="lq_latent_saliency")
    
    # Build rate-distortion optimizer
    rd_optimizer = build_rate_distortion_optimizer(
        img_shape, 
        {'hq': base_latent_dim * 2, 'lq': base_latent_dim},
        name="rd_optimizer"
    )
    
    # Model flows:
    # 1. Encode image
    hq_encoder_outputs = hq_encoder(img_input)
    lq_encoder_outputs = lq_encoder(img_input)
    
    # 2. Extract latent and skip connections
    hq_latent = hq_encoder_outputs[0]
    hq_skip1 = hq_encoder_outputs[1]
    hq_skip2 = hq_encoder_outputs[2]
    hq_skip3 = hq_encoder_outputs[3]
    
    lq_latent = lq_encoder_outputs[0]
    lq_skip1 = lq_encoder_outputs[1]
    lq_skip2 = lq_encoder_outputs[2]
    lq_skip3 = lq_encoder_outputs[3]
    
    # 3. Compute latent saliency scores
    hq_latent_saliency = latent_saliency_hq(hq_latent)
    lq_latent_saliency = latent_saliency_lq(lq_latent)
    
    # 4. Get rate-distortion parameters
    if has_target_bpp:
        rd_params = rd_optimizer([img_input, saliency_input, target_bpp_input])
    else:
        rd_params = rd_optimizer([img_input, saliency_input, default_bpp])
    
    # 5. Apply adaptive quantization based on RD parameters
    # Create a more direct connection between target BPP and outputs

    target_bpp_normalized = layers.Lambda(
    lambda x: tf.clip_by_value(x / 5.0, 0.0, 1.0)  # Increased from 3.0 to 5.0
    )(target_bpp_input)


    # Extract RD parameters (values between 0 and 1)
    overall_compression = layers.Lambda(
        lambda x: 1.0 - 0.8 * x  # Higher BPP = lower compression
    )(target_bpp_normalized)

    # Make HQ/LQ threshold directly dependent on target BPP
    hq_lq_threshold = layers.Lambda(
    lambda x: 0.9 - 0.85 * x  # More dynamic range
    )(target_bpp_normalized)
    
    # Make quantization strength inversely proportional to target BPP
    quant_strength = layers.Lambda(
        lambda x: 0.9 - 0.8 * x  # Higher BPP = finer quantization
    )(target_bpp_normalized)

    enhanced_saliency = layers.Lambda(lambda x: tf.pow(x, 0.7))(saliency_input)


    # Modified version with greater dynamic range
    dynamic_threshold = layers.Lambda(
        lambda x: tf.sigmoid((x[0] - tf.reshape(x[1], [-1, 1, 1, 1])) * 20.0)  # Increased slope for sharper transition
    )([enhanced_saliency, hq_lq_threshold])



    # In build_adaptive_compression_model:
    # Apply quantization to latent vectors - create separate instances for HQ and LQ
    adaptive_quantize_layer_hq = AdaptiveQuantizationLayer(name="quantize_hq")
    adaptive_quantize_layer_lq = AdaptiveQuantizationLayer(name="quantize_lq")
    hq_latent_quantized = adaptive_quantize_layer_hq([hq_latent, hq_latent_saliency, quant_strength])
    lq_latent_quantized = adaptive_quantize_layer_lq([lq_latent, lq_latent_saliency, quant_strength])
    
    # 6. Generate reconstructed outputs
    hq_output = hq_generator([hq_latent_quantized, hq_skip1, hq_skip2, hq_skip3])
    lq_output = lq_generator([lq_latent_quantized, lq_skip1, lq_skip2, lq_skip3])
    
    # 7. Blend outputs based on saliency and RD parameters
    # Convert pixel-based saliency and latent-based saliency to a combined saliency map
    
    # Create enhanced saliency mask
    

    # Create a soft HQ/LQ boundary - simpler approach
 
    
    # Blend outputs
    weighted_hq = layers.Multiply()([hq_output, dynamic_threshold])
    weighted_lq = layers.Multiply()([lq_output, 1.0 - dynamic_threshold])
    blended_output = layers.Add()([weighted_hq, weighted_lq])
    
    # Create full model
    if has_target_bpp:
        adaptive_model = keras.Model(
            inputs=[img_input, saliency_input, target_bpp_input],
            outputs=[
                blended_output, 
                hq_latent_quantized, 
                lq_latent_quantized,
                rd_params,
                dynamic_threshold
            ],
            name="adaptive_compression_model"
        )
    else:
        adaptive_model = keras.Model(
            inputs=[img_input, saliency_input],
            outputs=[
                blended_output, 
                hq_latent_quantized, 
                lq_latent_quantized,
                rd_params,
                dynamic_threshold
            ],
            name="adaptive_compression_model"
        )
    
    # Return all components
    return {
        'adaptive_model': adaptive_model,
        'hq_encoder': hq_encoder,
        'hq_generator': hq_generator,
        'lq_encoder': lq_encoder,
        'lq_generator': lq_generator,
        'latent_saliency_hq': latent_saliency_hq,
        'latent_saliency_lq': latent_saliency_lq,
        'rd_optimizer': rd_optimizer
    }

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

def visualize_results(original, saliency_map, compressed, save_path=None, bit_allocation=None):
    """
    Visualize original image, saliency map, compressed image, and optional bit allocation map.
    
    Args:
        original: Original image (normalized to [-1, 1])
        saliency_map: Saliency map with values in [0, 1]
        compressed: Compressed image (normalized to [-1, 1])
        save_path: If provided, save the visualization to this path
        bit_allocation: Optional bit allocation map (HQ/LQ regions)
    """
    # Convert images from [-1, 1] to [0, 1] for display
    orig_display = (original + 1) / 2
    comp_display = (compressed + 1) / 2
    
    # Determine number of subplots based on presence of bit allocation
    if bit_allocation is not None:
        num_plots = 4
    else:
        num_plots = 3
    
    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))
    
    axes[0].imshow(orig_display)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(saliency_map, cmap='hot')
    axes[1].set_title('Saliency Map')
    axes[1].axis('off')
    
    axes[2].imshow(comp_display)
    axes[2].set_title('Compressed')
    axes[2].axis('off')
    
    if bit_allocation is not None:
        axes[3].imshow(bit_allocation, cmap='viridis')
        axes[3].set_title('Bit Allocation (HQ/LQ)')
        axes[3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

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


def visualize_bit_allocation_by_bpp(image, mask, model, save_path=None):
    """Visualize how bit allocation changes with target BPP."""
    try:
        # Convert inputs to TensorFlow tensors if they aren't already
        if not isinstance(image, tf.Tensor):
            image = tf.convert_to_tensor(image, dtype=tf.float32)
        if not isinstance(mask, tf.Tensor):
            mask = tf.convert_to_tensor(mask, dtype=tf.float32)
            
        test_bpps = [0.1, 1.0, 2.0]
        
        # Create figure
        fig, axes = plt.subplots(1, len(test_bpps) + 1, figsize=(5 * (len(test_bpps) + 1), 5))
        
        # Show original image
        img_np = image.numpy() if hasattr(image, 'numpy') else image
        axes[0].imshow((img_np + 1) / 2)  # Convert from [-1,1] to [0,1]
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Add batch dimension if not already present
        img_batch = tf.expand_dims(image, 0) if len(image.shape) == 3 else image
        
        # Ensure mask has right dimensions [batch, h, w, 1]
        if len(mask.shape) == 2:  # [h, w]
            mask_batch = tf.expand_dims(tf.expand_dims(mask, -1), 0)
        elif len(mask.shape) == 3 and mask.shape[-1] == 1:  # [h, w, 1]
            mask_batch = tf.expand_dims(mask, 0)
        elif len(mask.shape) == 3 and mask.shape[0] == 1:  # [1, h, w]
            mask_batch = tf.expand_dims(mask, -1)
        else:
            mask_batch = mask
            
        # Show bit allocation maps at different BPPs
        for i, bpp in enumerate(test_bpps):
            print(f"Processing BPP: {bpp}")
            bpp_tensor = tf.constant([[bpp]], dtype=tf.float32)
            
            # Run model prediction with error handling
            try:
                # Use model's call method directly in eager mode
                outputs = model([img_batch, mask_batch, bpp_tensor], training=False)
                bit_allocation = outputs[4]
                
                # Convert to numpy for plotting
                bit_alloc_np = bit_allocation[0].numpy()
                
                axes[i+1].imshow(bit_alloc_np, cmap='viridis')
                axes[i+1].set_title(f'Bit Allocation at {bpp} BPP')
                axes[i+1].axis('off')
            except Exception as e:
                print(f"Error predicting for BPP {bpp}: {str(e)}")
                # Still create a subplot but show error
                axes[i+1].text(0.5, 0.5, f"Error: {str(e)}", 
                             horizontalalignment='center',
                             verticalalignment='center',
                             transform=axes[i+1].transAxes)
                axes[i+1].set_title(f'Failed at {bpp} BPP')
                axes[i+1].axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            print(f"Saved visualization to {save_path}")
            plt.close()
        else:
            plt.show()
    except Exception as e:
        print(f"Error in visualize_bit_allocation_by_bpp: {str(e)}")
        import traceback
        traceback.print_exc()
