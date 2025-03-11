import os
# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import VGG16 #type:ignore
from tensorflow.keras.applications import VGG19, VGG16 #type:ignore
from tensorflow.keras.applications.vgg16 import preprocess_input #type:ignore
from keras import layers
import glob
import time
import matplotlib.pyplot as plt
from GAN_functions import (
    load_and_preprocess_image, compute_saliency_map, create_saliency_mask,
    build_discriminator, build_adaptive_compression_model, create_directories,
    save_image, visualize_results, build_encoder, build_generator, 
    build_latent_saliency_model, build_rate_distortion_optimizer,
    SelfAttention
)

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configuration
IMG_SIZE = (256, 256)
IMG_SHAPE = (IMG_SIZE[0], IMG_SIZE[1], 3)
BASE_LATENT_DIM = 512  # Base dimension for non-salient regions
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 1e-4
BPP_RANGE = [0.1, 1.0, 2.0]  # Different target bitrates to train for

# Set memory growth for GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Found {len(gpus)} GPU(s), memory growth enabled")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(f"Error setting memory growth: {e}")

# Directories
TRAIN_DIR = "dataset"
RESULTS_DIR = "results"
MODEL_DIR = "models"
create_directories([RESULTS_DIR, MODEL_DIR])

# Load training images
def load_training_data(train_dir, max_images=None):
    print(f"Loading training images from {train_dir}...")
    image_paths = glob.glob(os.path.join(train_dir, "*.jpg")) + \
                 glob.glob(os.path.join(train_dir, "*.png")) + \
                 glob.glob(os.path.join(train_dir, "*.jpeg"))
    
    if max_images:
        image_paths = image_paths[:max_images]
    
    print(f"Found {len(image_paths)} images.")
    
    images = []
    for path in image_paths:
        try:
            img = load_and_preprocess_image(path, IMG_SIZE)
            images.append(img)
        except Exception as e:
            print(f"Error loading {path}: {e}")
    
    return np.array(images)

# Generate saliency masks for all images - with both traditional and learned methods
def prepare_training_data(images):
    print("Generating saliency masks...")
    images_with_masks = []
    
    for i, img in enumerate(images):
        # Use the combined method for traditional saliency
        saliency_map = compute_saliency_map(img, method='combined')
        mask = create_saliency_mask(saliency_map, smooth=True)
        # Expand mask to match image dimensions (H, W, 1)
        mask = np.expand_dims(mask, axis=-1)
        images_with_masks.append((img, mask))
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(images)} images")
    
    return images_with_masks

def create_tf_dataset(images_with_masks, batch_size, target_bpp=None):
    """
    Create a TensorFlow dataset with optional target bitrate.
    
    Args:
        images_with_masks: List of (image, mask) tuples
        batch_size: Batch size
        target_bpp: Target bits per pixel (if None, random values will be used)
    """
    def generator():
        for img, mask in images_with_masks:
            if target_bpp is None:
                # Select a random target BPP from the range for varied training
                random_bpp = np.random.choice(BPP_RANGE)
                yield (img, mask, np.array([random_bpp], dtype=np.float32)), img
            else:
                # Use the specified target BPP
                yield (img, mask, np.array([target_bpp], dtype=np.float32)), img
    
    output_signature = (
        (
            tf.TensorSpec(shape=IMG_SHAPE, dtype=tf.float32),  # Image
            tf.TensorSpec(shape=(IMG_SIZE[0], IMG_SIZE[1], 1), dtype=tf.float32),  # Mask
            tf.TensorSpec(shape=(1,), dtype=tf.float32)  # Target BPP
        ),
        tf.TensorSpec(shape=IMG_SHAPE, dtype=tf.float32)  # Output image (for reconstruction)
    )
    
    return tf.data.Dataset.from_generator(
        generator,
        output_signature=output_signature
    ).batch(batch_size).shuffle(len(images_with_masks))

# Training loop
def train_gan(dataset, epochs, steps_per_epoch):
    # Initialize history dictionary to store losses
    history = {
        "d_loss": [],
        "g_loss": [],
        "reconstruction_loss": [],
        "gan_loss": [],
        "perceptual_loss": [],
        "l1_loss": [],
        "latent_saliency_loss": [],
        "rd_loss": []
    }
    
    # Build models
    models = build_adaptive_compression_model(IMG_SHAPE, BASE_LATENT_DIM, target_bpp=True)
    adaptive_model = models['adaptive_model']
    hq_encoder = models['hq_encoder']
    hq_generator = models['hq_generator']
    lq_encoder = models['lq_encoder']
    lq_generator = models['lq_generator']
    latent_saliency_hq = models['latent_saliency_hq']
    latent_saliency_lq = models['latent_saliency_lq']
    rd_optimizer = models['rd_optimizer']
    
    # Build discriminator
    discriminator = build_discriminator(IMG_SHAPE)
    
    # Create optimizer
    d_optimizer = keras.optimizers.Adam(LEARNING_RATE, clipnorm=1.0)
    g_optimizer = keras.optimizers.Adam(LEARNING_RATE, clipnorm=1.0)
    ls_optimizer = keras.optimizers.Adam(LEARNING_RATE * 0.5, clipnorm=1.0)  # Slower learning for saliency
    rd_optimizer_opt = keras.optimizers.Adam(LEARNING_RATE * 0.5, clipnorm=1.0)  # Slower learning for RD
    
    # Define loss functions
    mse_loss = keras.losses.MeanSquaredError()
    bce_loss = keras.losses.BinaryCrossentropy()
    mae_loss = keras.losses.MeanAbsoluteError()
    
    # Build perceptual model for VGG-based perceptual loss
    def build_perceptual_model():
        # Use VGG19 instead of VGG16
        vgg = VGG19(include_top=False, weights='imagenet', input_shape=(256, 256, 3))
        
        # Set to non-trainable
        for layer in vgg.layers:
            layer.trainable = False
        
        # Different layers for VGG19
        feature_layers = ['block1_conv2', 'block2_conv2', 'block3_conv4', 'block4_conv4', 'block5_conv4']
        feature_weights = [0.1, 0.1, 0.2, 0.3, 0.3]
        
        outputs = [vgg.get_layer(layer).output for layer in feature_layers]
        feature_model = keras.Model(inputs=vgg.input, outputs=outputs)
        
        return feature_model, feature_weights

    # Create perceptual model

    try:
        perceptual_model, feature_weights = build_perceptual_model()  # <-- Unpack both return values
        use_perceptual_loss = True
        print("Successfully created perceptual loss model using VGG16")
    except:
        print("WARNING: Could not create VGG16 perceptual model. Falling back to standard loss.")
        use_perceptual_loss = False
        perceptual_model = None
        feature_weights = None
    
    # Define training step using tf.function for efficiency
    @tf.function
    def train_d_step(batch_data):
        (images, masks, target_bpp), _ = batch_data
        
        with tf.GradientTape() as d_tape:
            # Generate compressed images using the adaptive model
            compressed_output, _, _, _, _ = adaptive_model([images, masks, target_bpp])
            
            # Get discriminator predictions
            real_preds = discriminator(images)
            fake_preds = discriminator(compressed_output)
            
            # Create labels with smoothing
            batch_size = tf.shape(images)[0]
            real_labels = tf.ones((batch_size, 1)) * 0.9
            fake_labels = tf.zeros((batch_size, 1)) + 0.1
            
            # Calculate losses
            d_loss_real = bce_loss(real_labels, real_preds)
            d_loss_fake = bce_loss(fake_labels, fake_preds)
            d_loss = 0.5 * (d_loss_real + d_loss_fake)
        
        # Calculate gradients and update discriminator
        d_gradients = d_tape.gradient(d_loss, discriminator.trainable_weights)
        d_optimizer.apply_gradients(zip(d_gradients, discriminator.trainable_weights))
        
        return d_loss
    
    @tf.function
    def train_g_step(batch_data):
        (images, masks, target_bpp), _ = batch_data
        
        # Combined generator training: encoders, generators, latent saliency, and RD optimizer
        with tf.GradientTape() as g_tape, tf.GradientTape() as ls_tape, tf.GradientTape() as rd_tape:
            # Forward pass through the entire model
            compressed_output, hq_latent, lq_latent, rd_params, bit_allocation = adaptive_model([images, masks, target_bpp])
            
            # Compute traditional saliency score for comparison with learned saliency
            # This would need to be implemented as a TF op for efficiency
            # For the function version, we'll use the existing saliency maps
            
            # Get discriminator predictions for fake images
            fake_preds = discriminator(compressed_output)
            
            # Create target labels for GAN
            batch_size = tf.shape(images)[0]
            real_labels = tf.ones((batch_size, 1))
            
            # Calculate reconstruction loss
            recon_loss = mse_loss(images, compressed_output)
            
            # Calculate GAN loss
            adv_loss = bce_loss(real_labels, fake_preds)
            
            # Calculate L1 loss
            l1_loss = mae_loss(images, compressed_output)
            
            # Calculate perceptual loss if available
            if use_perceptual_loss:
                # Convert images from [-1, 1] to [0, 255] for VGG
                orig_vgg_input = (images + 1) * 127.5
                gen_vgg_input = (compressed_output + 1) * 127.5
                
                # Get features
                orig_features = perceptual_model(preprocess_input(orig_vgg_input))
                gen_features = perceptual_model(preprocess_input(gen_vgg_input))
                
                # Initialize perceptual loss
                perceptual_loss = 0
                layer_perceptual_losses = []
                

                for i, (orig_feat, gen_feat) in enumerate(zip(orig_features, gen_features)):
                    # Normalize features to have unit variance
                    orig_mean = tf.reduce_mean(orig_feat)
                    orig_std = tf.math.reduce_std(orig_feat) + 1e-8
                    gen_mean = tf.reduce_mean(gen_feat)
                    gen_std = tf.math.reduce_std(gen_feat) + 1e-8
                    
                    # Normalize features to zero mean and unit variance
                    orig_norm = (orig_feat - orig_mean) / orig_std
                    gen_norm = (gen_feat - gen_mean) / gen_std
                    
                    # Compute MSE on normalized features
                    layer_loss = tf.reduce_mean(tf.square(orig_norm - gen_norm)) * feature_weights[i]
                    layer_perceptual_losses.append(layer_loss)
                    perceptual_loss += layer_loss
                
                # Scale the perceptual loss - increased from 0.005
                perceptual_loss *= 0.5  # Try 0.00001 or even 0.000001
                
                # Optional: Log individual layer losses (outside of tf.function or use tf.summary)
                # Note: tf.print inside tf.function should be used carefully
                for i, loss in enumerate(layer_perceptual_losses):
                    tf.print(f"Layer {i} perceptual loss:", loss)
            else:
                perceptual_loss = tf.constant(0.0)
            
            # Latent saliency losses - encourage latent saliency to match pixel saliency
            # Calculate latent saliency loss - encourage consistency with traditional saliency
            # We want high latent saliency scores where the mask has high values
            flat_masks = tf.reduce_mean(masks, axis=[1, 2, 3])  # Average saliency per image
            
            # Get latent saliency predictions
            hq_saliency = latent_saliency_hq(hq_latent)
            lq_saliency = latent_saliency_lq(lq_latent)
            
            # We want HQ saliency to be high where mask is high, LQ saliency to be low where mask is high
            # Simplified approach: HQ should match mask, LQ should match inverse of mask
            latent_saliency_loss_hq = tf.reduce_mean(tf.square(hq_saliency - flat_masks))
            latent_saliency_loss_lq = tf.reduce_mean(tf.square(lq_saliency - (1.0 - flat_masks)))
            latent_saliency_loss = latent_saliency_loss_hq + latent_saliency_loss_lq
            
            # Rate-distortion loss - penalize excessive bits for the target bitrate
            # Simplified approach: encourage bit allocation to match the target bitrate
            target_compression = 1.0 - target_bpp / 4.0  # Normalize to [0, 1] range assuming max 4 bpp
            target_compression = tf.clip_by_value(target_compression, 0.1, 0.9)
            
            # Extract the compression parameter from rd_params
            actual_compression = tf.slice(rd_params, [0, 0], [-1, 1])
            rd_loss = tf.reduce_mean(tf.square(actual_compression - target_compression))

            # Calculate actual BPP based on the model's output
            hq_ratio = tf.reduce_mean(bit_allocation, axis=[1, 2, 3])
            lq_ratio = 1.0 - hq_ratio

            # Calculate bits used for HQ and LQ regions (32 bits per float)
            hq_bits = hq_ratio * (BASE_LATENT_DIM * 2) * 32
            lq_bits = lq_ratio * BASE_LATENT_DIM * 32
            total_bits = hq_bits + lq_bits

            # Calculate actual BPP (bits per pixel)
            actual_bpp = total_bits / (IMG_SIZE[0] * IMG_SIZE[1])

            # Add explicit bitrate control loss with higher weight
            bitrate_control_loss = tf.reduce_mean(tf.abs(actual_bpp - target_bpp)) * 1.0  # Higher weight and using absolute error

            bitrate_underutilization_penalty = tf.nn.relu(target_bpp - actual_bpp) * tf.nn.relu(target_bpp - 1.0) * 2.0





            # After calculating all the individual losses but before calculating the combined g_loss
            # Add this to print the component values:
            tf.print("Loss components - Recon:", recon_loss, 
                    "L1:", l1_loss, 
                    "Perceptual:", perceptual_loss, 
                    "Adv:", adv_loss, 
                    "BPP control:", bitrate_control_loss)


            # Combined generator loss (with weights)
            g_loss = (
            0.35 * recon_loss +         # Slightly reduced
            0.15 * l1_loss +            # Increased
            0.15 * perceptual_loss +    # Reduced slightly
            0.15 * adv_loss +           # Increased
            0.20 * bitrate_control_loss # Unchanged
            )

            # Add this after calculating your g_loss
            g_loss_offset = tf.maximum(0.0, 0.5 - g_loss) * 0.5  # Add up to 0.25 to get to 0.5 minimum
            g_loss += g_loss_offset
            
            # Latent saliency and RD losses are separate for different optimizers
            ls_loss = 0.1 * latent_saliency_loss
            rd_opt_loss = 0.3 * rd_loss + 0.7 * bitrate_control_loss + bitrate_underutilization_penalty

            
        # Get trainable weights for each component
        g_trainable_weights = (
            hq_encoder.trainable_weights + 
            hq_generator.trainable_weights + 
            lq_encoder.trainable_weights + 
            lq_generator.trainable_weights
        )
        ls_trainable_weights = (
            latent_saliency_hq.trainable_weights +
            latent_saliency_lq.trainable_weights
        )
        rd_trainable_weights = rd_optimizer.trainable_weights
        
        # Calculate gradients and update weights for each component
        g_gradients = g_tape.gradient(g_loss, g_trainable_weights)
        g_optimizer.apply_gradients(zip(g_gradients, g_trainable_weights))
        
        ls_gradients = ls_tape.gradient(ls_loss, ls_trainable_weights)
        ls_optimizer.apply_gradients(zip(ls_gradients, ls_trainable_weights))
        
        rd_gradients = rd_tape.gradient(rd_opt_loss, rd_trainable_weights)
        rd_optimizer_opt.apply_gradients(zip(rd_gradients, rd_trainable_weights))
        
        return g_loss, recon_loss, adv_loss, perceptual_loss, l1_loss, latent_saliency_loss, rd_loss, bitrate_control_loss
    
    # Create sample directories
    SAMPLES_DIR = os.path.join(RESULTS_DIR, "training_samples")
    create_directories([SAMPLES_DIR])
    
    # Get sample images for visualization
    sample_images = []
    sample_masks = []
    sample_bpps = []
    for ((batch_images, batch_masks, batch_bpps), _) in dataset.take(1):
        num_samples = min(4, batch_images.shape[0])
        for i in range(num_samples):
            sample_images.append(batch_images[i].numpy())
            sample_masks.append(batch_masks[i].numpy())
            sample_bpps.append(batch_bpps[i].numpy())
    
    print(f"Selected {len(sample_images)} sample images for training visualization")
    
    # Training loop
    for epoch in range(epochs):
        start_time = time.time()
        
        # Lists to store batch losses
        d_losses = []
        g_losses = []
        recon_losses = []
        gan_losses = []
        perceptual_losses = []
        l1_losses = []
        latent_saliency_losses = []
        rd_losses = []
        
        for step, batch_data in enumerate(dataset.take(steps_per_epoch)):
            # Train discriminator only every other batch
            if step % 2 == 0:
                d_loss = train_d_step(batch_data)
                d_losses.append(d_loss)
            
            # Train generator
            g_loss, recon_loss, adv_loss, perceptual_loss, l1_loss, latent_saliency_loss, rd_loss, bitrate_control_loss = train_g_step(batch_data)

            
            g_losses.append(g_loss)
            recon_losses.append(recon_loss)
            gan_losses.append(adv_loss)
            l1_losses.append(l1_loss)
            latent_saliency_losses.append(latent_saliency_loss)
            rd_losses.append(rd_loss)
            
            if perceptual_loss is not None:
                perceptual_losses.append(perceptual_loss)
            
            # Print progress
            if (step + 1) % 1 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Step {step+1}/{steps_per_epoch}: "
                      f"D Loss: {np.mean(d_losses):.4f}, G Loss: {np.mean(g_losses):.4f}, "
                      f"Recon Loss: {np.mean(recon_losses):.4f}, RD Loss: {np.mean(rd_losses):.4f}")
        
        # Visualize samples at the end of each epoch
        if sample_images:
            print(f"Generating sample visualizations for epoch {epoch+1}...")
            for i, (img, mask, bpp) in enumerate(zip(sample_images, sample_masks, sample_bpps)):
                img_batch = np.expand_dims(img, axis=0)
                mask_batch = np.expand_dims(mask, axis=0)
                bpp_batch = np.expand_dims(bpp, axis=0)
                
                # Generate compressed image using the adaptive model
                compressed_output, _, _, _, bit_allocation = adaptive_model.predict([img_batch, mask_batch, bpp_batch])
                compressed_img = compressed_output[0]
                bit_alloc_map = bit_allocation[0]
                
                # Save visualization
                vis_path = os.path.join(SAMPLES_DIR, f"sample_{i+1}_epoch_{epoch+1}_bpp_{bpp[0]:.2f}.png")
                visualize_results(img, mask[:,:,0], compressed_img, vis_path, bit_alloc_map)

         # Add the call here, right after visualizations
        if (epoch + 1) % 1 == 0:  # Test every 5 epochs
            print(f"Testing rate control gradients at epoch {epoch+1}...")
            test_img = tf.convert_to_tensor(sample_images[0])
            test_mask = tf.convert_to_tensor(sample_masks[0])
            test_rate_control_gradients(adaptive_model, test_img, test_mask)
    

        # Calculate epoch average losses
        epoch_d_loss = np.mean([loss.numpy() for loss in d_losses]) if d_losses else float('nan')
        epoch_g_loss = np.mean([loss.numpy() for loss in g_losses]) if g_losses else float('nan')
        epoch_recon_loss = np.mean([loss.numpy() for loss in recon_losses]) if recon_losses else float('nan')
        epoch_gan_loss = np.mean([loss.numpy() for loss in gan_losses]) if gan_losses else float('nan')
        epoch_l1_loss = np.mean([loss.numpy() for loss in l1_losses]) if l1_losses else float('nan')
        epoch_latent_saliency_loss = np.mean([loss.numpy() for loss in latent_saliency_losses]) if latent_saliency_losses else float('nan')
        epoch_rd_loss = np.mean([loss.numpy() for loss in rd_losses]) if rd_losses else float('nan')
        epoch_perceptual_loss = np.mean([loss.numpy() for loss in perceptual_losses]) if perceptual_losses else float('nan')
        
        # Update history
        history["d_loss"].append(epoch_d_loss)
        history["g_loss"].append(epoch_g_loss)
        history["reconstruction_loss"].append(epoch_recon_loss)
        history["gan_loss"].append(epoch_gan_loss)
        history["l1_loss"].append(epoch_l1_loss)
        history["latent_saliency_loss"].append(epoch_latent_saliency_loss)
        history["rd_loss"].append(epoch_rd_loss)
        history["perceptual_loss"].append(epoch_perceptual_loss if perceptual_losses else float('nan'))
        
        # Print epoch summary
        time_taken = time.time() - start_time
        print(f"Epoch {epoch+1}/{epochs} completed in {time_taken:.2f}s - "
              f"D Loss: {epoch_d_loss:.4f}, G Loss: {epoch_g_loss:.4f}, "
              f"Recon Loss: {epoch_recon_loss:.4f}, RD Loss: {epoch_rd_loss:.4f}, "
              f"Latent Saliency Loss: {epoch_latent_saliency_loss:.4f}")
        
        # Plot and save loss history
        plt.figure(figsize=(15, 12))
        
        # Plot 1: GAN Losses
        plt.subplot(2, 2, 1)
        plt.plot(history["d_loss"], label="Discriminator Loss")
        plt.plot(history["g_loss"], label="Generator Loss")
        plt.legend()
        plt.title("GAN Losses")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        
        # Plot 2: Generator Components
        plt.subplot(2, 2, 2)
        plt.plot(history["reconstruction_loss"], label="Reconstruction Loss")
        plt.plot(history["gan_loss"], label="GAN Component Loss")
        if history["perceptual_loss"]:
            plt.plot(history["perceptual_loss"], label="Perceptual Loss")
        plt.plot(history["l1_loss"], label="L1 Loss")
        plt.legend()
        plt.title("Generator Loss Components")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        
        # Plot 3: Latent Saliency and RD Losses
        plt.subplot(2, 2, 3)
        plt.plot(history["latent_saliency_loss"], label="Latent Saliency Loss")
        plt.plot(history["rd_loss"], label="Rate-Distortion Loss")
        plt.legend()
        plt.title("Adaptive Compression Losses")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        
        # Plot 4: Total Generator Loss
        plt.subplot(2, 2, 4)
        plt.plot(history["g_loss"], label="Total Generator Loss")
        plt.title("Total Generator Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"loss_history_epoch_{epoch+1}.png"))
        plt.close()

        
        # Save models every 5 epochs or at the end
        if (epoch + 1) % 5 == 0 or epoch + 1 == epochs:
            print(f"Saving models at epoch {epoch+1}...")
            try:
                # Save adaptive model and all components
                adaptive_model.save(os.path.join(MODEL_DIR, f"adaptive_model_epoch_{epoch+1}.h5"))
                discriminator.save(os.path.join(MODEL_DIR, f"discriminator_epoch_{epoch+1}.h5"))
                
                # Save individual components
                hq_encoder.save(os.path.join(MODEL_DIR, f"hq_encoder_epoch_{epoch+1}.h5"))
                hq_generator.save(os.path.join(MODEL_DIR, f"hq_generator_epoch_{epoch+1}.h5"))
                lq_encoder.save(os.path.join(MODEL_DIR, f"lq_encoder_epoch_{epoch+1}.h5"))
                lq_generator.save(os.path.join(MODEL_DIR, f"lq_generator_epoch_{epoch+1}.h5"))
                latent_saliency_hq.save(os.path.join(MODEL_DIR, f"latent_saliency_hq_epoch_{epoch+1}.h5"))
                latent_saliency_lq.save(os.path.join(MODEL_DIR, f"latent_saliency_lq_epoch_{epoch+1}.h5"))
                rd_optimizer.save(os.path.join(MODEL_DIR, f"rd_optimizer_epoch_{epoch+1}.h5"))
            except Exception as e:
                print(f"Error saving models: {e}")
    
    # Save final models
    print("Saving final models...")
    try:
        adaptive_model.save(os.path.join(MODEL_DIR, "adaptive_model_final.h5"))
        discriminator.save(os.path.join(MODEL_DIR, "discriminator_final.h5"))
        
        # Save individual components
        hq_encoder.save(os.path.join(MODEL_DIR, "hq_encoder_final.h5"))
        hq_generator.save(os.path.join(MODEL_DIR, "hq_generator_final.h5"))
        lq_encoder.save(os.path.join(MODEL_DIR, "lq_encoder_final.h5"))
        lq_generator.save(os.path.join(MODEL_DIR, "lq_generator_final.h5"))
        latent_saliency_hq.save(os.path.join(MODEL_DIR, "latent_saliency_hq_final.h5"))
        latent_saliency_lq.save(os.path.join(MODEL_DIR, "latent_saliency_lq_final.h5"))
        rd_optimizer.save(os.path.join(MODEL_DIR, "rd_optimizer_final.h5"))
    except Exception as e:
        print(f"Error saving final models: {e}")
    
    # Plot final loss history with all metrics
    plt.figure(figsize=(20, 15))
    
    # Plot all metrics in separate subplots
    metrics = [
        ("d_loss", "Discriminator Loss"),
        ("g_loss", "Generator Loss"),
        ("reconstruction_loss", "Reconstruction Loss"),
        ("gan_loss", "GAN Component Loss"),
        ("l1_loss", "L1 Loss"),
        ("perceptual_loss", "Perceptual Loss"),
        ("latent_saliency_loss", "Latent Saliency Loss"),
        ("rd_loss", "Rate-Distortion Loss")
    ]
    
    for i, (metric_key, metric_title) in enumerate(metrics):
        plt.subplot(4, 2, i+1)
        if metric_key in history and len(history[metric_key]) > 0:
            plt.plot(history[metric_key], 'b-')
            plt.title(metric_title)
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "final_loss_history.png"), dpi=300)
    plt.close()
    
    return {
        'adaptive_model': adaptive_model, 
        'discriminator': discriminator, 
        'hq_encoder': hq_encoder, 
        'hq_generator': hq_generator, 
        'lq_encoder': lq_encoder, 
        'lq_generator': lq_generator,
        'latent_saliency_hq': latent_saliency_hq,
        'latent_saliency_lq': latent_saliency_lq,
        'rd_optimizer': rd_optimizer,
        'history': history
    }

def test_rate_control_gradients(model, test_image, test_mask):
    """Test if target BPP changes actually affect the model's behavior."""
    test_bpps = [0.1, 1.0, 2.0]
    results = []
    
    img_batch = tf.expand_dims(test_image, axis=0)
    mask_batch = tf.expand_dims(tf.expand_dims(test_mask, axis=-1), axis=0)
    
    for bpp in test_bpps:
        with tf.GradientTape() as tape:
            bpp_tensor = tf.constant([[bpp]], dtype=tf.float32)
            tape.watch(bpp_tensor)
            outputs = model([img_batch, mask_batch, bpp_tensor])
            bit_allocation = outputs[4]  # Assuming this is where bit allocation is
            hq_ratio = tf.reduce_mean(bit_allocation)
        
        # Check gradients
        grads = tape.gradient(hq_ratio, bpp_tensor)
        results.append({
            'target_bpp': bpp,
            'hq_ratio': hq_ratio.numpy(),
            'gradient': grads.numpy() if grads is not None else np.array([[0.0]])
        })
    
    print("\nRate Control Gradient Test:")
    for r in results:
        print(f"  BPP: {r['target_bpp']}, HQ Ratio: {r['hq_ratio']:.4f}, Gradient: {r['gradient'][0][0]:.6f}")

    # Also call the visualization function
    from GAN_functions import visualize_bit_allocation_by_bpp
    visualize_bit_allocation_by_bpp(
        test_image.numpy(), 
        test_mask.numpy(), 
        model, 
        save_path="rate_control_test.png"
    )
    print("Rate control visualization saved to 'rate_control_test.png'")

def main():
    print("Starting training process...")
    
    # Load and prepare training data
    images = load_training_data(TRAIN_DIR)
    if len(images) == 0:
        print("No images found in the training directory!")
        return
    
    # Generate saliency maps and prepare dataset
    images_with_masks = prepare_training_data(images)
    
    # Create dataset with random target bitrates for varied training
    dataset = create_tf_dataset(images_with_masks, BATCH_SIZE)
    
    # Calculate steps per epoch
    steps_per_epoch = len(images) // BATCH_SIZE
    
    # Train the models
    models = train_gan(dataset, EPOCHS, steps_per_epoch)
    
    # Optional: Train on specific target bitrates
    for target_bpp in BPP_RANGE:
        print(f"\nFine-tuning model for target bitrate: {target_bpp} bpp...")
        # Create dataset with specific target bitrate
        dataset_bpp = create_tf_dataset(images_with_masks, BATCH_SIZE, target_bpp=target_bpp)
        
        # Train for a few epochs with this target bitrate
        fine_tune_epochs = 3
        train_gan(dataset_bpp, fine_tune_epochs, steps_per_epoch)
    
    print("Training completed!")

if __name__ == "__main__":
    main()
