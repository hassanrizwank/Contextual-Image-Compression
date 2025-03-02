import os
# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import VGG16 # type: ignore
from tensorflow.keras.applications.vgg16 import preprocess_input # type: ignore
from keras import layers
import glob
import time
import matplotlib.pyplot as plt
from GAN_functions import (
    load_and_preprocess_image, compute_saliency_map, create_saliency_mask,
    build_discriminator, build_adaptive_compression_model, create_directories,
    save_image, visualize_results, build_encoder, build_generator, 
)

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configuration
IMG_SIZE = (256, 256)
IMG_SHAPE = (IMG_SIZE[0], IMG_SIZE[1], 3)
BASE_LATENT_DIM = 512  # Base dimension for non-salient regions
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 1e-4

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

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

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

# Generate saliency masks for all images
def prepare_training_data(images):
    print("Generating saliency masks...")
    images_with_masks = []
    
    for i, img in enumerate(images):
        # Use the combined method instead of the default
        saliency_map = compute_saliency_map(img, method='combined')
        mask = create_saliency_mask(saliency_map, smooth=True)
        # Expand mask to match image dimensions (H, W, 1)
        mask = np.expand_dims(mask, axis=-1)
        images_with_masks.append((img, mask))
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(images)} images")
    
    return images_with_masks

def create_tf_dataset(images_with_masks, batch_size):
    def generator():
        for img, mask in images_with_masks:
            yield (img, mask), img
    
    return tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            (
                tf.TensorSpec(shape=IMG_SHAPE, dtype=tf.float32),
                tf.TensorSpec(shape=(IMG_SIZE[0], IMG_SIZE[1], 1), dtype=tf.float32)
            ),
            tf.TensorSpec(shape=IMG_SHAPE, dtype=tf.float32)
        )
    ).batch(batch_size).shuffle(len(images_with_masks))

# Build models
def build_models():
    print("Building models...")
    
    # Create encoders and generators directly
    hq_encoder = build_encoder(IMG_SHAPE, BASE_LATENT_DIM * 2, name="hq_encoder")
    hq_generator = build_generator(BASE_LATENT_DIM * 2, IMG_SHAPE, name="hq_generator")
    lq_encoder = build_encoder(IMG_SHAPE, BASE_LATENT_DIM, name="lq_encoder")
    lq_generator = build_generator(BASE_LATENT_DIM, IMG_SHAPE, name="lq_generator")
    
    # Create discriminator
    discriminator = build_discriminator(IMG_SHAPE)
    
    # Freeze discriminator for generator training
    discriminator.trainable = False
    
    # Build the combined model directly without using adaptive_model
    img_input = keras.Input(shape=IMG_SHAPE)
    saliency_input = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 1))


    hq_outputs = hq_encoder(img_input)
    lq_outputs = lq_encoder(img_input)

    # Extract latent and skip connections
    hq_latent = hq_outputs[0]
    hq_skips = hq_outputs[1:]

    lq_latent = lq_outputs[0]
    lq_skips = lq_outputs[1:]

    # Feed to generators with skip connections
    hq_output = hq_generator([hq_latent] + hq_skips)
    lq_output = lq_generator([lq_latent] + lq_skips)
    
    
    # Ensure saliency mask matches output dimensions
    if hq_output.shape[1:3] != img_input.shape[1:3]:
        resized_saliency = layers.Resizing(
            hq_output.shape[1], 
            hq_output.shape[2]
        )(saliency_input)
        expanded_saliency = resized_saliency
    else:
        expanded_saliency = saliency_input
    
    # Blend outputs based on saliency
    hq_mask = expanded_saliency
    lq_mask = layers.Lambda(lambda x: 1.0 - x)(expanded_saliency)
    
    weighted_hq = layers.Multiply()([hq_output, hq_mask])
    weighted_lq = layers.Multiply()([lq_output, lq_mask])
    blended_output = layers.Add()([weighted_hq, weighted_lq])
    
    # Feed the blended output to the discriminator
    validity = discriminator(blended_output)
    
    # Create the adaptive model (for convenience in inference)
    adaptive_model = keras.Model(
        inputs=[img_input, saliency_input],
        outputs=[blended_output, hq_latent, lq_latent],
        name="adaptive_compression_model"
    )
    
    # Create the combined model for training
    combined_model = keras.Model(
        inputs=[img_input, saliency_input],
        outputs=[blended_output, validity],
        name="gan_model"
    )
    
    # Make sure all models are built (have weights)
    dummy_img = np.zeros((1, IMG_SIZE[0], IMG_SIZE[1], 3))
    dummy_mask = np.zeros((1, IMG_SIZE[0], IMG_SIZE[1], 1))
    _ = hq_encoder.predict(dummy_img)
    _ = hq_generator.predict(np.zeros((1, BASE_LATENT_DIM * 2)))
    _ = lq_encoder.predict(dummy_img)
    _ = lq_generator.predict(np.zeros((1, BASE_LATENT_DIM)))
    _ = combined_model.predict([dummy_img, dummy_mask])
    
    # Print model summaries for debugging
    print(f"HQ Encoder trainable weights: {len(hq_encoder.trainable_weights)}")
    print(f"HQ Generator trainable weights: {len(hq_generator.trainable_weights)}")
    print(f"LQ Encoder trainable weights: {len(lq_encoder.trainable_weights)}")
    print(f"LQ Generator trainable weights: {len(lq_generator.trainable_weights)}")
    print(f"Combined model trainable weights: {len(combined_model.trainable_weights)}")
    
    # Compile models
    discriminator.trainable = True
    discriminator.compile(
        optimizer=keras.optimizers.Adam(LEARNING_RATE),
        loss='binary_crossentropy'
    )
    
    # For training the generator, keep discriminator frozen
    discriminator.trainable = False
    combined_model.compile(
        optimizer=keras.optimizers.Adam(LEARNING_RATE),
        loss=['mse', 'binary_crossentropy'],
        loss_weights=[0.9, 0.1]  # Prioritize reconstruction over GAN loss
    )
    
    return adaptive_model, discriminator, combined_model, hq_encoder, hq_generator, lq_encoder, lq_generator

# Training loop
def train_gan(dataset, epochs, steps_per_epoch):
    # Build models
    # In the train_gan function, make sure both encoder and generator use the same dimensions
    hq_encoder = build_encoder(IMG_SHAPE, BASE_LATENT_DIM * 2, name="hq_encoder")  # 2x instead of 3x
    hq_generator = build_generator(BASE_LATENT_DIM * 2, IMG_SHAPE, name="hq_generator")  # Also 2x to match
    lq_encoder = build_encoder(IMG_SHAPE, BASE_LATENT_DIM, name="lq_encoder")
    lq_generator = build_generator(BASE_LATENT_DIM, IMG_SHAPE, name="lq_generator")
    discriminator = build_discriminator(IMG_SHAPE)
    
    # Create optimizer
    d_optimizer = keras.optimizers.Adam(LEARNING_RATE, clipnorm=1.0)
    g_optimizer = keras.optimizers.Adam(LEARNING_RATE, clipnorm=1.0)
    
    # Define loss functions
    mse_loss = keras.losses.MeanSquaredError()
    bce_loss = keras.losses.BinaryCrossentropy()
    
    # Build perceptual model for VGG-based perceptual loss
    def build_perceptual_model():
        # Load pretrained VGG16 without classification layers
        vgg = VGG16(include_top=False, weights='imagenet', input_shape=(256, 256, 3))
        
        # Set to non-trainable
        for layer in vgg.layers:
            layer.trainable = False
            
        # Use intermediate layers for perceptual loss
        feature_layers = ['block3_conv3', 'block4_conv3']
        outputs = [vgg.get_layer(layer).output for layer in feature_layers]
        
        # Create feature extraction model
        feature_model = keras.Model(inputs=vgg.input, outputs=outputs)
        
        return feature_model

    # Create perceptual model
    try:
        perceptual_model = build_perceptual_model()
        use_perceptual_loss = True
        print("Successfully created perceptual loss model using VGG16")
    except:
        print("WARNING: Could not create VGG16 perceptual model. Falling back to standard loss.")
        use_perceptual_loss = False
    
    # Create adaptive model for inference only
    img_input = keras.Input(shape=IMG_SHAPE)
    saliency_input = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 1))
    
    # Build model graph
    hq_latent = hq_encoder(img_input)
    lq_latent = lq_encoder(img_input)
    hq_output = hq_generator(hq_latent)
    lq_output = lq_generator(lq_latent)
    
    # Create blend
    expanded_saliency = saliency_input
    # Apply non-linear enhancement to saliency
    enhanced_saliency = layers.Lambda(lambda x: tf.pow(x, 0.7))(expanded_saliency)
    weighted_hq = layers.Multiply()([hq_output, enhanced_saliency])
    weighted_lq = layers.Multiply()([lq_output, 1 - enhanced_saliency])
    blended_output = layers.Add()([weighted_hq, weighted_lq])
    
    # Create model for inference
    adaptive_model = keras.Model(
        inputs=[img_input, saliency_input],
        outputs=[blended_output, hq_latent, lq_latent],
        name="adaptive_compression_model"
    )
    
    print(f"HQ Encoder trainable weights: {len(hq_encoder.trainable_weights)}")
    print(f"HQ Generator trainable weights: {len(hq_generator.trainable_weights)}")
    print(f"LQ Encoder trainable weights: {len(lq_encoder.trainable_weights)}")
    print(f"LQ Generator trainable weights: {len(lq_generator.trainable_weights)}")
    
    # Define training step using tf.function for efficiency
    @tf.function
    def train_d_step(images, masks):
        # Generate compressed images
        with tf.GradientTape() as d_tape:
            # Forward pass through model
            hq_latent_vals = hq_encoder(images)
            lq_latent_vals = lq_encoder(images)
            hq_gen_imgs = hq_generator(hq_latent_vals)
            lq_gen_imgs = lq_generator(lq_latent_vals)
            
            # Blend outputs using enhanced saliency
            enhanced_masks = tf.pow(masks, 0.7)
            weighted_hq = hq_gen_imgs * enhanced_masks
            weighted_lq = lq_gen_imgs * (1 - enhanced_masks)
            gen_imgs = weighted_hq + weighted_lq
            
            # Get discriminator predictions
            real_preds = discriminator(images)
            fake_preds = discriminator(gen_imgs)
            
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
    def train_g_step(images, masks):
        with tf.GradientTape() as g_tape:
            # Forward pass through encoder
            hq_encoder_outputs = hq_encoder(images)
            lq_encoder_outputs = lq_encoder(images)
            
            # Extract latent and skip connections
            hq_latent = hq_encoder_outputs[0]
            hq_skip1 = hq_encoder_outputs[1]
            hq_skip2 = hq_encoder_outputs[2]
            hq_skip3 = hq_encoder_outputs[3]
            
            lq_latent = lq_encoder_outputs[0]
            lq_skip1 = lq_encoder_outputs[1]
            lq_skip2 = lq_encoder_outputs[2]
            lq_skip3 = lq_encoder_outputs[3]
            
            # Generate images
            hq_gen_imgs = hq_generator([hq_latent, hq_skip1, hq_skip2, hq_skip3])
            lq_gen_imgs = lq_generator([lq_latent, lq_skip1, lq_skip2, lq_skip3])
            
            # Blend outputs using enhanced saliency
            enhanced_masks = tf.pow(masks, 0.7)
            weighted_hq = hq_gen_imgs * enhanced_masks
            weighted_lq = lq_gen_imgs * (1 - enhanced_masks)
            gen_imgs = weighted_hq + weighted_lq
            
            # Get discriminator predictions for fake images
            fake_preds = discriminator(gen_imgs)
            
            # Create target labels
            batch_size = tf.shape(images)[0]
            real_labels = tf.ones((batch_size, 1))
            
            # Calculate losses
            recon_loss = mse_loss(images, gen_imgs)
            adv_loss = bce_loss(real_labels, fake_preds)
            
            # Calculate perceptual loss if available
            if use_perceptual_loss:
                # Convert images from [-1, 1] to [0, 255] for VGG
                orig_vgg_input = (images + 1) * 127.5
                gen_vgg_input = (gen_imgs + 1) * 127.5
                
                # Get features
                orig_features = perceptual_model(preprocess_input(orig_vgg_input))
                gen_features = perceptual_model(preprocess_input(gen_vgg_input))
                
                # Calculate feature loss
                perceptual_loss = 0
                for orig_feat, gen_feat in zip(orig_features, gen_features):
                    # Normalize the feature differences by the feature map size
                    feature_size = tf.cast(tf.reduce_prod(tf.shape(orig_feat)[1:]), tf.float32)
                    perceptual_loss += tf.reduce_sum(tf.square(orig_feat - gen_feat)) / feature_size
                perceptual_loss /= len(orig_features)

                perceptual_loss *= 0.15  # Increase from 0.0005 to 0.05
                # Clamp perceptual loss to avoid extreme values
                perceptual_loss = tf.clip_by_value(perceptual_loss, 0.0, 10.0)
                l1_loss = tf.reduce_mean(tf.abs(images - gen_imgs))

                # Combine losses with perceptual component
                g_loss = 0.5 * recon_loss + 0.15 * l1_loss + 0.30 * perceptual_loss + 0.05 * adv_loss
                # Modified debug print statement that will work with TensorFlow tensors
                print(f"Recon: {float(recon_loss):.4f}, L1: {float(l1_loss):.4f}, "
                    f"Perceptual: {float(perceptual_loss):.4f}, Adv: {float(adv_loss):.4f}")
            else:
                # Combine losses without perceptual component
                g_loss = 0.95 * recon_loss + 0.05 * adv_loss
        
        # Get all trainable weights from generator components
        g_trainable_weights = (
            hq_encoder.trainable_weights + 
            hq_generator.trainable_weights + 
            lq_encoder.trainable_weights + 
            lq_generator.trainable_weights
        )
        
        # Calculate gradients and update generator
        g_gradients = g_tape.gradient(g_loss, g_trainable_weights)
        g_optimizer.apply_gradients(zip(g_gradients, g_trainable_weights))
        
        return g_loss, recon_loss, adv_loss
    
    # Create sample directories
    SAMPLES_DIR = os.path.join(RESULTS_DIR, "training_samples")
    create_directories([SAMPLES_DIR])
    
    # Get sample images for visualization
    sample_images = []
    sample_masks = []
    for ((batch_images, batch_masks), _) in dataset.take(1):
        num_samples = min(4, batch_images.shape[0])
        for i in range(num_samples):
            sample_images.append(batch_images[i].numpy())
            sample_masks.append(batch_masks[i].numpy())
    
    print(f"Selected {len(sample_images)} sample images for training visualization")
    history = {"d_loss": [], "g_loss": [], "reconstruction_loss": [], "gan_loss": []}
    
    # Training loop
    for epoch in range(epochs):
        start_time = time.time()
        
        # Lists to store batch losses
        d_losses = []
        g_losses = []
        recon_losses = []
        gan_losses = []
        
        for step, ((batch_images, batch_masks), _) in enumerate(dataset.take(steps_per_epoch)):
            # Train discriminator only every other batch
            if step % 2 == 0:
                d_loss = train_d_step(batch_images, batch_masks)
                d_losses.append(d_loss)
            
            # Train generator
            g_loss, recon_loss, adv_loss = train_g_step(batch_images, batch_masks)
            g_losses.append(g_loss)
            recon_losses.append(recon_loss)
            gan_losses.append(adv_loss)
            
            # Print progress
            if (step + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Step {step+1}/{steps_per_epoch}: "
                      f"D Loss: {np.mean(d_losses):.4f}, G Loss: {np.mean(g_losses):.4f}, "
                      f"Recon Loss: {np.mean(recon_losses):.4f}")
        
        # Visualize samples at the end of each epoch
        if sample_images:
            print(f"Generating sample visualizations for epoch {epoch+1}...")
            for i, (img, mask) in enumerate(zip(sample_images, sample_masks)):
                img_batch = np.expand_dims(img, axis=0)
                mask_batch = np.expand_dims(mask, axis=0)
                
                # Generate compressed image using our models
                hq_latent_val = hq_encoder.predict(img_batch)
                lq_latent_val = lq_encoder.predict(img_batch)
                hq_output = hq_generator.predict(hq_latent_val)
                lq_output = lq_generator.predict(lq_latent_val)
                
                # Blend outputs
                blended = hq_output * mask_batch + lq_output * (1 - mask_batch)
                compressed_img = blended[0]
                
                # Save visualization
                vis_path = os.path.join(SAMPLES_DIR, f"sample_{i+1}_epoch_{epoch+1}.png")
                visualize_results(img, mask[:,:,0], compressed_img, vis_path)
        
        # Calculate epoch average losses
        epoch_d_loss = np.mean([loss.numpy() for loss in d_losses]) if d_losses else float('nan')
        epoch_g_loss = np.mean([loss.numpy() for loss in g_losses]) if g_losses else float('nan')
        epoch_recon_loss = np.mean([loss.numpy() for loss in recon_losses]) if recon_losses else float('nan')
        epoch_gan_loss = np.mean([loss.numpy() for loss in gan_losses]) if gan_losses else float('nan')
        
        # Update history
        history["d_loss"].append(epoch_d_loss)
        history["g_loss"].append(epoch_g_loss)
        history["reconstruction_loss"].append(epoch_recon_loss)
        history["gan_loss"].append(epoch_gan_loss)
        
        # Print epoch summary
        time_taken = time.time() - start_time
        print(f"Epoch {epoch+1}/{epochs} completed in {time_taken:.2f}s - "
              f"D Loss: {epoch_d_loss:.4f}, G Loss: {epoch_g_loss:.4f}, "
              f"Recon Loss: {epoch_recon_loss:.4f}, GAN Loss: {epoch_gan_loss:.4f}")
        
        # Plot and save loss history
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(history["d_loss"], label="Discriminator Loss")
        plt.plot(history["g_loss"], label="Generator Loss")
        plt.legend()
        plt.title("GAN Losses")
        
        plt.subplot(2, 1, 2)
        plt.plot(history["reconstruction_loss"], label="Reconstruction Loss")
        plt.plot(history["gan_loss"], label="GAN Component Loss")
        plt.legend()
        plt.title("Generator Loss Components")
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"loss_history_epoch_{epoch+1}.png"))
        plt.close()
        
        # Save models every 5 epochs or at the end
        if (epoch + 1) % 5 == 0 or epoch + 1 == epochs:
            print(f"Saving models at epoch {epoch+1}...")
            try:
                # Save adaptive model
                adaptive_model.save(os.path.join(MODEL_DIR, f"adaptive_model_epoch_{epoch+1}.h5"))
                discriminator.save(os.path.join(MODEL_DIR, f"discriminator_epoch_{epoch+1}.h5"))
                
                # Save individual components
                hq_encoder.save(os.path.join(MODEL_DIR, f"hq_encoder_epoch_{epoch+1}.h5"))
                hq_generator.save(os.path.join(MODEL_DIR, f"hq_generator_epoch_{epoch+1}.h5"))
                lq_encoder.save(os.path.join(MODEL_DIR, f"lq_encoder_epoch_{epoch+1}.h5"))
                lq_generator.save(os.path.join(MODEL_DIR, f"lq_generator_epoch_{epoch+1}.h5"))
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
    except Exception as e:
        print(f"Error saving final models: {e}")
    
    return adaptive_model, discriminator, hq_encoder, hq_generator, lq_encoder, lq_generator

def main():
    print("Starting training process...")
    
    # Load and prepare training data
    images = load_training_data(TRAIN_DIR)
    if len(images) == 0:
        print("No images found in the training directory!")
        return
    
    images_with_masks = prepare_training_data(images)
    dataset = create_tf_dataset(images_with_masks, BATCH_SIZE)
    
    # Calculate steps per epoch
    steps_per_epoch = len(images) // BATCH_SIZE
    
    # Train the model
    history = train_gan(dataset, EPOCHS, steps_per_epoch)
    
    print("Training completed!")

if __name__ == "__main__":
    main()