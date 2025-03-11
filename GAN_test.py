import os
# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from tensorflow import keras
import glob
import time
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import csv
from GAN_functions import (
    load_and_preprocess_image, compute_saliency_map, create_saliency_mask,
    compute_metrics, estimate_compression_ratio, visualize_results,
    save_image, create_directories, build_latent_saliency_model,
    build_rate_distortion_optimizer, build_encoder, build_generator, visualize_bit_allocation_by_bpp, 
    SelfAttention, AdaptiveQuantizationLayer
)

# Configuration
IMG_SIZE = (256, 256)
IMG_SHAPE = (IMG_SIZE[0], IMG_SIZE[1], 3)
BASE_LATENT_DIM = 512  # Should match training config
HQ_LATENT_DIM = BASE_LATENT_DIM * 2  # Match high-quality latent dimension

# Target bitrates for testing
BPP_VALUES = [0.1, 1.0, 2.0]

# Directories
TEST_DIR = "test_dataset"
RESULTS_DIR = "test_results"
MODEL_DIR = "models"
create_directories([RESULTS_DIR])

def load_models(model_dir):
    """Load the trained models."""
    print(f"Loading models from {model_dir}...")
    
    # Load individual components
    try:
        # Try to load the full adaptive model first
        adaptive_model = keras.models.load_model(
            os.path.join(model_dir, "adaptive_model_final.h5"),
            custom_objects={
                'SelfAttention': SelfAttention,
                'AdaptiveQuantizationLayer': AdaptiveQuantizationLayer
            }
        )
        
        # Load component models
        hq_encoder = keras.models.load_model(
            os.path.join(model_dir, "hq_encoder_final.h5"),
            custom_objects={'SelfAttention': SelfAttention}
        )
        hq_generator = keras.models.load_model(os.path.join(model_dir, "hq_generator_final.h5"))
        lq_encoder = keras.models.load_model(
            os.path.join(model_dir, "lq_encoder_final.h5"),
            custom_objects={'SelfAttention': SelfAttention}
        )
        lq_generator = keras.models.load_model(os.path.join(model_dir, "lq_generator_final.h5"))
        
        # Load the new components
        latent_saliency_hq = keras.models.load_model(os.path.join(model_dir, "latent_saliency_hq_final.h5"))
        latent_saliency_lq = keras.models.load_model(os.path.join(model_dir, "latent_saliency_lq_final.h5"))
        rd_optimizer = keras.models.load_model(os.path.join(model_dir, "rd_optimizer_final.h5"))
        
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
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Trying to load from the last saved epoch...")
        
        # Find the latest epoch by checking hq_encoder files
        saved_epochs = []
        for file in os.listdir(model_dir):
            if file.startswith("hq_encoder_epoch_") and file.endswith(".h5"):
                try:
                    epoch = int(file.split("_")[-1].split(".")[0])
                    saved_epochs.append(epoch)
                except:
                    pass
        
        if saved_epochs:
            latest_epoch = max(saved_epochs)
            print(f"Loading models from epoch {latest_epoch}...")
            
            # Load each component individually
            hq_encoder = keras.models.load_model(
                os.path.join(model_dir, f"hq_encoder_epoch_{latest_epoch}.h5"),
                custom_objects={'SelfAttention': SelfAttention}
            )
            hq_generator = keras.models.load_model(
                os.path.join(model_dir, f"hq_generator_epoch_{latest_epoch}.h5")
            )
            lq_encoder = keras.models.load_model(
                os.path.join(model_dir, f"lq_encoder_epoch_{latest_epoch}.h5"),
                custom_objects={'SelfAttention': SelfAttention}
            )
            lq_generator = keras.models.load_model(
                os.path.join(model_dir, f"lq_generator_epoch_{latest_epoch}.h5")
            )
            
            # Load other components if available
            try:
                latent_saliency_hq = keras.models.load_model(
                    os.path.join(model_dir, f"latent_saliency_hq_epoch_{latest_epoch}.h5")
                )
                latent_saliency_lq = keras.models.load_model(
                    os.path.join(model_dir, f"latent_saliency_lq_epoch_{latest_epoch}.h5")
                )
                rd_optimizer = keras.models.load_model(
                    os.path.join(model_dir, f"rd_optimizer_epoch_{latest_epoch}.h5")
                )
            except Exception as e:
                print(f"Could not load all advanced components: {e}")
                # If we can't load the advanced components, rebuild them
                latent_saliency_hq = build_latent_saliency_model(BASE_LATENT_DIM * 2, name="hq_latent_saliency")
                latent_saliency_lq = build_latent_saliency_model(BASE_LATENT_DIM, name="lq_latent_saliency")
                rd_optimizer = build_rate_distortion_optimizer(
                    IMG_SHAPE, {'hq': BASE_LATENT_DIM * 2, 'lq': BASE_LATENT_DIM}, name="rd_optimizer"
                )
            
            # Rebuild the adaptive model
            try:
                adaptive_model = keras.models.load_model(
                    os.path.join(model_dir, f"adaptive_model_epoch_{latest_epoch}.h5"),
                    custom_objects={
                        'SelfAttention': SelfAttention,
                        'AdaptiveQuantizationLayer': AdaptiveQuantizationLayer
                    }
                )
            except Exception as e:
                print(f"Could not load adaptive model: {e}")
                print("Creating a new one from components.")
                # Rebuild from components
                from tensorflow.keras import layers # type:ignore 
                
                # Input layers
                img_input = keras.Input(shape=IMG_SHAPE, name="image_input")
                saliency_input = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 1), name="saliency_input")
                target_bpp_input = keras.Input(shape=(1,), name="target_bpp_input")
                
                # Forward pass
                hq_encoder_outputs = hq_encoder(img_input)
                lq_encoder_outputs = lq_encoder(img_input)
                
                # Extract outputs
                hq_latent = hq_encoder_outputs[0]
                hq_skip1 = hq_encoder_outputs[1]
                hq_skip2 = hq_encoder_outputs[2]
                hq_skip3 = hq_encoder_outputs[3]
                
                lq_latent = lq_encoder_outputs[0]
                lq_skip1 = lq_encoder_outputs[1]
                lq_skip2 = lq_encoder_outputs[2]
                lq_skip3 = lq_encoder_outputs[3]
                
                # Get rate-distortion parameters
                rd_params = rd_optimizer([img_input, saliency_input, target_bpp_input])
                
                # Extract RD parameters
                overall_compression = layers.Lambda(lambda x: x[:, 0:1])(rd_params)
                hq_lq_threshold = layers.Lambda(lambda x: x[:, 1:2])(rd_params)
                quant_strength = layers.Lambda(lambda x: x[:, 2:3])(rd_params)
                
                # Compute latent saliency scores
                hq_latent_saliency = latent_saliency_hq(hq_latent)
                lq_latent_saliency = latent_saliency_lq(lq_latent)
                
                # Apply adaptive quantization using layer
                adaptive_quantize_layer = AdaptiveQuantizationLayer()
                hq_latent_quantized = adaptive_quantize_layer([hq_latent, hq_latent_saliency, quant_strength])
                lq_latent_quantized = adaptive_quantize_layer([lq_latent, lq_latent_saliency, quant_strength])
                
                # Generate outputs
                hq_output = hq_generator([hq_latent_quantized, hq_skip1, hq_skip2, hq_skip3])
                lq_output = lq_generator([lq_latent_quantized, lq_skip1, lq_skip2, lq_skip3])
                
                # Create enhanced saliency mask
                enhanced_saliency = layers.Lambda(lambda x: tf.pow(x, 0.7))(saliency_input)
                
                # Create a soft HQ/LQ boundary based on the RD threshold - with broadcast
                dynamic_threshold = layers.Lambda(
                    lambda x: tf.cast(x[0] > tf.expand_dims(tf.expand_dims(x[1], 1), 1), tf.float32)
                )([enhanced_saliency, hq_lq_threshold])
                
                # Blend outputs
                weighted_hq = layers.Multiply()([hq_output, dynamic_threshold])
                weighted_lq = layers.Multiply()([lq_output, 1.0 - dynamic_threshold])
                blended_output = layers.Add()([weighted_hq, weighted_lq])
                
                # Create adaptive model
                adaptive_model = keras.Model(
                    inputs=[img_input, saliency_input, target_bpp_input],
                    outputs=[blended_output, hq_latent_quantized, lq_latent_quantized, rd_params, dynamic_threshold],
                    name="adaptive_compression_model"
                )
            
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
        else:
            raise ValueError("No models found! Please train the models first.")

def load_test_images(test_dir):
    """Load test images."""
    print(f"Looking for test images in: {test_dir}")
    if not os.path.exists(test_dir):
        print(f"Warning: Test directory {test_dir} does not exist!")
        # Create the directory instead of failing
        create_directories([test_dir])
        print(f"Created {test_dir}. Please add test images and run again.")
        return [], [], []
        
    image_paths = glob.glob(os.path.join(test_dir, "*.jpg")) + \
                 glob.glob(os.path.join(test_dir, "*.png")) + \
                 glob.glob(os.path.join(test_dir, "*.jpeg"))
    
    print(f"Found {len(image_paths)} test images.")
    
    if len(image_paths) == 0:
        print(f"No images found in {test_dir}. Please add test images.")
        return [], [], []
    
    test_images = []
    file_names = []
    original_sizes = []
    
    for path in image_paths:
        try:
            print(f"Loading image: {path}")
            # Get original file size
            original_size = os.path.getsize(path)
            original_sizes.append(original_size)
            
            # Load and preprocess image
            img = load_and_preprocess_image(path, IMG_SIZE)
            test_images.append(img)
            
            # Get file name
            file_name = os.path.basename(path)
            file_names.append(file_name)
        except Exception as e:
            print(f"Error loading {path}: {e}")
    
    return test_images, file_names, original_sizes

def compress_and_reconstruct(img, models, target_bpp=1.0):
    """
    Compress and reconstruct an image using the adaptive model.
    
    Args:
        img: Input image (normalized to [-1, 1])
        models: Dictionary of models
        target_bpp: Target bits per pixel
        
    Returns:
        Dictionary with compression outputs and metrics
    """
    # Generate saliency map
    print(f"Computing saliency map for image with target BPP={target_bpp}")
    saliency_map = compute_saliency_map(img, method='combined')
    mask = create_saliency_mask(saliency_map, smooth=True)
    
    # Convert to batch format
    img_batch = np.expand_dims(img, axis=0)
    mask_batch = np.expand_dims(np.expand_dims(mask, axis=-1), axis=0)
    target_bpp_batch = np.array([[target_bpp]], dtype=np.float32)
    
    # Get the adaptive model
    adaptive_model = models['adaptive_model']
    
    # Run compression
    print("Running compression through adaptive model...")
    compressed_output, hq_latent, lq_latent, rd_params, bit_allocation = adaptive_model.predict(
        [img_batch, mask_batch, target_bpp_batch],
        verbose=0
    )
    
    # Calculate bit allocation metrics
    # Extract first element from each batch
    compressed_img = compressed_output[0]
    hq_latent_val = hq_latent[0]
    lq_latent_val = lq_latent[0]
    rd_params_val = rd_params[0]
    bit_alloc_map = bit_allocation[0]
    
    # Calculate metrics
    quality_metrics = compute_metrics(img, compressed_img)
    
    # Estimate compression ratio based on latent sizes and saliency
    # Calculate proportion of image using HQ vs LQ encoding
    hq_ratio = np.mean(bit_alloc_map)
    lq_ratio = 1.0 - hq_ratio
    
    # Calculate effective compression
    hq_bits = hq_ratio * HQ_LATENT_DIM * 32  # 32 bits per float
    lq_bits = lq_ratio * BASE_LATENT_DIM * 32
    total_bits = hq_bits + lq_bits
    
    # Original image size in bits (3 channels, 8 bits per channel)
    original_bits = IMG_SIZE[0] * IMG_SIZE[1] * 3 * 8
    
    # Compression ratio
    compression_ratio = original_bits / total_bits
    
    # Actual bits per pixel
    actual_bpp = total_bits / (IMG_SIZE[0] * IMG_SIZE[1])
    
    return {
        'saliency_map': mask,
        'compressed_img': compressed_img,
        'hq_latent': hq_latent_val,
        'lq_latent': lq_latent_val,
        'rd_params': rd_params_val,
        'bit_allocation': bit_alloc_map,
        'metrics': quality_metrics,
        'compression_ratio': compression_ratio,
        'actual_bpp': actual_bpp,
        'target_bpp': target_bpp,
        'hq_ratio': hq_ratio,
        'lq_ratio': lq_ratio
    }

def test_compression(test_images, file_names, original_sizes, models):
    """
    Test compression on all test images at different bitrates.
    
    Args:
        test_images: List of preprocessed test images
        file_names: List of file names corresponding to test images
        original_sizes: List of original file sizes in bytes
        models: Dictionary of models
        
    Returns:
        Dictionary with metrics for each target bitrate
    """
    print("Testing compression at different bitrates...")
    
    # Create subdirectories for each bitrate
    results_by_bpp = {}
    for bpp in BPP_VALUES:
        bpp_dir = os.path.join(RESULTS_DIR, f"bpp_{bpp}")
        output_dir = os.path.join(bpp_dir, "compressed")
        vis_dir = os.path.join(bpp_dir, "visualizations")
        create_directories([bpp_dir, output_dir, vis_dir])
        
        results_by_bpp[bpp] = {
            'psnr': [],
            'ssim': [],
            'mse': [],
            'compression_ratio': [],
            'actual_bpp': [],
            'hq_ratio': []
        }
    
    # Test on each image
    for i, (img, file_name, original_size) in enumerate(zip(test_images, file_names, original_sizes)):
        print(f"\nProcessing image {i+1}/{len(test_images)}: {file_name}")
        
        # Test at different bitrates
        for bpp in BPP_VALUES:
            print(f"  Target bitrate: {bpp} bpp")
            bpp_dir = os.path.join(RESULTS_DIR, f"bpp_{bpp}")
            output_dir = os.path.join(bpp_dir, "compressed")
            vis_dir = os.path.join(bpp_dir, "visualizations")
            
            # Compress and reconstruct
            result = compress_and_reconstruct(img, models, target_bpp=bpp)
            
            # Save compressed image
            output_path = os.path.join(output_dir, file_name)
            save_image(result['compressed_img'], output_path)
            
            # Save visualization
            vis_name = f"{os.path.splitext(file_name)[0]}_vis.png"
            vis_path = os.path.join(vis_dir, vis_name)
            visualize_results(
                img, 
                result['saliency_map'], 
                result['compressed_img'], 
                vis_path,
                result['bit_allocation']
            )
            
            # Store metrics
            results_by_bpp[bpp]['psnr'].append(result['metrics']['psnr'])
            results_by_bpp[bpp]['ssim'].append(result['metrics']['ssim'])
            results_by_bpp[bpp]['mse'].append(result['metrics']['mse'])
            results_by_bpp[bpp]['compression_ratio'].append(result['compression_ratio'])
            results_by_bpp[bpp]['actual_bpp'].append(result['actual_bpp'])
            results_by_bpp[bpp]['hq_ratio'].append(result['hq_ratio'])
            
            # Print metrics
            print(f"    PSNR: {result['metrics']['psnr']:.2f} dB, "
                  f"SSIM: {result['metrics']['ssim']:.4f}, "
                  f"BPP: {result['actual_bpp']:.4f} (target: {bpp}), "
                  f"HQ ratio: {result['hq_ratio']*100:.2f}%")
    
    # Calculate and save average metrics for each bitrate
    avg_metrics = {}
    for bpp, results in results_by_bpp.items():
        avg_metrics[bpp] = {
            'psnr': np.mean(results['psnr']) if results['psnr'] else 0,
            'ssim': np.mean(results['ssim']) if results['ssim'] else 0,
            'mse': np.mean(results['mse']) if results['mse'] else 0,
            'compression_ratio': np.mean(results['compression_ratio']) if results['compression_ratio'] else 0,
            'actual_bpp': np.mean(results['actual_bpp']) if results['actual_bpp'] else 0,
            'hq_ratio': np.mean(results['hq_ratio']) if results['hq_ratio'] else 0,
        }
        
        # Save metrics to file
        metrics_path = os.path.join(RESULTS_DIR, f"bpp_{bpp}", "metrics.txt")
        print(f"Saving metrics to {metrics_path}")
        with open(metrics_path, "w") as f:
            f.write(f"Target BPP: {bpp}\n\n")
            f.write(f"Average PSNR: {avg_metrics[bpp]['psnr']:.2f} dB\n")
            f.write(f"Average SSIM: {avg_metrics[bpp]['ssim']:.4f}\n")
            f.write(f"Average MSE: {avg_metrics[bpp]['mse']:.6f}\n")
            f.write(f"Average Compression Ratio: {avg_metrics[bpp]['compression_ratio']:.2f}x\n")
            f.write(f"Average Actual BPP: {avg_metrics[bpp]['actual_bpp']:.4f}\n")
            f.write(f"Average HQ Region Ratio: {avg_metrics[bpp]['hq_ratio']*100:.2f}%\n\n")
            
            f.write("Image-by-image metrics:\n")
            for i, file_name in enumerate(file_names):
                f.write(f"\n{file_name}:\n")
                f.write(f"  PSNR: {results['psnr'][i]:.2f} dB\n")
                f.write(f"  SSIM: {results['ssim'][i]:.4f}\n")
                f.write(f"  MSE: {results['mse'][i]:.6f}\n")
                f.write(f"  Compression Ratio: {results['compression_ratio'][i]:.2f}x\n")
                f.write(f"  Actual BPP: {results['actual_bpp'][i]:.4f}\n")
                f.write(f"  HQ Region Ratio: {results['hq_ratio'][i]*100:.2f}%\n")
    
    return {
        'by_bpp': results_by_bpp,
        'avg_metrics': avg_metrics
    }

def plot_rate_distortion_curve(avg_metrics):
    """
    Plot rate-distortion curves based on test results.
    
    Args:
        avg_metrics: Dictionary with average metrics for each bitrate
    """
    # Sort bitrates
    bitrates = sorted(avg_metrics.keys())
    
    # Extract data for plotting
    psnrs = [avg_metrics[bpp]['psnr'] for bpp in bitrates]
    ssims = [avg_metrics[bpp]['ssim'] for bpp in bitrates]
    actual_bpps = [avg_metrics[bpp]['actual_bpp'] for bpp in bitrates]
    hq_ratios = [avg_metrics[bpp]['hq_ratio'] * 100 for bpp in bitrates]
    
    print("Generating rate-distortion plots...")
    
    # Create subplots
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Rate-Distortion Curve (PSNR)
    plt.subplot(2, 2, 1)
    plt.plot(actual_bpps, psnrs, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Bits per Pixel (BPP)')
    plt.ylabel('PSNR (dB)')
    plt.title('Rate-Distortion Curve (PSNR)')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Rate-Distortion Curve (SSIM)
    plt.subplot(2, 2, 2)
    plt.plot(actual_bpps, ssims, 'ro-', linewidth=2, markersize=8)
    plt.xlabel('Bits per Pixel (BPP)')
    plt.ylabel('SSIM')
    plt.title('Rate-Distortion Curve (SSIM)')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Target vs Actual BPP
    plt.subplot(2, 2, 3)
    plt.plot(bitrates, actual_bpps, 'go-', linewidth=2, markersize=8)
    plt.plot(bitrates, bitrates, 'k--', alpha=0.5)  # Identity line
    plt.xlabel('Target BPP')
    plt.ylabel('Actual BPP')
    plt.title('Bitrate Control Accuracy')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: HQ Region Ratio vs BPP
    plt.subplot(2, 2, 4)
    plt.plot(actual_bpps, hq_ratios, 'mo-', linewidth=2, markersize=8)
    plt.xlabel('Bits per Pixel (BPP)')
    plt.ylabel('HQ Region Ratio (%)')
    plt.title('HQ/LQ Region Allocation')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, 'rate_distortion_curves.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Rate-distortion plots saved to {plot_path}")
    
    # Also save the data as CSV for further analysis
    csv_path = os.path.join(RESULTS_DIR, 'rate_distortion_data.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Target BPP', 'Actual BPP', 'PSNR', 'SSIM', 'HQ Ratio (%)'])
        for i, bpp in enumerate(bitrates):
            writer.writerow([
                bpp, 
                actual_bpps[i], 
                psnrs[i], 
                ssims[i], 
                hq_ratios[i]
            ])
    
    print(f"Rate-distortion data saved to {csv_path}")

def test_rate_control(models, test_images, file_names):
    """Test rate control across multiple images and target bitrates."""
    test_bpps = np.linspace(0.1, 2.0, 10)  # 10 points between 0.1 and 2.0
    adaptive_model = models['adaptive_model']
    
    results = {
        'target_bpp': [],
        'actual_bpp': [],
        'hq_ratio': [],
        'image': []
    }
    
    # Create directory for detailed rate control tests
    rate_control_dir = os.path.join(RESULTS_DIR, "rate_control_test")
    create_directories([rate_control_dir])
    
    print("\nRunning detailed rate control tests...")
    
    for img_idx, (img, file_name) in enumerate(zip(test_images[:4], file_names[:4])):  # Test on first 4 images
        # Generate saliency map
        saliency_map = compute_saliency_map(img, method='combined')
        mask = create_saliency_mask(saliency_map, smooth=True)
        
        # Convert to batch format
        img_batch = np.expand_dims(img, axis=0)
        mask_batch = np.expand_dims(np.expand_dims(mask, axis=-1), axis=0)
        
        print(f"Testing rate control on image: {file_name}")
        
        # Create visualization for this image
        vis_path = os.path.join(rate_control_dir, f"{os.path.splitext(file_name)[0]}_bit_allocation.png")
        visualize_bit_allocation_by_bpp(img, mask, adaptive_model, save_path=vis_path)
        
        for bpp in test_bpps:
            bpp_tensor = np.array([[bpp]], dtype=np.float32)
            _, hq_latent, lq_latent, _, bit_allocation = adaptive_model.predict(
                [img_batch, mask_batch, bpp_tensor], 
                verbose=0
            )
            
            # Calculate metrics
            hq_ratio = np.mean(bit_allocation)
            lq_ratio = 1.0 - hq_ratio
            
            # Calculate bits based on latent dimensions
            hq_bits = hq_ratio * (BASE_LATENT_DIM * 2) * 32  # 32 bits per float
            lq_bits = lq_ratio * BASE_LATENT_DIM * 32
            total_bits = hq_bits + lq_bits
            
            # Calculate actual BPP
            actual_bpp = total_bits / (IMG_SIZE[0] * IMG_SIZE[1])
            
            results['target_bpp'].append(bpp)
            results['actual_bpp'].append(actual_bpp)
            results['hq_ratio'].append(hq_ratio)
            results['image'].append(file_name)
    
    # Plot results
    plt.figure(figsize=(12, 6))
    
    # Group by image for plotting with different colors
    unique_images = list(set(results['image']))
    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
    
    for i, img_name in enumerate(unique_images):
        idx = [j for j, x in enumerate(results['image']) if x == img_name]
        plt.scatter(
            [results['target_bpp'][j] for j in idx],
            [results['actual_bpp'][j] for j in idx],
            color=colors[i % len(colors)],
            alpha=0.7,
            label=img_name
        )
    
    plt.plot([0, 2], [0, 2], 'k--')  # Identity line
    plt.xlabel('Target BPP')
    plt.ylabel('Actual BPP')
    plt.title('Rate Control Accuracy')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    rate_curve_path = os.path.join(rate_control_dir, 'rate_control_accuracy.png')
    plt.savefig(rate_curve_path)
    plt.close()
    
    # Also plot HQ ratio
    plt.figure(figsize=(12, 6))
    
    for i, img_name in enumerate(unique_images):
        idx = [j for j, x in enumerate(results['image']) if x == img_name]
        plt.scatter(
            [results['target_bpp'][j] for j in idx],
            [results['hq_ratio'][j] for j in idx],
            color=colors[i % len(colors)],
            alpha=0.7,
            label=img_name
        )
    
    plt.xlabel('Target BPP')
    plt.ylabel('HQ Region Ratio')
    plt.title('Bit Allocation vs. Target BPP')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    hq_ratio_path = os.path.join(rate_control_dir, 'hq_ratio_by_bpp.png')
    plt.savefig(hq_ratio_path)
    plt.close()
    
    print(f"Rate control test results saved to {rate_control_dir}")
    return {
        'target_bpp': results['target_bpp'],
        'actual_bpp': results['actual_bpp'],
        'hq_ratio': results['hq_ratio']
    }

def main():
    print("\n===== Starting GAN-based Content-Adaptive Image Compression Testing =====\n")
    
    # Check if model directory exists
    if not os.path.exists(MODEL_DIR):
        print(f"Warning: Model directory {MODEL_DIR} does not exist. Creating it.")
        create_directories([MODEL_DIR])
        print(f"Please train the model first or place model files in {MODEL_DIR}")
        return
    
    # Load models
    print("Loading compression models...")
    try:
        models = load_models(MODEL_DIR)
        print("Models loaded successfully!")
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Please train the model first or check model files.")
        return
    
    # Load test images
    test_images, file_names, original_sizes = load_test_images(TEST_DIR)
    
    if len(test_images) == 0:
        print("\nNo test images found! Please add test images to the test_dataset directory.")
        return
    
    print(f"\nSuccessfully loaded {len(test_images)} test images.")
    
    # Test compression at different bitrates
    try:
        results = test_compression(test_images, file_names, original_sizes, models)
        rate_results = test_rate_control(models, test_images, file_names)

        # Plot rate-distortion curves
        if results['avg_metrics']:
            plot_rate_distortion_curve(results['avg_metrics'])
        
        print("\nTesting complete! Results saved to:")
        for bpp in BPP_VALUES:
            print(f"  - {os.path.join(RESULTS_DIR, f'bpp_{bpp}')}")
        print(f"  - {os.path.join(RESULTS_DIR, 'rate_distortion_curves.png')}")
        print(f"  - {os.path.join(RESULTS_DIR, 'rate_distortion_data.csv')}")
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
