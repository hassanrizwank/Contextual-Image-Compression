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
from GAN_functions import (
    load_and_preprocess_image, compute_saliency_map, create_saliency_mask,
    compute_metrics, estimate_compression_ratio, visualize_results,
    save_image, create_directories
)

# Configuration
IMG_SIZE = (256, 256)
IMG_SHAPE = (IMG_SIZE[0], IMG_SIZE[1], 3)
BASE_LATENT_DIM = 512  # Should match training config
HQ_LATENT_DIM = BASE_LATENT_DIM * 4

# Directories
TEST_DIR = "test_dataset"
RESULTS_DIR = "test_results"
MODEL_DIR = "models"
OUTPUT_DIR = os.path.join(RESULTS_DIR, "compressed_images")
VISUALIZATION_DIR = os.path.join(RESULTS_DIR, "visualizations")
create_directories([RESULTS_DIR, OUTPUT_DIR, VISUALIZATION_DIR])

def load_models(model_dir):
    """Load the trained models."""
    print(f"Loading models from {model_dir}...")
    
    # Load individual components
    try:
        hq_encoder = keras.models.load_model(os.path.join(model_dir, "hq_encoder_final.h5"))
        hq_generator = keras.models.load_model(os.path.join(model_dir, "hq_generator_final.h5"))
        lq_encoder = keras.models.load_model(os.path.join(model_dir, "lq_encoder_final.h5"))
        lq_generator = keras.models.load_model(os.path.join(model_dir, "lq_generator_final.h5"))
        
        return hq_encoder, hq_generator, lq_encoder, lq_generator
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Trying to load from the last saved epoch...")
        
        # Find the latest epoch
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
            
            hq_encoder = keras.models.load_model(
                os.path.join(model_dir, f"hq_encoder_epoch_{latest_epoch}.h5")
            )
            hq_generator = keras.models.load_model(
                os.path.join(model_dir, f"hq_generator_epoch_{latest_epoch}.h5")
            )
            lq_encoder = keras.models.load_model(
                os.path.join(model_dir, f"lq_encoder_epoch_{latest_epoch}.h5")
            )
            lq_generator = keras.models.load_model(
                os.path.join(model_dir, f"lq_generator_epoch_{latest_epoch}.h5")
            )
            
            return hq_encoder, hq_generator, lq_encoder, lq_generator
        else:
            raise ValueError("No models found! Please train the models first.")

def load_test_images(test_dir):
    """Load test images."""
    print(f"Loading test images from {test_dir}...")
    image_paths = glob.glob(os.path.join(test_dir, "*.jpg")) + \
                 glob.glob(os.path.join(test_dir, "*.png")) + \
                 glob.glob(os.path.join(test_dir, "*.jpeg"))
    
    print(f"Found {len(image_paths)} test images.")
    
    test_images = []
    file_names = []
    original_sizes = []
    
    for path in image_paths:
        try:
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

def compress_and_reconstruct(img, hq_encoder, hq_generator, lq_encoder, lq_generator):
    """
    Compress and reconstruct an image using the adaptive compression model.
    """
    # Generate saliency map
    saliency_map = compute_saliency_map(img)
    mask = create_saliency_mask(saliency_map, smooth=True)
    
    # Create expanded mask for blending

    enhanced_mask = np.power(mask, 0.7)
    expanded_mask = np.expand_dims(enhanced_mask, axis=-1)
    expanded_mask = np.expand_dims(expanded_mask, axis=0)  # Add batch dimension
    
    # Add batch dimension to image
    img_batch = np.expand_dims(img, axis=0)
    
    # Encode with both encoders
    hq_latent = hq_encoder.predict(img_batch)
    lq_latent = lq_encoder.predict(img_batch)
    
    # Decode with both generators
    hq_output = hq_generator.predict(hq_latent)
    lq_output = lq_generator.predict(lq_latent)
    
    # Blend outputs based on saliency
    blended_output = hq_output * expanded_mask + lq_output * (1 - expanded_mask)
    
    # Return all outputs and the saliency map for analysis
    return {
        'saliency_map': mask,
        'hq_latent': hq_latent[0],  # Remove batch dimension
        'lq_latent': lq_latent[0],  # Remove batch dimension
        'hq_output': hq_output[0],  # Remove batch dimension
        'lq_output': lq_output[0],  # Remove batch dimension
        'blended_output': blended_output[0],  # Remove batch dimension
    }

def test_compression(test_images, file_names, original_sizes, hq_encoder, hq_generator, lq_encoder, lq_generator):
    """
    Test compression on all test images and compute metrics.
    """
    print("Testing compression on all images...")
    
    results = {
        'psnr': [],
        'ssim': [],
        'mse': [],
        'compression_ratio': [],
        'percentage_reduction': []
    }
    
    for i, (img, file_name, original_size) in enumerate(zip(test_images, file_names, original_sizes)):
        print(f"Processing image {i+1}/{len(test_images)}: {file_name}")
        
        # Compress and reconstruct
        outputs = compress_and_reconstruct(img, hq_encoder, hq_generator, lq_encoder, lq_generator)
        
        # Compute image quality metrics
        metrics = compute_metrics(img, outputs['blended_output'])
        
        # Calculate latent space sizes        
        mask_sum = np.sum(outputs['saliency_map']) / (outputs['saliency_map'].shape[0] * outputs['saliency_map'].shape[1])
        hq_size = HQ_LATENT_DIM * mask_sum * 4  # 4 bytes per float32
        lq_size = BASE_LATENT_DIM * (1 - mask_sum) * 4  # 4 bytes per float32
        total_latent_size = hq_size + lq_size
        
        # Estimate compression ratio
        compression_ratio, percentage_reduction = estimate_compression_ratio(
            original_size, total_latent_size
        )
        
        # Save results
        results['psnr'].append(metrics['psnr'])
        results['ssim'].append(metrics['ssim'])
        results['mse'].append(metrics['mse'])
        results['compression_ratio'].append(compression_ratio)
        results['percentage_reduction'].append(percentage_reduction)
        
        # Save compressed image
        output_path = os.path.join(OUTPUT_DIR, f"compressed_{file_name}")
        save_image(outputs['blended_output'], output_path)
        
        # Visualize and save visualization
        vis_path = os.path.join(VISUALIZATION_DIR, f"visualization_{file_name.split('.')[0]}.png")
        visualize_results(img, outputs['saliency_map'], outputs['blended_output'], vis_path)
        
        print(f"  PSNR: {metrics['psnr']:.2f}, SSIM: {metrics['ssim']:.4f}, MSE: {metrics['mse']:.6f}")
        print(f"  Compression ratio: {compression_ratio:.2f}x, Reduction: {percentage_reduction:.2f}%")
    
    # Calculate average metrics
    avg_results = {
        'psnr': np.mean(results['psnr']),
        'ssim': np.mean(results['ssim']),
        'mse': np.mean(results['mse']),
        'compression_ratio': np.mean(results['compression_ratio']),
        'percentage_reduction': np.mean(results['percentage_reduction'])
    }
    
    print("\nAverage metrics across all test images:")
    print(f"  PSNR: {avg_results['psnr']:.2f} dB")
    print(f"  SSIM: {avg_results['ssim']:.4f}")
    print(f"  MSE: {avg_results['mse']:.6f}")
    print(f"  Compression ratio: {avg_results['compression_ratio']:.2f}x")
    print(f"  Size reduction: {avg_results['percentage_reduction']:.2f}%")
    
    # Save metrics to file
    with open(os.path.join(RESULTS_DIR, "metrics.txt"), "w") as f:
        f.write("Image-by-image metrics:\n")
        for i, file_name in enumerate(file_names):
            f.write(f"\n{file_name}:\n")
            f.write(f"  PSNR: {results['psnr'][i]:.2f} dB\n")
            f.write(f"  SSIM: {results['ssim'][i]:.4f}\n")
            f.write(f"  MSE: {results['mse'][i]:.6f}\n")
            f.write(f"  Compression ratio: {results['compression_ratio'][i]:.2f}x\n")
            f.write(f"  Size reduction: {results['percentage_reduction'][i]:.2f}%\n")
        
        f.write("\nAverage metrics:\n")
        f.write(f"  PSNR: {avg_results['psnr']:.2f} dB\n")
        f.write(f"  SSIM: {avg_results['ssim']:.4f}\n")
        f.write(f"  MSE: {avg_results['mse']:.6f}\n")
        f.write(f"  Compression ratio: {avg_results['compression_ratio']:.2f}x\n")
        f.write(f"  Size reduction: {avg_results['percentage_reduction']:.2f}%\n")
    
    return avg_results

def plot_metrics(results, file_names):
    """Plot metrics for each test image."""
    metrics = ['psnr', 'ssim', 'mse', 'compression_ratio', 'percentage_reduction']
    titles = ['PSNR (dB)', 'SSIM', 'MSE', 'Compression Ratio', 'Size Reduction (%)']
    
    for metric, title in zip(metrics, titles):
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(file_names)), results[metric])
        plt.xticks(range(len(file_names)), file_names, rotation=45, ha='right')
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"{metric}_plot.png"))
        plt.close()

def main():
    print("Starting testing process...")
    
    # Load models
    hq_encoder, hq_generator, lq_encoder, lq_generator = load_models(MODEL_DIR)
    
    # Load test images
    test_images, file_names, original_sizes = load_test_images(TEST_DIR)
    
    if len(test_images) == 0:
        print("No test images found!")
        return
    
    # Test compression and get results
    results = test_compression(test_images, file_names, original_sizes, 
                             hq_encoder, hq_generator, lq_encoder, lq_generator)
    
    print("Testing completed!")
    print("\nFinal results:")
    print(f"  Average PSNR: {results['psnr']:.2f} dB")
    print(f"  Average SSIM: {results['ssim']:.4f}")
    print(f"  Average MSE: {results['mse']:.6f}")
    print(f"  Average compression ratio: {results['compression_ratio']:.2f}x")
    print(f"  Average size reduction: {results['percentage_reduction']:.2f}%")
    
    print(f"\nCompressed images saved to: {OUTPUT_DIR}")
    print(f"Visualizations saved to: {VISUALIZATION_DIR}")
    print(f"Detailed metrics saved to: {os.path.join(RESULTS_DIR, 'metrics.txt')}")

if __name__ == "__main__":
    main()