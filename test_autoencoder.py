import os
import glob
import cv2
import numpy as np
import tensorflow as tf
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TensorFlow logs except errors

# Define function to load images
def load_images_from_folder(folder, target_size):
    """
    Loads and preprocesses images from a specified folder.
    - Resizes images to the target size.
    - Normalizes pixel values to [0, 1].
    """
    image_paths = glob.glob(os.path.join(folder, "*.jpg"))
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is not None:
            img = cv2.resize(img, target_size)
            img = img.astype('float32') / 255.0  # Normalize to [0,1]
            images.append((img, path))  # Store (image, file_path) tuple
    return images

# Load the trained model with the correct loss function
model_path = "autoencoder_model.h5"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Trained model not found: {model_path}")

autoencoder = tf.keras.models.load_model(model_path, compile=False)
autoencoder.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())

# Set the test dataset folder
test_dataset_folder = "./test_dataset"  # Update if your test images are in a different folder
target_size = (128, 128)  # Ensure same size as training data

# Load test images
test_images = load_images_from_folder(test_dataset_folder, target_size)
if len(test_images) == 0:
    raise ValueError(f"No images found in test dataset folder: {test_dataset_folder}")

print(f"Testing on {len(test_images)} images from {test_dataset_folder}")

# Define metrics for evaluation
def calculate_mse(image1, image2):
    return np.mean((image1 - image2) ** 2)

def calculate_psnr(image1, image2):
    if image1.dtype == np.uint8 or image2.dtype == np.uint8:
        data_range = 255
    else:
        data_range = 1.0  # Assuming normalized input
    return peak_signal_noise_ratio(image1, image2, data_range=data_range)

def calculate_ssim(image1, image2):
    if image1.dtype == np.uint8 or image2.dtype == np.uint8:
        data_range = 255
    else:
        data_range = 1.0
    image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    return structural_similarity(image1_gray, image2_gray, data_range=data_range)

def calculate_size_reduction(original_path, compressed_path):
    original_size = os.path.getsize(original_path)
    compressed_size = os.path.getsize(compressed_path)
    if original_size == 0:
        raise ValueError("Original image size is 0. Check the file path.")
    return (1 - (compressed_size / original_size)) * 100

# Initialize accumulators for average calculations
total_mse = 0
total_psnr = 0
total_ssim = 0
total_size_reduction = 0
num_images = len(test_images)

# Iterate over test images
for img, file_path in test_images:
    # Compress image using autoencoder
    compressed_img = autoencoder.predict(np.expand_dims(img, axis=0))[0]
    
    # Convert back to 0-255 range for saving
    compressed_img_uint8 = (compressed_img * 255).astype("uint8")

    # Save compressed image
    compressed_path = os.path.join("compressed_outputs", os.path.basename(file_path))
    os.makedirs("compressed_outputs", exist_ok=True)
    cv2.imwrite(compressed_path, compressed_img_uint8)

    # Load original image (uint8 format for comparison)
    original_img = (img * 255).astype("uint8")

    # Compute quantitative metrics
    mse = calculate_mse(original_img, compressed_img_uint8)
    psnr = calculate_psnr(original_img, compressed_img_uint8)
    ssim = calculate_ssim(original_img, compressed_img_uint8)
    size_reduction = calculate_size_reduction(file_path, compressed_path)

    # Accumulate totals
    total_mse += mse
    total_psnr += psnr
    total_ssim += ssim
    total_size_reduction += size_reduction

# Compute averages
avg_mse = total_mse / num_images
avg_psnr = total_psnr / num_images
avg_ssim = total_ssim / num_images
avg_size_reduction = total_size_reduction / num_images

# Print overall results
print("\n=== Overall Compression Performance ===")
print(f"  - Average Mean Squared Error (MSE): {avg_mse:.4f}")
print(f"  - Average Peak Signal-to-Noise Ratio (PSNR): {avg_psnr:.2f} dB")
print(f"  - Average Structural Similarity Index (SSIM): {avg_ssim:.4f}")
print(f"  - Average Size Reduction: {avg_size_reduction:.2f}%")

print("\nTesting complete. Compressed images saved in 'compressed_outputs' folder.")
