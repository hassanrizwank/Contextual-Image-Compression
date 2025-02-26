import os
import glob
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.models import Model

def build_autoencoder(input_shape):
    """Builds an autoencoder model for image compression with skip connections."""
    input_img = Input(shape=input_shape)

    # Encoder
    x1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x1 = MaxPooling2D((2, 2), padding='same')(x1)  # Downsample 1
    
    x2 = Conv2D(64, (3, 3), activation='relu', padding='same')(x1)
    encoded = MaxPooling2D((2, 2), padding='same')(x2)  # Downsample 2

    # Decoder
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)  # Upsample 1
    x = concatenate([x, x2])  # Skip connection from encoder layer

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)  # Upsample 2
    x = concatenate([x, x1])  # Skip connection from encoder layer

    decoded = Conv2D(input_shape[2], (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

def load_images_from_folder(folder, target_size):
    """
    Loads and preprocesses images from a specified folder.
    - Resizes images to the target size.
    - Normalizes pixel values to the range [0, 1].
    """
    image_paths = glob.glob(os.path.join(folder, "*.jpg"))
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is not None:
            img = cv2.resize(img, target_size)
            img = img.astype('float32') / 255.0  # Normalize to [0,1]
            images.append(img)
    return np.array(images)

def main():
    # Define the dataset folder and target image size (width, height)
    dataset_folder = "./dataset"  # Ensure this folder exists and contains .jpg images
    target_size = (128, 128)  # Change dimensions if needed

    # Load and preprocess the dataset
    images = load_images_from_folder(dataset_folder, target_size)
    if images.size == 0:
        print("No images found in the dataset folder:", dataset_folder)
        return
    print(f"Loaded {len(images)} images from {dataset_folder}")

    # Build the autoencoder model based on the input image shape
    input_shape = images[0].shape  # e.g., (128, 128, 3)
    autoencoder = build_autoencoder(input_shape)
    autoencoder.summary()

    # Train the autoencoder
    history = autoencoder.fit(
        images, images,
        epochs=50,         # Increase or decrease as needed
        batch_size=16,     # Adjust batch size based on available memory
        shuffle=True,
        validation_split=0.1
    )

    # Save the trained model for later use (e.g., in a test script)
    model_save_path = "autoencoder_model.h5"
    autoencoder.save(model_save_path)
    print("Model saved to:", model_save_path)

if __name__ == "__main__":
    main()
