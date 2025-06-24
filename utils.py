import cv2
import numpy as np

def preprocess_signature(image_path, image_size=(224, 224)):
    """
    Loads and preprocesses a signature image. Assumes the image is already
    grayscale and binarized. This function will resize and normalize it.

    Args:
        image_path (str): The path to the signature image file.
        image_size (tuple): The target size (height, width) for the output image.

    Returns:
        np.array: The preprocessed image as a NumPy array.
    """
    # Read the image. The IMREAD_GRAYSCALE flag ensures it's loaded as a single channel.
    # This is robust and will work even if some images are not pre-converted.
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    # Resize the image to the target size
    resized_img = cv2.resize(img, image_size, interpolation=cv2.INTER_AREA)

    # Normalize pixel values to be between 0 and 1
    normalized_img = resized_img.astype('float32') / 255.0

    # Add a channel dimension to make it (1, H, W) for PyTorch
    return np.expand_dims(normalized_img, axis=0)