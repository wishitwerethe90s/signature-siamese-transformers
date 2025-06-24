import cv2
import numpy as np
from PIL import Image, ImageEnhance
import random

def advanced_preprocess_signature(image_path, image_size=(224, 224), augment=False):
    """Enhanced preprocessing with augmentation and contrast enhancement."""
    
    # Load image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Contrast enhancement
    img = cv2.convertScaleAbs(img, alpha=1.2, beta=10)
    
    # Noise reduction
    img = cv2.medianBlur(img, 3)
    
    # Binarization with adaptive thresholding
    img = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    if augment:
        img = apply_augmentation(img)
    
    # Resize
    img = cv2.resize(img, image_size, interpolation=cv2.INTER_AREA)
    
    # Normalize
    img = img.astype('float32') / 255.0
    
    return np.expand_dims(img, axis=0)

def apply_augmentation(img):
    """Apply random augmentations suitable for signatures."""
    # Random rotation (-10 to 10 degrees)
    if random.random() < 0.5:
        angle = random.uniform(-10, 10)
        center = (img.shape[1]//2, img.shape[0]//2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(img, matrix, (img.shape[1], img.shape[0]))
    
    # Slight scaling
    if random.random() < 0.3:
        scale = random.uniform(0.95, 1.05)
        new_size = (int(img.shape[1] * scale), int(img.shape[0] * scale))
        img = cv2.resize(img, new_size)
        # Crop or pad to original size
        img = cv2.resize(img, (img.shape[1], img.shape[0]))
    
    return img