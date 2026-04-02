# scripts/preprocess_sar.py

import numpy as np
from skimage import io, transform, exposure
from scipy.ndimage import uniform_filter
from scipy.ndimage import variance

"""
This module provides functions for preprocessing Synthetic Aperture Radar (SAR) images
for oil spill detection models. The pipeline includes speckle noise reduction,
contrast enhancement, and normalization.
"""

def lee_filter(img: np.ndarray, size: int) -> np.ndarray:
    """
    Apply a Lee filter for speckle noise reduction.
    The Lee filter is an adaptive filter that preserves edges while smoothing noise.
    It is particularly effective for multiplicative noise like speckle in SAR images.
    
    Args:
        img (np.ndarray): Input image as a NumPy array.
        size (int): The size of the filter window (e.g., 7 for a 7x7 window).
        
    Returns:
        np.ndarray: The filtered image.
    """
    # Ensure image is float type for calculations
    img = img.astype(np.float64)
    
    # Calculate local mean
    img_mean = uniform_filter(img, (size, size))
    
    # Calculate local squared mean
    img_sqr_mean = uniform_filter(img**2, (size, size))
    
    # Calculate local variance
    img_variance = img_sqr_mean - img_mean**2
    
    # Estimate overall variance (noise variance)
    overall_variance = variance(img)
    
    # Calculate weights for the filter
    # Weights are inversely proportional to local variance
    img_weights = img_variance / (img_variance + overall_variance)
    
    # Apply the filter formula
    filtered_img = img_mean + img_weights * (img - img_mean)
    
    return filtered_img

def preprocess_image(input_path: str, target_size: tuple = (512, 512)) -> np.ndarray:
    """
    Main function to execute the full preprocessing pipeline on a single SAR image.
    
    Pipeline steps:
    1. Load the image and convert to grayscale float.
    2. Apply the Lee filter to reduce speckle noise.
    3. Enhance contrast using histogram equalization.
    4. Resize the image to the model's expected input dimensions.
    5. Normalize pixel values to the  range.
    6. Expand dimensions to create a batch of 1 for model input.
    
    Args:
        input_path (str): The file path to the input SAR image.
        target_size (tuple): The target (height, width) for the model input.
        
    Returns:
        np.ndarray: The preprocessed image tensor of shape (1, H, W, 1).
    """
    # 1. Image Loading and Conversion
    # Use scikit-image's io.imread for robust image loading
    try:
        image = io.imread(input_path, as_gray=True)
    except Exception as e:
        print(f"Error loading image at {input_path}: {e}")
        raise
        
    # Convert image to a floating point type for processing
    image = image.astype(np.float64)

    # 2. Speckle Noise Reduction (Lee Filter)
    # A window size of 7 is a common starting point
    lee_filtered_image = lee_filter(image, size=7)

    # 3. Contrast Enhancement (Histogram Equalization)
    # This stretches the intensity range to improve visibility of features
    equalized_image = exposure.equalize_hist(lee_filtered_image)

    # 4. Resizing
    # Resize to the fixed input size required by the CNN
    resized_image = transform.resize(equalized_image, target_size, anti_aliasing=True)
    
    # 5. Normalization
    # Ensure values are strictly between 0 and 1, which is already handled by
    # equalize_hist and resize, but we can clip just in case.
    normalized_image = np.clip(resized_image, 0.0, 1.0)

    # 6. Dimension Expansion
    # The model expects a 4D tensor: (batch_size, height, width, channels)
    # Here, batch_size is 1 and channels is 1 (grayscale)
    model_input = np.expand_dims(normalized_image, axis=-1) # Add channel dimension
    model_input = np.expand_dims(model_input, axis=0)      # Add batch dimension
    
    return model_input.astype(np.float32)