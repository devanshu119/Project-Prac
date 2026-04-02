# file: scripts/refine_mask.py
import cv2
import numpy as np

def refine_segmentation_mask(mask_path, min_area_threshold=100):
    """
    Refines a binary segmentation mask using morphological operations and
    connected components analysis.

    Args:
        mask_path (str): Path to the input binary mask image.
        min_area_threshold (int): The minimum number of pixels for a component
                                  to be retained.

    Returns:
        numpy.ndarray: The refined binary mask.
    """
    # Read the mask as a grayscale image
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask file not found at {mask_path}")

    # Ensure the mask is binary (0 or 255)
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # 1. Morphological Operations
    # Use a small kernel for subtle cleaning
    kernel = np.ones((3, 3), np.uint8)
    
    # Opening to remove salt-and-pepper noise
    opened_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Closing to fill small holes within the spill
    closed_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 2. Connected Components Analysis to remove small blobs
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed_mask, 4, cv2.CV_32S)

    refined_mask = np.zeros_like(closed_mask)

    # Start from 1 to ignore the background (label 0)
    for i in range(1, num_labels):
        area = stats
        if area >= min_area_threshold:
            # Add the component to the final mask if it's large enough
            refined_mask[labels == i] = 255
            
    return refined_mask

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Refine a segmentation mask.')
    parser.add_argument('-i', '--input', required=True, help='Path to the input mask image.')
    parser.add_argument('-o', '--output', required=True, help='Path to save the refined mask image.')
    parser.add_argument('--min-area', type=int, default=100, help='Minimum area threshold for components.')
    args = parser.parse_args()

    refined_output = refine_segmentation_mask(args.input, args.min_area)
    cv2.imwrite(args.output, refined_output)
    print(f"Refined mask saved to {args.output}")