# file: scripts/quantify_spill.py
import rasterio
import numpy as np
from skimage.measure import label, regionprops
import cv2

def quantify_spill_properties(geotiff_path, refined_mask_path):
    """
    Calculates real-world properties (area, centroid) of detected spills.

    Args:
        geotiff_path (str): Path to the original georeferenced GeoTIFF image.
        refined_mask_path (str): Path to the refined binary mask.

    Returns:
        list: A list of dictionaries, where each dictionary contains the 
              properties of one detected spill component.
    """
    spill_properties = []

    with rasterio.open(geotiff_path) as src:
        transform = src.transform
        crs = src.crs
        pixel_size_x, pixel_size_y = transform, -transform
        pixel_area = pixel_size_x * pixel_size_y

    mask = cv2.imread(refined_mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask file not found at {refined_mask_path}")

    # Label each contiguous spill region in the mask
    labeled_mask = label(mask > 0)
    regions = regionprops(labeled_mask)

    for i, props in enumerate(regions):
        area_pixels = props.area
        area_sq_meters = area_pixels * pixel_area
        area_sq_km = area_sq_meters / 1_000_000

        # Get centroid in pixel coordinates (row, col)
        centroid_row, centroid_col = props.centroid

        # Convert pixel coordinates to geographic coordinates (lon, lat)
        lon, lat = rasterio.transform.xy(transform, centroid_row, centroid_col)
        
        spill_properties.append({
            'spill_id': i + 1,
            'area_km2': round(area_sq_km, 4),
            'centroid_pixel': (centroid_row, centroid_col),
            'centroid_coords': (lon, lat),
            'crs': str(crs)
        })
        
    return spill_properties

if __name__ == '__main__':
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description='Quantify spill properties from a mask.')
    parser.add_argument('--geotiff', required=True, help='Path to the original GeoTIFF.')
    parser.add_argument('--mask', required=True, help='Path to the refined mask.')
    parser.add_argument('--output', required=True, help='Path to save the output JSON file.')
    args = parser.parse_args()

    properties = quantify_spill_properties(args.geotiff, args.mask)
    
    with open(args.output, 'w') as f:
        json.dump(properties, f, indent=4)
        
    print(f"Spill properties saved to {args.output}")
    for prop in properties:
        print(f"  Spill ID {prop['spill_id']}: Area = {prop['area_km2']} km^2, Centroid = {prop['centroid_coords']}")