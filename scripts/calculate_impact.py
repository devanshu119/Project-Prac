# file: scripts/calculate_impact.py
import geopandas as gpd
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape, MultiPolygon
import pandas as pd

def calculate_distance_to_coast(geotiff_path, refined_mask_path, coastline_shapefile_path):
    """
    Calculates the minimum distance from each detected spill to the nearest coastline.

    Args:
        geotiff_path (str): Path to the original georeferenced GeoTIFF.
        refined_mask_path (str): Path to the refined binary mask.
        coastline_shapefile_path (str): Path to a coastline shapefile.

    Returns:
        geopandas.GeoDataFrame: A GeoDataFrame with spill polygons and their distance to the coast.
    """
    # 1. Convert raster mask to vector polygons
    with rasterio.open(refined_mask_path) as src_mask:
        # Read the mask data
        mask_data = src_mask.read(1)
        # Get the transform and CRS from the original geotiff to ensure alignment
        with rasterio.open(geotiff_path) as src_geotiff:
            transform = src_geotiff.transform
            crs = src_geotiff.crs

        # Generate shapes (polygons) from the raster mask
        results = [
            {'properties': {'raster_val': v}, 'geometry': s}
            for i, (s, v) in enumerate(shapes(mask_data, mask=mask_data > 0, transform=transform))
        ]

    if not results:
        print("No spill polygons found in the mask.")
        return None

    # Create a GeoDataFrame for the spills
    spills_gdf = gpd.GeoDataFrame.from_features(results, crs=crs)

    # 2. Load coastline data
    coastline_gdf = gpd.read_file(coastline_shapefile_path)

    # 3. Ensure both GeoDataFrames use the same projected CRS for accurate distance calculation
    # A world projected CRS like EPSG:3857 is suitable for global distance approximation
    # For regional analysis, a local UTM zone would be more accurate.
    projected_crs = "EPSG:3857"
    spills_gdf_proj = spills_gdf.to_crs(projected_crs)
    coastline_gdf_proj = coastline_gdf.to_crs(projected_crs)
    
    # Combine all coastline geometries into a single MultiPolygon/MultiLineString for efficiency
    coastline_unary = coastline_gdf_proj.unary_union

    # 4. Calculate the minimum distance from each spill to the coastline
    # The distance method calculates the shortest distance from each spill polygon
    # to the unified coastline geometry.
    spills_gdf_proj['distance_to_coast_m'] = spills_gdf_proj.geometry.apply(lambda geom: geom.distance(coastline_unary))
    spills_gdf_proj['distance_to_coast_km'] = spills_gdf_proj['distance_to_coast_m'] / 1000

    return spills_gdf_proj[['geometry', 'distance_to_coast_km']]

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Calculate distance from spills to coastline.')
    parser.add_argument('--geotiff', required=True, help='Path to the original GeoTIFF.')
    parser.add_argument('--mask', required=True, help='Path to the refined mask.')
    parser.add_argument('--coastline', required=True, help='Path to the coastline shapefile.')
    parser.add_argument('--output', required=True, help='Path to save the output GeoJSON file.')
    args = parser.parse_args()

    impact_gdf = calculate_distance_to_coast(args.geotiff, args.mask, args.coastline)
    
    if impact_gdf is not None:
        impact_gdf.to_file(args.output, driver='GeoJSON')
        print(f"Impact analysis saved to {args.output}")
        print(impact_gdf.head())