# file: webapp/app.py
import os
import uuid
import cv2
from flask import Flask, request, render_template, jsonify, url_for
from werkzeug.utils import secure_filename

# Import the pipeline functions from our scripts
# Note: For a production app, these would be better organized into a package
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.preprocess_sar import preprocess_image
from scripts.predict import predict_image
from scripts.refine_mask import refine_segmentation_mask
from scripts.quantify_spill import quantify_spill_properties
from scripts.calculate_impact import calculate_distance_to_coast

# Model loading (should be done once at startup)
from models.unet_model import UNet
import torch

# --- Configuration ---
UPLOAD_FOLDER = os.path.join('static', 'uploads')
RESULTS_FOLDER = os.path.join('static', 'results')
ALLOWED_EXTENSIONS = {'tif', 'tiff', 'safe'} # Allow SAR data formats

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB limit

# Create directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# --- Model Loading ---
# Load the trained model once when the app starts
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Assuming a trained UNet model for 3 classes
net = UNet(n_channels=1, n_classes=3) 
model_path = '../saved_models/best_model.pth' # Path to your trained model
if os.path.exists(model_path):
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.to(device=device)
    net.eval()
    print("Model loaded successfully.")
else:
    print(f"Warning: Model file not found at {model_path}. Prediction will fail.")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1).lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle file upload and run the full analysis pipeline."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        # Generate a unique filename to avoid conflicts
        ext = file.filename.rsplit('.', 1).lower()
        unique_id = str(uuid.uuid4())
        
        # We assume the user uploads a raw Sentinel-1 scene (e.g., a.zip file from download)
        # For simplicity in this example, let's assume the user uploads a pre-processed GeoTIFF
        # A full implementation would run the `preprocess_sentinel1_scene` function here.
        
        # Let's save the uploaded pre-processed GeoTIFF
        filename = secure_filename(f"{unique_id}.{ext}")
        input_geotiff_path = os.path.join(app.config, filename)
        file.save(input_geotiff_path)

        try:
            # --- Pipeline Execution ---
            # 1. Predict raw mask
            
            # Use preprocess_image to get tensor
            preprocessed_tensor = preprocess_image(input_geotiff_path, target_size=(256, 256))
            
            # Infer segmentation mask using predict_image
            raw_mask_np = predict_image(model=net, image_tensor=preprocessed_tensor, device=device)
            
            raw_mask_path = os.path.join(app.config['RESULTS_FOLDER'], f'{unique_id}_raw_mask.png')
            
            # Save raw mask directly, mapping classes to distinct grayscale intensities or colors
            mask_img_save = (raw_mask_np * (255 // max(1, raw_mask_np.max()))).astype(np.uint8)
            cv2.imwrite(raw_mask_path, mask_img_save)

            # 2. Refine the mask
            refined_mask_np = refine_segmentation_mask(raw_mask_path, min_area_threshold=100)
            refined_mask_path = os.path.join(app.config['RESULTS_FOLDER'], f'{unique_id}_refined_mask.png')
            cv2.imwrite(refined_mask_path, refined_mask_np)

            # 3. Quantify spill properties
            spill_props = quantify_spill_properties(input_geotiff_path, refined_mask_path)
            total_area = sum(p['area_km2'] for p in spill_props)

            # 4. Calculate distance to coast (requires a coastline shapefile)
            coastline_shp = '../data/coastline/ne_10m_coastline.shp' # Path to coastline data
            impact_gdf = calculate_distance_to_coast(input_geotiff_path, refined_mask_path, coastline_shp)
            min_dist = impact_gdf['distance_to_coast_km'].min() if impact_gdf is not None and not impact_gdf.empty else 'N/A'

            # 5. Prepare response for frontend
            import rasterio
            with rasterio.open(input_geotiff_path) as src:
                bounds = src.bounds
                # Leaflet expects bounds in [[lat_min, lon_min], [lat_max, lon_max]]
                # We need to convert from the image's CRS to WGS84 (lat/lon)
                from rasterio.warp import transform_bounds
                wgs84_bounds = transform_bounds(src.crs, 'EPSG:4326', *bounds)
                # wgs84_bounds is (west, south, east, north) -> (lon_min, lat_min, lon_max, lat_max)
                leaflet_bounds = [[wgs84_bounds, wgs84_bounds], [wgs84_bounds, wgs84_bounds]]

            # Create a semi-transparent version of the mask for overlay
            refined_mask_rgba = cv2.cvtColor(refined_mask_np, cv2.COLOR_GRAY2BGRA)
            refined_mask_rgba[:, :, 3] = (refined_mask_np > 0) * 150 # Set alpha channel (0-255)
            overlay_path = os.path.join(app.config['RESULTS_FOLDER'], f'{unique_id}_overlay.png')
            cv2.imwrite(overlay_path, refined_mask_rgba)

            response_data = {
                'success': True,
                'mask_image_url': url_for('static', filename=f'results/{unique_id}_overlay.png'),
                'bounds': leaflet_bounds,
                'analysis': {
                    'spill_area_km2': round(total_area, 2),
                    'distance_to_coast_km': round(min_dist, 2) if isinstance(min_dist, float) else min_dist,
                    'num_spills': len(spill_props)
                }
            }
            return jsonify(response_data)

        except Exception as e:
            print(f"Error during processing: {e}")
            return jsonify({'error': 'An error occurred during processing.'}), 500

    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True)