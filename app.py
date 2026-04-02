import streamlit as st
import torch
import numpy as np
from PIL import Image
import cv2

from models.unet_model import UNet
from models.deeplab_model import create_deeplabv3


# -------------------------
# Utility: Load Model
# -------------------------
@st.cache_resource
def load_model(checkpoint_path, model_choice, n_classes, n_channels, device):
    if model_choice == "UNet":
        model = UNet(n_channels=n_channels, n_classes=n_classes)
    elif model_choice == "DeepLabV3":
        model = create_deeplabv3(n_classes=n_classes, n_channels=n_channels)
    else:
        raise ValueError("Invalid model choice")

    state_dict = torch.load(checkpoint_path, map_location=device)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        st.error(
            f"❌ Checkpoint is incompatible with {model_choice}. "
            f"Please select the correct model or matching checkpoint."
        )
        raise e

    model.to(device)
    model.eval()
    return model


# -------------------------
# Utility: Predict
# -------------------------
def predict(model, image_tensor, device):
    with torch.no_grad():
        output = model(image_tensor.to(device))
        if isinstance(output, dict):  # DeepLab returns dict
            output = output["out"]
        pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
    return pred_mask


# -------------------------
# Utility: Highlight Spill
# -------------------------
def highlight_spill(original_img, pred_mask, spill_class=1):
    img = np.array(original_img.convert("RGB"))  # 3-channel for drawing
    spill_mask = (pred_mask == spill_class).astype(np.uint8) * 255

    contours, _ = cv2.findContours(
        spill_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(img, contours, -1, (255, 0, 0), thickness=2)  # red outline

    return Image.fromarray(img)


# -------------------------
# Streamlit NASA-style Layout
# -------------------------
st.set_page_config(
    page_title=" Oil Spill Detection",
    layout="wide",
    page_icon="🛰",
)

# Custom CSS for NASA theme
st.markdown(
    """
    <style>
    body {
        background-color: #0b0c10;
        color: #c5c6c7;
    }
    .stApp {
        background-color: #0b0c10;
    }
    .big-title {
        font-size: 38px;
        font-weight: 700;
        color: #66fcf1;
        text-align: center;
    }
    .subtitle {
        font-size: 20px;
        color: #c5c6c7;
        text-align: center;
        margin-bottom: 30px;
    }
    .card {
        padding: 20px;
        border-radius: 12px;
        background-color: #1f2833;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown('<div class="big-title"> SAR Oil Spill Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Deep Learning Segmentation with U-Net & DeepLabV3</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("🔧 Control Panel")
uploaded_file = st.sidebar.file_uploader("📂 Upload SAR Image", type=["png", "jpg", "jpeg"])
model_choice = st.sidebar.radio("Select Model", ["UNet", "DeepLabV3"])
checkpoint_path = st.sidebar.text_input(
    "Checkpoint Path (.pth)", "checkpoints/unet_epoch1.pth"
)
spill_class = st.sidebar.number_input("Spill Class Index", min_value=1, max_value=3, value=1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.sidebar.markdown("---")
st.sidebar.info("Tip: Use **DeepLabV3** for large-scale generalization.\nUse **UNet** for lightweight inference.")

# Main Workflow
if uploaded_file is not None and checkpoint_path:
    # Load image
    image = Image.open(uploaded_file).convert("L")  # grayscale
    img_resized = image.resize((256, 256))
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

    # Load model
    model = load_model(checkpoint_path, model_choice, n_classes=3, n_channels=1, device=device)

    # Predict
    pred_mask = predict(model, img_tensor, device)

    # Generate results
    outlined_img = highlight_spill(img_resized, pred_mask, spill_class=spill_class)

    # Color segmentation mask
    colors = {
        0: (0, 0, 0),        # background
        1: (255, 0, 0),      # oil spill
        2: (0, 255, 0),      # other/noise
    }
    seg_vis = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
    for cls, color in colors.items():
        seg_vis[pred_mask == cls] = color
    seg_img = Image.fromarray(seg_vis)

    # Overlay mask on image (NASA-style heatmap)
    overlay = np.array(img_resized.convert("RGB"))
    overlay_mask = cv2.addWeighted(overlay, 0.7, seg_vis, 0.3, 0)
    overlay_img = Image.fromarray(overlay_mask)

    # Layout in columns
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.image(image, caption="📥 Input SAR Image", use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.image(seg_img, caption="🎨 Segmentation Mask", use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.image(outlined_img, caption="🔴 Outlined Oil Spill", use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Full-width overlay (NASA-style figure)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.image(overlay_img, caption="🌌 SAR + Spill Heatmap Overlay", use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
