import streamlit as st
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import segmentation_models_pytorch as smp
from scipy.ndimage import label, binary_erosion, binary_closing, binary_fill_holes
import matplotlib.pyplot as plt
import os
import gdown
import tifffile as tiff

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Semantic Segmentation Urban Change Detection", page_icon="🛰️", layout="wide")

st.title("🛰️ SOTA Urban Change Detection")
st.markdown("### DeepLabV3+ Early Fusion | EfficientNet-b4 Backbone")
st.info("System optimized for LEVIR-CD+ 0.5m/px high-resolution satellite imagery.")

# --- 2. ROBUST IMAGE LOADER ---
def load_image_robust(file):
    try:
        # Robust handling for professional GeoTIFF/Mosaic formats
        img_array = tiff.imread(file)
    except:
        # Standard handling for PNG/JPG
        return Image.open(file).convert("RGB")
    
    # Handle multi-band imagery (strip to first 3 channels)
    if len(img_array.shape) == 3 and img_array.shape[-1] > 3:
        img_array = img_array[:, :, :3]
    # Normalize 16-bit radiometric depth to 8-bit for AI processing
    if img_array.dtype == np.uint16:
        img_array = (img_array / 65535.0 * 255).astype(np.uint8)
    # Handle single-channel grayscale
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array]*3, axis=-1)
    return Image.fromarray(img_array)

# --- 3. MODEL LOADER (CACHED) ---
@st.cache_resource
def load_model():
    model_path = 'best_levir_deeplabv3_effb4.pth'
    file_id = '1iV91GcRBVgfu39IA4CVkQ4rH_iw6W8PV' 
    
    # Automatic Cloud Fetching
    if not os.path.exists(model_path):
        with st.spinner("Downloading 150MB DeepLabV3+ Weights..."):
            gdown.download(id=file_id, output=model_path, quiet=False)

    model = smp.DeepLabV3Plus(
        encoder_name="efficientnet-b4",
        encoder_weights=None, 
        in_channels=6,
        classes=1,
    )
    
    # Load to CPU for Streamlit Cloud compatibility
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# --- 4. SIDEBAR: GRANULAR OPTIMIZATION ---
st.sidebar.header("⚖️ Optimization Control")
st.sidebar.markdown("Tune the AI sensitivity to prioritize building count vs. total area.")

#sidebar for the tweaking of accuracy of no of buildings and area covered
threshold = st.sidebar.slider(
    "Set Detection Sensitivity:", 
    min_value=0.05, 
    max_value=0.95, 
    value=0.50, 
    step=0.05
)

# Dynamic labeling for the demo
if threshold <= 0.25:
    mode_label = "🔴 Mode: Max Area Coverage"
    st.sidebar.warning("High Recall: Best for total footprint. Buildings may merge.")
elif 0.25 < threshold <= 0.70:
    mode_label = "🟡 Mode: Balanced Analysis"
    st.sidebar.info("Standard baseline for LEVIR-CD+ imagery.")
else:
    mode_label = "🟢 Mode: Max Count Accuracy"
    st.sidebar.success("High Precision: Best for individual building counts.")

st.sidebar.subheader(mode_label)
st.sidebar.divider()

file1 = st.sidebar.file_uploader("Upload Time 1 (Before)", type=['png', 'jpg', 'tif'])
file2 = st.sidebar.file_uploader("Upload Time 2 (After)", type=['png', 'jpg', 'tif'])

# --- 5. MAIN LOGIC PIPELINE ---
if file1 and file2:
    image1 = load_image_robust(file1)
    image2 = load_image_robust(file2)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Time 1 (Before)")
        st.image(image1, use_column_width=True)
    with col2:
        st.subheader("Time 2 (After)")
        st.image(image2, use_column_width=True)
        
    if st.button("Run Comprehensive Change Analysis", use_container_width=True):
        with st.spinner(f"Inference at {threshold} Threshold..."):
            
            # --- PREPROCESSING (NATIVE SCALING) ---
            # Model trained on 512x512 windows
            t1_img = image1.resize((512, 512))
            t2_img = image2.resize((512, 512))
            
            # THE 0-255 FIX: Critical for high-confidence detections
            t1_np = np.array(t1_img).astype(np.float32)
            t2_np = np.array(t2_img).astype(np.float32)
            
            # Native tensor conversion without scaling to [0,1]
            x1 = torch.from_numpy(t1_np).permute(2, 0, 1)
            x2 = torch.from_numpy(t2_np).permute(2, 0, 1)
            input_tensor = torch.cat([x1, x2], dim=0).unsqueeze(0)
            
            # --- AI INFERENCE ---
            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.sigmoid(output).squeeze().cpu().numpy()
                mask = (probs > threshold).astype(np.uint8)
            
            # --- POST-PROCESSING (MORPHOLOGICAL REFINEMENT) ---
            # 1. Visual Refinement (Fills holes in roofs)
            mask = binary_closing(mask, structure=np.ones((3,3))).astype(np.uint8)
            mask = binary_fill_holes(mask).astype(np.uint8)

            # 2. Connectivity Resolution (Erosion to separate buildings for count)
            # We use a stronger kernel at low thresholds to prevent merging
            erosion_kernel = np.ones((5,5)) if threshold < 0.25 else np.ones((3,3))
            count_mask = binary_erosion(mask, structure=erosion_kernel).astype(np.uint8)
            _, num_buildings = label(count_mask)
            
            # 3. Geo-Metric Calculation
            # 0.5m resolution means 1px = 0.25 sq meters
            area = np.sum(mask) * 0.25 
            
            # --- 6. DISPLAY RESULTS ---
            st.divider()
            st.subheader(f"📊 Quantitative Analysis ({opt_mode if 'opt_mode' in locals() else 'Manual'} Mode)")
            
            m1, m2 = st.columns(2)
            m1.metric("Building Footprints Detected", num_buildings)
            m2.metric("Total Urbanized Area", f"{area:,.1f} m²")
            
            # --- VISUAL OVERLAY ---
            st.subheader("Building Detection Highlights")
            
            # PIL Alpha Blending for clean yellow overlay
            base = Image.fromarray(np.array(t2_img)).convert("RGBA")
            yellow_overlay = Image.new("RGBA", base.size, (255, 255, 0, 0))
            overlay_pixels = yellow_overlay.load()
            
            for y in range(512):
                for x in range(512):
                    if mask[y, x] == 1:
                        # Semi-transparent yellow (160/255)
                        overlay_pixels[x, y] = (255, 255, 0, 160) 

            combined = Image.alpha_composite(base, yellow_overlay)
            
            st.image(combined, use_column_width=True, caption=f"Detection Mask (Sensitivity: {threshold})")
            st.success("Analysis successful. Building instances isolated via morphological erosion.")
            
else:
    st.info(" Please upload the satellite image pair in the Control Panel to begin.")
