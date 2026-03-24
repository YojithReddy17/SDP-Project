import streamlit as st
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import segmentation_models_pytorch as smp
from scipy.ndimage import label
import matplotlib.pyplot as plt
import os
import gdown
import tifffile as tiff

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="SOTA Urban Change AI", page_icon="🛰️", layout="wide")

st.title("🛰️ High-Resolution Urban Change Detection")
st.markdown("Processing large-scale mosaic imagery using Sliding Window Tiling.")

# --- 2. ROBUST IMAGE LOADER & NORMALIZER ---
def load_satellite_image(file):
    try:
        img_array = tiff.imread(file)
    except Exception:
        return np.array(Image.open(file).convert("RGB"))
    
    # Slice to RGB if multi-band (keeping first 3)
    if len(img_array.shape) == 3 and img_array.shape[-1] > 3:
        img_array = img_array[:, :, :3]
        
    # Normalization for 16-bit imagery
    if img_array.dtype == np.uint16:
        img_array = (img_array / 65535.0 * 255).astype(np.uint8)
    
    # Handle Grayscale
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array]*3, axis=-1)
        
    return img_array

# --- 3. MODEL LOADER (FROM GOOGLE DRIVE) ---
@st.cache_resource
def load_model():
    model_path = 'best_levir_deeplabv3_effb4.pth'
    file_id = '1iV91GcRBVgfu39IA4CVkQ4rH_iw6W8PV' 
    
    if not os.path.exists(model_path):
        with st.spinner("Downloading AI weights from Cloud..."):
            gdown.download(id=file_id, output=model_path, quiet=False)

    model = smp.DeepLabV3Plus(
        encoder_name="efficientnet-b4",
        encoder_weights=None, 
        in_channels=6,
        classes=1,
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# --- 4. THE TILING ENGINE ---
def run_tiled_inference(img1, img2, model, tile_size=512):
    h, w, _ = img1.shape
    
    # Create an empty mask for the results
    full_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Calculate total tiles for progress bar
    steps_y = range(0, h, tile_size)
    steps_x = range(0, w, tile_size)
    total_tiles = len(steps_y) * len(steps_x)
    
    progress_bar = st.progress(0)
    current_tile = 0
    
    with torch.no_grad():
        for y in steps_y:
            for x in steps_x:
                # Extract tiles (handling edges)
                end_y = min(y + tile_size, h)
                end_x = min(x + tile_size, w)
                
                tile1 = img1[y:end_y, x:end_x]
                tile2 = img2[y:end_y, x:end_x]
                
                # Resize tile to exactly 512x512 for the model
                t1_p = TF.to_tensor(Image.fromarray(tile1).resize((tile_size, tile_size)))
                t2_p = TF.to_tensor(Image.fromarray(tile2).resize((tile_size, tile_size)))
                
                input_tensor = torch.cat([t1_p, t2_p], dim=0).unsqueeze(0)
                
                # Inference
                output = model(input_tensor)
                pred = (torch.sigmoid(output) > 0.5).squeeze().cpu().numpy().astype(np.uint8)
                
                # Resize back to original tile dimensions and stitch
                pred_resized = np.array(Image.fromarray(pred).resize((end_x - x, end_y - y)))
                full_mask[y:end_y, x:end_x] = pred_resized
                
                current_tile += 1
                progress_bar.progress(current_tile / total_tiles)
                
    return full_mask

# --- 5. SIDEBAR ---
st.sidebar.header("Data Input")
file1 = st.sidebar.file_uploader("Time 1 (Before)", type=['png', 'jpg', 'tif'])
file2 = st.sidebar.file_uploader("Time 2 (After)", type=['png', 'jpg', 'tif'])

# --- 6. MAIN APP LOGIC ---
if file1 and file2:
    img1 = load_satellite_image(file1)
    img2 = load_satellite_image(file2)
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Mosaic: Time 1")
        st.image(img1, use_column_width=True)
    with c2:
        st.subheader("Mosaic: Time 2")
        st.image(img2, use_column_width=True)
        
    if st.button("🚀 Analyze Full Mosaic", use_container_width=True):
        st.info("Large Mosaic detected. Running Tiled Analysis (This preserves building-level detail).")
        
        final_mask = run_tiled_inference(img1, img2, model)
        
        # Calculate Metrics
        _, num_buildings = label(final_mask)
        area = np.sum(final_mask) * 0.25 # Based on LEVIR-CD resolution
        
        st.divider()
        m1, m2 = st.columns(2)
        m1.metric("Total Buildings Detected", num_buildings)
        m2.metric("Total Urbanization Area", f"{area:,.1f} m²")
        
        st.subheader("Global Change Mask")
        st.image(final_mask * 255, use_column_width=True, clamp=True)
        
        st.success("Analysis Complete!")
else:
    st.info("👈 Upload your satellite mosaics to begin.")
