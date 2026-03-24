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
st.set_page_config(page_title="Urban Change Detection AI", page_icon="🏙️", layout="wide")

st.title("🛰️ Urban Change Detection AI")
st.markdown("Upload bi-temporal satellite imagery to detect new construction and urbanization.")

# --- 2. DEFINE THE ROBUST LOADER ---
def load_image_robust(file):
    try:
        # Try reading with tifffile first (best for GeoTIFFs)
        img_array = tiff.imread(file)
    except Exception:
        # Fallback to standard Pillow if tifffile fails
        return Image.open(file).convert("RGB")
    
    # Handle multi-band (Keep RGB)
    if len(img_array.shape) == 3 and img_array.shape[-1] > 3:
        img_array = img_array[:, :, :3]
        
    # Handle 16-bit to 8-bit normalization
    if img_array.dtype == np.uint16:
        img_array = (img_array / 65535.0 * 255).astype(np.uint8)
    
    # Handle grayscale
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array]*3, axis=-1)

    return Image.fromarray(img_array)

# --- 3. LOAD MODEL (CACHED) ---
@st.cache_resource
def load_model():
    model_path = 'best_levir_deeplabv3_effb4.pth'
    file_id = '1iV91GcRBVgfu39IA4CVkQ4rH_iw6W8PV' 
    
    if not os.path.exists(model_path):
        with st.spinner("Downloading AI Model weights..."):
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

# Global initialization
model = load_model()

# --- 4. SIDEBAR UPLOADS ---
st.sidebar.header("Data Input Panel")
img_file1 = st.sidebar.file_uploader("Upload Time 1 (Before)", type=['png', 'jpg', 'jpeg', 'tif', 'tiff'])
img_file2 = st.sidebar.file_uploader("Upload Time 2 (After)", type=['png', 'jpg', 'jpeg', 'tif', 'tiff'])

# --- 5. MAIN LOGIC (Only runs if files are present) ---
if img_file1 and img_file2:
    
    # Load and display images
    image1 = load_image_robust(img_file1)
    image2 = load_image_robust(img_file2)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Time 1")
        st.image(image1, use_column_width=True)
    with col2:
        st.subheader("Time 2")
        st.image(image2, use_column_width=True)
        
    if st.button("🚀 Run Analysis"):
        with st.spinner("Analyzing Satellite Data..."):
            # Prepare Tensors
            t1 = TF.to_tensor(TF.resize(image1, (512, 512)))
            t2 = TF.to_tensor(TF.resize(image2, (512, 512)))
            input_tensor = torch.cat([t1, t2], dim=0).unsqueeze(0)
            
            # AI Inference
            with torch.no_grad():
                output = model(input_tensor)
                mask = (torch.sigmoid(output) > 0.5).squeeze().cpu().numpy().astype(np.uint8)
            
            # Metrics
            _, num_buildings = label(mask)
            area = np.sum(mask) * 0.25 # 0.5m resolution
            
            # Display Results
            st.divider()
            m1, m2 = st.columns(2)
            m1.metric("New Buildings", num_buildings)
            m2.metric("Est. Urbanization Area", f"{area:,.1f} m²")
            
            fig, ax = plt.subplots()
            ax.imshow(mask, cmap='magma')
            ax.axis('off')
            st.pyplot(fig)
            st.success("Analysis Complete!")
else:
    st.info("👈 Please upload both images to begin.")
