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

st.title("🛰️ Semantic Segmentation: Satellite Urban Change Detection")
st.markdown("""
Upload bi-temporal satellite imagery (Time 1 and Time 2) to automatically detect new construction, count structural footprints, and estimate urbanization area.
*Powered by DeepLabV3+ and EfficientNet-b4.*
""")

# --- 2. LOAD MODEL (WITH GOOGLE DRIVE FETCH) ---
@st.cache_resource
def load_model():
    model_path = 'best_levir_deeplabv3_effb4.pth'
    
    # 🚨 REPLACE THIS WITH YOUR EXACT GOOGLE DRIVE FILE ID 🚨
    file_id = '1iV91GcRBVgfu39IA4CVkQ4rH_iw6W8PV' 
    
    # Download from Drive if it doesn't exist on the server yet
    if not os.path.exists(model_path):
        with st.spinner("Downloading 150MB AI Model from Google Drive... (This takes about 30 seconds on first boot)"):
            gdown.download(id=file_id, output=model_path, quiet=False)

    # Initialize the DeepLabV3+ Architecture
    model = smp.DeepLabV3Plus(
        encoder_name="efficientnet-b4",
        encoder_weights=None, 
        in_channels=6,
        classes=1,
    )
    
    # Load weights onto the CPU (Streamlit Community Cloud does not use GPUs)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Replace your image loading lines with this:
def load_image_robust(file):
    # Read the file using tifffile
    img_array = tiff.imread(file)
    
    # 1. Handle multi-band (e.g., 4-band RGB-NIR). Keep only first 3 bands (RGB).
    if len(img_array.shape) == 3 and img_array.shape[-1] > 3:
        img_array = img_array[:, :, :3]
        
    # 2. Handle 16-bit to 8-bit normalization (Standardizes the brightness)
    if img_array.dtype == np.uint16:
        img_array = (img_array / 65535.0 * 255).astype(np.uint8)
    
    # 3. Handle grayscale (Repeat channel 3 times for RGB)
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array]*3, axis=-1)

    return Image.fromarray(img_array)


image1 = load_image_robust(img_file1)
image2 = load_image_robust(img_file2)

with st.spinner("Initializing System & Checking Weights..."):
    model = load_model()

# --- 3. SIDEBAR UPLOADS ---
st.sidebar.header("Data Input Panel")
st.sidebar.info("Upload standard RGB images (e.g., .png or .jpg). Images will be automatically resized for analysis.")

img_file1 = st.sidebar.file_uploader("Upload Time 1 Image (Before)", type=['png', 'jpg', 'jpeg', 'tif', 'tiff'])
img_file2 = st.sidebar.file_uploader("Upload Time 2 Image (After)", type=['png', 'jpg', 'jpeg', 'tif', 'tiff'])

# --- 4. MAIN APPLICATION LOGIC ---
if img_file1 and img_file2:
    
    col1, col2 = st.columns(2)
    
    image1 = Image.open(img_file1).convert("RGB")
    image2 = Image.open(img_file2).convert("RGB")
    
    with col1:
        st.subheader("Time 1 (Before)")
        st.image(image1, use_column_width=True)
        
    with col2:
        st.subheader("Time 2 (After)")
        st.image(image2, use_column_width=True)
        
    if st.button("🚀 Run AI Change Detection Analysis", use_container_width=True):
        
        with st.spinner("Processing 6-Channel Early Fusion Tensor & Running Inference..."):
            
            # Resize to 512x512 (Dimensions must be divisible by 32 for DeepLab)
            process_size = (512, 512)
            t1 = TF.resize(image1, process_size)
            t2 = TF.resize(image2, process_size)
            
            # Convert to tensors
            t1_tensor = TF.to_tensor(t1)
            t2_tensor = TF.to_tensor(t2)
            
            # Stack into 6 channels: [1, 6, 512, 512]
            input_tensor = torch.cat([t1_tensor, t2_tensor], dim=0).unsqueeze(0)
            
            # Run the AI
            with torch.no_grad():
                output = model(input_tensor)
                pred_mask = torch.sigmoid(output) > 0.5
            
            pred_mask_np = pred_mask.squeeze().cpu().numpy().astype(np.uint8)
            
            # --- ALGORITHMS: METRICS CALCULATIONS ---
            # 1. Count isolated buildings using Connected Components
            labeled_mask, num_buildings = label(pred_mask_np)
            
            # 2. Calculate Area (Assuming LEVIR-CD+ ~0.5m/pixel resolution -> 0.25 sq meters per pixel)
            total_changed_pixels = np.sum(pred_mask_np)
            estimated_sq_meters = total_changed_pixels * 0.25
            
            # --- DISPLAY RESULTS ---
            st.divider()
            st.subheader("📊 Analysis Results")
            
            m1, m2, m3 = st.columns(3)
            m1.metric(label="New Buildings Detected", value=num_buildings)
            m2.metric(label="Changed Pixels", value=f"{total_changed_pixels:,}")
            m3.metric(label="Est. Urbanization Area", value=f"{estimated_sq_meters:,.1f} m²")
            
            st.subheader("AI Prediction Mask")
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(pred_mask_np, cmap='magma')
            ax.axis('off')
            st.pyplot(fig)
            
            st.success("Analysis Complete! Ready for next upload.")
else:
    st.info("👈 Please upload both images in the sidebar to begin analysis.")
