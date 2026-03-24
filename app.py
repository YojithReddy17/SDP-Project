import streamlit as st
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import segmentation_models_pytorch as smp
from scipy.ndimage import label
import os
import gdown

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="LEVIR-CD+ Change Detection", page_icon="🏙️", layout="wide")

st.title("🛰️ SOTA Urban Change Detection")
st.markdown("""
### DeepLabV3+ with EfficientNet-b4 Backbone
Upload native high-resolution **LEVIR-CD+** images (Time 1 and Time 2) in **PNG** format to detect new construction.
""")

# --- 2. RESTORED MODEL LOADER (WITH DRIVE FETCH) ---
@st.cache_resource
def load_model():
    model_path = 'best_levir_deeplabv3_effb4.pth'
    
    # Keeping your existing Google Drive ID for the champion DeepLabV3+ model
    file_id = '1iV91GcRBVgfu39IA4CVkQ4rH_iw6W8PV' 
    
    # Download weights if they don't exist on the server
    if not os.path.exists(model_path):
        with st.spinner("Downloading 150MB DeepLabV3+ model weights from Google Drive... (This takes ~30s on first boot)"):
            gdown.download(id=file_id, output=model_path, quiet=False)

    # Reinitialize the champion architecture
    model = smp.DeepLabV3Plus(
        encoder_name="efficientnet-b4",
        encoder_weights=None, # None because we are loading custom weights
        in_channels=6,
        classes=1,
    )
    
    # Load weights onto the CPU (required for free Streamlit Community Cloud)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

with st.spinner("Initializing System & Checking Weights..."):
    model = load_model()

# --- 3. SIDEBAR UPLOADERS (NORMAL FORMAT) ---
st.sidebar.header("Data Input Panel")
st.sidebar.info("Upload standard high-res PNG images.")

file1 = st.sidebar.file_uploader("Upload Time 1 (Before)", type=['png', 'jpg', 'jpeg'])
file2 = st.sidebar.file_uploader("Upload Time 2 (After)", type=['png', 'jpg', 'jpeg'])

# --- 4. MAIN APPLICATION LOGIC ---
if file1 and file2:
    
    # Load images immediately for display
    image1 = Image.open(file1).convert("RGB")
    image2 = Image.open(file2).convert("RGB")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Time 1 (Before)")
        st.image(image1, use_column_width=True)
    with col2:
        st.subheader("Time 2 (After)")
        st.image(image2, use_column_width=True)
        
    if st.button("🚀 Run Analysis", use_container_width=True):
        
        with st.spinner("Processing 6-Channel Early Fusion Tensor & Running Native Inference..."):
            
            # --- RESTORED NATIVE INFERENCE PIPELINE ---
            
            # 1. Resize images to 512x512. Model must infer at training dimension.
            # This handles any input dimension and rescales it to the native model scale.
            process_size = (512, 512)
            t1 = TF.resize(image1, process_size)
            t2 = TF.resize(image2, process_size)
            
            # 2. Convert to tensors
            t1_tensor = TF.to_tensor(t1)
            t2_tensor = TF.to_tensor(t2)
            
            # 3. Stack into 6 channels and add Batch Dimension [1, 6, 512, 512]
            input_tensor = torch.cat([t1_tensor, t2_tensor], dim=0).unsqueeze(0)
            
            # 4. Direct Inference (Normal Format - no tiling)
            with torch.no_grad():
                output = model(input_tensor)
                # Sigmoid to convert to probabilities, threshold at 0.5
                pred_mask = torch.sigmoid(output) > 0.5
            
            # Post-Processing: Convert back to Numpy
            pred_mask_np = pred_mask.squeeze().cpu().numpy().astype(np.uint8)
            
            # --- URBANIZATION METRICS ALGORITHMS ---
            # 1. Count Buildings (Connected Components)
            labeled_mask, num_buildings = label(pred_mask_np)
            
            # 2. Calculate Area
            # LEVIR-CD+ resolution is ~0.5 meters per pixel. 1 pixel = 0.25 sq meters.
            total_changed_pixels = np.sum(pred_mask_np)
            estimated_sq_meters = total_changed_pixels * 0.25
            
            # --- DISPLAY RESULTS ---
            st.divider()
            st.subheader("📊 Analysis Results")
            
            # Display Metrics dynamically
            m1, m2, m3 = st.columns(3)
            m1.metric(label="New Buildings Detected", value=num_buildings)
            m2.metric(label="Changed Pixels", value=f"{total_changed_pixels:,}")
            m3.metric(label="Est. Urbanization Area", value=f"{estimated_sq_meters:,.1f} m²")
            
            # Display Prediction Mask in black and white
            st.subheader("DeepLabV3+ Change Mask")
            st.image(pred_mask_np * 255, use_column_width=True, clamp=True)
            
            st.success("Analysis Complete! Ready for next upload.")
else:
    st.info("👈 Please upload both LEVIR-CD+ PNG images to begin.")
