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
st.set_page_config(page_title="Urban Change Detection AI", page_icon="🛰️", layout="wide")

st.title("🛰️ SOTA Urban Change Detection")
st.markdown("DeepLabV3+ with EfficientNet-b4 backbone. Optimized for LEVIR-CD+ native resolution.")

# --- 2. ROBUST IMAGE LOADER ---
def load_image_robust(file):
    try:
        img_array = tiff.imread(file)
    except:
        return Image.open(file).convert("RGB")
    
    if len(img_array.shape) == 3 and img_array.shape[-1] > 3:
        img_array = img_array[:, :, :3]
    if img_array.dtype == np.uint16:
        img_array = (img_array / 65535.0 * 255).astype(np.uint8)
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array]*3, axis=-1)
    return Image.fromarray(img_array)

# --- 3. MODEL LOADER (CACHED) ---
@st.cache_resource
def load_model():
    model_path = 'best_levir_deeplabv3_effb4.pth'
    file_id = '1iV91GcRBVgfu39IA4CVkQ4rH_iw6W8PV' 
    if not os.path.exists(model_path):
        with st.spinner("Fetching AI weights..."):
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

# --- 4. SIDEBAR SETTINGS ---
st.sidebar.header("Control Panel")
st.sidebar.markdown("Adjust sensitivity to tune the detection engine.")
threshold = st.sidebar.slider("Detection Sensitivity", 0.1, 0.9, 0.5, 0.05)

file1 = st.sidebar.file_uploader("Upload Time 1 (Before)", type=['png', 'jpg', 'tif'])
file2 = st.sidebar.file_uploader("Upload Time 2 (After)", type=['png', 'jpg', 'tif'])

# --- 5. MAIN LOGIC ---
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
        
    if st.button("🚀 Run Analysis", use_container_width=True):
        with st.spinner("Processing 0-255 Native Tensors..."):
            # 1. Resize for model input (Native scale)
            t1_img = image1.resize((512, 512))
            t2_img = image2.resize((512, 512))
            
            # 2. Convert to Numpy Float32 (Keep 0-255 range)
            t1_np = np.array(t1_img).astype(np.float32)
            t2_np = np.array(t2_img).astype(np.float32)
            
            # 3. Create Tensors
            x1 = torch.from_numpy(t1_np).permute(2, 0, 1)
            x2 = torch.from_numpy(t2_np).permute(2, 0, 1)
            
            # 4. Early Fusion Stack
            input_tensor = torch.cat([x1, x2], dim=0).unsqueeze(0)
            
            # 5. Inference
            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.sigmoid(output).squeeze().cpu().numpy()
                mask = (probs > threshold).astype(np.uint8)
            
            # 6. Metrics Calculation
            _, num_buildings = label(mask)
            area = np.sum(mask) * 0.25 
            
            st.divider()
            st.subheader("📊 Analysis Results")
            m1, m2 = st.columns(2)
            m1.metric("Buildings Detected", num_buildings)
            m2.metric("Est. Urbanization Area", f"{area:,.1f} m²")
            
            # --- THE HIGHLIGHT OVERLAY (Visual Polish) ---
            st.subheader("AI Analysis: Building Highlights")
            
            # Use PIL for clean alpha blending (Yellow Overlay)
            base = Image.fromarray(np.array(t2_img)).convert("RGBA")
            yellow_mask = Image.new("RGBA", base.size, (255, 255, 0, 0))
            mask_pixels = yellow_mask.load()
            
            for y in range(512):
                for x in range(512):
                    if mask[y, x] == 1:
                        # Bright yellow with semi-transparency
                        mask_pixels[x, y] = (255, 255, 0, 160) 

            combined = Image.alpha_composite(base, yellow_mask)
            
            st.image(combined, use_column_width=True, caption=f"Detection Overlay (Confidence Threshold: {threshold})")
            st.success(f"Analysis Complete! Successfully identified {num_buildings} changes.")
            
else:
    st.info("👈 Upload LEVIR-CD+ images in the sidebar to begin analysis.")
