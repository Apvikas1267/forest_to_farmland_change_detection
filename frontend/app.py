import streamlit as st
import os
from PIL import Image
import numpy as np
import pandas as pd
import tempfile
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.inference import load_model, predict_image
from backend.change_detector import detect_forest_to_farmland
from backend.gis_processor import overlay_mask_on_image, colorize_prediction
import rasterio
from rasterio.io import MemoryFile

def load_image_from_upload(uploaded_file):
    """Safely loads an uploaded file (JPG, PNG, TIF) into an RGB PIL Image."""
    file_bytes = uploaded_file.read()
    
    # Try rasterio first for TIF/TIFF or complex formats
    if uploaded_file.name.lower().endswith(('.tif', '.tiff')):
        try:
            with MemoryFile(file_bytes) as memfile:
                with memfile.open() as dataset:
                    img_array = dataset.read() # (bands, H, W)
                    
            # Move bands to last dimension for PIL (H, W, bands)
            img_array = np.transpose(img_array, (1, 2, 0))
            
            # Normalize to 0-255 if it's float or uint16 (like NDVI maps)
            if img_array.dtype != np.uint8:
                img_array = np.nan_to_num(img_array) # handle NaNs
                min_val = img_array.min()
                max_val = img_array.max()
                if max_val > min_val:
                    img_array = ((img_array - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                else:
                    img_array = np.zeros_like(img_array, dtype=np.uint8)
                    
            # Handle 1-band (Grayscale to RGB)
            if img_array.shape[2] == 1:
                img_array = np.repeat(img_array, 3, axis=2)
            # Handle >3 bands (Take first 3 for RGB)
            elif img_array.shape[2] > 3:
                img_array = img_array[:, :, :3]
                
            return Image.fromarray(img_array).convert('RGB')
        except Exception as e:
            print(f"Rasterio failed: {e}")
            # Fallback to PIL
            pass
            
    # Standard PIL open for JPG/PNG
    uploaded_file.seek(0)
    return Image.open(uploaded_file).convert('RGB')

# ---- Configuration ----
st.set_page_config(
    page_title="Forest-to-Farmland Change Detection",
    page_icon="🌲",
    layout="wide"
)

# ---- Load Model ----
@st.cache_resource
def get_unet_model():
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'saved_models', 'unet_eurosat.pth')
    return load_model(model_path=model_path)

model = get_unet_model()

# ---- UI Structure ----
st.title("🌲 AI-Based Forest-to-Farmland Change Detection 🌾")
st.markdown("""
This tool uses a PyTorch U-Net Deep Learning model to automatically detect locations where 
forest land has been converted into agricultural land using Before (T1) and After (T2) imagery.
""")

st.header("1. Upload Imagery")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Before Image (T1)")
    file_t1 = st.file_uploader("Upload T1 image (JPG/PNG/TIF)", type=["jpg", "png", "jpeg", "tif", "tiff"], key='t1')

with col2:
    st.subheader("After Image (T2)")
    file_t2 = st.file_uploader("Upload T2 image (JPG/PNG/TIF)", type=["jpg", "png", "jpeg", "tif", "tiff"], key='t2')

if file_t1 and file_t2:
    # Read images directly from uploaded files using the robust helper
    img_t1 = load_image_from_upload(file_t1)
    img_t2 = load_image_from_upload(file_t2)

    # Automatically resize T2 to match T1 if they are different sizes
    if img_t1.size != img_t2.size:
        img_t2 = img_t2.resize(img_t1.size, Image.Resampling.LANCZOS)
        st.warning(f"⚠️ Note: T2 image was automatically resized to perfectly match T1 dimensions ({img_t1.size[0]}x{img_t1.size[1]}).")

    st.success("Images uploaded successfully! Running AI models...")

    # Run inference
    with st.spinner("Classifying Land Cover (U-Net)..."):
        pred_t1 = predict_image(model, img_t1)
        pred_t2 = predict_image(model, img_t2)
    
    with st.spinner("Detecting Changes..."):
        conversion_mask, stats = detect_forest_to_farmland(pred_t1, pred_t2)

    # Display Results
    st.header("2. GIS Viewer")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(img_t1, caption="T1 Image", use_column_width=True)
        st.image(colorize_prediction(pred_t1), caption="T1 Land Cover", use_column_width=True)
    with c2:
        st.image(img_t2, caption="T2 Image", use_column_width=True)
        st.image(colorize_prediction(pred_t2), caption="T2 Land Cover", use_column_width=True)
    with c3:
        conversion_overlay = overlay_mask_on_image(img_t2, conversion_mask, color=(255, 0, 0))
        st.image(conversion_overlay, caption="Detected Conversion (Red)", use_column_width=True)

    st.header("3. Statistics & Analysis")
    
    s1, s2, s3, s4 = st.columns(4)
    s1.metric(label="Forest Area T1", value=f"{stats['Forest Area T1 (ha)']:.2f} ha")
    s2.metric(label="Forest Area T2", value=f"{stats['Forest Area T2 (ha)']:.2f} ha")
    s3.metric(label="Converted Area", value=f"{stats['Converted Area (ha)']:.2f} ha", delta=f"-{stats['Converted Area (ha)']:.2f} ha", delta_color="inverse")
    s4.metric(label="Conversion %", value=f"{stats['Percentage Converted (%)']:.2f} %")

    # Bar chart of land cover changes
    chart_data = pd.DataFrame({
        'Time': ['T1 (Before)', 'T2 (After)'],
        'Forest (ha)': [stats['Forest Area T1 (ha)'], stats['Forest Area T2 (ha)']],
        'Farmland (ha)': [stats['Farmland Area T1 (ha)'], stats['Farmland Area T2 (ha)']]
    }).set_index('Time')

    st.bar_chart(chart_data)

    st.header("4. Report Export")
    df_stats = pd.DataFrame([stats])
    csv = df_stats.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Results as CSV",
        data=csv,
        file_name='conversion_report.csv',
        mime='text/csv',
    )
    
    st.info("⚠️ This system provides decision support and should not be used as the sole basis for legal enforcement.")
