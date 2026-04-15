import streamlit as st
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib # To save and load your model

# --- UI SETTINGS ---
st.set_page_config(page_title="Geospatial AI - Urban Analyzer", layout="wide")

st.title("🌍 Urban vs. Vegetation Analyzer")
st.markdown("""
This tool uses a **Random Forest Classifier** to quantify land use from high-resolution satellite imagery. 
Built as part of my pivot from **Town Planning** to **Computer Science**.
""")

# --- SIDEBAR: Upload & Settings ---
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Choose a Google Earth Screenshot...", type=["jpg", "png", "jpeg"])

# --- CORE LOGIC: The Brain ---
def extract_features(tile):
    avg_color = np.mean(tile, axis=(0, 1))
    std_color = np.std(tile, axis=(0, 1))
    return np.hstack([avg_color, std_color])

# We load the model you trained earlier
# (Make sure to save your model first using: joblib.dump(model, 'my_model.pkl'))
@st.cache_resource
def load_model():
    return joblib.load('my_model.pkl')

try:
    model = load_model()
except:
    st.error("Please train and save your 'my_model.pkl' first!")

# --- MAIN INTERFACE ---
if uploaded_file is not None:
    # Convert uploaded file to OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image_rgb, use_container_width=True)
        
    with col2:
        st.subheader("AI Analysis")
        with st.spinner("Analyzing neighborhood..."):
            # Run your scanner logic here
            # (Simplified for display)
            h, w, _ = image_rgb.shape
            window_size = 64
            results = []
            
            for y in range(0, h - window_size, window_size):
                for x in range(0, w - window_size, window_size):
                    tile = image_rgb[y:y+window_size, x:x+window_size]
                    features = extract_features(tile)
                    results.append(model.predict([features])[0])
            
            veg_pct = (results.count('Vegetation') / len(results)) * 100
            urban_pct = (results.count('Urban') / len(results)) * 100
            
            # --- METRICS ---
            st.metric("Vegetation Cover", f"{veg_pct:.1f}%")
            st.metric("Urban / Concrete Cover", f"{urban_pct:.1f}%")
            
            # --- ACTIONABLE INSIGHT (The "Town Planner" Touch) ---
            if veg_pct < 20:
                st.warning("⚠️ Critical: Low greenery detected. High Risk of Urban Heat Island effect.")
            else:
                st.success("✅ Healthy vegetation balance for this altitude.")