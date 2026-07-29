import streamlit as st
import requests


API_URL = "api_url"

# Page Setup
st.set_page_config(page_title="Caltech Classifier", page_icon="🖼️", layout="centered")

st.title("🖼️ Caltech Image Classifier")
st.write("Upload an image to classify it using the MLflow PyTorch backend.")

# File Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    
    # Classify Button
    if st.button("Classify Image", type="primary", use_container_width=True):
        with st.spinner("Running inference via FastAPI..."):
            try:
                # Prepare the file payload for FastAPI
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                
                # Call the backend API
                response = requests.post(API_URL, files=files)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    st.subheader("Predictions")
                    
                    # Iterate through the predictions and display progress bars
                    for pred in data["predictions"]:
                        label = pred["label"]
                        confidence = pred["confidence"]
                        
                        # Layout for label and percentage
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write(f"**{label.title()}**")
                        with col2:
                            st.write(f"{confidence * 100:.1f}%")
                            
                        # Visual confidence bar
                        st.progress(float(confidence))
                    
                    # Display metrics from MLflow/Backend
                    st.divider()
                    metrics = data["metrics"]
                    st.caption(
                        f"⚡ **Backend Latency:** {metrics['total_backend_ms']} ms | "
                        f"📦 **Model Source:** `{metrics['source_run']}`"
                    )
                else:
                    st.error(f"API Error {response.status_code}: {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("🚨 Failed to connect to the backend API. Please ensure the FastAPI Docker container is running on port 8000.")