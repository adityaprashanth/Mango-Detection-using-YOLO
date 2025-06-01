import streamlit as st
import os
import time
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import tempfile
from ultralytics import YOLO
import io

st.set_page_config(page_title="YOLO Model Comparison", layout="wide")

def main():
    st.title("YOLO Model Comparison Tool")
    
    # Sidebar for upload and settings
    with st.sidebar:
        st.header("Settings")
        
        # Device selection - moved to the top
        device_options = ["cpu", "cuda", "mps"]
        device = st.selectbox("Select Device", device_options)
        
        # Confidence threshold - moved to the top
        confidence = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
        
        # Model upload section
        st.subheader("Upload YOLO Models")
        uploaded_models = st.file_uploader(
            "Upload .pt model files",
            type=["pt"],
            accept_multiple_files=True
        )
            
    # Main content area
    st.header("Image Analysis")
    
    # Image upload section
    uploaded_image = st.file_uploader("Upload an image for inference", type=["jpg", "jpeg", "png"])
    
    if not uploaded_models:
        st.warning("Please upload at least one YOLO model (.pt file)")
        return
    
    if not uploaded_image:
        st.info("Please upload an image for analysis")
        return
    
    # Save models to temporary files
    model_paths = []
    model_names = []
    for model_file in uploaded_models:
        # Store the original filename
        model_names.append(model_file.name)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp:
            tmp.write(model_file.getvalue())
            model_paths.append(tmp.name)
    
    # Save image to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
        tmp.write(uploaded_image.getvalue())
        image_path = tmp.name
    
    # Display the original image
    original_img = cv2.imread(image_path)
    original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    st.subheader("Original Image")
    st.image(original_img_rgb, caption="Original Image", use_column_width=True)
    
    # Run button
    if st.button("Run Inference"):
        inference_results = run_inference(model_paths, model_names, image_path, confidence, device)
        
        # Display results
        display_results(inference_results, original_img)
    
    # Clean up temporary files when the app closes
    for path in model_paths:
        if os.path.exists(path):
            os.unlink(path)
    
    if os.path.exists(image_path):
        os.unlink(image_path)

def run_inference(model_paths, model_names, image_path, confidence, device):
    """
    Perform inference on the uploaded image using multiple models
    """
    st.write("Running inference with", len(model_paths), "models...")
    
    # Process each model
    inference_results = {}
    
    progress_bar = st.progress(0)
    
    for i, (model_path, model_name) in enumerate(zip(model_paths, model_names)):
        # Use original filename instead of temporary path
        
        with st.spinner(f"Processing model {i+1}/{len(model_paths)}: {model_name}"):
            try:
                # Load model
                model = YOLO(model_path)
                
                # Run inference
                start_time = time.time()
                results = model.predict(
                    source=image_path,
                    conf=confidence,
                    save=False,
                    device=device
                )
                end_time = time.time()
                inference_time = end_time - start_time
                
                # Process results
                result = results[0]
                boxes = result.boxes
                
                # Store results with original model name
                inference_results[model_name] = {
                    'num_detections': len(boxes),
                    'time': inference_time,
                    'boxes': boxes,
                    'result': result,
                    'model': model
                }
                
                st.success(f"Model {model_name}: {len(boxes)} detections in {inference_time:.4f} seconds")
                
            except Exception as e:
                st.error(f"Error during inference with model {model_name}: {str(e)}")
        
        # Update progress bar
        progress_bar.progress((i + 1) / len(model_paths))
    
    progress_bar.empty()
    
    return inference_results

def display_results(inference_results, original_img):
    """
    Display the inference results
    """
    if not inference_results:
        st.warning("No valid inference results to display")
        return
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["Detections", "Performance Comparison"])
    
    with tab1:
        st.subheader("Detection Results")
        
        # Create columns for each model
        cols = st.columns(len(inference_results))
        
        for i, (model_name, data) in enumerate(inference_results.items()):
            with cols[i]:
                # Create a copy of the original image for visualization
                img_with_boxes = original_img.copy()
                
                # Draw bounding boxes
                for box in data['boxes']:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())
                    cls_name = data['model'].names[cls]
                    
                    # Draw rectangle
                    cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Add label
                    label = f"{cls_name}: {conf:.2f}"
                    cv2.putText(img_with_boxes, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Convert to RGB for display
                img_with_boxes_rgb = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)
                
                # Display the image with bounding boxes
                st.image(img_with_boxes_rgb, caption=f"{model_name}: {data['num_detections']} detections", use_column_width=True)
                st.write(f"Inference time: {data['time']:.4f} seconds")
    
    with tab2:
        st.subheader("Performance Comparison")
        
        if len(inference_results) > 1:
            # Prepare data for comparison
            comparison_data = []
            
            for model_name, data in inference_results.items():
                comparison_data.append({
                    'Model': model_name,
                    'Detections': data['num_detections'],
                    'Inference Time (s)': data['time']
                })
            
            df_comparison = pd.DataFrame(comparison_data)
            
            # Display comparison table
            st.dataframe(df_comparison, use_container_width=True)
            
            # Create comparison charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Number of Detections")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x='Model', y='Detections', data=df_comparison, ax=ax)
                plt.xticks(rotation=45)
                st.pyplot(fig)
            
            with col2:
                st.subheader("Inference Time (seconds)")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x='Model', y='Inference Time (s)', data=df_comparison, ax=ax)
                plt.xticks(rotation=45)
                st.pyplot(fig)
        else:
            st.info("Upload multiple models to see comparison charts")

if __name__ == "__main__":
    main()