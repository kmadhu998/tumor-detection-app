import os
import streamlit as st
import numpy as np
from PIL import Image
import cv2
import plotly.graph_objects as go
import requests

# Try imports for TensorFlow / Keras and image preprocessing
TF_AVAILABLE = False
try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing import image as tf_image
    TF_AVAILABLE = True
except Exception:
    # TensorFlow not available (common in some cloud envs)
    try:
        # standalone Keras
        from keras.models import load_model as keras_load_model
        from keras.preprocessing import image as tf_image
    except Exception:
        tf = None
        keras_load_model = None
        tf_image = None

# Page configuration
st.set_page_config(
    page_title="Soft Tissue Tumor Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)


# ===== Model loading utility =====
@st.cache_resource
def download_file(url: str, dest_path: str, timeout: int = 300):
    """Download a file to dest_path (used for model download)."""
    r = requests.get(url, stream=True, timeout=timeout)
    r.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    return dest_path


@st.cache_resource
def load_model_cached(model_filename: str = "tumor_transfer_model.h5"):
    """
    Load model from local path; if missing, try to download using st.secrets["MODEL"]["model_url"].
    Returns a loaded model (either TF Keras model or standalone Keras model) or None on failure.
    """
    # 1) local file
    if os.path.exists(model_filename):
        try:
            if TF_AVAILABLE:
                return tf.keras.models.load_model(model_filename)
            else:
                # use standalone keras loader if available
                if keras_load_model is not None:
                    return keras_load_model(model_filename)
        except Exception as e:
            st.warning(f"Local model found but failed to load: {e}")

    # 2) try download from secrets
    try:
        model_url = st.secrets.get("MODEL", {}).get("model_url")
    except Exception:
        model_url = None

    if model_url:
        try:
            download_file(model_url, model_filename)
            if TF_AVAILABLE:
                return tf.keras.models.load_model(model_filename)
            else:
                if keras_load_model is not None:
                    return keras_load_model(model_filename)
        except Exception as e:
            st.error(f"Failed to download or load model from URL: {e}")
            return None

    st.error(
        "Model file not found locally and MODEL.model_url is not set in Streamlit secrets. "
        "Either add `tumor_transfer_model.h5` to the repo or configure the MODEL.model_url secret."
    )
    return None


# ===== Prediction helper =====
def predict_tumor(model, img):
    """
    img: PIL Image
    returns: numpy array of probabilities
    """
    img_resized = img.resize((224, 224))
    # use tf_image reference (either from tensorflow.keras or keras.preprocessing)
    x = tf_image.img_to_array(img_resized)
    x = np.expand_dims(x, axis=0) / 255.0
    # model.predict should work for both Keras and TF Keras models
    preds = model.predict(x, verbose=0)
    return preds[0]


# ===== Grad-CAM: only when TensorFlow is available =====
def generate_gradcam(model, img_array, class_idx):
    """
    img_array: a batch-shaped numpy array (1, H, W, C) scaled [0,1]
    class_idx: integer class index
    Returns heatmap (H,W) or raises Exception if unavailable.
    """
    if not TF_AVAILABLE:
        raise RuntimeError("Grad-CAM requires TensorFlow (not available in this environment).")

    try:
        # locate nested base model if present
        base_model = None
        for layer in model.layers:
            if hasattr(layer, "layers"):
                base_model = layer
                break
        if base_model is None:
            base_model = model

        # find a conv layer
        last_conv_layer_name = None
        for layer_name in ['out_relu', 'Conv_1', 'Conv_1_bn', 'top_activation']:
            try:
                _ = base_model.get_layer(layer_name)
                last_conv_layer_name = layer_name
                break
            except Exception:
                continue

        if last_conv_layer_name is None:
            # fallback: search for the last layer with 4D output
            for layer in reversed(base_model.layers):
                try:
                    output_shape = layer.output_shape
                    if isinstance(output_shape, tuple) and len(output_shape) == 4:
                        last_conv_layer_name = layer.name
                        break
                except Exception:
                    continue

        if last_conv_layer_name is None:
            raise ValueError("Could not find a suitable convolutional layer for Grad-CAM.")

        # build grad model
        grad_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=[base_model.get_layer(last_conv_layer_name).output, model.output]
        )

        # ensure img_array is a tensor
        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_tensor)
            loss = predictions[:, class_idx]

        grads = tape.gradient(loss, conv_outputs)
        if grads is None:
            raise ValueError("Could not compute gradients for Grad-CAM.")

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0].numpy()
        pooled_grads = pooled_grads.numpy()

        for i in range(pooled_grads.shape[-1]):
            conv_outputs[:, :, i] *= pooled_grads[i]

        heatmap = np.mean(conv_outputs, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        if np.max(heatmap) != 0:
            heatmap /= np.max(heatmap)

        return heatmap

    except Exception as e:
        raise Exception(f"Grad-CAM generation failed: {str(e)}")


# ===== Main app =====
def main():
    # Header
    st.markdown('<p class="main-header">🧠 Soft Tissue Tumor Detection System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Medical Image Analysis</p>', unsafe_allow_html=True)

    # Sidebar info
    with st.sidebar:
        st.header("ℹ️ About")
        st.info("""
        This AI system detects and classifies brain tumors from MRI scans using deep learning.

        **Tumor Types:**
        - Glioma
        - Meningioma
        - Pituitary
        - No Tumor
        """)

        st.warning("""
        ⚠️ **Medical Disclaimer**

        This tool is for research and educational purposes only. 
        It should NOT be used as a substitute for professional medical diagnosis.
        Always consult qualified healthcare professionals.
        """)

        st.header("📊 Model Info")
        st.write("**Architecture:** MobileNetV2 (expected)")
        st.write("**Training Data:** Brain MRI Scans")
        st.write("**Input Size:** 224x224 pixels")

    # Load model
    model = load_model_cached()

    if model is None:
        st.error("⚠️ Model not found! Please ensure 'tumor_transfer_model.h5' is accessible (repo root or MODEL.model_url in Secrets).")
        st.stop()

    # File upload
    st.header("📤 Upload MRI Scan")
    uploaded_file = st.file_uploader(
        "Choose a brain MRI image...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a brain MRI scan in JPG or PNG format"
    )

    # Class names
    class_names = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']

    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📷 Uploaded Image")
            img = Image.open(uploaded_file).convert("RGB")
            st.image(img, use_container_width=True)

        if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
            with st.spinner("Analyzing MRI scan..."):
                # Prediction
                try:
                    predictions = predict_tumor(model, img)
                except Exception as e:
                    st.error(f"Failed to run prediction: {e}")
                    return

                predicted_class_idx = int(np.argmax(predictions))
                predicted_class = class_names[predicted_class_idx]
                confidence = float(predictions[predicted_class_idx] * 100)

                # Display results
                with col2:
                    st.subheader("📊 Analysis Results")
                    if confidence > 80:
                        st.success(f"**Prediction:** {predicted_class}")
                    elif confidence > 60:
                        st.warning(f"**Prediction:** {predicted_class}")
                    else:
                        st.error(f"**Prediction:** {predicted_class} (Low confidence)")

                    st.metric("Confidence Score", f"{confidence:.2f}%")

                    # Confidence gauge
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=confidence,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        gauge={
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 60], 'color': "lightgray"},
                                {'range': [60, 80], 'color': "gray"},
                                {'range': [80, 100], 'color': "lightblue"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 90
                            }
                        },
                        title={'text': "Confidence Level"}
                    ))
                    fig.update_layout(height=250)
                    st.plotly_chart(fig, use_container_width=True)

                # Probability bar chart
                st.subheader("📈 Detailed Probability Distribution")
                prob_data = {class_names[i]: float(predictions[i] * 100) for i in range(len(class_names))}
                fig2 = go.Figure(data=[go.Bar(
                    x=list(prob_data.keys()),
                    y=list(prob_data.values()),
                    marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
                    text=[f"{v:.2f}%" for v in prob_data.values()],
                    textposition='auto',
                )])
                fig2.update_layout(title="Probability for Each Class", xaxis_title="Tumor Type", yaxis_title="Probability (%)",
                                   yaxis_range=[0, 100], height=400)
                st.plotly_chart(fig2, use_container_width=True)

                # Grad-CAM visualization (guarded)
                st.subheader("🔥 Grad-CAM Heatmap (Region of Interest)")
                with st.spinner("Generating heatmap..."):
                    try:
                        # Prepare array for gradcam function
                        img_resized = img.resize((224, 224))
                        img_array = tf_image.img_to_array(img_resized)
                        img_array = np.expand_dims(img_array, axis=0) / 255.0

                        if TF_AVAILABLE:
                            heatmap = generate_gradcam(model, img_array, predicted_class_idx)

                            # Convert original image to array for overlay
                            orig_img = np.array(img)
                            heatmap_resized = cv2.resize(heatmap, (orig_img.shape[1], orig_img.shape[0]))
                            heatmap_colored = np.uint8(255 * heatmap_resized)
                            heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
                            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

                            orig_img_array = np.array(orig_img)
                            if len(orig_img_array.shape) == 2:
                                orig_img_array = cv2.cvtColor(orig_img_array, cv2.COLOR_GRAY2RGB)

                            superimposed = cv2.addWeighted(orig_img_array, 0.6, heatmap_colored, 0.4, 0)

                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.image(orig_img, caption="Original", use_container_width=True)
                            with col2:
                                st.image(heatmap_resized, caption="Heatmap", use_container_width=True, clamp=True)
                            with col3:
                                st.image(superimposed, caption="Overlay", use_container_width=True)

                            st.info("🔍 The heatmap highlights regions the AI focused on for classification.")
                        else:
                            st.warning("Grad-CAM visualization is unavailable because TensorFlow is not installed in this environment.")
                            st.info("Predictions are still provided above.")
                    except Exception as e:
                        st.warning(f"⚠️ Could not generate Grad-CAM visualization: {str(e)}")
                        st.info("Prediction results are still shown above.")

                # Recommendations
                st.subheader("💡 Recommendations")
                if "No Tumor" in predicted_class and confidence > 80:
                    st.success("✅ No tumor detected with high confidence. Regular check-ups recommended.")
                elif confidence < 60:
                    st.warning("⚠️ Low confidence prediction. Please consult a radiologist for expert analysis.")
                else:
                    st.warning(f"⚠️ {predicted_class} detected. Immediate consultation with a neurologist/neurosurgeon is recommended.")
    else:
        st.info("""
        👆 **Get Started:**
        1. Upload a brain MRI scan using the file uploader above
        2. Click the "Analyze Image" button
        3. View the AI analysis results and heatmap visualization

        Supported formats: JPG, JPEG, PNG
        """)

    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666; padding: 1rem;'>
            <p>🏥 Soft Tissue Tumor Detection System | For research and educational purposes only</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()


