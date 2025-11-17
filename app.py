import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import plotly.graph_objects as go
from tensorflow.keras.preprocessing import image

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

# Load model with caching
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('tumor_transfer_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Prediction function
def predict_tumor(model, img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    predictions = model.predict(img_array, verbose=0)
    return predictions[0]

# Grad-CAM function
def generate_gradcam(model, img_array, class_idx):
    try:
        base_model = None
        for layer in model.layers:
            if hasattr(layer, 'layers'):
                base_model = layer
                break
        
        if base_model is None:
            base_model = model
        
        last_conv_layer_name = None
        for layer_name in ['out_relu', 'Conv_1', 'Conv_1_bn', 'top_activation']:
            try:
                layer = base_model.get_layer(layer_name)
                last_conv_layer_name = layer_name
                break
            except:
                continue
        
        if last_conv_layer_name is None:
            for layer in reversed(base_model.layers):
                try:
                    if hasattr(layer, 'output_shape'):
                        output_shape = layer.output_shape
                        if isinstance(output_shape, tuple) and len(output_shape) == 4:
                            last_conv_layer_name = layer.name
                            break
                except:
                    continue
        
        if last_conv_layer_name is None:
            raise ValueError("Could not find a suitable convolutional layer")
        
        grad_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=[base_model.get_layer(last_conv_layer_name).output, model.output]
        )
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            class_channel = predictions[:, class_idx]
        
        grads = tape.gradient(class_channel, conv_outputs)
        
        if grads is None:
            raise ValueError("Gradients could not be computed")
        
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

# Main app
def main():
    st.markdown('<p class="main-header">🧠 Soft Tissue Tumor Detection System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Medical Image Analysis</p>', unsafe_allow_html=True)

    with st.sidebar:
        st.header("ℹ️ About")
        st.info("""
        Detects brain tumors from MRI scans using deep learning.
        
        **Detected Tumor Types:**
        - Glioma
        - Meningioma
        - Pituitary
        - No Tumor
        """)
        
        st.warning("""
        ⚠️ Medical Disclaimer  
        Research & educational purposes only.
        """)
        
        st.header("📊 Model Info")
        st.write("**Architecture:** MobileNetV2")
        st.write("**Training Data:** Brain MRI Scans")
        st.write("**Input Size:** 224x224 pixels")

    model = load_model()
    if model is None:
        st.error("Model not found.")
        st.stop()

    st.header("📤 Upload MRI Scan")
    uploaded_file = st.file_uploader("Choose an MRI image...", type=['jpg', 'jpeg', 'png'])
    
    class_names = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']

    if uploaded_file is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📷 Uploaded Image")
            img = Image.open(uploaded_file)
            st.image(img, use_container_width=True)

        if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
            with st.spinner("Analyzing..."):
                
                predictions = predict_tumor(model, img)
                predicted_class_idx = np.argmax(predictions)
                predicted_class = class_names[predicted_class_idx]
                confidence = predictions[predicted_class_idx] * 100

                with col2:
                    st.subheader("📊 Results")

                    if confidence > 80:
                        st.success(f"Prediction: **{predicted_class}**")
                    elif confidence > 60:
                        st.warning(f"Prediction: **{predicted_class}**")
                    else:
                        st.error(f"Prediction: **{predicted_class}** (Low confidence)")

                    st.metric("Confidence Score", f"{confidence:.2f}%")

                # --------------------------------------------------------------------
                # ⭐ IF TUMOR DETECTED → Show Graph + Heatmap
                # ⭐ IF NO TUMOR → Hide BOTH
                # --------------------------------------------------------------------
                if predicted_class.strip().lower() != "no tumor":

                    # ----------------- CONFIDENCE GAUGE -----------------
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=confidence,
                        gauge={
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                        },
                        title={'text': "Confidence Level"}
                    ))
                    fig.update_layout(height=250)
                    st.plotly_chart(fig, use_container_width=True)

                    # ---------------- PROBABILITY BAR CHART ---------------
                    st.subheader("📈 Detailed Probability Distribution")
                    prob_data = {class_names[i]: predictions[i] * 100 for i in range(len(class_names))}

                    fig2 = go.Figure(data=[
                        go.Bar(
                            x=list(prob_data.keys()),
                            y=list(prob_data.values()),
                            orientation="v",   # <-- fixes sideways bars
                            marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
                            text=[f"{v:.2f}%" for v in prob_data.values()],
                            textposition='auto',
                        )
                    ])
                    fig2.update_layout(
                        xaxis_title="Tumor Type",
                        yaxis_title="Probability (%)",
                        yaxis_range=[0, 100],
                        height=400
                    )
                    st.plotly_chart(fig2, use_container_width=True)

                    # -------------------- GRAD-CAM ------------------------
                    st.subheader("🔥 Grad-CAM Heatmap")
                    try:
                        img_resized = img.resize((224, 224))
                        img_array = image.img_to_array(img_resized)
                        img_array = np.expand_dims(img_array, axis=0) / 255.0

                        heatmap = generate_gradcam(model, img_array, predicted_class_idx)

                        orig_img = np.array(img)
                        heatmap_resized = cv2.resize(heatmap, (orig_img.shape[1], orig_img.shape[0]))
                        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
                        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

                        superimposed = cv2.addWeighted(orig_img, 0.6, heatmap_colored, 0.4, 0)

                        c1, c2, c3 = st.columns(3)
                        c1.image(orig_img, caption="Original", use_container_width=True)
                        c2.image(heatmap_resized, caption="Heatmap", use_container_width=True, clamp=True)
                        c3.image(superimposed, caption="Overlay", use_container_width=True)

                    except Exception as e:
                        st.warning("Could not generate heatmap.")

                # --------------------------------------------------------------------
                # ⭐ END OF CONDITIONAL SECTION ⭐
                # --------------------------------------------------------------------

                st.subheader("💡 Recommendations")
                if predicted_class.strip().lower() == "no tumor":
                    st.success("No tumor detected. Regular check-ups recommended.")
                elif confidence < 60:
                    st.warning("Low confidence. Consult a radiologist for confirmation.")
                else:
                    st.warning(f"{predicted_class} detected. Please consult a neurologist.")

    else:
        st.info("""
        👆 Upload an MRI scan to begin analysis.
        """)

    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666; padding: 1rem;'>
            <p>🏥 Soft Tissue Tumor Detection System</p>
            <p style='font-size: 0.9rem;'>For research and educational purposes only</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
