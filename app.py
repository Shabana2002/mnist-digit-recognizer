import streamlit as st
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
from utils.predict import predict_digit
import numpy as np

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="🎨 MNIST Digit Fun",
    layout="wide",
    page_icon="🖌️"
)

st.title("🖌️ MNIST Digit Recognizer")
st.markdown("Draw a digit (0–9) or upload an image. The AI will guess it! 🎯")

# -----------------------------
# Sidebar options
# -----------------------------
st.sidebar.header("Canvas Options")
stroke_width = st.sidebar.slider("Brush size", 5, 50, 15)
stroke_color = st.sidebar.color_picker("Brush color", "#FFFFFF")
bg_color = "#000000"  # MNIST-style background (black)
canvas_size = 280

# -----------------------------
# Session state
# -----------------------------
if "canvas_data" not in st.session_state:
    st.session_state.canvas_data = None

if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0

# -----------------------------
# Helper functions
# -----------------------------
def is_canvas_empty(img_array: np.ndarray, threshold: int = 10):
    """Check if the canvas is blank."""
    total = img_array[:, :, :3].sum()
    return total < threshold

def prepare_pil_image(img_array: np.ndarray) -> Image.Image:
    """Convert RGBA canvas NumPy array to MNIST-style grayscale PIL image."""
    # Convert to uint8
    img_array = img_array.astype("uint8")

    # Handle transparency (alpha channel)
    if img_array.shape[-1] == 4:
        rgb = img_array[:, :, :3]
        alpha = img_array[:, :, 3] / 255.0
        # Blend with black background
        for c in range(3):
            rgb[:, :, c] = rgb[:, :, c] * alpha
        img = Image.fromarray(rgb.astype("uint8"))
    else:
        img = Image.fromarray(img_array)

    # Convert to grayscale
    img = ImageOps.grayscale(img)

    # Invert if background is dark (ensure white digit on black)
    if np.mean(np.array(img)) < 127:
        img = ImageOps.invert(img)

    return img

def display_prediction(img: Image.Image, show_debug: bool = True):
    """Display model prediction and optional debug view."""
    if show_debug:
        st.image(img, caption="🧠 What the model sees", width=150)

    digit, confidence = predict_digit(img)

    # Determine color and message
    if confidence == 1.0:
        color = "green"
        emoji = "💯✅"
        msg = "I am completely sure!"
        st.balloons()  # 🎉 Confetti!
    elif confidence > 0.7:
        color = "orange"
        emoji = "❄️"
        msg = "I think it’s..."
        st.snow()  # ❄️ Snow effect for confidence > 0.7
    else:
        color = "red"
        emoji = "❌"
        msg = "Hmm… I’m not sure"

    # Display result
    st.markdown(f"### {msg} **{digit}** {emoji}")
    st.markdown(
        f"<div style='background-color:#ddd; width:100%; height:20px; border-radius:5px;'>"
        f"<div style='width:{confidence*100}%; height:100%; background-color:{color}; border-radius:5px;'></div>"
        f"</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<span style='color:{color}; font-weight:bold;'>Confidence: {confidence:.2f}</span>",
        unsafe_allow_html=True
    )

# -----------------------------
# Layout
# -----------------------------
col1, col2 = st.columns([1, 1])

with col1:
    canvas_result = st_canvas(
        fill_color=stroke_color,
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        width=canvas_size,
        height=canvas_size,
        drawing_mode="freedraw",
        key=f"canvas_widget_{st.session_state.canvas_key}"
    )

    if canvas_result.image_data is not None:
        st.session_state.canvas_data = canvas_result.image_data

with col2:
    uploaded_file = st.file_uploader("Or upload an image...", type=["png", "jpg", "jpeg"])

# -----------------------------
# Uploaded image handling
# -----------------------------
if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="📤 Uploaded Digit", width=150)
    display_prediction(img)
    uploaded_file = None

# -----------------------------
# Canvas drawing handling
# -----------------------------
if uploaded_file is None and st.session_state.canvas_data is not None:
    img_array = st.session_state.canvas_data.astype("uint8")

    if not is_canvas_empty(img_array):
        img = prepare_pil_image(img_array)
        display_prediction(img)
    else:
        st.info("✏️ Canvas is empty. Draw a digit or upload an image.")

# -----------------------------
# Clear button
# -----------------------------
if st.button("🧹 Clear Canvas"):
    st.session_state.canvas_data = None
    st.session_state.canvas_key += 1
