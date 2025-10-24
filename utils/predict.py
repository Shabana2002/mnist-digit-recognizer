import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps

# -----------------------------
# Load the model (only once)
# -----------------------------
model = load_model('model/mnist_cnn.h5')

# -----------------------------
# Helper: Preprocess image
# -----------------------------
def preprocess_image(img: Image.Image):
    """Preprocess a PIL image for MNIST CNN model."""
    # 1️⃣ Convert to grayscale
    img = img.convert("L")

    # 2️⃣ Invert (MNIST digits are white-on-black)
    img = ImageOps.invert(img)

    # 3️⃣ Convert to numpy array
    img_array = np.array(img)

    # 4️⃣ Crop the digit region (remove excess black borders)
    non_empty = np.where(img_array > 10)
    if non_empty[0].size > 0:
        top, bottom = non_empty[0].min(), non_empty[0].max()
        left, right = non_empty[1].min(), non_empty[1].max()
        img_array = img_array[top:bottom + 1, left:right + 1]

    # 5️⃣ Resize cropped image to fit 20x20 box (preserves aspect ratio)
    img = Image.fromarray(img_array)
    img.thumbnail((20, 20), Image.Resampling.LANCZOS)

    # 6️⃣ Create new 28x28 black canvas and paste centered digit
    new_img = Image.new("L", (28, 28), color=0)
    paste_x = (28 - img.width) // 2
    paste_y = (28 - img.height) // 2
    new_img.paste(img, (paste_x, paste_y))

    # 7️⃣ Normalize (0–1) and reshape for CNN input
    img_array = np.array(new_img).astype("float32") / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    return img_array

# -----------------------------
# Main: Predict digit
# -----------------------------
def predict_digit(img: Image.Image):
    """Predict the digit in a PIL image and return (digit, confidence)."""
    img_array = preprocess_image(img)
    pred = model.predict(img_array, verbose=0)
    digit = int(np.argmax(pred))
    confidence = float(np.max(pred))
    return digit, confidence
