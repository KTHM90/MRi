from pathlib import Path
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

st.set_page_config(
    page_title="Brain Tumor MRI Classifier", page_icon="🧠", layout="wide"
)

MODEL_PATH = Path(__file__).parent / "brain_tumor_cnn.keras"

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]


@st.cache_resource
def load_model():
  if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
  return tf.keras.models.load_model(MODEL_PATH)


def preprocess_image(image):
  image = image.convert("RGB")
  image = image.resize((128, 128))
  image = np.array(image, dtype=np.float32)
  return np.expand_dims(image, axis=0)


st.title("🧠 Brain Tumor MRI Classifier")
st.write("Upload a brain MRI image to see the CNN model's prediction.")

st.warning(
    "Research and educational use only. This model is not a medical diagnostic"
    " tool."
)

try:
  model = load_model()
  st.success("CNN model loaded successfully")
except Exception as e:
  st.error(f"Could not load the model: {e}")
  st.stop()

uploaded_file = st.file_uploader(
    "Upload a brain MRI image", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
  image = Image.open(uploaded_file)

  left, right = st.columns(2)

  with left:
    st.subheader("Uploaded MRI")
    # width="stretch" makes the image fill the column width cleanly
    st.image(image, caption="Input image", width="stretch")

  with right:
    st.subheader("Prediction")

    if st.button("Classify MRI", type="primary"):
      with st.spinner("Analyzing image..."):
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image, verbose=0)[0]

        if len(predictions) != len(CLASS_NAMES):
          st.error(
              "The model output does not match the four configured classes."
          )
          st.stop()

        predicted_index = int(np.argmax(predictions))
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = float(predictions[predicted_index])

        st.metric(
            "Predicted class", predicted_class.replace("_", " ").title()
        )

        st.metric("Model score", f"{confidence * 100:.2f}%")

        st.subheader("Class scores")

        for name, score in zip(CLASS_NAMES, predictions):
          st.write(f"{name.title()}: {score * 100:.2f}%")
          st.progress(float(np.clip(score, 0.0, 1.0)))

        st.caption(
            "Scores are model outputs, not verified medical probabilities."
        )