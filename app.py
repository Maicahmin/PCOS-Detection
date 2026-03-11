import streamlit as st
from ultralytics import YOLO
from PIL import Image

# Load YOLO model
model = YOLO("best.pt")

st.title("PCOS Detection from Ultrasound Images")

st.write("Upload an ultrasound image to detect ovarian follicles.")

uploaded_file = st.file_uploader("Upload Ultrasound Image", type=["jpg","jpeg","png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Run detection
    results = model(image)

    for r in results:
        st.image(r.plot(), caption="Detection Result")