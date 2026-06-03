import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile

# ----------------------------------------------------
# PAGE TITLE
# ----------------------------------------------------

st.title("🌊 Marine Plastic Detection System")

st.write(
    "Upload an underwater image and detect plastic waste."
)

# ----------------------------------------------------
# LOAD MODEL
# ----------------------------------------------------

model = YOLO("best2.pt")

# ----------------------------------------------------
# IMAGE UPLOAD
# ----------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

# ----------------------------------------------------
# IF IMAGE IS UPLOADED
# ----------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image")

    # Save temporary image
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".jpg"
    ) as tmp_file:

        image.save(tmp_file.name)

        # Run detection
        results = model.predict(
            source=tmp_file.name,
            conf=0.25
        )

    # Draw bounding boxes
    plotted_image = results[0].plot()

    # Display result
    st.image(
        plotted_image,
        caption="Image After Detection",
        channels="BGR"
    )