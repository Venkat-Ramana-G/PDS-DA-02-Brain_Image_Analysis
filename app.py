# import streamlit as st
# import cv2
# import numpy as np
# from tensorflow.keras.models import load_model
# from PIL import Image

# # Load the saved model
# model = load_model('brain_tumor_model.h5')

# # Define constants
# IMG_SIZE = 128  # Make sure this matches the size used in training

# # Define function to predict tumor presence and severity
# def predict_tumor(image, model):
#     # Resize and preprocess the image
#     image = cv2.resize(np.array(image), (IMG_SIZE, IMG_SIZE))
#     image = image / 255.0  # Normalize
#     image = np.expand_dims(image, axis=0)  # Add batch dimension

#     # Perform prediction
#     prediction = model.predict(image)

#     # Assuming the model outputs a single probability for the tumor being present
#     tumor_probability = prediction[0][0]  # Adjust index based on your model's output

#     # Determine severity
#     severity = ""
#     if tumor_probability > 0.7:
#         severity = "High Severity"
#     elif tumor_probability > 0.4:
#         severity = "Moderate Severity"
#     else:
#         severity = "Low Severity"

#     # Determine if a tumor is detected
#     tumor_detected = "Tumor Detected" if tumor_probability > 0.5 else "No Tumor Detected"

#     return tumor_detected, severity

# # Streamlit app layout
# st.title("Brain Tumor Detection")

# # Display the custom message in bold
# st.markdown("<h1 style='text-align: center; font-weight: bold;'>PDS DA 2 by 24MAI0091 & 24MAI0070</h1>", unsafe_allow_html=True)

# # File upload widget
# uploaded_file = st.file_uploader("Upload a brain MRI image", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     # Display the uploaded image
#     image = Image.open(uploaded_file)
#     st.image(image, caption="Uploaded MRI Image", use_column_width=True)

#     # Run prediction
#     if st.button("Predict"):
#         tumor_result, severity = predict_tumor(image, model)
#         st.write(tumor_result)
#         st.write("Severity Level:", severity)


import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import gdown

# Download the model if not present
gdown.download('https://drive.google.com/uc?id=1hlDW7RGM3X3KPgt2pdrBbVqUqMtmHUCG', 'brain_tumor_model.h5', quiet=False)

# Load the saved model
model = load_model('brain_tumor_model.h5')

# Define constants
IMG_SIZE = 128  # Make sure this matches the size used in training

# Define function to predict tumor presence
def predict_tumor(image, model):
    # Resize and preprocess the image
    image = cv2.resize(np.array(image), (IMG_SIZE, IMG_SIZE))
    image = image / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Perform prediction
    prediction = model.predict(image)
    severity = "High" if prediction > 0.7 else "Moderate" if prediction > 0.4 else "Low"
    return "Tumor detected" if prediction > 0.5 else "No tumor", severity

# Streamlit app layout
st.title("Brain Tumor Detection")

# File upload widget
uploaded_file = st.file_uploader("Upload a brain MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Image", use_column_width=True)

    # Run prediction
    if st.button("Predict"):
        result, severity = predict_tumor(image, model)
        st.write(result)
        if result == "Tumor detected":
            st.write(f"Severity Level: {severity}")
