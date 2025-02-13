import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import joblib


@st.cache_resource
def load_model():
    model_data = joblib.load("cnn_model.joblib")
    model = tf.keras.models.model_from_json(model_data["architecture"])
    model.set_weights(model_data["weights"])
    model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    return model

model = load_model()

def preprocess_image(image):
    image = image.resize((128, 128)) 
    image = np.array(image)
    if image.shape[-1] == 4:  
        image = image[..., :3]
    image = image / 255.0  
    image = np.expand_dims(image, axis=0) 
    return image


st.title("POTATO LEAF DISEASE PREDICTION")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction, axis=1)[0]
    

    class_names = {0: "Healthy", 1: "Early Blight", 2: "Late Blight"}
    
    st.write(f"**Prediction:** {class_names[predicted_class]}")
    st.write("**Confidence:** {:.2f}%".format(100 * np.max(prediction)))