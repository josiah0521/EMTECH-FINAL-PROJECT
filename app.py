import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

model.save('model.hdf5')


@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('model.hdf5')
    return model

model = load_model()

st.write("""
# Flower Detection Classifier
""")

file = st.file_uploader("Choose plant photo from computer", type=["jpg", "png"])

def import_and_predict(image_data, model):
    size = (64, 64)  # Ensure this matches your model's input size
    image = ImageOps.fit(image_data, size)
    img = np.asarray(image)
    img = img / 255.0  # Normalize the image if your model expects normalized input
    img_reshape = np.expand_dims(img, axis=0)  # Add batch dimension
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file).convert('RGB')  # Ensure image is in RGB format
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    class_names = ['dandelion', 'daisy', 'rose', 'sunflower', 'tulip']
    string = "OUTPUT : " + class_names[np.argmax(prediction)]
    st.success(string)
