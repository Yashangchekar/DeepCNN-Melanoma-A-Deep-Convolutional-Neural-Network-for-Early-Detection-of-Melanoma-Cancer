# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 12:50:37 2024

@author: yash
"""

import streamlit as st
import cv2
import numpy as np
import pickle
# Load your model here
# Assuming you have loaded the model and named it 'model'
model = pickle.load(open('C:/Users/yash/Documents/OpenCvclass/project/Tumorclassification/model100-85.sav','rb'))
st.title('Cancer Detection App')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    input_image = cv2.imdecode(file_bytes, 1)

    # Resize, scale, and reshape image
    input_image_resized = cv2.resize(input_image, (224, 224))
    input_image_scaled = input_image_resized / 255
    input_image_reshaped = np.reshape(input_image_scaled, [1, 224, 224, 3])
    input_prediction = model.predict(input_image_reshaped)
    print(input_prediction)
    if input_prediction > 0.5:
        
       
        st.header("Malignant")
    else:
        
       
        st.header('Benign')