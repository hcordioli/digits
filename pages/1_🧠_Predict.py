import streamlit as st
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os

st.markdown("# Load image and predict its class")

col1, col2 = st.columns(2)  # 1: Select File button;  2: Predict Button
col3, col4 = st.columns(2)  # 3: Loaded Image; 4: Prediction

filepath = "./data/*.jpeg" 
file_list = [os.path.basename(f) for f in glob.glob(filepath)]

with col1:
    picked_image = st.selectbox('Pick a file:', file_list)
    input_filename = ".\data\\" + picked_image
    img=mpimg.imread(input_filename)

with col2:
    predict_button = st.button("Predict",disabled= not st.session_state.get("model_loaded"))
    if (predict_button):
        img = img/255.0
        img = img.reshape(-1,28,28,1)
        y_pred = st.session_state['hdr_model'].predict(img)
        prediction = np.argmax(y_pred, axis=1)
        st.write(prediction)
        
with col3:
    st.image(img, caption='Handwriten image')
    
