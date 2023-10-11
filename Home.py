import streamlit as st
import tensorflow as tf

st.set_page_config(
    page_title="Handwriten Digit Recognizer",
    page_icon="🧠",
)

if 'model_loaded' not in st.session_state:
    st.session_state['model_loaded'] = False
        
st.write("# Welcome to Handwriten Digit Recognizer! 🧠")
#st.sidebar.success("Home")

st.markdown(
    """
    #Handwriten Digit Recognizer
"""
)

load_model = st.button("Load Model")
if load_model:
    try:
        fname = "./data/digit_recognizer.h5"
        hdr_model = tf.keras.saving.load_model(fname)
        st.session_state['model_loaded'] = True
        if 'hdr_model' not in st.session_state:
            st.session_state['hdr_model'] = hdr_model
        st.write ("Now you can predict an image")
    except OSError:
        err_msg = "Could not find model file: "+ fname
        st.write (err_msg)
