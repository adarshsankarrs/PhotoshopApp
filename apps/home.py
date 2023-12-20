import streamlit as st
from PIL import Image
def app():
    image = Image.open('imgs/lake.jpeg')
    st.image(image, caption='Welcome to our webapp!', use_column_width=True)
    st.subheader('19AIE303	Signal And Image Processing Project')
    st.subheader("Group 9", anchor=None)
    st.subheader("Photoshop tool using Python and OpenCV", anchor=None)
    st.subheader("Team Members", anchor=None)
    st.markdown('Ashwin V   &emsp;&emsp; &emsp; &emsp;19018  <br> Devagopal AM &emsp; &emsp;19025  <br> Vishal Menon &nbsp;&nbsp;&emsp; &emsp;19070', unsafe_allow_html=True)
    st.subheader("Submitted to", anchor=None)
    st.markdown("Dr. Uma G")
