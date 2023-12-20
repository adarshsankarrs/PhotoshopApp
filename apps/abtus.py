import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image

def app():
    # im1 = Image.open('/Users/ashwinv/Documents/SEM5/Signal/project/code/apps/VCODE.png')

    # st.header('Meet the Team')
    # st.image(im1, width=750)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Ashwin V")
        st.markdown('*AM.EN.U4AIE19018*')
        st.image("propics/ash.jpg")

    with col2:
        st.subheader("Devagopal AM")
        st.markdown('*AM.EN.U4AIE19025*')
        st.image("propics/deva.jpg")

    with col3:
        st.subheader("Vishal Menon")
        st.markdown('*AM.EN.U4AIE19070*')
        st.image("propics/Vishal.png")

    st.markdown('<center>Department of Computer Science and Engineering </center>', unsafe_allow_html=True)
    st.markdown("<center> Amrita Viswa Vidyapeetham </center>", unsafe_allow_html=True)
