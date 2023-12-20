import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import cv2

# path to input image is specified and
# image is loaded with imread command


# cv2.cvtColor is applied over the
# image input with applied parameters
# to convert the image in grayscale
def app():
    DEMO_IMAGE = 'imgs/scannedimg.jpeg'
    @st.cache
    def adap(img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # applying different thresholding
    # techniques on the input image


        thresh2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY,15,5)

        return thresh2

    st.title('Document enhancer')
    img_file_buffer = st.file_uploader("Upload a page ", type=[ "jpg", "jpeg",'png'])
    if img_file_buffer is not None:
            image = np.array(Image.open(img_file_buffer))
    else:
            demo_image = DEMO_IMAGE
            image = np.array(Image.open(demo_image))

    st.image(image, caption=f"Uploaded page",use_column_width= False)

    if st.button("Enhance image"):
            
            newimg = adap(image)

            st.image(
            newimg, caption=f"Enhanced image", use_column_width=False)


            
