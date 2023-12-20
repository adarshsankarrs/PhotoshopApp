# organizing imports
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import cv2

def app():
        DEMO_IMAGE = 'imgs/impai.png'
        DEMO_IMAGE_MASK = 'imgs/maskn.png'
        @st.cache
        def inpaintt(img,mask):
                #inpainting using the inpaint function
                dst = cv2.inpaint(img,mask,3,cv2.INPAINT_TELEA)
                return dst

        st.title('Image inpaint')
        #uploading the file
        img_file_buffer = st.file_uploader("Upload an image file", type=[ "jpg", "jpeg",'png'])
        if img_file_buffer is not None:
                #converting to numpy array using PIL
                image = np.array(Image.open(img_file_buffer))
        else:
            demo_image = DEMO_IMAGE
            image = np.array(Image.open(demo_image))

                
        img_file_mask = st.file_uploader("Upload the mask file", type=[ "jpg", "jpeg",'png'])
        if img_file_mask is not None:
                mask = Image.open(img_file_mask)
                mask = ImageOps.grayscale(mask)
                mask = np.array(mask) 
        else:
                demo_image_mask = DEMO_IMAGE_MASK
                mask = Image.open(demo_image_mask)
                mask = ImageOps.grayscale(mask)
                mask = np.array(mask)
        st.image(image, caption=f"Original Image",use_column_width= False)
        st.image(mask, caption=f"Mask Image",use_column_width= False)

        if st.button("inpaint image"):

                newimg = inpaintt(image,mask)
                st.image(newimg, caption=f"Inpainted Image",use_column_width= False)



