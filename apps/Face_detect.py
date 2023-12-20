import streamlit as st
from PIL import Image
import cv2 
import numpy as np
import copy


def app():
    DEMO_IMAGE = 'imgs/Person.jpg'

    def face_detection():
    
        st.header("Face Detection using haarcascade")

        img_file_buffer = st.file_uploader("Upload an image", type=[ "jpg", "jpeg",'png'])
        if img_file_buffer is not None:
                image = np.array(Image.open(img_file_buffer))
        else:
            demo_image = DEMO_IMAGE
            image = np.array(Image.open(demo_image))

        st.image(image, caption=f"Original Image",use_column_width= True)
        
        image2 = image

        face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(image2)
        print(f"{len(faces)} faces detected in the image.")
        for x, y, width, height in faces:
            cv2.rectangle(image2, (x, y), (x + width, y + height), color=(255, 0, 0), thickness=2)
        
        st.image(image2, use_column_width=True,clamp = True)

    face_detection() 
    
