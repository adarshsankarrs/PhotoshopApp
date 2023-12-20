import streamlit as st
from PIL import Image
import cv2 
import numpy as np
import copy
DEMO_IMAGE = 'imgs/Person.jpg'

def app():

    def load_image():
        img_file_buffer = st.file_uploader("Upload an image", type=[ "jpg", "jpeg",'png'])
        if img_file_buffer is not None:
                image = np.array(Image.open(img_file_buffer))
        else:
            demo_image = DEMO_IMAGE
            image = np.array(Image.open(demo_image))

        st.image(image, caption=f"Original Image",use_column_width= True)
        return image
    

    def feature_detection():
        image = load_image()
        img2 = copy.deepcopy(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()    
        keypoints = sift.detect(gray, None)
        
        st.write("Number of keypoints Detected: ",len(keypoints))
        image = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        st.image(image, use_column_width=True,clamp = True)
        
        
        st.write("FAST")
        image_fast = img2
        gray = cv2.cvtColor(image_fast, cv2.COLOR_BGR2GRAY)
        fast = cv2.FastFeatureDetector_create()
        keypoints = fast.detect(gray, None)
        st.write("Number of keypoints Detected: ",len(keypoints))
        image_  = cv2.drawKeypoints(image_fast, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        st.image(image_, use_column_width=True,clamp = True)
    
    feature_detection()
