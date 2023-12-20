# Photoshop Tool using Python and OpenCV

A photoshop web app deployed in streamlit having various filters and image processing capabilities built using Python and OpenCV modules.



[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
![issues](https://img.shields.io/github/issues/adarshsankarrs/PhotoshopApp)
![forks](https://img.shields.io/github/forks/adarshsankarrs/PhotoshopApp)
![stars](https://img.shields.io/github/stars/adarshsankarrs/PhotoshopApp)



## Author

- [Adarsh Sankar R S](https://github.com/adarshsankarrs)

## Table of contents
- [Features](#Features)
- [Installation](#Installation-steps)
- [Screenshots](#Screenshots)
- [Examples](#examples)
<!-- - [Demo](#demo) -->



## Features

- Filter Modules
  - Bright
  - Detail Enhance
  - Invert
  - Summer
  - Winter
  - Daylight
  - High Contrast
  - Sepia
  - Splash
  - Emboss
  - 60s TV
  - Dual tone
  - Cartoon
  - Pencil Drawing
  - Comic

- Converting Image to Sketch
- Image Inpainting
- Document Scanner
- Adding Titles to Images
- Crop Images
- Edge and contour detection
- Face detection
- Feature Detection

## Installation steps

#### Install Anaconda

```
  https://www.anaconda.com/products/individual
```


#### Create a conda environment and activate it

```
  $ conda create streamlitapp
  $ conda activate streamlitapp
```

#### Install required packages from requirements.txt

```
  # Clone this repository and cd into it
  $ cd 
  $ pip install -r requirements.txt
```

#### Run the streamlit app

```
  $ streamlit run app.py  
```




## Screenshots

### Filter Modules:

#### High Contrast:
	Input Types Accepted: jpg, jpeg, png
	(Webpage interface is same as Bright and Detail enhancement pages)

<p align="center">
<img style="display: block; margin: auto;"
src="MDimgs/a1.png"<br>
</p>


#### Bright:
	Input Types Accepted: jpg, jpeg, png

<p align="center">
<img style="display: block; margin: auto;"
src="MDimgs/a3.png"<br>
</p>



#### Detail Enhancement:
	Input Types Accepted: jpg, jpeg, png


<p align="center">
<img style="display: block; margin: auto;"
src="MDimgs/a4.png"<br>
</p>
 
<p align="center">
<img style="display: block; margin: auto;"
src="MDimgs/a5.png"<br>
</p>
 




#### Invert:
	Input Types Accepted: jpg, jpeg, png
	(Webpage interface is same as Bright and Detail enhancement pages)

<p align="center">
<img style="display: block; margin: auto;"
src="MDimgs/a6.png"<br>
</p>
 
   

#### Summer:
	Input Types Accepted: jpg, jpeg, png
	(Webpage interface is same as Bright and Detail enhancement pages)

	 
<p align="center">
<img style="display: block; margin: auto;"
src="MDimgs/a7.png"<br>
</p>
 

#### Winter:
	Input Types Accepted: jpg, jpeg, png
	(Webpage interface is same as Bright and Detail enhancement pages)

<p align="center">
<img style="display: block; margin: auto;"
src="MDimgs/a8.png"<br>
</p>
 
	
#### Daylight:
	Input Types Accepted: jpg, jpeg, png
	(Webpage interface is same as Bright and Detail enhancement pages)
	


<p align="center">
<img style="display: block; margin: auto;"
src="MDimgs/a9.png"<br>
</p>
 



#### Sepia:
	Input Types Accepted: jpg, jpeg, png
	(Webpage interface is same as Bright and Detail enhancement pages)


<p align="center">
<img style="display: block; margin: auto;"
src="MDimgs/a10.png"<br>
</p>

	   
#### Splash:
	Input Types Accepted: jpg, jpeg, png
	(Webpage interface is same as Bright and Detail enhancement pages)
Note: The splash filter only works successfully for images with objects having high contrast colors (Eg: Yellow and Blue). The image given below is a good example over which the splash filter works successfully.


 
<p align="center">
<img style="display: block; margin: auto;"
src="MDimgs/a11.png"<br>
</p>


 

#### Emboss:
	Input Types Accepted: jpg, jpeg, png
	(Webpage interface is same as Bright and Detail enhancement pages)
<p align="center">
<img style="display: block; margin: auto;"
src="MDimgs/a12.png"<br>
</p>
	   

#### 60s TV:
	Input Types Accepted: jpg, jpeg, png
	(Webpage interface is same as Bright and Detail enhancement pages)
Note: This filter also consists of 2 other input parameters the noise and threshold values. Our Webpage provides a slider widget to set these parameter values seamlessly. The image example given below shows the interface and outputs.

<p align="center">
<img style="display: block; margin: auto;"
src="MDimgs/a13.png"<br>
</p>

        

#### Dual Tone:
	Input Types Accepted: jpg, jpeg, png
	(Webpage interface is same as Bright and Detail enhancement pages)

	  
   
<p align="center">
<img style="display: block; margin: auto;"
src="MDimgs/a14.png"<br>
</p>




	   

#### Pencil Drawing:
	Input Types Accepted: jpg, jpeg, png
	(Webpage interface is same as Bright and Detail enhancement pages)
<p align="center">
<img style="display: block; margin: auto;"
src="MDimgs/a15.png"<br>
</p>
	  

#### Comic (Using K-Means):
	Input Types Accepted: jpg, jpeg, png
	(Webpage interface is same as Bright and Detail enhancement pages)

<p align="center">
<img style="display: block; margin: auto;"
src="MDimgs/a16.png"<br>
</p>




	 

#### Image to Sketch Module:
	Input Types Accepted: jpg, jpeg, png
	(Webpage interface is same as Bright and Detail enhancement pages)

<p align="center">
<img style="display: block; margin: auto;"
src="MDimgs/a18.png"<br>
</p>


       

#### Image inpainting Module:
	Input Types Accepted: jpg, jpeg, png
	(Webpage interface is same as Bright and Detail enhancement pages)


<p align="center">
<img style="display: block; margin: auto;"
src="MDimgs/a19.png"<br>
</p>




Note: This module takes 2 images as its inputs: the first benign our original image and the second is the mask image of the section to be removed or inpainted over. The images given below shows the type of input images and the output generated.

   <p align="center">
<img style="display: block; margin: auto;"
src="MDimgs/a20.png"<br>
</p>
      



#### Doc Scanner Module:
	Input Types Accepted: jpg, jpeg, png
	(Webpage interface is same as Bright and Detail enhancement pages)


<p align="center">
<img style="display: block; margin: auto;"
src="MDimgs/a21.png"<br>
</p>

               

#### Add Title to Image Module:
	Input Types Accepted: jpg, jpeg, png
	(Webpage interface is same as Bright and Detail enhancement pages)
<p align="center">
<img style="display: block; margin: auto;"
src="MDimgs/a22.png"<br>
</p>

#### Edge and contour module
<p align="center">           
<img style="display: block; margin: auto;"
src="MDimgs/a23.png"<br>
</p>

#### Crop Image Module:
	Input Types Accepted: jpg, jpeg, png
<p align="center">           
<img style="display: block; margin: auto;"
src="MDimgs/a25.png"<br>
</p>



#### Face and Feature Detection Module:
	Input Types Accepted: jpg, jpeg, png
<p align="center">           
<img style="display: block; margin: auto;"
src="MDimgs/a24.png"<br>
</p>
