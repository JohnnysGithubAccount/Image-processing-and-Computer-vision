![image](https://github.com/user-attachments/assets/9d83f295-c844-48b7-8fdc-36454e6950af)![image](https://github.com/user-attachments/assets/0d776186-0b2a-4929-9fce-771af1e9bf9a)# Image Processing and Computer Vision

Welcome to my repository for Image Processing and Computer Vision projects! This repository contains two main projects that demonstrate various techniques and applications in the field.

## ⚙️ Technologies Used
- Python
- OpenCV
- NumPy
- Matplotlib

## 📚 Table of Contents
- [1. Simple Stereo Camera Setup](#1-simple-stereo-camera-setup)
- [2. Object Counting in Video Processing](#2-object-counting-in-video-processing)

## 📂 Contents

### 1. Simple Stereo Camera Setup
In this project, I used my phone to simulate a stereo camera setup to measure distance of an object. The main objectives of this project are:
- **Camera Calibration**: Performed calibration for my phone's camera to improve the accuracy of depth calculations. Using 28 images of chessboard and OpenCV to find my phone camera matrix in order to extract the fx for using later in the depth calculation.

![image](https://github.com/user-attachments/assets/01daab1d-6450-4595-91d2-4b280d411cd3)

- **Theory**: A little bit about the theory, this use Triangulation with 2 camera principle for identify the object depth. Theorically, each pixel from an image will be an outgoing ray, using the same pixel on 2 images taken side by side will result in the rays meet in the real 3D space where the object is. From the idea I just said, you can have the following formula for calculating depth.
![image](https://github.com/user-attachments/assets/5ee713fc-6ac4-4745-854f-a1fb843babe1)

![image](https://github.com/user-attachments/assets/29f65363-0eb3-4373-ac17-77d4c3288123)

- **Images**: The images will be taken twice using my phone, and the phone will be move horizontally to simulate 2 camera placing side by side. The baseline (distance between 2 cameras is 10 centimeters).

![image](https://github.com/user-attachments/assets/5ee1a31e-2ee7-42a9-a118-d506dc4d970c)

- **LoFTR Architecture**: Detector-Free Local Feature Matching with Transformer, a light model for feature matching purpose, can be found on HuggingFace. 

![image](https://github.com/user-attachments/assets/608869e6-7717-40f3-a658-fb432ad0d32e)
 
- **Feature Matching**: Applied a LoFTR model for feature matching between the two images.

![image](https://github.com/user-attachments/assets/af069375-634b-4b22-98b7-643f4b2464e6)

- **Depth Calculation**: Calculated the depth of objects within specified bounding boxes using the formula provided above, calculate for all pairs of features matched in the bounding box, then calcuate the mean distance calculated from those matching pixels.

![image](https://github.com/user-attachments/assets/07c63731-e692-4091-b7c8-4cc8c47ffa26)

![image](https://github.com/user-attachments/assets/23b1a6a6-51d0-498d-962c-2ba822e995a9)

Here is the distribution of the distance calculated. Everything is around 18, which is the real measured distance. 

![image](https://github.com/user-attachments/assets/2a0105f3-87fb-40b2-9787-050ba6db5913)

**Optional**: You could try to filter out the outliers with Z-score. 

![image](https://github.com/user-attachments/assets/97abe409-457d-4f3b-a2a7-85a613fb436d)


### 2. Object Counting in Video Processing
This project focuses on counting objects (either squares or circles) on a board using a combination of traditional image processing techniques. Key components of this project include:

![image](https://github.com/user-attachments/assets/d62c34bf-ba17-4bef-8d41-2b00fb072eb8)

- **Thresholding**: Employed traditional thresholding methods to segment objects from the background. First changed the frame to grey scale, then use OTSU method to threshold the frame and get the binary map, using closing and opening to clean the map.
- **Labeling**: Used labeling techniques to identify and distinguish different objects in the frame. Use label() and find_object() from scipy the get the object bounding box, then calcuate the white pixel percentage in the bounding box to identify the object and count accordingly.
- **Tracking**: Implemented object tracking using motpy to maintain the count of objects over time in the video feed.
- **Logging**: Logging output in-case the system crash.
