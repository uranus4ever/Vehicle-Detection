# **Vehicle Detection Project**

[//]: # (Image References)
[img1]: ./Image/car_not_car.png
[img2]: ./Image/HOG_example.jpg
[img3]: ./Image/sliding_windows.jpg
[img4]: ./Image/sliding_window.jpg
[img5]: ./Image/bboxes_and_heat.png
[img6]: ./Image/labels_map.png
[img7]: ./Image/output_bboxes.png
[img8]: ./Image/yolo-box.PNG
[img9]: ./Image/find_car_yolo.png
[gif]: ./Image/yolo.gif
[video-SVM]: ./project_video.mp4
[video-yolo]: ./project_video.mp4

---
## Overview

Vehicle detection used machine learning and computer vision techniques, and combined [advanced lane detection](https://github.com/uranus4ever/Advanced-Lane-Detection) techniques in my previous project.

![yolo-gif][gif]

The steps of this project are the following:

**SVM**

 - Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier.
 - Implement an effiecient sliding-window technique and use  trained SVM classifier to search for vehicles in images.
 - Run your pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles. [see full video][video-SVM]

**YOLO**

 - Construct a Keras network and implement a pre-trained model to predict in images.
 - Run a pipeline on a video stream and create a console to monitor lane status and detections. [see full video][video-yolo]

### Usage

`Project-SVM.py` and `helper.py` contain the code for SVM classifier stracture and pipeline.
`dist.p` contains a trained SVM classifier based on YUV color features and HOG features with 17,000+ car and not-car pictures.
`Project-yolo.py` and `helper_yolo.py` contain the code for Keras network and pipeline.
`weights/yolo-tiny.weights` is a trained weight file (170 M) used for tiny yolo network.

### Dependencies
 - Numpy
 - cv2
 - sklearn
 - scipy
 - skimage
 - keras

---

### SVM Algorithm

SVM (Support Vector Machine) is a powerful machine learning technique. Here in this project it is trained and used for classification of car and not-car.

#### 1. Collecting data
My training data is mainly downloaded from [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html) and [KITTI vision benchmark](http://www.cvlibs.net/datasets/kitti/) websites, which contain about 8700 pictures of car and 8900 of not-car. In addition, in order to increase detection accurancy, I create multiple not-car pics from video.

![car and not-car][img1]

#### 2. Extracting features

I explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`) and made a comparison.

| Color Space | Accuracy | Training Time (CPU) |
|:--:|:--:|:--:|:--:|
| YUV | 97.75% | 65 s |
| YCrCb | 98.11% | 51 s |
| LUV | 98.23% | 59 s |
| HLS | 98% | 60 s |
| HSV | 97.8% | 112 s |

I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 3. Training classifier
#### 4. Sliding window
#### 5. Filtering False-positive by heatmap

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and...

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using...

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

####2. Show some Image of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]

### YOLO

[YOLO](https://arxiv.org/pdf/1506.02640) (You Look Only Once) is a popular end-to-end **Real-time Object Detection** algorith based on deep learning. Compared with other object recognition method, such as Fast R-CNN, YOLO integrates target area and object classification into a single neural network. The most outstanding point is its fast speed with preferably high accuracy, nearly 45 fps of base version and up to 155 fps in FastYOLO, quite favourable for real-time applications, for example, computer vision of self-driving car.

#### 1. Principle

YOLO uses an unified single neural network, which makes full use of the whole image infomation as bounding box identification and classification. It divides the image into an *SxS* grid and for each grid celll predicts *B* bounding boxes, confidence for those boxes, and *C* class probailities. The output is a 1470 vector, containing probability, confidence and box coordinates. 
![model][image8]

It has 20 classes as the following:
```
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
```

In this project, I used tiny YOLO v1 as it is easy to implement and impressively fast.

#### 2. Pipeline

First to construct a convolutional neuarl network architecture based on Keras, containing 9 convolution layers and 3 full connected layers.
```
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
convolution2d_1 (Convolution2D)  (None, 16, 448, 448)  448         convolution2d_input_1[0][0]      
____________________________________________________________________________________________________
leakyrelu_1 (LeakyReLU)          (None, 16, 448, 448)  0           convolution2d_1[0][0]            
____________________________________________________________________________________________________
maxpooling2d_1 (MaxPooling2D)    (None, 16, 224, 224)  0           leakyrelu_1[0][0]                
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 32, 224, 224)  4640        maxpooling2d_1[0][0]             
____________________________________________________________________________________________________
leakyrelu_2 (LeakyReLU)          (None, 32, 224, 224)  0           convolution2d_2[0][0]            
____________________________________________________________________________________________________
maxpooling2d_2 (MaxPooling2D)    (None, 32, 112, 112)  0           leakyrelu_2[0][0]                
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 64, 112, 112)  18496       maxpooling2d_2[0][0]             
____________________________________________________________________________________________________
leakyrelu_3 (LeakyReLU)          (None, 64, 112, 112)  0           convolution2d_3[0][0]            
____________________________________________________________________________________________________
maxpooling2d_3 (MaxPooling2D)    (None, 64, 56, 56)    0           leakyrelu_3[0][0]                
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 128, 56, 56)   73856       maxpooling2d_3[0][0]             
____________________________________________________________________________________________________
leakyrelu_4 (LeakyReLU)          (None, 128, 56, 56)   0           convolution2d_4[0][0]            
____________________________________________________________________________________________________
maxpooling2d_4 (MaxPooling2D)    (None, 128, 28, 28)   0           leakyrelu_4[0][0]                
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 256, 28, 28)   295168      maxpooling2d_4[0][0]             
____________________________________________________________________________________________________
leakyrelu_5 (LeakyReLU)          (None, 256, 28, 28)   0           convolution2d_5[0][0]            
____________________________________________________________________________________________________
maxpooling2d_5 (MaxPooling2D)    (None, 256, 14, 14)   0           leakyrelu_5[0][0]                
____________________________________________________________________________________________________
convolution2d_6 (Convolution2D)  (None, 512, 14, 14)   1180160     maxpooling2d_5[0][0]             
____________________________________________________________________________________________________
leakyrelu_6 (LeakyReLU)          (None, 512, 14, 14)   0           convolution2d_6[0][0]            
____________________________________________________________________________________________________
maxpooling2d_6 (MaxPooling2D)    (None, 512, 7, 7)     0           leakyrelu_6[0][0]                
____________________________________________________________________________________________________
convolution2d_7 (Convolution2D)  (None, 1024, 7, 7)    4719616     maxpooling2d_6[0][0]             
____________________________________________________________________________________________________
leakyrelu_7 (LeakyReLU)          (None, 1024, 7, 7)    0           convolution2d_7[0][0]            
____________________________________________________________________________________________________
convolution2d_8 (Convolution2D)  (None, 1024, 7, 7)    9438208     leakyrelu_7[0][0]                
____________________________________________________________________________________________________
leakyrelu_8 (LeakyReLU)          (None, 1024, 7, 7)    0           convolution2d_8[0][0]            
____________________________________________________________________________________________________
convolution2d_9 (Convolution2D)  (None, 1024, 7, 7)    9438208     leakyrelu_8[0][0]                
____________________________________________________________________________________________________
leakyrelu_9 (LeakyReLU)          (None, 1024, 7, 7)    0           convolution2d_9[0][0]            
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 50176)         0           leakyrelu_9[0][0]                
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 256)           12845312    flatten_1[0][0]                  
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 4096)          1052672     dense_1[0][0]                    
____________________________________________________________________________________________________
leakyrelu_10 (LeakyReLU)         (None, 4096)          0           dense_2[0][0]                    
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 1470)          6022590     leakyrelu_10[0][0]               
====================================================================================================
Total params: 45,089,374
Trainable params: 45,089,374
Non-trainable params: 0
____________________________________________________________________________________________________
```

Then to load the pre-trained weights ([link](https://github.com/uranus4ever/Vehicle-Detection/tree/master/weights/yolo-tiny.weights)) from website as network training is really time consuming.

After weight loading, detected bounding boxes could be draw onto the images and finally applied into video stream pipeline with a confidence *threshold=0.2*.

![find_car][image9]

---

### Reflection

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

### Reference
1. J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/pdf/1506.02640), arXiv:1506.02640 (2015).
2. [dark flow](https://github.com/thtrieu/darkflow) 
3. [yolo_tensorflow](https://github.com/hizhangp/yolo_tensorflow)
3. [xslittlegrass](https://github.com/xslittlegrass/CarND-Vehicle-Detection)
4. [JunshengFu](https://github.com/JunshengFu/vehicle-detection)

