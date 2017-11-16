# **Vehicle Detection Project**

## Overview

Vehicle detection project used machine learning and computer vision techniques, and combined [advanced lane detection](https://github.com/uranus4ever/Advanced-Lane-Detection) techniques.

![yolo-gif][gif]

I applied two different methods for detection. The steps of this project are the following:

**1) SVM Algorithm**

 - Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier.
 - Implement an effiecient sliding-window technique and use  trained SVM classifier to search for vehicles in images.
 - Run a pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles. [watch full video][video-SVM]

**2) YOLO Algorithm**

 - Construct a Keras based neural network and implement a pre-trained model to predict images.
 - Run a pipeline on a video stream and create a console to monitor lane status and detections. [watch full video][video-yolo]

### Usage

 - `Project-SVM.py` and `helper.py` contain the code for SVM classifier stracture and pipeline. 
 - `dist.p` contains a trained SVM classifier based on YUV color features and HOG features with 17,000+ car and not-car pictures.
 - `Project-yolo.py` and `helper_yolo.py` contain the code for Keras network and pipeline. 

### Dependencies
 - Numpy
 - cv2
 - sklearn
 - scipy
 - skimage
 - keras

---

### **1) SVM Algorithm**

SVM (Support Vector Machine) is a powerful machine learning technique. Here in this project it is trained and used for classification of car and not-car.

#### 1. Collecting Data
My main training data is downloaded from [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html) and [KITTI vision benchmark](http://www.cvlibs.net/datasets/kitti/) websites, which contain about 8700 pictures of car and 8900 of not-car. In addition, in order to increase detection accurancy, I create about 20 not-car pictures from video.

![car and not-car][img1]

#### 2. Extracting Features

I explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`) and made a comparison.

| Color Space | Accuracy | Training Time (CPU) |
|:--:|:--:|:--:|
| YUV | 97.75% | 65 s |
| YCrCb | 98.11% | 51 s |
| LUV | 98.23% | 59 s |
| HLS | 98% | 60 s |
| HSV | 97.8% | 112 s |


The above table indicates that accuracy performance in different color space are almost same. Considering less false-positive, I chose `YUV` to extract color features.

Here is an example using the `YUV` color space and HOG parameters of `orientations=15`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![feature][img2]

#### 3. Training classifier

I trained a linear SVM using the following code:

```
car_features = extract_features(cars, color_space, orient, pix_per_cell, cell_per_block, spatial_feat=False, hist_feat=False, hog_channel=hog_channel)
notcar_features = extract_features(notcars, color_space, orient, pix_per_cell, cell_per_block, spatial_feat=False, hist_feat=False, hog_channel=hog_channel)
# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)    
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)
# Use a linear SVC
svc = LinearSVC()
svc.fit(X_train, y_train)
```

Note that feature normalization through `sklearn.prepocessing.StandardScaler` is one of the key steps before training. Here is a comparison:

![normalization][img3]

#### 4. Sliding window

An efficient method for sliding window search is applied, one that allows me to only have to extract the HOG features once.

```
def find_cars(img, ystart=400, ystop=656, scale=1.5):

    draw_img = np.copy(img)
    img = img.astype(np.float32) / 255

    img_tosearch = img[ystart:ystop, :, :]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YUV')
    cspace = 'YUV'
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient * cell_per_block ** 2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    bbox_list = []

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # Get color features
            spatial_features = bin_spatial(subimg, color_space=cspace, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(
                np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                cv2.rectangle(draw_img, (xbox_left, ytop_draw + ystart),
                              (xbox_left + win_draw, ytop_draw + win_draw + ystart), (0, 0, 255), 6)
                bbox_list.append(((xbox_left, ytop_draw + ystart), (xbox_left + win_draw, ytop_draw + win_draw + ystart)))

    return bbox_list, draw_img
```

Additionally, multiple-scaled search windows is applied with different scale values.

![multi box][img4]

#### 5. Filtering False-positive by heatmap

Heatmap with a certain threshold is a good helper to filter false positives and deal with multiple detections. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.

```
def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    return heatmap
```
```
heatmap = threshold(heatmap, 2)
labels = label(heatmap)
```

![heatmap filter][img5]

![label][img6]

#### 6. Video Implementation

Video stream is a series of image process with the techniques above. In order to make detection between frames more smooth, I built a simple historical heatmap queue to set up connections.

```
history = deque(maxlen=8)
current_heatmap = np.clip(heat, 0, 255)
history.append(current_heatmap)
```

### **2) YOLO Algorithm**

[YOLO](https://arxiv.org/pdf/1506.02640) (You Look Only Once) is a popular end-to-end **Real-time Object Detection** algorithm based on deep learning. Compared with other object recognition methods, such as Fast R-CNN, YOLO integrates target area and object classification into a single neural network. The most outstanding point is its fast speed with preferably high accuracy, nearly 45 fps in base version and up to 155 fps in FastYOLO, quite favourable for real-time applications, for example, computer vision of self-driving car.

#### 1. Principle

YOLO uses an unified single neural network, which makes full use of the whole image infomation as bounding box identification and classification. It divides the image into an *SxS* grid and for each grid celll predicts *B* bounding boxes, confidence for those boxes, and *C* class probailities. The output is a 1470 vector, containing probability, confidence and box coordinates. 
![model][img8]

It has 20 classes as the following:
```
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle",
"bus", "car", "cat", "chair", "cow", 
"diningtable", "dog", "horse", "motorbike", "person", 
"pottedplant", "sheep", "sofa", "train", "tvmonitor"]
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

Then to load the pre-trained weights (172 MB, [link](https://drive.google.com/file/d/0B1tW_VtY7onibmdQWE1zVERxcjQ/view?usp=sharing)) from website as network training is really time-consuming.

After weight loading, detected bounding boxes could be draw onto the images and finally applied into video stream pipeline with a confidence *threshold=0.2*.

![find_car][img9]


## Reflection

### 1. Discussion

SVM has an acceptable accuracy of detection, however, it has two shorcomings:

 1. Due to heatmap threshold application, the bounding box usually appears unstable and smaller than the actual size of object cars.
 2. The processing speed is only up to 2 fps on account of sliding window search. Even with GPU parallel computing, it is not favorable for real-time application.

YOLO is much more preferable by reason of its strength against SVM's shortcomings:

 1. Better detection and more stable bounding box position.
 2. Real-time processing speed, nearly 40 fps.

Note that the limitation of YOLO is as follows:

 1. Only two boxes and one class in each grid are predicted, which causes detection accuracy of adjacent objects decreases. 
 2. YOLO is learned from pre-trained data. As a result, it performs poor with new objects or usual view angle.

### 2. Next Plan

 1. As the false detection types of YOLO and Fast R-CNN are different, we can integrate these two models to enhance performance.
 2. Explore YOLO to more videos to try other classifications in addition to cars.

## Reference
1. J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/pdf/1506.02640), arXiv:1506.02640 (2015).
2. [dark flow](https://github.com/thtrieu/darkflow) 
3. [yolo_tensorflow](https://github.com/hizhangp/yolo_tensorflow)
3. [YOLO Introduction](https://zhuanlan.zhihu.com/p/25045711)

[//]: # (Image References)
[img1]: ./Image/car_not_car.PNG
[img2]: ./Image/feature.png
[img3]: ./Image/Normalize_Feature_HSV.png
[img4]: ./Image/find_car.png
[img5]: ./Image/heatmap1.png
[img6]: ./Image/heatmap.png
[img8]: ./Image/yolo-box.PNG
[img9]: ./Image/find_car_yolo.png
[gif]: ./Image/yolo.gif
[video-SVM]: ./project_video.mp4
[video-yolo]: ./project_video.mp4
