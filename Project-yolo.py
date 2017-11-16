import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import keras
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Flatten, Dense, Activation, Reshape
import time
from helper_yolo import *

keras.backend.set_image_dim_ordering('th')

model = Sequential()
model.add(Convolution2D(16, 3, 3,input_shape=(3, 448, 448), border_mode='same', subsample=(1, 1)))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(32, 3, 3, border_mode='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))
model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))
model.add(Convolution2D(128, 3, 3, border_mode='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))
model.add(Convolution2D(256, 3, 3, border_mode='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))
model.add(Convolution2D(512, 3, 3, border_mode='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))
model.add(Convolution2D(1024, 3, 3, border_mode='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(Convolution2D(1024, 3, 3, border_mode='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(Convolution2D(1024, 3, 3, border_mode='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(Flatten())
model.add(Dense(256))
model.add(Dense(4096))
model.add(LeakyReLU(alpha=0.1))
model.add(Dense(1470))

print('model loaded.')

# model.summary()

load_weights(model, './weights/yolo-tiny.weights')
print('weight loaded.')

# predict = draw_test_img('./test_img/test*.jpg', model)


def process_image(image):
    crop = image[300:650, 500:, :]
    resized = cv2.resize(crop, (448, 448))
    batch = np.array([resized[:, :, 0], resized[:, :, 1], resized[:, :, 2]])
    batch = 2 * (batch/255.) - 1
    batch = np.expand_dims(batch, axis=0)
    out = model.predict(batch)
    boxes = yolo_boxes(out[0], threshold=0.2)

    # draw result
    img_cp = np.copy(image)
    img_cp = draw_background_highlight(img_cp, image)
    img_cp = draw_box(boxes, np.copy(img_cp), [[500, 1280], [300, 650]])

    return img_cp

input_path = './input_videos/project.mp4'
video_output = './output_videos/project_yolo_0.2.mp4'

# input_path = './input_videos/test_video.mp4'
# video_output = './output_videos/test_yolo.mp4'

clip1 = VideoFileClip(input_path)
# clip1 = VideoFileClip(input_path).subclip(29, 39)

t = time.time()
final_clip = clip1.fl_image(process_image)
final_clip.write_videofile(video_output, audio=False)
t2 = time.time()
print(round(t2 - t, 2), 'Seconds to process video...')

# image = mpimg.imread('./test_img/test5.jpg')
# plt.figure()
# plt.imshow(process_image(image))
# plt.show()
