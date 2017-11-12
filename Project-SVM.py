# import numpy as np
# import matplotlib.pyplot as plt
# import pickle
# import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
from collections import deque
from helper import *


def process_image(image, heat_thres=1):
    global dist_pickle
    boxes = []

    multibox1, img_multibox1 = find_cars(image, dist_pickle, ystart=400, ystop=500, scale=1.0)
    multibox2, img_multibox2 = find_cars(image, dist_pickle, ystart=400, ystop=500, scale=1.3)
    multibox3, img_multibox3 = find_cars(image, dist_pickle, ystart=420, ystop=556, scale=1.6)
    multibox4, img_multibox4 = find_cars(image, dist_pickle, ystart=430, ystop=556, scale=2.0)
    multibox5, img_multibox5 = find_cars(image, dist_pickle, ystart=500, ystop=656, scale=3.0)
    multibox6, img_multibox6 = find_cars(image, dist_pickle, ystart=410, ystop=500, scale=1.4)
    multibox7, img_multibox7 = find_cars(image, dist_pickle, ystart=430, ystop=556, scale=1.8)
    multibox8, img_multibox8 = find_cars(image, dist_pickle, ystart=440, ystop=556, scale=1.9)
    multibox9, img_multibox9 = find_cars(image, dist_pickle, ystart=400, ystop=556, scale=2.2)

    boxes.extend(multibox1)
    boxes.extend(multibox2)
    boxes.extend(multibox3)
    boxes.extend(multibox4)
    boxes.extend(multibox5)
    boxes.extend(multibox6)
    boxes.extend(multibox7)
    boxes.extend(multibox8)
    boxes.extend(multibox9)

    heat_zero = np.zeros_like(image[:, :, 0]).astype(np.float)
    heat = add_heat(heat_zero, boxes)
    heat = apply_threshold(heat, threshold=heat_thres)
    current_heatmap = np.clip(heat, 0, 255)

    # HM.current_heat = heatmap
    # merged_heat = HM.merge_heat()
    # heatmap = apply_threshold(merged_heat, threshold=framenum*heat_thres)

    history.append(current_heatmap)
    heatmap = np.zeros_like(current_heatmap).astype(np.float)
    for heat in history:
        heatmap += heat

    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(image), labels)

    return draw_img

dist_pickle = pickle.load(open("dist.p", "rb"))
history = deque(maxlen=8)

input_path = './input_videos/project.mp4'
video_output = './output_videos/project_SVM.mp4'

# input_path = './input_videos/test_video.mp4'
# video_output = './output_videos/test_SVM.mp4'

clip1 = VideoFileClip(input_path)
# clip1 = VideoFileClip(input_path).subclip(4, 16)

t = time.time()
# final_clip = clip1.fl_image(process_image)
# final_clip.write_videofile(video_output, audio=False)
t2 = time.time()
print(round(t2 - t, 2), 'Seconds to process video...')

image = mpimg.imread('./test_img/test5.jpg')
plt.figure()
plt.imshow(process_image(image, heat_thres=1))
plt.show()
