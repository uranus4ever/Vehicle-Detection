# import numpy as np
# import matplotlib.pyplot as plt
# import pickle
# import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
from collections import deque
from helper import *


class BoxList(object):
    def __init__(self):
        # box detected? in last frame
        self.detected_list = None
        self.box_list = None
        self.last_box_list = None
        self.coordinate = ((0, 0), (0, 0))
        self.last_center_list = deque(maxlen=5*framenum)
        self.current_center_list = deque(maxlen=5)
        self.new_box_list = deque(maxlen=5)

    def get_box_center(self):
        self.current_center_list = []
        self.last_center_list = []
        for box_num in range(len(self.box_list)):
            x = (self.box_list[box_num][0][0] + self.box_list[box_num][1][0]) // 2
            y = (self.box_list[box_num][0][1] + self.box_list[box_num][1][1]) // 2
            self.current_center_list.append((x, y))
        for box_num in range(len(self.last_box_list)):
            x = (self.last_box_list[box_num][0][0] + self.last_box_list[box_num][1][0]) // 2
            y = (self.last_box_list[box_num][0][1] + self.last_box_list[box_num][1][1]) // 2
            self.last_center_list.append((x, y))
        return self.current_center_list, self.last_center_list

    def confidence_filter(self, confidence_thres):
        pix = 20
        self.current_center_list, self.last_center_list = self.get_box_center()
        confid_list = np.zeros(len(self.current_center_list))
        for c_idx in range(len(self.current_center_list)):
            for l_idx in range(len(self.last_center_list)):
                c_x, c_y = self.current_center_list[c_idx][0], self.current_center_list[c_idx][1]
                l_x, l_y = self.last_center_list[l_idx][0], self.last_center_list[l_idx][1]
                if (l_x - pix < c_x < l_x + pix) and (l_y - pix < c_y < l_y + pix):
                    confid_list[c_idx] += 1

        self.new_box_list = self.current_center_list[np.where(confid_list > confidence_thres)]
        return self.new_box_list


class HeatMap(object):
    def __init__(self):
        self.current_heat = np.zeros((720, 1280)).astype(np.float)
        self.previous_heat = deque(maxlen=framenum)

    def merge_heat(self):
        merged_heat = np.zeros_like(self.current_heat).astype(np.float)
        self.previous_heat.append(self.current_heat)
        for num in range(len(self.previous_heat)):
            merged_heat += self.previous_heat[~num]

        return merged_heat


def draw_labeled_boxes_2(img, labels):
    box_list = []
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
        # box_list.append(bbox)

    # return img, box_list
    return img


def process_image(image):
    global dist_pickle, confidence_thres

    multibox, img_multibox = find_cars(image, dist_pickle)

    heat = np.zeros_like(image[:, :, 0]).astype(np.float)
    heat = add_heat(heat, multibox)
    heat = apply_threshold(heat, threshold=3)
    heatmap = np.clip(heat, 0, 255)

    HM.current_heat = heatmap
    merged_heat = HM.merge_heat()
    heatmap = apply_threshold(merged_heat, threshold=framenum)

    labels = label(heatmap)
    draw_img = draw_labeled_boxes_2(np.copy(image), labels)

    # box.current_center_list = box_list
    # box.new_box_list = box.confidence_filter(confidence_thres)
    return draw_img

dist_pickle = pickle.load(open("dist.p", "rb"))

framenum = 10
confidence_thres = 2
# box = BoxList()
HM = HeatMap()

input_path = './input_videos/project.mp4'
video_output = './videos/project_SVM.mp4'

clip1 = VideoFileClip(input_path)
# clip1 = VideoFileClip(input_path).subclip(4, 16)

t = time.time()
final_clip = clip1.fl_image(process_image)
final_clip.write_videofile(video_output, audio=False)
t2 = time.time()
print(round(t2 - t, 2), 'Seconds to process video...')

# image = mpimg.imread('./test_img/test6.jpg')
# plt.imshow(process_image(image))
