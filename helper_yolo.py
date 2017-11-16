import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import glob


def draw_test_img(imgs_path, model):
    images = [plt.imread(file) for file in glob.glob(imgs_path)]
    batch = np.array([np.transpose(cv2.resize(image[300:650, 500:, :], (448, 448)), (2, 0, 1))
                      for image in images])
    batch = 2 * (batch / 255.) - 1
    out = model.predict(batch)
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 6))
    for i, ax in zip(range(len(batch)), [ax1, ax2, ax3, ax4]):
        boxes = yolo_boxes(out[i], threshold=0.17)
        ax.imshow(draw_box(boxes, images[i], [[500, 1280], [300, 650]]))

    return out


def load_weights(model, yolo_weight_file):
    data = np.fromfile(yolo_weight_file, np.float32)
    data = data[4:]

    index = 0
    for layer in model.layers:
        shape = [w.shape for w in layer.get_weights()]
        if shape != []:
            kshape, bshape = shape
            bia = data[index:index + np.prod(bshape)].reshape(bshape)
            index += np.prod(bshape)
            ker = data[index:index + np.prod(kshape)].reshape(kshape)
            index += np.prod(kshape)
            layer.set_weights([ker, bia])


class Box:
    def __init__(self):
        self.x, self.y = float(), float()
        self.w, self.h = float(), float()
        self.c = float()
        self.prob = float()


def overlap(x1, w1, x2, w2):
    l1 = x1 - w1 / 2.
    l2 = x2 - w2 / 2.
    left = max(l1, l2)
    r1 = x1 + w1 / 2.
    r2 = x2 + w2 / 2.
    right = min(r1, r2)
    return right - left


def box_intersection(a, b):
    w = overlap(a.x, a.w, b.x, b.w)
    h = overlap(a.y, a.h, b.y, b.h)
    if w < 0 or h < 0:
        return 0
    area = w * h
    return area


def box_union(a, b):
    i = box_intersection(a, b)
    u = a.w * a.h + b.w * b.h - i
    return u


def box_iou(a, b):
    return box_intersection(a, b) / box_union(a, b)


def yolo_boxes(net_out, threshold=0.2, sqrt=1.8, C=20, B=2, S=7):
    classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
                "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
                "sheep", "sofa", "train","tvmonitor"]
    class_num = 6
    boxes = []
    SS = S * S  # number of grid cells
    prob_size = SS * C  # class probabilities
    conf_size = SS * B  # confidences for each grid cell

    probs = net_out[0: prob_size]
    confs = net_out[prob_size: (prob_size + conf_size)]
    cords = net_out[(prob_size + conf_size):]
    probs = probs.reshape([SS, C])
    confs = confs.reshape([SS, B])
    cords = cords.reshape([SS, B, 4])

    for grid in range(SS):
        for b in range(B):
            bx = Box()
            bx.c = confs[grid, b]
            bx.x = (cords[grid, b, 0] + grid % S) / S
            bx.y = (cords[grid, b, 1] + grid // S) / S
            bx.w = cords[grid, b, 2] ** sqrt
            bx.h = cords[grid, b, 3] ** sqrt
            p = probs[grid, :] * bx.c

            if p[class_num] >= threshold:
                bx.prob = p[class_num]
                boxes.append(bx)

    # combine boxes that are overlap
    boxes.sort(key=lambda b: b.prob, reverse=True)
    for i in range(len(boxes)):
        boxi = boxes[i]
        if boxi.prob == 0:
            continue
        for j in range(i + 1, len(boxes)):
            boxj = boxes[j]
            if box_iou(boxi, boxj) >= .4:
                boxes[j].prob = 0.
    boxes = [b for b in boxes if b.prob > 0.]

    return boxes


def draw_box(boxes, im, crop_dim):
    imgcv = np.copy(im)
    [xmin, xmax] = crop_dim[0]
    [ymin, ymax] = crop_dim[1]
    for i, b in enumerate(boxes, 1):
        h, w, _ = imgcv.shape
        left = int((b.x - b.w / 2.) * w)
        right = int((b.x + b.w / 2.) * w)
        top = int((b.y - b.h / 2.) * h)
        bot = int((b.y + b.h / 2.) * h)
        left = int(left * (xmax - xmin) / w + xmin)
        right = int(right * (xmax - xmin) / w + xmin)
        top = int(top * (ymax - ymin) / h + ymin)
        bot = int(bot * (ymax - ymin) / h + ymin)

        left = max(left, 0)
        right = min(right, w-1)
        top = max(top, 0)
        bot = min(bot, h-1)

        cv2.rectangle(imgcv, (left, top), (right, bot), (0, 0, 255), thickness=3)

        # draw label
        label = 'car ' + str(i)
        cv2.rectangle(imgcv, (left, top - 30), (right, top), (125, 125, 125), -1)
        cv2.putText(imgcv, label, (left + 5, top - 7), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

        # draw thumbnail in highlight title
        thumbnail = im[top:bot, left:right]
        vehicle_thumb = cv2.resize(thumbnail, dsize=(120, 80))  # width=120, height=80
        start_x = 750 + (i-1) * 30 + (i-1) * 120  # offset=30
        imgcv[60:60+80, start_x:start_x+120, :] = vehicle_thumb

    cv2.putText(imgcv, 'Lane', (280, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(imgcv, 'Detected Vehicles', (800, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2,
                cv2.LINE_AA)

    return imgcv


def draw_background_highlight(image, draw_img, w=1280):

    mask = cv2.rectangle(np.copy(image), (0, 0), (w, 155), (0, 0, 0), thickness=cv2.FILLED)

    return cv2.addWeighted(src1=mask, alpha=0.3, src2=draw_img, beta=0.8, gamma=0)


def draw_thumbnails(img_cp, img, window_list, thumb_w=120, thumb_h=80, off_x=30, off_y=30):
    cv2.putText(img_cp, 'Lane', (280, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(img_cp, 'Detected Vehicles', (600, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
    for i, bbox in enumerate(window_list):
        thumbnail = img[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]]
        vehicle_thumb = cv2.resize(thumbnail, dsize=(thumb_w, thumb_h))
        start_x = 640 + (i+1) * off_x + i * thumb_w
        img_cp[off_y + 30:off_y + thumb_h + 30, start_x:start_x + thumb_w, :] = vehicle_thumb

