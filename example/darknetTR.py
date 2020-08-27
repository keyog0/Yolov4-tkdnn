#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from ctypes import *
import cv2
import numpy as np
import argparse
import os
from threading import Thread
import time

class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("cl", c_int),
                ("bbox", BOX),
                ("prob", c_float),
                ("name", c_char*20),
                ]

lib = CDLL("./build/libdarknetTR.so", RTLD_GLOBAL)

load_network = lib.load_network
load_network.argtypes = [c_char_p, c_int, c_int]
load_network.restype = c_void_p

copy_image_from_bytes = lib.copy_image_from_bytes
copy_image_from_bytes.argtypes = [IMAGE,c_char_p]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

do_inference = lib.do_inference
do_inference.argtypes = [c_void_p, IMAGE]

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_float, c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

# cfg = 'yolo4_fp16.rt'
# netMain = load_network(cfg.encode("ascii"), 80, 1)  # batch size = 1
#
#
# darknet_image = make_image(512, 512, 3)
# image = cv2.imread('/home/juzheng/dataset/mask/image/20190821004325_55.jpg')
# frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# image = cv2.resize(frame_rgb,
#                    (512, 512),
#                    interpolation=cv2.INTER_LINEAR)
#
# # frame_data = np.asarray(image, dtype=np.uint8)
# # print(frame_data.shape)
# frame_data = image.ctypes.data_as(c_char_p)
# copy_image_from_bytes(darknet_image, frame_data)
#
# num = c_int(0)
#
# pnum = pointer(num)
# do_inference(netMain, darknet_image)
# dets = get_network_boxes(netMain, 0.5, 0, pnum)
# print('end')
# print(dets[0].cl, dets[0].prob)


def resizePadding(image, height, width):
    desized_size = height, width
    old_size = image.shape[:2]
    max_size_idx = old_size.index(max(old_size))
    ratio = float(desized_size[max_size_idx]) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    if new_size > desized_size:
        min_size_idx = old_size.index(min(old_size))
        ratio = float(desized_size[min_size_idx]) / min(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])

    image = cv2.resize(image, (new_size[1], new_size[0]))
    delta_w = desized_size[1] - new_size[1]
    delta_h = desized_size[0] - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)
    return image

def detect_image(net, meta, darknet_image, thresh=.5):
    num = c_int(0)

    pnum = pointer(num)
    do_inference(net, darknet_image)
    dets = get_network_boxes(net, 0.5, 0, pnum)
    res = []
    for i in range(pnum[0]):
        b = dets[i].bbox
        res.append((dets[i].name.decode("ascii"), dets[i].prob, (b.x, b.y, b.w, b.h)))

    return res

def retbox(detections,i) :
    #print(detections)
    label = detections[i][0]
    score = detections[i][1]
    #classes = labels_arr.index(label)

    # x1 = int(round((detections[i][2][0]) - (detections[i][2][2]/2.0))) # top left x1 
    # y1 = int(round((detections[i][2][1]) - (detections[i][2][3]/2.0))) # top left y1 
    # x2 = int(round((detections[i][2][0]) + (detections[i][2][2]/2.0))) # bottom right x2 
    # y2 = int(round((detections[i][2][1]) + (detections[i][2][3]/2.0))) # bottom right y2 
    x1 = int(round((detections[i][2][0]))) # top left x1 
    y1 = int(round((detections[i][2][1]))) # top left y1 
    x2 = int(round((detections[i][2][2]))) # bottom right x2 
    y2 = int(round((detections[i][2][3]))) # bottom right y2 
                
    box = np.array([x1,y1,x2,y2])

    return label, score, box 


def loop_detect(detect_m, video_path):
    stream = cv2.VideoCapture(video_path)
    start = time.time()
    cnt = 0
    while stream.isOpened():
        ret, image = stream.read()
        if ret is False:
            break
        # image = resizePadding(image, 512, 512)
        # frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image,
                           (416, 416),
                           interpolation=cv2.INTER_LINEAR)
        detections = detect_m.detect(image, need_resize=False)
        for i in range(len(detections)) :
            label , score , box = retbox(detections,i)
            print(box)

        #for box in boxes:
        x, y = int(box[0]), int(box[1])
        w, h = int(box[2]), int(box[3])
        cv2.rectangle(image, (x, y), (w+x, h+y), (0,255,0), 3)

        cnt += 1

        cv2.imshow('test',image)
        k = cv2.waitKey(1)
        if k == 27: 
            break
        for det in detections:
            print(det)
    end = time.time()
    print("frame:{},time:{:.3f},FPS:{:.2f}".format(cnt, end-start, cnt/(end-start)))
    stream.release()


# class myThread(threading.Thread):
#    def __init__(self, func, args):
#       threading.Thread.__init__(self)
#       self.func = func
#       self.args = args
#    def run(self):
#       # print ("Starting " + self.args[0])
#       self.func(*self.args)
#       print ("Exiting " )


class YOLO4RT(object):
    def __init__(self,
                 input_size=416,
                 weight_file='./yolo4_fp16.rt',
                 metaPath='Models/yolo4/coco.data',
                 nms=0.2,
                 conf_thres=0.3,
                 device='cuda'):
        self.input_size = input_size
        self.metaMain =None
        self.model = load_network(weight_file.encode("ascii"), 80, 1)
        self.darknet_image = make_image(input_size, input_size, 3)
        self.thresh = conf_thres
        # self.resize_fn = ResizePadding(input_size, input_size)
        # self.transf_fn = transforms.ToTensor()

    def detect(self, image, need_resize=True, expand_bb=5):
        try:
            if need_resize:
                frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(frame_rgb,
                                   (self.input_size, self.input_size),
                                   interpolation=cv2.INTER_LINEAR)
            frame_data = image.ctypes.data_as(c_char_p)
            copy_image_from_bytes(self.darknet_image, frame_data)

            detections = detect_image(self.model, self.metaMain, self.darknet_image, thresh=self.thresh)
            #print('@@@@@@@@@@@@@@@@',detections)
            # cvDrawBoxes(detections, image)
            # cv2.imshow("1", image)
            # cv2.waitKey(1)
            # detections = self.filter_results(detections, "person")
            return detections
        except Exception as e_s:
            print(e_s)

def parse_args():
    parser = argparse.ArgumentParser(description='tkDNN detect')
    parser.add_argument('weight', help='rt file path')
    parser.add_argument('--video',  type=str, help='video path')
    args = parser.parse_args()

    return args



if __name__ == '__main__':
    #args = parse_args()
    detect_m = YOLO4RT(weight_file='./build/ground_yolov4_fp32.rt',metaPath='./ground.data')
    t = Thread(target=loop_detect, args=(detect_m, './bacon_potato_pizza.mp4'), daemon=True)

    # thread1 = myThread(loop_detect, [detect_m])

    # Start new Threads
    t.start()
    t.join()