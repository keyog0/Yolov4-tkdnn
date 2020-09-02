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

lib = CDLL("../libdarknetTR.so", RTLD_GLOBAL)

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
def reproduce_bbox(x,y,w,h,image_w,image_h,model_size = 416) :
    rx = (x/model_size) * image_w
    rw = (w/model_size) * image_w
    ry = (y/model_size) * image_h
    rh = (h/model_size) * image_h
    #print(rx,rw,ry,rh)

    return rx,ry,rw,rh


def detect_image(net, meta, darknet_image,image_w,image_h,thresh=.5):

    num = c_int(0)

    pnum = pointer(num)
    do_inference(net, darknet_image)
    dets = get_network_boxes(net, 0.5, 0, pnum)
    res = []

    for i in range(pnum[0]):
        b = dets[i].bbox
        #print(b.x,b.y,b.w,b.h)
        b.x, b.y, b.w, b.h = reproduce_bbox(b.x, b.y, b.w, b.h,image_w,image_h)
        res.append((dets[i].cl, dets[i].prob, (b.x, b.y, b.w, b.h)))

    return res


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
            image_w = image.shape[1]
            image_h = image.shape[0]
            if need_resize:
                frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(frame_rgb,
                                   (self.input_size, self.input_size),
                                   interpolation=cv2.INTER_LINEAR)
            frame_data = image.ctypes.data_as(c_char_p)
            copy_image_from_bytes(self.darknet_image, frame_data)

            detections = detect_image(self.model, self.metaMain, self.darknet_image,image_w,image_h,thresh=self.thresh)

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
