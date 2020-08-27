# -*- encoding: utf-8 -*-
# Author : Keunyoung Jung

import darknetTR
import cv2
import numpy as np
import time

def retbox(detections,i,labels_arr) :
	#print(detections)
	label = detections[i][0]
	score = detections[i][1]
	classes = labels_arr.index(label)

	x1 = int(round((detections[i][2][0]))) # top left x1 
	y1 = int(round((detections[i][2][1]))) # top left xy 
	x2 = int(round((detections[i][2][0]) + (detections[i][2][2]))) # bottom right x2 
	y2 = int(round((detections[i][2][1]) + (detections[i][2][3]))) # bottom right y2 
                
	box = np.array([x1,y1,x2,y2])

	return label, score, box ,classes

if __name__ == "__main__" :

	class_num = 80
	
	detector = darknetTR.YOLO4RT(weight_file='custom_yolov4_fp32.rt', 
					metaPath='custom.data',nms=0.2,conf_thres=0.3)
	
	font = cv2.FONT_HERSHEY_DUPLEX
	COLORS = []
	for i in range(class_num) :
		COLORS.append((np.random.randint(0,255),
				np.random.randint(0,255),
				np.random.randint(0,255)))

	
	vidfile = 'yolo_test.mp4'
	cap = cv2.VideoCapture(vid_file)


	while True :
		start = time.time()
		
		_,frame = cap.read()
		rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

		detections = detector.detect(rgb_frame)

		for i in range(len(detections)) :
			label , score , box ,classes = retbox(detections,i,labels_arr)
			left,top,right,bottom=box
			cv2.rectangle(frame, (x, y), (w, h), color[idx], 2)
			cv2.putText(frame,label,(x,y+30),font,1,color[idx],1)

		process_time = str(round(time.time() - start,3))
		frame_per_second = str(round(1/(time.time() - start),3))

		cv2.putText(frame,'process time :'+process_time+'sec',(frame.shape[1]-420,frame.shape[0]-30),font,1,(0,255,0),2)
		cv2.putText(frame,'fps :'+frame_per_second,(frame.shape[1]-420,frame.shape[0]-60),font,1,(0,255,0),2)

		cv2.imshow('frame',frame)
		k = cv2.waitKey(1)
		if k == 27 :
			break

	
