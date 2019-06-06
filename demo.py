#!/usr/bin/env python

import cv2
import time
import os
from core.detectors import CornerNet_Saccade
from core.detectors import CornerNet_Squeeze
from core.detectors import RetailNet
from core.vis_utils import draw_bboxes


def demoJpg():
        detector = CornerNet_Saccade()
        image    = cv2.imread("demo.jpg")

        bboxes = detector(image)
        image  = draw_bboxes(image, bboxes)
        cv2.imwrite("demo_out.jpg", image)

def demoOneRetail():
    detector = RetailNet()
    image = cv2.imread("20180824-15-35-19-51.jpg")
    bboxes = detector(image)
    thresh = 0.61
    for cat_name in bboxes:
        keep_inds = bboxes[cat_name][:, -1] > thresh
        print(cat_name,keep_inds,bboxes[cat_name][:, -1],sep=":")
    image  = draw_bboxes(image, bboxes,thresh=thresh)
    cv2.imwrite("20180824-15-35-19-51_out.jpg", image)

def demoRetail():
    detector = RetailNet()
    valDir = '/home/myu/retail_project/CornerNet-Lite/data/retail/images/val2019/'
    imgs = [i for i in os.listdir(valDir) if os.path.splitext(i)[1]=='.jpg']
    print(f"found imgs:{len(imgs)}")
    for img in imgs:
        print(f"validating img:{valDir+img}")
        image = cv2.imread(valDir+img)
        bboxes = detector(image)
        image  = draw_bboxes(image, bboxes,thresh=0.61)
        cv2.imwrite(f"./validateRetailResults/{os.path.splitext(img)[0]}_out.jpg", image)

def demoCamera():
	cap = cv2.VideoCapture("rtsp://user:pass@192.168.1.3:554/Streaming/Channels/1/")
	if not cap.isOpened():
		print("Unable to open camera")
		exit(-1)
	detector = CornerNet_Saccade()
	while True:
		res, img = cap.read()
		if res:
			bboxes = detector(img)
			img  = draw_bboxes(img, bboxes)
			cv2.imshow("camera", img)
			cv2.waitKey(1)
		else:
             		print("Unable to read image")
             		break
	cv2.destroyAllWindows()
	cap.release()

if __name__ == '__main__':
	demoRetail()
