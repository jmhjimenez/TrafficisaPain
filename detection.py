import cv2
from glob import glob
import numpy as np
import os

def get_first():
    for i in np.arange(300000, 310000, 100):
        first_image = '../data/frames/0{}.jpg'.format(i)
        print(first_image)
        first_image = cv2.imread(first_image, 0)
        cv2.imshow('pouet', first_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


first_image = '../data/frames/0302600.jpg'
first_image = cv2.imread(first_image, 0)
for i in np.arange(300000, 340000, 100):
    image = '../data/frames/0{}.jpg'.format(i)
    ori = cv2.imread(image, 0)
    image = cv2.GaussianBlur(ori, (21, 21), 0)
    frame_delta = cv2.absdiff(first_image, image)
    frame_delta[frame_delta>60] = 255
    frame_delta[frame_delta<=60] = 0
    # frame_delta = cv2.threshold(frame_delta, 50, 255, cv2.THRESH_BINARY)[1]
    kernel = np.ones((10,10),np.uint8)
    closing = cv2.morphologyEx(frame_delta, cv2.MORPH_CLOSE, kernel)
    kernel = np.ones((10,10),np.uint8)
    frame_delta = cv2.erode(frame_delta,kernel,iterations = 1)
    kernel = np.ones((10,10),np.uint8)
    frame_delta = cv2.dilate(frame_delta,kernel,iterations = 1)

    cv2.imshow('pouet', frame_delta)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cnts = cv2.findContours(frame_delta.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)[1]
    print(cnts)
    for c in cnts:
        a = cv2.contourArea(c)
        if a > 100:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(ori, (x, y), (x + w, y + h), 255, 3)
    cv2.imshow('pouet', ori)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
