import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('data/16.png')
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# Initiate STAR detector
orb = cv2.ORB_create()
# find the keypoints with ORB
kp = orb.detect(gray,None)
# compute the descriptors with ORB
kp, des = orb.compute(gray, kp)
# draw only keypoints location,not size and orientation
img1=cv2.drawKeypoints(gray,kp,None,color=(0,0,255), flags=0)
plt.imshow(img1)
plt.show()