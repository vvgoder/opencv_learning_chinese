import numpy as np
import cv2
from matplotlib import pyplot as plt

# %%
# FAST角点检测
img = cv2.imread('data/seu.png')
grey=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# Initiate FAST object with default values
# nonmaxSuppression=True启用极大值抑制或者非极大值
fast = cv2.FastFeatureDetector_create(nonmaxSuppression=True)
# find and draw the keypoints
kp = fast.detect(grey,None)
img=cv2.drawKeypoints(grey, kp, None,color=(255,0,0))
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('data/16.png')
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# 初始化STAR检测器
star = cv2.xfeatures2d.StarDetector_create()
# 初始化BRIEF特征提取器
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
# 使用STAR寻找特征点
kp = star.detect(gray,None)
# 计算特征描述符
kp, des = brief.compute(gray, kp)

img=cv2.drawKeypoints(gray,kp,None,color=(255,0,0))
cv2.imshow('p',img)
cv2.waitKey(0)


# %%
