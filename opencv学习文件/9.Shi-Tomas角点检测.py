# %%
# Shi-TOmasi角点检测（一种适应追踪的角点检测方式）
import numpy as np
import cv2
from matplotlib import pyplot as plt

# %%
img = cv2.imread('data/calibresult.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
"""goodFeaturesToTrack函数：
    参数： （1）gray：输入灰度图
           （2） 25：想要检测的角点数量
           （3） 0：角点最低水平 （0到1）
           （4） 10：两个角点的最短欧式距离
"""
corners = cv2.goodFeaturesToTrack(gray,25,0.01,10)
# 返回的结果是[[ 311., 250.]] 两层括号的数组。
corners = np.int0(corners)
for corner in corners:
    x,y = corner.ravel()
    cv2.circle(img,(x,y),3,255,-1)  #汇出角点
plt.imshow(img),plt.show()

# %%
