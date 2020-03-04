import cv2
import numpy as np

# %%
# SIFT opencv实现
img = cv2.imread('data/16.png')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# 创建一个SIFT算法检测器
detector = cv2.xfeatures2d.SIFT_create()
# 检测点，生成关键点
keypoints = detector.detect(gray,None)
# 描述符（numpy）
des=detector.compute(gray,keypoints)
print(des)
# 或者同时生成关键点和描述符
keypoints,des=detector.detectAndCompute(gray,None)
#在灰度图上画出关键点，生成img
# cv2.drawKeypoints(gray,keypoints,img)
# 此flags可以绘制关键点的方向
img=cv2.drawKeypoints(gray,keypoints,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('test',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# %%
# SIFT 速度比较慢，因为使用 DoG来逼近 LoG
# SURF 速度 比 SIFT快，因为使用了盒子滤波器来逼近LoG
img = cv2.imread('data/16.png')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# Hessian Threshold to 400
# 设置upright属性，使得所有的特征点的方向一致
# extended属性，使得特征点的描述符维度增加
detector = cv2.xfeatures2d.SURF_create(400)
# 或者同时生成关键点和描述符
keypoints,des=detector.detectAndCompute(gray,None)
print(len(keypoints)) #1082
# 改变Hessian Threshold ，增加会减少后面的关键点
detector1 = cv2.xfeatures2d.SURF_create(4000)
# 或者同时生成关键点和描述符
keypoints1,des1=detector1.detectAndCompute(gray,None)
print(len(keypoints1)) #1082
img=cv2.drawKeypoints(gray,keypoints1,None,color=(255,0,0),
                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                  )
cv2.imshow('test',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
