'''
@Author: your name
@Date: 2020-02-02 16:39:05
@LastEditTime : 2020-02-02 18:26:22
@LastEditors  : Please set LastEditors
@Description: In User Settings Edit
@FilePath: \pactise_opencv_python\姿势估计.py
'''
import cv2
import numpy as np
import glob
import os
# %%
# Load previously saved data
_, mtx, dist, _, _=np.load('B.npy',allow_pickle=True)

# %%
def draw(img, corners, imgpts):
    """
    img :输入图片
    corners：棋盘上的角点，这里取第一个角点
    imgpts：
    """
    corner = tuple(corners[0].ravel())
    imgpts=np.array(imgpts,dtype=np.int32)
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img
# %%
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
# 轴在实际空间坐标系的坐标点，-3表示垂直于摄像机方向
axis = np.array([[3,0,0], [0,3,0], [0,0,-3]],dtype=np.float).reshape(-1,3)

# %%
# 每个图片有不同的旋转和变换矩阵
for fname in glob.glob('qipan/left12.jpg'):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (7,6),None)
    if ret == True:
        # 寻找更精确的角点
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        # Find the rotation and translation vectors.
        # 返回旋转和变换矩阵,利用RANSAC实现物体位置的3维坐标和2维坐标之间的转换
        _,rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, mtx, dist)
        print('tvecs',tvecs)
        print('rvecs',rvecs)
        # project 3D points to image plane
        # 将3d坐标转为图片上的2d坐标
        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
        print('imgpts',imgpts)
        img = draw(img,corners2,imgpts)
        cv2.imshow('img',img)
        k = cv2.waitKey(0) & 0xff
        if k == 's':
            cv2.imwrite(os.path.join('qipan',fname[:6]+'.png'), img)
cv2.destroyAllWindows()

# %%
# 上面是画坐标轴
# 下面画立方体
def draw_cube(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)
    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)
    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)
        # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)
    return img

# %%
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
# 轴在实际空间坐标系的坐标点，-3表示垂直于摄像机方向
axis = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],[0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3] ])

# %%
# 每个图片有不同的旋转和变换矩阵
for fname in glob.glob('qipan/left12.jpg'):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (7,6),None)
    if ret == True:
        # 寻找更精确的角点
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        # Find the rotation and translation vectors.
        # 返回旋转和变换矩阵
        _,rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, mtx, dist)
        # project 3D points to image plane
        # 将3d坐标转为图片上的2d坐标
        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
        img = draw_cube(img,corners2,imgpts)
        cv2.imshow('img',img)
        k = cv2.waitKey(0) & 0xff
        if k == 's':
            cv2.imwrite(os.path.join('qipan',fname[:6]+'.png'), img)
cv2.destroyAllWindows()

# %%
