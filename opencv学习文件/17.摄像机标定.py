'''
@Author: your name
@Date: 2020-02-02 15:03:08
@LastEditTime : 2020-02-05 20:43:07
@LastEditors  : Please set LastEditors
@Description: In User Settings Edit
@FilePath: \pactise_opencv_python\摄像机标定.py
'''
import numpy as np
import cv2
import glob
# %%
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
# 以下记录棋盘的真实3d位置，z都设为0
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('qipan/left12.jpg')
for fname in images:
    img= cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    # 找到角点后
    ret, corners = cv2.findChessboardCorners(gray, (7,6),None)
    print('ret',ret)
    print('corners',corners)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        # 使用cv2.cornerSubPix（亚像素角点）增加寻找角点的准确性
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (7,6), corners2,ret)
        cv2.imshow('img',img)
        cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
# 标定，返回摄像机矩阵(综合所有图片)，畸变系数，旋转和变换向量(每个图片的旋转和变换向量不一样)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
# mtx为摄像机矩阵，rvecs, tvecs 为旋转和变换向量

# %%
# 在畸变矫正之前，我们使用函数cv2.getOptimalNewCameraMatrix()
# 得到的自由缩放系数对摄像机矩阵进行优化
img = cv2.imread('qipan/left12.jpg')
h, w = img.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

# %%
# 第一种修正方法
# undistort
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
# crop the image
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('calibresult.png',dst)

# %%
# 第二种修正方法
# undistort
mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
# crop the image
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('calibresult.png',dst)


# %%
# 使用反向投影误差评估估计参数好坏，越接近0越好
total_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    total_error += error
print ("total error: ", total_error/len(objpoints))

# %%
np.save('B.npy',[ret, mtx, dist, rvecs, tvecs])


# %%
