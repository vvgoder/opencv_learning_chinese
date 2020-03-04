'''
@Author: your name
@Date: 2020-01-17 20:37:07
@LastEditTime : 2020-01-18 16:44:10
@LastEditors  : Please set LastEditors
@Description: 对比度增强的一些方法
@FilePath: \pactise_opencv_python\4.对比度增强.py
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
# %%
# 灰度直方图的绘制与可视化
def calcGrayHist(image):
    rows,cols=image.shape
    grayHist=np.zeros([256],np.uint64)
    for i in range(rows):
        for j in range(cols):
            grayHist[image[i][j]]+=1
    return grayHist
image=cv2.imread('15.png',0)
grayHist=calcGrayHist(image)
plt.plot(range(256),grayHist,'r',linewidth=2,c='black')
# 设置坐标轴的范围
y_maxvalue=np.max(grayHist)
plt.axis([0,255,0,y_maxvalue])
# 设置坐标轴的标签
plt.xlabel('gray level')
plt.ylabel('number of pixels')
plt.show()

# %%
# 直接使用plt.hist对图片直接绘制灰度直方图
rows,cols=image.shape
pixelSequence=image.reshape([rows*cols,])#reshape成行向量
# 组数
numbins=256
# 计算灰度直方图
histgray,bins,patch=plt.hist(pixelSequence,numbins,facecolor='black',histtype='bar')
# 设置坐标轴的范围
y_maxvalue=np.max(grayHist)
plt.axis([0,255,0,y_maxvalue])
# 设置坐标轴的标签
plt.xlabel('gray level')
plt.ylabel('number of pixels')
plt.show()

# %%
# 线性变换，相当于拉伸，增加对比度，原理看书p100
a=2
output=float(a)*image
output[output>255]=255
# 数据类型转换
output=np.round(output)
output=output.astype(np.uint8) #没有这一步就显示不出图像
# 显示原图和线性变换的图
cv2.imshow('I',image)
cv2.imshow('O',output)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
# 上述线性变换最头疼的是找合适的a，b
# 所以下面使用直方图正规化来对直方图进行拉伸，原理见书p105
# 先求Imax和Imin
Imax=np.max(image)
Imin=np.min(image)
# 设定输出灰度级的最小和最大
Omax=255
Omin=0
# 计算a，b
a=float(Omax-Omin)/(Imax-Imin)
b=Omin-a*Imin
# 对矩阵进行线性变换
output=a*output+b
# 类型转换
output=output.astype(np.uint8)
# 显示出原图和直方图正规化的效果
cv2.imshow('I',image)
cv2.imshow('O',output)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
# 使用normlize获取和上述的直方图正规化一样的效果
src=cv2.imread('2.jpg',cv2.IMREAD_ANYCOLOR)
dst=cv2.normalize(src,255,0,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8U)
# 显示出原图和直方图正规化的效果
cv2.imshow('I',src)
cv2.imshow('O',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
# 伽马变换
# 先将图片灰度归一化，除以255或者最大值
# 然后作指数运算，原理见书p111
# 指数在（0，1）的时候，对比度增强
img=image/255
# 伽马变换
gamma=0.5
output=np.power(img,gamma)
# 显示出原图和伽马变换的效果
cv2.imshow('I',image)
cv2.imshow('O',output)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
# 全局直方图均衡化
# 可自动调节对比度
# 原理见书p114
def equalHist(image):
    # 灰度图像矩阵的高、宽
    rows,cols=image.shape
    # 第一步：计算灰度直方图
    grayHist=calcGrayHist(image)
    # 第二步：计算累加灰度直方图
    zeroCumuMoment=np.zeros([256],np.uint32)
    for p in range(256):
        if p==0:
            zeroCumuMoment[p]=grayHist[0]
        else:
            zeroCumuMoment[p]=zeroCumuMoment[p-1]+grayHist[p]
    # 第三步：根据累加灰度直方图得到输入灰度级和输出灰度级之间的映射关系
    output_q=np.zeros([256],np.uint8)
    cofficient=256.0/(rows*cols)
    for p in range(256):
        q=cofficient*float(zeroCumuMoment[p])-1
        if q>=0:
            output_q[p]=math.floor(q)
        else:
            output_q[p]=0
    # 第四步：得到直方图均衡化后的图像
    equalHistImage=np.zeros(image.shape,np.uint8)
    for r in range(rows):
        for c in range(cols):
            equalHistImage[r][c]=output_q[image[r][c]]
    return equalHistImage
src=cv2.imread('2.bmp',0)
dst=equalHist(src)
# 显示
cv2.imshow('I',src)
cv2.imshow('O',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
# 限制对比度的自适应直方图均衡化
# Contrast Limited Adaptive Histogram Equalization
src=cv2.imread('2.bmp',0)
# 创建CLAHE对象
clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
# 限制对比度的自适应阈值均衡化
dst=clahe.apply(src)
# 显示
cv2.imshow('I',src)
cv2.imshow('O',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()


# %%
