"""
本节主要讲一下：图像梯度的一些常用算子：Roberts、Prewitt、Sobel、Scharr、Kirsch、Robinson、Laplacian、
在重点讲一下Canny边缘检测、LoG、DoG
说白了梯度图像梯度：就是基于方向的差分
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
# %%
# Roberts算子和 Prewitt算子看书p224到p234
# 差分方向（或者叫做梯度方向）和得到的边缘方向是垂直的
# 注意多种算子结果的融合方式，见p224

# %%
# Sobel算子其实是高斯卷积核和差分卷积核的合体，基于Prewitt是使用均值平滑，效果不佳
# 由此Sobel算子的高斯卷积核效果不错
# Sobel算子是在一个坐标轴方向进行非归一化的高斯平滑，在另一个坐标轴上进行差分处理
# 所以Sobel算子是可分离的
# 手动搭建sobel算子见书p237
img=cv2.imread('data/seu.png')
# 参数1,0 为只在x 方向求一阶导数，最大可以求2 阶导数。
sobelx=cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
# 参数0,1 为只在y 方向求一阶导数，最大可以求2 阶导数。
sobely=cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
# 将x,y方向的融合，为总的梯度大小
sobel=np.sqrt(np.power(sobelx,2.0)+np.power(sobely,2.0))
sobel=sobel/sobel.max()
sobel=sobel*255
sobel=sobel.astype(np.uint8)
# 图片展示
cv2.imshow('sobelx',sobelx)
cv2.imshow('sobely',sobely)
cv2.imshow('sobel',sobel)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
# scharr算子：不可分离算子
# 高斯梯度加上差分算分，有x方向，y方向，45和135
# 就kszie=3的时候，效果比Sobel好
img=cv2.imread('data/seu.png')
scharr_x=cv2.Scharr(img,cv2.CV_64F,1,0,scale=5)
scharr_y=cv2.Scharr(img,cv2.CV_64F,0,1,scale=5)
# 融合为scharr
scharr=np.sqrt(np.power(scharr_x,2.0)+np.power(scharr_y,2.0))
scharr=scharr/scharr.max()
scharr=scharr*255
scharr=scharr.astype(np.uint8)
# 图片展示
cv2.imshow('scharrx',scharr_x)
cv2.imshow('scharry',scharr_y)
cv2.imshow('scharr',scharr)
cv2.waitKey(0)
cv2.destroyAllWindows()


# %%
# Kirsch、Robinson 都是八个卷积算子共同处理
# 然后在每个位置上取得绝对值最大值即可
# 具体实现看书p244-p246

# %%
# canny边缘检测就上述算子的两个问题进行完善
# （1）上述算子没有利用边缘的梯度方向，所以canny提出了基于边缘梯度方向的非极大值抑制
# （2）上述算子最后输出的是边缘二值图，只是简单的进行了阈值处理，阈值过大，失去一些重要信息，阈值过小，会有很多噪声
#  所以canny使用双阈值的滞后阈值处理
# 具体内容看p248-p267

# %%
# Canny实战
import cv2
import numpy as np

minVal,maxVal = 0,0 #阈值

def nothing(x):#回调函数更新图像
    #获取滑条位置
    minVal = cv2.getTrackbarPos('minVal','Canny')
    maxVal = cv2.getTrackbarPos('maxVal','Canny')
    edges = cv2.Canny(img,minVal,maxVal)
    cv2.imshow('Canny',edges)

img = cv2.imread('data/seu.png',0)#读取图片
cv2.namedWindow('image')#显示原图
cv2.imshow('image',img)

cv2.namedWindow('Canny')#显示检测图
#创建两个滑条
cv2.createTrackbar('minVal','Canny',0,255,nothing)
cv2.createTrackbar('maxVal','Canny',0,255,nothing)
"""
cv2.canny函数参数分析
（1）原图
（2）阈值分割minval
（3）阈值分割maxval
（4）计算图像梯度的Sobel卷积核size
（5）最后一个参数是L2gradient，它可以用来设定求梯度大小的方程
"""
# edges = cv2.Canny(img,minVal,maxVal)
# cv2.imshow('Canny',edges)

while(True):
    #等待关闭
    k=cv2.waitKey(1)&0xFF
    if k==27:
        break
cv2.destroyAllWindows()

# %%
# Laplacian
# 见p268，Laplacian变换可以看作矩阵与拉普拉斯卷积核的卷积
# 根据拉普拉斯的公式推导设定拉普拉斯的卷积核：中心-4周  or 4周-中心
# 由于没有引入平滑，所以噪声较大
laplacian=cv2.Laplacian(img,cv2.CV_64F)

# %%
# 上面说到拉普拉斯算子没有平滑，效果不佳，所以引入高斯拉普拉斯算子（LoG）
# 没找到相关的API
# 原理推导见书上的p273,其实就是对二维高斯分布求拉普拉斯变换，对x的二阶导+对y的二阶导
def createLoGkernel(sigma,size):
    H,W=size
    r,c=np.mgrid[0:H:1,0:W:1]
    r=r-(H-1)/2
    c=c-(W-1)/2
    # 方差
    sigma2=pow(sigma,2.0)
    # 高斯拉普拉斯核
    norm2=np.power(r,2.0)+np.power(c,2.0)
    LoGkernel=(norm2/sigma2-2)*np.exp(-norm2/(2*sigma2))
    return LoGkernel

def LoG(image,sigma,size,_boundary='symm'):
    # 构建高斯拉普拉斯卷积核
    LoGkernel=createLoGkernel(sigma,size)
    # 图像矩阵与高斯拉普拉斯卷积核卷积
    img=convolve2d(image,LoGkernel,'same',boundary=_boundary)
    return img

image=cv2.imread('data/seu.png',0)
# size为 6*sigma+1
img_log=LoG(image,sigma=6,size=(37,37))
cv2.imshow('img',image)
cv2.imshow('img log',img_log)
# 为了更好的展示，做二值化处理
edge_binary=np.copy(img_log)
edge_binary[edge_binary>0]=255
edge_binary[edge_binary<=0]=0
edge_binary=edge_binary.astype(np.uint8)
cv2.imshow('binary',edge_binary)
while(True):
    #等待关闭
    k=cv2.waitKey(1)&0xFF
    if k==27:
        break
cv2.destroyAllWindows()

# %%
# 高斯拉普拉斯可以用差分拉普拉斯去等效
def gaussConv(I,size,sigma):
    #获取卷积核的高和宽
    H,W=size
    # 构建水平方向非归一化的高斯核
    xr,xc=np.mgrid[0:1,0:W]
    xc=xc-(W-1)/2
    xk=np.exp(-np.power(xc,2.0))
    # I和xk卷积
    I_xk=convolve2d(I,xk,'same','symm')
    # 构造垂直方向上的非归一化高斯卷积核
    yr,yc=np.mgrid[0:H,0:1]
    yc=yc-(H-1)/2
    yk=np.exp(-np.power(yc,2.0))
    # I_xk与yk卷积
    I_xk_yk=convolve2d(I_xk,yk,'same','symm')
    I_xk_yk*=1.0/(2*np.pi*pow(sigma,2.0))
    return I_xk_yk

# 定义DoG实现高斯差分
def DoG(image,size,sigma,k=1.1):
    #标准差为sigma的非归一化高斯卷积核
    Is=gaussConv(image,size,sigma)
     #标准差为 k*sigma的非归一化高斯卷积核
    Isk=gaussConv(image,size,k*sigma)
    # 两个高斯核的差分
    doG=Isk-Is
    doG/=(pow(sigma,2.0)*(k-1))
    return doG

image=cv2.imread('data/seu.png',0)
# size为 6*sigma+1
k=1.1
sigma=2
size=(13,13)
img_dog=DoG(image,size,sigma,k)
cv2.imshow('img',image)
cv2.imshow('img dog',img_dog)
# 为了更好的展示，做二值化处理
edge_binary=np.copy(img_dog)
edge_binary[edge_binary>0]=255
edge_binary[edge_binary<=0]=0
edge_binary=edge_binary.astype(np.uint8)
cv2.imshow('binary',edge_binary)
while(True):
    #等待关闭
    k=cv2.waitKey(1)&0xFF
    if k==27:
        break
cv2.destroyAllWindows()

# %%
# Marr-Hildreth边缘检测
# 基于拉普拉斯算子或者LoG、DoG进行相关的改进
# 具体见书上的p283