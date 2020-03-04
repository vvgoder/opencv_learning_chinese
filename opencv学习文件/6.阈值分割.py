'''
@Author: your name
@Date: 2020-01-30 17:31:10
@LastEditTime : 2020-01-30 20:45:42
@LastEditors  : Please set LastEditors
@Description:阈值分割+自适应阈值分割+OTSU阈值分割+
@FilePath: \pactise_opencv_python\6.阈值分割.py
'''
import cv2
import numpy as np
from matplotlib import pyplot as plt

# %%
# 阈值分割函数 threshold（图片，分割阈值，范围，分割类型）
img=cv2.imread('seu.png',0)
ret,thresh1=cv2.threshold(img,127,255,cv2.THRESH_BINARY) #大于阈值为白色（255）
ret,thresh2=cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV) #大于阈值为黑色（0）
ret,thresh3=cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
ret,thresh4=cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
ret,thresh5=cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)
titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
for i in range(6):
    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
    plt.show()

# %%
# 使用type=cv2.THRESH_OTSU（或者OTSU）或者 cv2.THRESH_TRIANGLE（和直方图阈值处理原理）
# OTSU阈值处理,有个初始值otsuthe=0
otsuthe=0
# 输出自己的otsu阈值和目标图像
otsuthe,dst_otsu =cv2.threshold(img,otsuthe,255,cv2.THRESH_OTSU)
cv2.imshow('I',img)
cv2.imshow('O',dst_otsu)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
# TRIANGLE阈值处理
triThe=0
triThe,dst_tri=cv2.threshold(img,triThe,255,cv2.THRESH_TRIANGLE)
cv2.imshow('I',img)
cv2.imshow('O',dst_tri)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
# 直方图技术法：原理见p187-188
# 适用条件：直方图存在双峰结构
# 代码实现如下：
def calcGrayHist(image):
    '''
    统计像素值
    :param image:
    :return:
    '''
    # 灰度图像的高，宽
    rows, cols = image.shape
    # 存储灰度直方图
    grayHist = np.zeros([256], np.uint64)
    for r in range(rows):
        for c in range(cols):
            grayHist[image[r][c]] += 1
    return grayHist
def threshTwoPeaks(image):
    # 计算灰度直方图
    histogram = calcGrayHist(image)
    # 找到灰度直方图的最大峰值对应的灰度值
    maxLoc = np.where(histogram == np.max(histogram))
    firstPeak = maxLoc[0][0]
    # 寻找灰度直方图的第二个峰值对应的灰度值
    measureDists = np.zeros([256], np.float32)
    for k in range(256):
        measureDists[k] = pow(k - firstPeak, 2)*histogram[k]
    maxLoc2 = np.where(measureDists == np.max(measureDists))
    secondPeak = maxLoc2[0][0]
    # 找两个峰值之间的最小值对应的灰度值，作为阈值
    thresh = 0
    if firstPeak > secondPeak:
        temp = histogram[int(secondPeak): int(firstPeak)]
        minLoc = np.where(temp == np.min(temp))
        thresh = secondPeak + minLoc[0][0] + 1
    else:
        temp = histogram[int(firstPeak): int(secondPeak)]
        minLoc = np.where(temp == np.min(temp))
        thresh = firstPeak + minLoc[0][0] + 1
    # 找到阈值，我们进行处理
    img = image.copy()
    img[img > thresh] = 255
    img[img <= thresh] = 0
    cv2.imshow('deal_image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == '__main__':
    image = cv2.imread('seu.png',0)
    threshTwoPeaks(image)


# %%
# 熵算法
# 原理见书p191-192
# 代码如下：
import math
def threshEntroy(image):
    rows, cols = image.shape
    # 求灰度直方图
    grayHist = calcGrayHist(image)
    # 归一化灰度直方图，即概率直方图
    normGrayHist = grayHist / float(rows*cols)
    # 第一步:计算累加直方图，也称零阶累积矩
    zeroCumuMoment = np.zeros([256], np.float32)
    for k in range(256):
        if k == 0:
            zeroCumuMoment[k] = normGrayHist[k]
        else:
            zeroCumuMoment[k] = zeroCumuMoment[k-1] + normGrayHist[k]
    # 第二步:计算各个灰度级的熵
    entropy = np.zeros([256], np.float32)
    for k in range(256):
        if k == 0:
            if normGrayHist[k] == 0:
                entropy[k] = 0
            else:
                entropy[k] = -normGrayHist[k]*math.log10(normGrayHist[k])
        else:
            if normGrayHist[k] == 0:
                entropy[k] = entropy[k-1]
            else:
                entropy[k] = entropy[k-1] - normGrayHist[k]*math.log10(normGrayHist[k])
    # 第三步:找阈值
    fT = np.zeros([256], np.float32)
    ft1, ft2 = 0.0, 0.0
    totalEntropy = entropy[255]
    for k in range(255):
        # 找最大值
        maxFront = np.max(normGrayHist[0: k+1])
        maxBack = np.max(normGrayHist[k+1: 256])
        if (maxFront == 0 or zeroCumuMoment[k] == 0
                or maxFront == 1 or zeroCumuMoment[k] == 1 or totalEntropy == 0):
            ft1 = 0
        else:
            ft1 = entropy[k] / totalEntropy*(math.log10(zeroCumuMoment[k])/math.log10(maxFront))
        if (maxBack == 0 or 1-zeroCumuMoment[k] == 0
                or maxBack == 1 or 1-zeroCumuMoment[k] == 1):
            ft2 = 0
        else:
            if totalEntropy == 0:
                ft2 = (math.log10(1-zeroCumuMoment[k]) / math.log10(maxBack))
            else:
                ft2 = (1-entropy[k]/totalEntropy)*(math.log10(1-zeroCumuMoment[k])/math.log10(maxBack))
        fT[k] = ft1 + ft2
    # 找最大值的索引，作为得到的阈值
    threshLoc = np.where(fT == np.max(fT))
    thresh = threshLoc[0][0]
    # 阈值处理
    threshold = np.copy(image)
    threshold[threshold > thresh] = 255
    threshold[threshold <= thresh] = 0
    return threshold
if __name__ == '__main__':
    image = cv2.imread('seu.png', 0)
    img = threshEntroy(image)
    cv2.imshow('origin', image)
    cv2.imshow('deal_image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# %%
# Otsu阈值分割
# 原则:选择的阈值应当使得前景区域的平均灰度和背景区域+整幅图的平均灰度差别最大
# 原理见书p195-196

image = cv2.imread('seu.png', cv2.IMREAD_GRAYSCALE)
maxval = 255
otsuThe = 0
otsuThe, dst_Otsu = cv2.threshold(image, otsuThe, maxval, cv2.THRESH_OTSU)
cv2.imshow('Otsu', dst_Otsu)
cv2.waitKey(0)
cv2.destroyAllWindows()


# %%
# 局部阈值分割（自适应阈值分割）
# 不同位置有不同的分割阈值
# 原理见书p200
# 手动实现代码如下
def adaptiveThresh(I, winSize, ratio=0.15):
    # 第一步:对图像矩阵进行均值平滑
    I_mean = cv2.boxFilter(I, cv2.CV_32FC1, winSize)
    # 第二步:原图像矩阵与平滑结果做差
    out = I - (1.0 - ratio) * I_mean
    # 第三步:当差值大于或等于0时，输出值为255；反之，输出值为0
    out[out >= 0] = 255
    out[out < 0] = 0
    out = out.astype(np.uint8)
    return out

if __name__ == '__main__':
    image = cv2.imread('seu.png', cv2.IMREAD_GRAYSCALE)
    img = adaptiveThresh(image, (5, 5))
    cv2.imshow('origin', image)
    cv2.imshow('deal_image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# %%
# 自适应自带函数
image = cv2.imread('dog.jpg', cv2.IMREAD_GRAYSCALE)
# adaptiveMethod有cv2.ADAPTIVE_THRESH_GAUSSIAN_C和 cv2.ADAPTIVE_THRESH_MEAN_C
img=cv2.adaptiveThreshold(image,maxValue=255,adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
thresholdType=cv2.THRESH_BINARY,
blockSize=5,C=0.15)
cv2.imshow('origin', image)
cv2.imshow('deal_image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
# 与运算进行抠图
img_bgr=cv2.imread('dog1.jpg', cv2.IMREAD_ANYCOLOR)
image = cv2.imread('dog1.jpg', cv2.IMREAD_GRAYSCALE)
maxval = 255
otsuThe = 0
otsuThe, dst_Otsu = cv2.threshold(image, otsuThe, maxval, cv2.THRESH_OTSU)
# 在彩色图中抠图
dst_and=cv2.bitwise_and(img_bgr,cv2.merge([dst_Otsu]*3))
cv2.imshow('Otsu', dst_and)
cv2.waitKey(0)
cv2.destroyAllWindows()


# %%
# 自适应阈值分割
import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('dog1.jpg',0)
# 中值滤波
img = cv2.medianBlur(img,5)
ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
#11 为Block size, 2 为C 值
th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
cv2.THRESH_BINARY,11,2)
th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
cv2.THRESH_BINARY,11,2)
titles = ['Original Image', 'Global Thresholding (v = 127)',
'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
    plt.show()

# %%
