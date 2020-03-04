'''
@Author: your name
@Date: 2020-01-18 16:44:54
@LastEditTime : 2020-02-04 17:42:34
@LastEditors  : Please set LastEditors
@Description: 图像平滑（又叫滤波技术），用来降低图片的噪声，并且可以保持边缘
              这里包含了：高斯平滑、均值平滑、基于统计学方法的中值平滑、双边滤波、导向滤波
@FilePath: \pactise_opencv_python\5.图像平滑.py
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
# %%
# 关于卷积的一些知识，详细参照书上的p122-128
# 三种卷积形式：Full,Same,Valid主要区别在于是否对原图进行padding
# 以及padding多少
# 输出的shape和输入的img的shape（n,n）和卷积核的shape(k,k)关系是
# o=[下取整（n+2p-k）/2]-1
# 在cv中，填充边界的方法为cv2.copyMakeBorder
# 该函数参数：第二个到第五个为四周填充的行数
#            填充类型为：cv2.BORDER_REPLACE 边界复制
#                       cv2.BORDER_CONSTANT 常数填充
#                       cv2.BORDER_REFLECT 反射扩充（包括边界）
#                       cv2.BORDER_REFLECT_101 反射扩充（不包括边界）
#                       cv2.BORDER_WRAP (平铺)
src=np.array([[1,2,3],[4,5,6],[7,8,9]])
dst=cv2.copyMakeBorder(src,2,2,2,2,borderType=cv2.BORDER_REFLECT_101)

# %%
# python的scipy提供的卷积
from scipy.signal import convolve2d
# 该函数有以下参数：
# 参数1：输入图像
# 参数2：卷积核
# 参数3：mode :full\valid\same
# 参数4：boundary： fill（常数）/wrap（平铺）/symm （反射）
# 参数5： fillvalue
k=np.array([[-1,-2],[2,1]])
dst=convolve2d(src,k,mode='full',boundary='symm')

# %%
# 我们知道卷积基如果可分离的话，可以减少计算量
#具体分离卷积核可见p134-p140

# %%
# 高斯平滑
# 确定高斯卷积核就可以了，具体公式看书p140
# p141自定义了个高斯卷积核函数
# 高斯卷积核是可分离的卷积核，所以只需要获取竖立的卷积核就能进行计算了
gk=cv2.getGaussianKernel(ksize=3,sigma=2,ktype=cv2.CV_64F)
# 第一个参数是行数，只要行数就行了,第二个就是高斯函数的标准差

# %%
# 高斯卷积核可以用二项式近似，具体见书p142
# 下面定义高斯平滑函数
def gaussBlur(image,sigma,H,W,_boundary='fill',_fillvalue=0):
    # 搭建水平方向上的高斯卷积核
    guassKenral_x=cv2.getGaussianKernel(W,sigma=sigma,ktype=cv2.CV_64F)
    # 转置
    guassKenral_x=np.transpose(guassKenral_x)
    # 图像与水平高斯核卷积运算
    guassBlur_x=convolve2d(image,guassKenral_x,mode='same',boundary=_boundary,fillvalue=_fillvalue)
    # 搭建垂直方向上的卷积核
    guassKenral_y=cv2.getGaussianKernel(H,sigma=sigma,ktype=cv2.CV_64F)
    # 再与垂直上的高斯核卷积运算
    gaussBlur_xy=convolve2d(guassBlur_x,guassKenral_y,mode='same',boundary=_boundary,fillvalue=_fillvalue)
    return gaussBlur_xy
if __name__=='__main__':
    image=cv2.imread('2.jpg',0)
    h,w=image.shape
    cv2.imshow('image',image)
    # 高斯平滑
    blurImage=gaussBlur(image,5,h,w,'symm')
    # 对blurImage进行灰度显示
    blurImage=np.round(blurImage)
    blurImage=blurImage.astype(np.uint8)
    cv2.imshow('GaussBlur',blurImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# %%
# 高斯滤波
img=cv2.imread('2.jpg',0)
blur = cv2.GaussianBlur(img,(5,5),0)
cv2.imshow('I',img)
cv2.imshow('O',blur)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
# 均值平滑核快速均值平滑的知识在P147-150中，需要的话自己看

# %%
# 中值平滑也很简单，就是取一块区域的中值，在去除椒盐噪声的时候很有用
# 具体原理和实现不妨看书p155
# 上述平滑可以去除一些噪声，但是同时也会模糊边缘
# 所以接下来我们会介绍保持边缘的平滑算法：双边滤波和导向滤波
img=cv2.imread('2.jpg',0)
median = cv2.medianBlur(img,5)
cv2.imshow('I',img)
cv2.imshow('O',median)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
#双边滤波
# 需要位置距离权重矩阵 + 相似性权重模板 ，在点乘为新的权重模板
# 具体的手动实现方式参考书上的p162-163
# 采用cv2自带的来完成双边滤波
#cv2.bilateralFilter(src, d, sigmaColor, sigmaSpace)
#d – Diameter of each pixel neighborhood that is used during filtering.
# If it is non-positive, it is computed from sigmaSpace
# 9为邻域直径，75是空间高斯函数标准差，0.5为灰度值相似性高斯函数标准差
img=cv2.imread('2.jpg',0)
blur = cv2.bilateralFilter(img,9,75,0.5)
cv2.imshow('I',img)
cv2.imshow('O',blur)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
# 联合双边滤波
# 和双边滤波的区别在于：搭建相似性权重模板的时候
# 双边滤波是根据原图的每一个位置计算的
# 联合双边滤波是根据高斯平滑后的原图的每一个位置
# 手动完成看p168-170
def getClosenessWeight(sigma_g,H,W):
    r,c = np.mgrid[0:H:1,0:W:1]
    r=r.astype(np.float64)
    c=c.astype(np.float64)
    r-=(H-1)/2
    c-=(W-1)/2
    closeWeight = np.exp(-0.5*(np.power(r,2)+np.power(c,2))/math.pow(sigma_g,2))
    return closeWeight
def jointBLF(I,H,W,sigma_g,sigma_d,borderType=cv2.BORDER_DEFAULT):
    closenessWeight = getClosenessWeight(sigma_g,H,W)
    #高斯平滑
    Ig = cv2.GaussianBlur(I,(W,H),sigma_g)
    cH = int((H-1)/2)
    cW = int((W-1)/2)
    Ip = cv2.copyMakeBorder(I,cH,cH,cW,cW,borderType)
    Igp = cv2.copyMakeBorder(Ig,cH,cH,cW,cW,borderType)
    rows,cols = I.shape
    i,j = 0,0
    jblf = np.zeros(I.shape,np.float64)
    for r in np.arange(cH,cH+rows,1):
        for c in np.arange(cW,cW+cols,1):
            pixel = Igp[r][c]
            rTop,rBottom = r-cH,r+cH
            cLeft,cRight = c-cW,c+cW
            region = Igp[rTop:rBottom+1,cLeft:cRight+1]
            similarityWeight = np.exp(-0.5*np.power(region-pixel,2.0)/math.pow(sigma_d,2.0))
            weight = closenessWeight*similarityWeight
            weight = weight/np.sum(weight)
            jblf[i][j] = np.sum(Ip[rTop:rBottom+1,cLeft:cRight+1]*weight)
            j+=1
        j = 0
        i+=1
    return jblf

I = cv2.imread('queban.jpeg',0)
fI = I.astype(np.float64)
jblf = jointBLF(fI,17,17,7,2)
jblf = np.round(jblf)
jblf = jblf.astype(np.uint8)
cv2.imshow('I',I)
cv2.imshow('jblf',jblf)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
# 导向滤波
# 一种不依赖权重模板、保持边缘的滤波方法
# 理论看p173
# python从零开始实现参考书上为p174-p175
def guideFilter(I, p, winSize, eps):
    #I的均值平滑
    mean_I = cv2.blur(I, winSize)
    #p的均值平滑
    mean_p = cv2.blur(p, winSize)
    #I*I和I*p的均值平滑
    mean_II = cv2.blur(I*I, winSize)
    mean_Ip = cv2.blur(I*p, winSize)
    #方差
    var_I = mean_II - mean_I * mean_I #方差公式
    #协方差
    cov_Ip = mean_Ip - mean_I * mean_p
    a = cov_Ip / (var_I + eps)
    b = mean_p - a*mean_I
    #对a、b进行均值平滑
    mean_a = cv2.blur(a, winSize)
    mean_b = cv2.blur(b, winSize)
    q = mean_a*I + mean_b
    return q

img=cv2.imread('queban.jpeg',0)
img_0_1=img/255.0
result=guideFilter(img_0_1,img_0_1,(17,17),pow(0.2,2.0))
cv2.imshow('I',img)
cv2.imshow('O',result)
cv2.waitKey(0)
cv2.destroyAllWindows()


# %%
# 快速导向滤波理论看p176
def fast_guideFilter(I, p, winSize, eps, s):
    #输入图像的高、宽
    h, w = I.shape[:2]
    #缩小图像
    size = (int(round(w*s)), int(round(h*s)))
    small_I = cv2.resize(I, size, interpolation=cv2.INTER_CUBIC)
    small_p = cv2.resize(I, size, interpolation=cv2.INTER_CUBIC)
    #缩小滑动窗口
    X = winSize[0]
    small_winSize = (int(round(X*s)), int(round(X*s)))
    #I的均值平滑
    mean_small_I = cv2.blur(small_I, small_winSize)
    #p的均值平滑
    mean_small_p = cv2.blur(small_p, small_winSize)
    #I*I和I*p的均值平滑
    mean_small_II = cv2.blur(small_I*small_I, small_winSize)
    mean_small_Ip = cv2.blur(small_I*small_p, small_winSize)
    #方差
    var_small_I = mean_small_II - mean_small_I * mean_small_I #方差公式
    #协方差
    cov_small_Ip = mean_small_Ip - mean_small_I * mean_small_p
    small_a = cov_small_Ip / (var_small_I + eps)
    small_b = mean_small_p - small_a*mean_small_I
    #对a、b进行均值平滑
    mean_small_a = cv2.blur(small_a, small_winSize)
    mean_small_b = cv2.blur(small_b, small_winSize)
    #放大
    size1 = (w, h)
    mean_a = cv2.resize(mean_small_a, size1, interpolation=cv2.INTER_LINEAR)
    mean_b = cv2.resize(mean_small_b, size1, interpolation=cv2.INTER_LINEAR)
    q = mean_a*I + mean_b
    return q

img=cv2.imread('queban.jpeg',0)
img_0_1=img/255.0
result=fast_guideFilter(img_0_1,img_0_1,(17,17),0.04,0.3)
cv2.imshow('I',img)
cv2.imshow('O',result)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
# 彩色图像的上述所有函数的处理
import cv2
import numpy as np
import math
def guideFilter(I, p, winSize, eps):
    #I的均值平滑
    mean_I = cv2.blur(I, winSize)
    #p的均值平滑
    mean_p = cv2.blur(p, winSize)
    #I*I和I*p的均值平滑
    mean_II = cv2.blur(I*I, winSize)
    mean_Ip = cv2.blur(I*p, winSize)
    #方差
    var_I = mean_II - mean_I * mean_I #方差公式
    #协方差
    cov_Ip = mean_Ip - mean_I * mean_p
    a = cov_Ip / (var_I + eps)
    b = mean_p - a*mean_I
    #对a、b进行均值平滑
    mean_a = cv2.blur(a, winSize)
    mean_b = cv2.blur(b, winSize)
    q = mean_a*I + mean_b
    return q

img=cv2.imread('queban.jpeg',cv2.IMREAD_ANYCOLOR)
img=img/255.0 #归一化
b, g, r = cv2.split(img)
gf1 = guideFilter(b, b, (36,36), math.pow(0.1,2.0))
gf2 = guideFilter(g, g, (36,36), math.pow(0.1,2.0))
gf3 = guideFilter(r, r, (36,36), math.pow(0.1,2.0))
gf = cv2.merge([gf1, gf2, gf3])
cv2.imshow('I',img)
# 以下这段有没有都一样
# gf=gf*255
# gf[gf>255]=255
# gf=np.round(gf).astype(np.uint8)
cv2.imshow('O',gf)
cv2.waitKey(0)
cv2.destroyAllWindows()

