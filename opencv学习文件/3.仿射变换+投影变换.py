import cv2
import numpy as np
import matplotlib.pyplot as plt
# %%
# cv2.warpAffine()仿射变换函数，可实现旋转，平移，缩放；变换后的平行线依旧平行
# cv2.warpAffine(src, M, dsize, dst=None, flags=None, borderMode=None, borderValue=None) --> dst
# src：输入图像     dst：输出图像
# M：2×3的变换矩阵
# dsize：变换后输出图像尺寸
# flag：插值方法
# borderMode：边界像素外扩方式
# borderValue：边界像素插值，默认用0填充

# %%
# 变换矩阵M可通过cv2.getAffineTransfrom(points1, points2)函数获得
# 变换矩阵的获取需要至少三组变换前后对应的点坐标，设取原图上的三个点组成矩阵points1，变换后的三个点组成的矩阵points2
# 如：
points1 = np.float32([ [30,30], [100,40], [40,100] ])
points2 = np.float32([ [60,60], [40,100], [80,20] ])
M=cv2.getAffineTransform(points1, points2)
print(M)

# %%
## 用Numpy 数组构建这个2*3的矩阵（数据类型是np.float32）
img=cv2.imread('2.jpg')
m_Translational=np.array([[1,0,50],[0,1,100]],np.float32) #一定要指定类型，而且一定是float型，这里是平移
                                                        # [[1,0,x],
                                                        # [0,1,y]]将图片平移（x,y)
# 这里的(2*img.shape[1], 2*img.shape[0])不是把图片放大两倍，而是将放置图片的背景放大两把
img_translational=cv2.warpAffine(img,m_Translational,(2*img.shape[1], 2*img.shape[0]),borderValue=125)
cv2.imshow('img',img)
cv2.imshow('img_translational',img_translational)
cv2.waitKey(0)

# %%
# 使用仿射矩阵进行缩放
m_suofang=np.array([[0.5,0,0],[0,0.5,0]],np.float32)#一定要指定类型，而且一定是float型，这里是平移
                                                        # [[sx,0,0],
                                                        # [0,sy,0]]将图片按着原点（sx,sy)
img_suofang=cv2.warpAffine(img,m_suofang,(img.shape[1], img.shape[0]),borderValue=125)
cv2.imshow('img',img)
cv2.imshow('img_suofang',img_suofang)
cv2.waitKey(0)

# %%
# 使用仿射矩阵进行先缩放后平移
m_sf_py=np.array([[0.5,0,50],[0,0.5,50]],np.float32)#一定要指定类型，而且一定是float型，这里是平移
                                                        # [[sx,0,x],
                                                        # [0,sy,y]]先缩小，再平移
img_sf_py=cv2.warpAffine(img,m_sf_py,(img.shape[1], img.shape[0]),borderValue=125)
cv2.imshow('img',img)
cv2.imshow('img_suofang',img_sf_py)
cv2.waitKey(0)

# %%
# 一般来说，图片进行旋转的流程是：
# 1、选择旋转中心
# 2、缩放
# 3、旋转
# 4、将中心回来原来
# 这样仿射矩阵多麻烦呀，所以用一个函数，来创建用来满足上面的要求的仿射矩阵
img=cv2.imread('2.jpg')
rows,cols,_=img.shape
# cv2.getRotationMatrix2D的
# 第一个参数为旋转中心
# 第二个为逆时针旋转角度
# 第三个为缩放比例
M=cv2.getRotationMatrix2D((cols/2,rows/2),45,0.6)
dst=cv2.warpAffine(img,M,(2*cols,2*rows))
while(1):
    cv2.imshow('img',dst)
    if cv2.waitKey(1)&0xFF==27:
        break
cv2.destroyAllWindows()

# %%
# 旋转的另一种方式
img=cv2.imread('2.jpg')
#这里的cv2.ROTATE_90_CLOCKWISE是顺时针90
# cv2.ROTATE_90_COUNTERCLOCKWISE是逆时针90
rimg=cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)
plt.subplot(121)
plt.imshow(img)
plt.title('Input')
plt.subplot(122)
plt.imshow(rimg)
plt.title('Output')
plt.show()

# %%
# 已经原图3个点和输出图像的对应点，创建仿射矩阵

img=cv2.imread('2.jpg')
rows,cols,ch=img.shape

pts1=np.float32([[50,50],[200,50],[50,200]])
pts2=np.float32([[10,100],[200,50],[100,250]])
M=cv2.getAffineTransform(pts1,pts2) #转移矩阵

dst=cv2.warpAffine(img,M,(cols,rows))

plt.subplot(121)
plt.imshow(img)
plt.title('Input')
plt.subplot(122)
plt.imshow(dst)
plt.title('Output')
plt.show()

# %%
# 透视变换:投影变换
# 我们需要一个3x3 变换矩阵。
# 你需要在输入图像上找4 个点，以及他们在输出图像上对应的位置。
# 这四个点中的任意三个都不能共线。
# 这个变换矩阵可以有函数cv2.getPerspectiveTransform() 构建。
# 然后把这个矩阵传给函数cv2.warpPerspective。
import cv2
import numpy as np
from matplotlib import pyplot as plt
img=cv2.imread('16.png')
rows,cols,ch=img.shape
pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]]) #原图四个点
pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]]) #在输出图像中的位置
M=cv2.getPerspectiveTransform(pts1,pts2) #透视转移矩阵
dst=cv2.warpPerspective(img,M,(300,300)) #转移矩阵*原图，size设置为(300,300)

plt.subplot(121)
plt.imshow(img)
plt.title('Input')
plt.subplot(122)
plt.imshow(dst)
plt.title('Output')
plt.show()

# %%
# 另一些明显的投影变换
# 类似于平面物体在三维空间进行旋转、平移
image=cv2.imread('data/2.jpg')
h,w,c=image.shape
# src=np.array([[0,0],[w-1,0],[0,h-1],[w-1,h-1]],np.float32)
# dst=np.array([[50,50],[w/3,50],[50,h-1],[w-1,h-1]],np.float32)
src=np.array([[200.8619 , 360.3923 ],
       [203.46745, 313.48337],
       [206.89188, 268.31717],
       [210.72318, 225.57188]], dtype=np.float32)
dst=np.array([[0., 0.],
       [1., 0.],
       [2., 0.],
       [3., 0.]], dtype=np.float32)
M=cv2.getPerspectiveTransform(src,dst)
dst=cv2.warpPerspective(image,M,(w,h),borderValue=125)
plt.imshow(dst)
plt.title('Output')
plt.show()

# %%
# 笛卡尔坐标和极坐标之间的相互转换
# （1）将笛卡尔坐标系转为极坐标系
# 例子：以（1，2）为中心，将（0，0），（1，1），（2，2）转为极坐标表示
import cv2
import numpy as np
# 这里转坐标系中心为（0，0）的坐标点取值
x=np.array([0,1,2],np.float)-1
y=np.array([0,1,2],np.float)-2
r,theta=cv2.cartToPolar(x,y,angleInDegrees=True)
print(r,theta)

# %%
# (2)将极坐标转为笛卡尔坐标系
# 例子：将（30，10）、（31，10）、（30，11）、（31，11）转为以（-12，15）为中心的笛卡尔坐标
angle=np.array([[30,31],[31,30],[30,31]],np.float)
r=np.array([[10,10],[11,11]],np.float)
x,y=cv2.polarToCart(r,angle,angleInDegrees=True)
# 中心偏移，上面的是以（0，0）为中心的，需要做转换
x+=-12
y+=15

# %%
# 如何将圆形图转为方型图，即极坐标转为笛卡尔坐标系，见书p85
# 线性极坐标函数和对数极坐标函数都可以将极坐标转为笛卡尔坐标系，见书p91-95