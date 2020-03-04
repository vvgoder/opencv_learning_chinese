'''
@Author: zhou wei
@Date: 2019-11-08 23:04:05
@LastEditTime : 2020-01-17 20:43:35
@LastEditors  : Please set LastEditors
@Description: 边界扩充+算术运算+图像混合+RGB转HSV
@FilePath: \pactise_opencv_python\practise_foundation.py
'''
# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt

# %%
image=cv2.imread('15.png') #type(image)=<class 'numpy.ndarray'>
print(type(image))
print (image.item(10,10,2)) #使用np.array.item(pos)获取位置的值，为标量
image.itemset((10,10,2),100) #更改某一位置的值
print (image.item(10,10,2))
#因为image的类型为numpy.array,所以可以使用shape获取图片的形状,也可以使用size获取多少个最小单元
print(image.shape)
print(image.size)
print(image.dtype) #获取图片的类型

# #拆开以及合并通道
b,g,r=cv2.split(image) #cv2.split() 是一个比较耗时的操作。只有真正需要时才用它，能用Numpy 索引就尽量用，毕竟numpy对矩阵操作加速
img=cv2.merge(b,g,r)
b=img[:,:,0]
img[:,:,2]=0 #r通道置为0

# %%
#为图像padding，扩充边界
import cv2
import numpy as np
from matplotlib import pyplot as plt
BLUE=[255,0,0]
img1=cv2.imread('15.png')
# top, bottom, left, right 对应边界的像素数目。
# borderType 要添加那种类型的边界，类型如下
#           – cv2.BORDER_CONSTANT 添加有颜色的常数值边界，还需下一个参数（value）， value 边界颜色
#           – cv2.BORDER_REFLECT 边界元素的镜像。比如: fedcba|abcdefgh|hgfedcb
#           – cv2.BORDER_REFLECT_101 or cv2.BORDER_DEFAULT， 跟上面一样，但稍作改动。例如: gfedcb|abcdefgh|gfedcba
#           – cv2.BORDER_REPLICATE 重复最后一个元素。例如: aaaaaa|abcdefgh|hhhhhhh
#           cv2.BORDER_WRAP 不知道怎么说了, 就像这样: cdefgh|abcdefgh|abcdefg

replicate = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REPLICATE)
reflect = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REFLECT)
reflect101 = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REFLECT_101)
wrap = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_WRAP)
constant= cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_CONSTANT,value=BLUE)
plt.subplot(231),plt.imshow(img1,'gray'),plt.title('ORIGINAL')
plt.subplot(232),plt.imshow(replicate,'gray'),plt.title('REPLICATE')
plt.subplot(233),plt.imshow(reflect,'gray'),plt.title('REFLECT')
plt.subplot(234),plt.imshow(reflect101,'gray'),plt.title('REFLECT_101')
plt.subplot(235),plt.imshow(wrap,'gray'),plt.title('WRAP')
plt.subplot(236),plt.imshow(constant,'gray'),plt.title('CONSTANT')
plt.show()

# %%
#图像上的算数运算
#cv2.add() 将两幅图像进行加法运算，当然也可以直接使用numpy，res=img1+img，和直接相加还是有区别的
# 两幅图像的大小，类型必须一致，或者第二个图像可以使一个简单的标量值。
x = np.uint8([250])
y = np.uint8([10])
print (cv2.add(x,y)) #输出为255，截断在255
print (x+y)  #输出为4，260%255=4

#图像混合
# 函数cv2.addWeighted() 可以对图片进行混合操作。
import cv2
img1=cv2.imread('15.png')
img2=cv2.imread('16.png')
dst=cv2.addWeighted(img1,0.1,img2,0.9,0) #线性求和加截取,要求shape一样
cv2.imshow('dst',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
#按位运算
import cv2
import numpy as np
# 加载图像
img1 = cv2.imread('2.jpg')
img2 = cv2.imread('seu.png')
img2=cv2.resize(img2,(100,100))
# I want to put logo on top-left corner, So I create a ROI
rows,cols,channels = img2.shape
roi = img1[0:rows, 0:cols ]
# Now create a mask of logo and create its inverse mask also
img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY) #转为灰度图
cv2.imshow('gray',img2gray)
cv2.waitKey(0) #等待鼠标取消这幅图像，在执行下面的code
ret, mask = cv2.threshold(img2gray, 220, 255, cv2.THRESH_BINARY) #转为二值图，175为在最大值，最大值为255（白色）,ret为二值化阈值，mask为二值图
cv2.imshow('mask',mask)
cv2.waitKey(0)
mask_inv = cv2.bitwise_not(mask) #二值图颠倒，逻辑取反
# 保留roi 中与mask 中不为零的值对应的像素的值，其他值为0
# 注意这里必须有mask=mask 或者mask=mask_inv, 其中的mask= 不能忽略
img1_bg = cv2.bitwise_and(roi,roi,mask = mask) #按位相与，mask中白色（255）保留，黑色（0）剔除，这里把背景图腾出空间
img2_fg = cv2.bitwise_and(img2,img2,mask = mask_inv) #这里同上，把前景图扣除物体
# Put logo in ROI and modify the main image
dst = cv2.add(img1_bg,img2_fg) #把后者（img_fg）放入前者中去
img1[0:rows, 0:cols ] = dst
cv2.imshow('res',img1)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
# 程序性能检测及优化
# cv2.getTickCount 函数返回从参考点到这个函数被执行的时钟数
# cv2.getTickFrequency 返回时钟频率，或者说每秒钟的时钟数
import cv2
import numpy as np
e1 = cv2.getTickCount()
for i in range(1000000):
    i=i+1
e2 = cv2.getTickCount()
time = (e2 - e1)/ cv2.getTickFrequency()
print('cv time count',time)

#使用time模块来计时，甚至还可以使用profile来获得更详细的运行报告
import time
from time import time
time_start=time()
for i in range(1000000):
    i=i+1
time_end=time()
time_dua=time_end-time_start
print('time time',time_dua)

# %%
# 你可以使用函数cv2.useOptimized()来查看优化是否被开启了
# 使用函数cv2.setUseOptimized() 来开启优化。

# 打开ipython，然后使用魔法工具，如：% time、%timeit 求解某一个运算的时间
# 注意：一般情况下OpenCV 的函数要比Numpy 函数快。所以对于相同的操作最好使用OpenCV 的函数。
# 当然也有例外，尤其是当使用Numpy 对视图（而非复制）进行操作时

# 1. 尽量避免使用循环，尤其双层三层循环，它们天生就是非常慢的。
# 2. 算法中尽量使用向量操作，因为Numpy 和OpenCV 都对向量操作进行
# 了优化。
# 3. 利用高速缓存一致性。
# 4. 没有必要的话就不要复制数组。使用视图来代替复制。数组复制是非常浪
# 费资源的。

# %%
# OpenCV 中的数学工具
# 颜色空间转换
#cv2有超过150中色彩转换，最常用的就是bgr转HSV，和bgr转灰度，就是cv2.COLOR_BGR2GRAY和cv2.COLOR_BGR2HSV
import cv2
flags=[i for i in dir(cv2) if i.startswith('COLOR_')]
print (flags)

# 在OpenCV 的HSV 格式中，H（色彩/色度）的取值范围是[0，179]，
# S（饱和度）的取值范围[0，255]，V（亮度）的取值范围[0，255]。
# 但是不同的软件使用的值可能不同。所以当你需要拿OpenCV 的HSV 值与其他软件的HSV 值进行对比时
# 一定要记得归一化。

# %%
# 一幅图像从BGR 转换到HSV 了，我们可以利用这一点来提取带有某个特定颜色的物体。
# 在HSV 颜色空间中要比在BGR 空间中更容易表示一个特定颜色，这样我们可以做特定颜色物体的追踪
# 当然，这是物体追踪就简单的办法
import cv2
import numpy as np
cap=cv2.VideoCapture(0)
while(1):
# 获取每一帧
    ret,frame=cap.read()
    # 转换到HSV
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    # 设定蓝色的阈值
    lower_blue=np.array([100,50,50]) #第一个维度是颜色空间
    upper_blue=np.array([130,255,255]) 
    # 根据阈值构建掩模，满足上面的为白色，不满足的为黑色
    mask=cv2.inRange(hsv,lower_blue,upper_blue)
    # 对原图像和掩模进行位运算
    res=cv2.bitwise_and(frame,frame,mask=mask) #保留蓝色的部分，其余的都不要
    # 显示图像
    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    k=cv2.waitKey(5)&0xFF
    if k==27:
        break
# 关闭窗口
cv2.destroyAllWindows()

# %%
#找到某一颜色的hsv，以下就以绿色为例子
import cv2
import numpy as np
green=np.uint8([0,255,0]) #这是bgr中的绿色，以下会报错，需要三维数据
hsv_green=cv2.cvtColor(green,cv2.COLOR_BGR2HSV) #报错

# 所以不能用[0,255,0]，而要用[[[0,255,0]]]
# 这里的三层括号应该分别对应于cvArray，cvMat，IplImage
import cv2
green=np.uint8([[[0,255,0]]])
hsv_green=cv2.cvtColor(green,cv2.COLOR_BGR2HSV)
print (hsv_green)  #输出为[[[ 60 255 255]]]

# %%
# 扩展图片
# 缩放时我们推荐使用cv2.INTER_AREA，
# 在扩展时我们推荐使用v2.INTER_CUBIC（慢) 和v2.INTER_LINEAR
# 默认情况下所有改变图像尺寸大小的操作使用的插值方法都是cv2.INTER_LINEAR
img=cv2.imread('2.jpg')
# 下面的None 本应该是输出图像的尺寸，但是因为后边我们设置了缩放因子
# 因此这里为None
# res=cv2.resize(img,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
#OR
# 这里呢，我们直接设置输出图像的尺寸，所以不用设置缩放因子
height,width=img.shape[:2]
res=cv2.resize(img,(2*width,2*height),interpolation=cv2.INTER_LINEAR)
while(1):
    cv2.imshow('res',res)
    cv2.imshow('img',img)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cv2.destroyAllWindows()

