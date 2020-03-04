import cv2
import numpy as np

# %%
# 腐蚀操作，白色区域变小，黑色变大，可以去除一些小白色噪声
# 腐蚀实质，卷积核区域取最小值
# 一般针对二值化图像处理（阈值分割后的）
img=cv2.imread('data/seu.png',0)
# 阈值分割
otsuthe=0
# 输出自己的otsu阈值和目标图像
otsuthe,dst_otsu =cv2.threshold(img,otsuthe,255,cv2.THRESH_OTSU)
kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(dst_otsu,kernel,iterations = 1)
cv2.imshow('erosion',erosion)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
# kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)) #椭圆卷积核
# kernel=cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5)) #十字卷积核
kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)) #方型卷积核
dst=cv2.erode(dst_otsu,kernel,iterations=1)
#边界提取
e=dst_otsu-dst
cv2.imshow('erosion',dst)
cv2.imshow('edge',e)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
# 膨胀：取卷积核内最大值
# 膨胀：白色区域变大，黑色区域变小，可以去除一些小黑色噪声
dilation = cv2.dilate(dst_otsu,kernel,iterations = 1)
cv2.imshow('dilation',dilation)
cv2.waitKey(0)
cv2.destroyAllWindows()


# %%
def nothing(*arg):
    pass
# trackerBar 动态调节结构元半径
img=cv2.imread('data/seu.png',0)
# 阈值分割
otsuthe=0
# 输出自己的otsu阈值和目标图像
otsuthe,dst_otsu =cv2.threshold(img,otsuthe,255,cv2.THRESH_OTSU)
# 调节结构元半径
cv2.namedWindow('dilate',1) #这是必要的
cv2.createTrackbar('r','dilate',1,20,nothing)
while True:
    r=cv2.getTrackbarPos('r','dilate')
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(2*r+1,2*r+1))
    erosion = cv2.erode(dst_otsu,kernel,iterations = 1)
    cv2.imshow('erosion',erosion)
    ch=cv2.waitKey(5)
    if ch==27: #ESC退出循环
        break
cv2.destroyAllWindows()

# %%
# 开运算：先腐蚀后膨胀
# 作用：消除有一些小的白噪声，在纤细处分分离物体，对大物体（白色）平滑边界
# 迭代次数越大，可以消除较大的异常点（白色区域）
closing = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel,iterations=10)

# %%
# 闭运算：先膨胀后腐蚀
# 作用：消除有一些小的黑噪声，在纤细处连接物体，对大物体（黑色）平滑边界
# 迭代次数越大，可以消除较大的异常点（黑色区域）
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel,iterations=10)

# %%
# 顶帽操作：原图-开运算
# 开运算可以消除较暗背景下较亮的区域 ，那么顶帽操作就是得到原图中灰度较亮的区域
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)

# %%
# 底帽操作：原图-闭运算
# 闭运算可以消除较亮背景下较暗的区域 ，那么底帽操作就是得到原图中灰度较暗的区域
blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)

# %%
# 形态学梯度：膨胀-腐蚀
# 获取轮廓信息
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
