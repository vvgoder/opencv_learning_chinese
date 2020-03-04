'''
@Author: zhou wei
@Date: 2019-11-08 23:04:05
@LastEditTime : 2020-01-17 20:44:54
@LastEditors  : Please set LastEditors
@Description:读取+展示+加字+画图+调节面板
@FilePath: \pactise_opencv_python\practise_foundation.py
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt

# %%
# read the image and display it
# cv2.IMREAD_COLOR：读入一副彩色图像。图像的透明度会被忽略，这是默认参数。
# cv2.IMREAD_GRAYSCALE：以灰度模式读入图像
# cv2.IMREAD_UNCHANGED：读入一幅图像，并且包括图像的alpha 通道
img=cv2.imread('e:/save_data/3.jpg',cv2.IMREAD_UNCHANGED)
cv2.namedWindow('image', cv2.WINDOW_NORMAL) #you can adjust the size of windows now
cv2.imshow('image',img) #if run here without following code,the image will disappeared very quickly,the name should be same as the window's name
cv2.waitKey(0)#cv2.waitKey() 是一个键盘绑定函数。需要指出的是它的时间尺度是毫秒级。函数等待特定的几毫秒，看是否有键盘输入。
cv2.destroyAllWindows() #cv2.destroyAllWindows() 可以轻易删除任何我们建立的窗口
#save the image
cv2.imwrite('ft.png',img) #the first parameter is saved path

# %%
img = cv2.imread('e:/save_data/3.jpg',0) #para 0 means it converts it to grey image
cv2.imshow('image',img)
k = cv2.waitKey(0)&0xFF #64-bit operation system
if k == 27: # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
    cv2.imwrite('ft.png', img)
    cv2.destroyAllWindows()

# %%
#using matplotlib.pyplot to draw grey image
img = cv2.imread('e:/save_data/3.jpg',0)
plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([]) # to hide tick values on X and Y axis
plt.show()

# %%
#notice:the image format is RGB when using plt read,but is BGR when opencv read
#maybe create some problems when using both of them
img = cv2.imread('e:/save_data/3.jpg')
b,g,r = cv2.split(img)
img2 = cv2.merge([r,g,b])
plt.subplot(121)
plt.imshow(img) # expects distorted(扭曲的） color
plt.subplot(122)
plt.imshow(img2) # expect true color
plt.show()
cv2.imshow('bgr image',img) # expects true color
cv2.imshow('rgb image',img2) # expects distorted color
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
#from Image to video,learn something about video processing
cap = cv2.VideoCapture(0) #0 means using Built-in camera,1 or other numbes means other Built-in camera
while(True):
# Capture frame-by-frame
    ret, frame = cap.read() #if read successfully ,the para ret is True ,else False
# Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

#some function as follows:
# cap.isOpened() checks Whether initialization succeeded
# cap.get(propId) is being used to obtain some parameter information of the video
# I can use cap.get(3) and cap.get(4) to see the width and height of each frame
# cap.get(3)=width ,cap.get(4)=height

# %%
cap = cv2.VideoCapture(0) #use built-in camera
cap=cv2.VideoCapture('e:/save_data/5.mp4') #using Local video file
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID') #video codec，MJPG results in high size video. X264 gives very small size video)
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480)) #分别表示：输出视频名称，编码器，帧率，帧宽和高

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        frame = cv2.flip(frame,0)
        # write the flipped frame
        out.write(frame) #If you want the saved video can play normally,Keep the image size the same as the original one
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()

# %%
#draw some figures on the picture
img=np.zeros((512,512,3), np.uint8) #black background
cv2.line(img,(0,0),(511,511),(255,0,0),5) #draw line ,the first parameter is img,the second is The starting point of the line,
#the third is the end point of the line ,the forth is the color of the line,the fifth is the linewidth of the line
cv2.imshow('line',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#draw rectangle
img=np.zeros((512,512,3), np.uint8) #black background
cv2.rectangle(img,(384,0),(510,128),(0,255,0),3) #the second parameter is the coordinates of upper left corner,
# the third para is  the coordinates of lower right corner,
cv2.circle(img,(447,63), 63, (0,0,255), -1) #the second para :the center of circle ;the third para:radius
cv2.ellipse(img,(256,256),(100,50),0,0,360,255,-1) #the second para:center of ellipse;
                                                    # the third para:major axes and minor axes
                                                    #the forth :The angle at which the ellipse rotates counterclockwise
                                                    #the sixth ：0 or 360 means complete ellipse
font=cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,'OpenCV',(10,500), font, 4,(255,255,255),2) #the second: content
                                                            #the third:position of text
                                                            #the forth:font of the content
                                                            #the fifth:the size of font
                                                            #the sixth:color of the font
                                                            #the seventh:the linewidth
cv2.imshow('line',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
# use mouse to draw circle everywhere
events=[i for i in dir(cv2) if 'EVENT'in i]
def draw_circle(event,x,y,flags,param):  #this is the main operation function, if you double-click the mouse,
                                            # you will draw a circle which focus on the location you click on
    if event==cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img,(x,y),100,(255,0,0),-1)
# 创建图像与窗口并将窗口与回调函数绑定
img=np.zeros((512,512,3),np.uint8)
cv2.namedWindow('image') #create the namewindows
cv2.setMouseCallback('image',draw_circle)
while(1):
    cv2.imshow('image',img)
    if cv2.waitKey(20)&0xFF==27:
        break
cv2.destroyAllWindows()

# %%
# 当鼠标按下时变为True
drawing=False
# 如果mode 为true 绘制矩形。按下'm' 变成绘制曲线。
mode=True
ix,iy=-1,-1
# 创建回调函数
def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing,mode
# 当按下左键是返回起始位置坐标,cv2.EVENT_LBUTTONDOWN是鼠标左键按下的信号
    if event==cv2.EVENT_LBUTTONDOWN:
        drawing=True
        ix,iy=x,y
# 当鼠标左键按下并移动是绘制图形。event 可以查看移动，flag 查看是否按下
    elif event==cv2.EVENT_MOUSEMOVE and flags==cv2.EVENT_FLAG_LBUTTON:
        if drawing==True:
            if mode==True:
                cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
            else:
# 绘制圆圈，小圆点连在一起就成了线，3 代表了笔画的粗细
                cv2.circle(img,(x,y),3,(0,0,255),-1)
# 下面注释掉的代码是起始点为圆心，起点到终点为半径的
# r=int(np.sqrt((x-ix)**2+(y-iy)**2))
# cv2.circle(img,(x,y),r,(0,0,255),-1)
# 当鼠标松开停止绘画。
    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False
# if mode==True:
# cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
# else:
# cv2.circle(img,(x,y),5,(0,0,255),-1)
img=np.zeros((512,512,3),np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)
while(1):
    cv2.imshow('image',img)
    k=cv2.waitKey(1)&0xFF
    if k==ord('m'):
        mode=not mode
    elif k==27:
        break

# %%
#绘制一个背景色可调的画板
def nothing(x):
    pass
# 创建一副黑色图像
img=np.zeros((300,512,3),np.uint8)
cv2.namedWindow('image')
cv2.createTrackbar('R','image',0,255,nothing) #第 一个参数是滑动条的名字，
                                                # 第二个参数是滑动条被放置窗口的名字，
                                                # 第三个参数是滑动条的默认位置
                                                #第四个参数是滑动条的最大值
                                                # 第五个函数是回调函数，每次滑动条的滑动都会调用回调函 数。
                                                # 回调函数通常都会含有一个默认参数，就是滑动条的位置。在本例中这个
                                                # 函数不用做任何事情，我们只需要pass 就可以了
cv2.createTrackbar('G','image',0,255,nothing)
cv2.createTrackbar('B','image',0,255,nothing)
switch='0:OFF\n1:ON'
cv2.createTrackbar(switch,'image',0,1,nothing) #开关进度条
while(1):
    cv2.imshow('image',img)
    k=cv2.waitKey(1)&0xFF
    if k==27:
        break
    r=cv2.getTrackbarPos('R','image')
    g=cv2.getTrackbarPos('G','image')
    b=cv2.getTrackbarPos('B','image')
    s=cv2.getTrackbarPos(switch,'image')
    if s==0: #如果为关，则黑色不变
        img[:]=0
    else: #如果为开，则bgr组合
        img[:]=[b,g,r]
cv2.destroyAllWindows()

# %%
#绘制一个笔色可调的画板
def nothing(x):
    pass
# 当鼠标按下时变为True
drawing=False
# 如果mode 为true 绘制矩形。按下'm' 变成绘制曲线。
mode=True
ix,iy=-1,-1
# 创建回调函数
def draw_circle(event,x,y,flags,param):
    r=cv2.getTrackbarPos('R','image')
    g=cv2.getTrackbarPos('G','image')
    b=cv2.getTrackbarPos('B','image')
    color=(b,g,r)
    global ix,iy,drawing,mode
# 当按下左键是返回起始位置坐标
    if event==cv2.EVENT_LBUTTONDOWN:
        drawing=True
        ix,iy=x,y
# 当鼠标左键按下并移动是绘制图形。event 可以查看移动，flag 查看是否按下
    elif event==cv2.EVENT_MOUSEMOVE and flags==cv2.EVENT_FLAG_LBUTTON:
        if drawing==True:
            if mode==True:
                cv2.rectangle(img,(ix,iy),(x,y),color,-1)
            else:
# 绘制圆圈，小圆点连在一起就成了线，3 代表了笔画的粗细
                cv2.circle(img,(x,y),3,color,-1)
# 下面注释掉的代码是起始点为圆心，起点到终点为半径的
# r=int(np.sqrt((x-ix)**2+(y-iy)**2))
# cv2.circle(img,(x,y),r,(0,0,255),-1)
# 当鼠标松开停止绘画。
    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False
# if mode==True:
# cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
# else:
# cv2.circle(img,(x,y),5,(0,0,255),-1)
img=np.zeros((512,512,3),np.uint8)
cv2.namedWindow('image')
cv2.createTrackbar('R','image',0,255,nothing)
cv2.createTrackbar('G','image',0,255,nothing)
cv2.createTrackbar('B','image',0,255,nothing)
cv2.setMouseCallback('image',draw_circle)
while(1):
    cv2.imshow('image',img)
    k=cv2.waitKey(1)&0xFF
    if k==ord('m'):
        mode=not mode
    elif k==27:break


