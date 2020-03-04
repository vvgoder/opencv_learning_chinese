'''
@Author: your name
@Date: 2020-02-02 21:42:32
@LastEditTime : 2020-02-02 22:22:08
@LastEditors  : Please set LastEditors
@Description: In User Settings Edit
@FilePath: \pactise_opencv_python\背景减除.py
'''
import numpy as np
import cv2

# %%
# BackgroundSubtractorMOG2
cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows = True)
while(1):
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    cv2.imshow('frame',fgmask)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()

# %%
import numpy as np
import cv2
cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorKNN()
while(1):
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    cv2.imshow('frame',fgmask)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()

# %%
# 静态背景图消除法
import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob

# %%
# 读取背景
background = cv2.imread('frame/3042.jpg', 0)
files = sorted(glob.glob('frame/*.jpg'))
for file in files:
    if file == 'frame/3042.jpg':
        continue

    img0 = cv2.imread(file, 0)
    # 其他图减去背景
    img = img0 - background
    # 高斯平滑（去噪声）
    imageGray = cv2.GaussianBlur(img,(15,15),25);
    # 阈值分割
    ret,thresh = cv2.threshold(imageGray,127,255,cv2.THRESH_BINARY_INV)
    # 轮廓检测
    im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    # cnt = np.row_stack((contours[i] for i in range(len(contours))))
    # # 轮廓画包围框
    # x, y, w, h = cv2.boundingRect(cnt)
    # cv2.rectangle(img0, (x, y), (x+w, y+h), (255, 0, 0), 4)

    # cv2.drawContours(img, contours, len(contours)-1, (255,0,0), 3)
    # cv2.drawContours(img, contours, len(contours)-1, (255,0,0), 3)
    # print(len(contours))
    plt.imshow(img)
    plt.show()

    # img0 = cv2.imread(file)
    # # roi = img0[x:x+w, y:y+h]
    # roi = img0[y:y+h, x:x+w]
    # cv2.imwrite(file.replace('bmp', 'jpg'), roi)


# %%
