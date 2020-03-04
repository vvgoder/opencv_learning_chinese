'''
@Author: your name
@Date: 2020-02-06 15:22:47
@LastEditTime : 2020-02-07 18:03:34
@LastEditors  : Please set LastEditors
@Description: In User Settings Edit8
@FilePath: \pactise_opencv_python\图像去黑边.py
'''
import os
import cv2
import numpy as np
from scipy.stats import mode
import time
import concurrent.futures

'''
    multi-process to crop pictures.
'''
# 定义裁剪函数
def crop(file_path_list):
    origin_path, save_path = file_path_list
    img = cv2.imread(origin_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    closed_1 = cv2.erode(gray, None, iterations=4)
    closed_1 = cv2.dilate(closed_1, None, iterations=4)
    blurred = cv2.blur(closed_1, (9, 9))
    # get the most frequent pixel
    # 找到出现次数最多的像素点
    num = mode(blurred.flat)[0][0] + 1
    # the threshold depends on the mode of your images' pixels
    num = num if num <= 30 else 1

    # 阈值分割后的二值图
    _, thresh = cv2.threshold(blurred, num, 255, cv2.THRESH_BINARY)

    # you can control the size of kernel according your need.
    kernel = np.ones((13, 13), np.uint8)
    closed_2 = cv2.erode(thresh, kernel, iterations=4)
    closed_2 = cv2.dilate(closed_2, kernel, iterations=4)

    # 返回边缘包围点坐标
    _, cnts, _ = cv2.findContours(closed_2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 从大到小排序
    c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

    # compute the rotated bounding box of the largest contour
    # 计算包围边缘c的最小面积矩阵参数，返回中心和宽高+旋转角度
    rect = cv2.minAreaRect(c)
    box = np.int0(cv2.boxPoints(rect))

    # draw a bounding box arounded the detected barcode and display the image
    # cv2.drawContours(img, [box], -1, (0, 255, 0), 3)
    # cv2.imshow("Image", img)
    # cv2.imwrite("pic.jpg", img)
    # cv2.waitKey(0)

    # 裁剪
    xs = [i[0] for i in box]
    ys = [i[1] for i in box]
    x1 = min(xs)
    x2 = max(xs)
    y1 = min(ys)
    y2 = max(ys)

    # 防止出现bug
    if x1<0:
        x1=0
    if y1<0:
        y1=0
    if x2<0:
        x2=0
    if y2<0:
        y2=0

    height = y2 - y1
    width = x2 - x1
    crop_img = img[y1:y1 + height, x1:x1 + width]
    cv2.imwrite(save_path, crop_img)
    # cv2.imshow("Image", crop_img)
    # cv2.waitKey(0)
    print(f'the {origin_path} finish crop, most frequent pixel is {num}')

def multi_process_crop(input_dir):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # 对input_dir中的每个元素都执行crop操作
        executor.map(crop , input_dir)

if __name__ == "__main__":
    data_dir = '119'
    save_dir = '119'
    path_list = [(os.path.join(data_dir, o), os.path.join(save_dir, o)) for o in os.listdir(data_dir)]
    start = time.time()
    multi_process_crop(path_list)
    print(f'Total cost {time.time()-start} seconds')