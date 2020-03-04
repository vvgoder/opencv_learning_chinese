import numpy as np
import cv2
from matplotlib import pyplot as plt
# %%
# 对ORB算法进行蛮力匹配
img1 = cv2.imread('data/zhouwei1.jpg', 0)
img2 = cv2.imread('data/zhouwei.jpg', 0)

orb = cv2.ORB_create()

kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# 定义一个匹配器
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# 匹配两张图片的特征符
matches = bf.match(des1, des2)

# 根据距离排序,越小越好
matches = sorted(matches, key=lambda x: x.distance)

draw_params = dict(matchColor=(0, 255, 0),
                   singlePointColor=(255, 0, 0),
                   flags=0)
# 绘制特征符
img3=cv2.drawMatches(img1, kp1, img2, kp2, matches[:20],None, **draw_params)

plt.imshow(img3, cmap='gray')
plt.title('Matched Result')
plt.axis('off')
plt.show()


# %%
# SIFT检测进行蛮力匹配
img1 = cv2.imread('data/zhouwei1.jpg', 0)  # queryImage
img2 = cv2.imread('data/zhouwei.jpg', 0)

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# BFMatcher with default params
# 匹配器，使用 Norm_L2 距离
bf = cv2.BFMatcher()
# knn匹配描述点
matches = bf.knnMatch(des1, des2, k=2)

good = []
# 挑选好的点
# Apply ratio test
# 比值测试，首先获取与A 距离最近的点B（最近）和C（次近），只有当B/C
# 小于阈值时（0.75）才被认为是匹配，因为假设匹配是一一对应的，真正的匹配的理想距离为0
for m, n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

# cv2.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good[:20], None, flags=2)
plt.imshow(img3, cmap='gray'), plt.title('Matched Result'), plt.axis('off')
plt.show()

# %%
# SIFT+FLANN匹配器
# FLANN 是快速最近邻搜索包（Fast_Library_for_Approximate_Nearest_Neighbors）的简称。
# 它是一个对大数据集和高维特征进行最近邻搜索的算法的集合，而且这些算法都已经被优化过了。
# 在面对大数据集时它的效果要好于BFMatcher
img1 = cv2.imread('data/zhouwei1.jpg', 0)
img2 = cv2.imread('data/zhouwei.jpg', 0)

sift = cv2.xfeatures2d.SIFT_create(nfeatures=50)

kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

FLANN_INDEX_KDTREE = 0  # kd树
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)   # or pass empty dictionary

# # 使用FLANN
flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1, des2, k=2)

# Need to draw only good matches, so create a mask
matchesMask = [[0, 0] for i in range(len(matches))]

# ratio test as per Lowe's paper
for i, (m, n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i] = [1, 0]

draw_params = dict(matchColor=(0, 255, 0),
                   singlePointColor=(255, 0, 0),
                   matchesMask=matchesMask,
                   flags=0)

img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)

plt.imshow(img3, cmap='gray'), plt.title('Matched Result'), plt.axis('off')
plt.show()