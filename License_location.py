# coding=gbk
"""
车牌定位模块，尝试将Sobel边缘和基于颜色HSV的两种定位方法互相筛选，实现车牌定位
"""
import cv2 as cv
import numpy as np


# 找出最有可能是车牌的位置
def getSatifyestBox(list_rate):
    for index, key in enumerate(list_rate):
        list_rate[index] = abs(key - 3)
    index = list_rate.index(min(list_rate))  # index函数作用是：若list_rate中存在index括号中的内容，则返回括号内字符串的索引值
    return index


def location(img):
    # 读取图片并统一尺寸
    img_resize = cv.resize(img, (640, 480), )
    # 高斯模糊+中值滤波
    img_gaus = cv.GaussianBlur(img_resize, (5, 5), 0)  # 高斯模糊
    img_med = cv.medianBlur(img_gaus, 5)  # 中值滤波

    # HSV模型处理，直至二值化
    # 转换为HSV模型
    img_hsv = cv.cvtColor(img_med, cv.COLOR_BGR2HSV)  # hsv模型
    lower_blue = np.array([100, 40, 50])
    higher_blue = np.array([140, 255, 255])
    mask = cv.inRange(img_hsv, lower_blue, higher_blue)  # 掩膜操作
    img_res = cv.bitwise_and(img_med, img_med, mask=mask)

    # 灰度化+二值化
    img_gray_h = cv.cvtColor(img_res, cv.COLOR_BGR2GRAY)  # 转换了灰度化
    ret1, img_thre_h = cv.threshold(img_gray_h, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # 进行Sobel算子运算，直至二值化
    img_gray_s = cv.cvtColor(img_med, cv.COLOR_BGR2GRAY)

    # sobel算子运算
    img_sobel_x = cv.Sobel(img_gray_s, cv.CV_32F, 1, 0, ksize=3)  # x轴Sobel运算
    img_sobel_y = cv.Sobel(img_gray_s, cv.CV_32F, 0, 1, ksize=3)
    img_ab_y = np.uint8(np.absolute(img_sobel_y))
    img_ab_x = np.uint8(np.absolute(img_sobel_x))  # 像素点取绝对值
    img_ab = cv.addWeighted(img_ab_x, 0.5, img_ab_y, 0.5, 0)  # 将两幅图像叠加在一起（按一定权值）
    # 考虑再加一次高斯去噪
    img_gaus_1 = cv.GaussianBlur(img_ab, (5, 5), 0)  # 高斯模糊

    # 二值化操作
    ret2, img_thre_s = cv.threshold(img_gaus_1, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)  # 正二值化

    # 颜色空间与边缘算子的图像互相筛选
    # 同时遍历两幅二值图片，若两者均为255，则置255
    img_1 = np.zeros(img_thre_h.shape, np.uint8)  # 重新拷贝图片
    height = img_resize.shape[0]  # 行数
    width = img_resize.shape[1]  # 列数
    for i in range(height):
        for j in range(width):
            h = img_thre_h[i][j]
            s = img_thre_s[i][j]
            if h == 255 and s == 255:
                img_1[i][j] = 255
            else:
                img_1[i][j] = 0
    # cv.imshow('threshold',img_1)
    # cv.waitKey(0)

    # 二值化后的图像进行闭操作
    kernel = np.ones((14, 18), np.uint8)
    img_close = cv.morphologyEx(img_1, cv.MORPH_CLOSE, kernel)  # 闭操作
    img_med_2 = cv.medianBlur(img_close, 5)
    # cv.imshow('close',img_med_2)
    # cv.waitKey(0)

    # 查找轮廓
    regions = []  # 区域
    list_rate = []
    img_input = img_med_2.copy()
    contours, hierarchy = cv.findContours(img_input, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    #   筛选面积最小的
    for contour in contours:
        # 计算该轮廓的面积
        area = cv.contourArea(contour)
        # 面积小的都筛选掉
        if area <= 2100:
            continue
        # 轮廓近似,epsilon，是从轮廓到近似轮廓的最大距离。是一个准确率参数，好的epsilon的选择可以得到正确的输出。True决定曲线是否闭合。
        epslion = 1e-3 * cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, epslion, True)  # 曲线折线化
        # 找到最小的矩形，该矩形可能有方向
        rect = cv.minAreaRect(contour)
        # box是四个点的坐标
        box = cv.boxPoints(rect)
        box = np.intp(box)
        # 计算高和宽
        height = abs(box[0][1] - box[2][1])
        width = abs(box[0][0] - box[2][0])
        # 车牌正常情况下长高比为2-5之间（精确一点可为（2.2,3.6））
        ratio = float(width) / float(height)
        if 2 < ratio < 5:
            regions.append(box)
            list_rate.append(ratio)
    # 输出车牌的轮廓
    print('[INF0]:Detect %d license plates' % len(regions))  # 输出疑似车牌图块的数量
    index = getSatifyestBox(list_rate)
    region = regions[index]
    # 用绿线画出这些找到的轮廓
    # 重新申请空间拷贝，因为drawcontours会改变原图片
    img_2 = np.zeros(img_resize.shape, np.uint8)
    img_2 = img_resize.copy()
    cv.drawContours(img_2, [region], 0, (0, 255, 0), 2)
    # cv.imshow('result',img_2)
    # cv.waitKey(0)

    # 定位后需对车牌图像做后面的字符分割等处理，因此需要将车牌图块单独截取出来，截取轮廓
    Xs = [i[0] for i in region]
    YS = [i[1] for i in region]
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(YS)
    y2 = max(YS)
    height_1 = y2 - y1
    width_1 = x2 - x1
    img_crop = img_resize[y1:y1 + height_1, x1:x1 + width_1]
    # cv.imshow('resultcut',img_crop)
    # cv.waitKey(0)

    # 后面是自己的一些想法，希望能够对截取到的车牌图块再细致处理一下，使其仅保留车牌部分，但作用貌似也不大（苦笑）
    # 假设再进行一次HSV
    img_hsv_1 = cv.cvtColor(img_crop, cv.COLOR_BGR2HSV)  # hsv模型
    lower_blue_1 = np.array([100, 90, 90])
    higher_blue_1 = np.array([140, 255, 255])
    mask_1 = cv.inRange(img_hsv_1, lower_blue_1, higher_blue_1)  # 掩膜操作
    img_res_1 = cv.bitwise_and(img_crop, img_crop, mask=mask_1)

    # 灰度化+二值化
    img_gray_1 = cv.cvtColor(img_res_1, cv.COLOR_BGR2GRAY)  # 转换了灰度化
    ret3, img_thre_1 = cv.threshold(img_gray_1, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    height_2 = img_thre_1.shape[0]  # 获得行数
    width_2 = img_thre_1.shape[1]  # 获得列数
    white_min = []
    white_max = []
    a = 0
    b = 0
    # 从每行开始遍历，记录每行第一个及最后一个白色像素点的列数
    for i in range(height_2):
        for j in range(width_2):
            h = img_thre_1[i, j]
            if h == 255:
                a = j
                white_min.append(a)
                break
    a = min(white_min)
    for q in range(height_2 - 1, -1, -1):
        for w in range(width_2 - 1, -1, -1):
            ps = img_thre_1[q, w]
            if ps == 255:
                b = w
                white_max.append(b)
                break
    b = max(white_max)
    white_min1 = []
    white_max1 = []
    c = 0
    d = 0
    # 从每一列开始遍历，记录每一行第一个白色像素点及最后一个像素点的行数
    for i in range(width_2):
        for j in range(height_2):
            h = img_thre_1[j, i]
            if h == 255:
                c = j
                white_max1.append(c)
                break
    c = min(white_max1)
    for q in range(width_2 - 1, -1, -1):
        for w in range(height_2 - 1, -1, -1):
            ps = img_thre_1[w, q]
            if ps == 255:
                d = w
                white_min1.append(d)
                break
    d = max(white_min1)
    img_cut = img_crop[c:d, a:b]
    return img_cut


if __name__ == "__main__":
    img = cv.imread("C:\\study\\homework\\test_picture\\car10.jpg")  # 输入图片
    cv.imshow("plate", img)
    cv.waitKey(0)
    img_p = location(img)
    cv.imshow("plate", img_p)
    cv.waitKey(0)


