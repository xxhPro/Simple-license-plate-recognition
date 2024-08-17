# coding=gbk
"""
���ƶ�λģ�飬���Խ�Sobel��Ե�ͻ�����ɫHSV�����ֶ�λ��������ɸѡ��ʵ�ֳ��ƶ�λ
"""
import cv2 as cv
import numpy as np


# �ҳ����п����ǳ��Ƶ�λ��
def getSatifyestBox(list_rate):
    for index, key in enumerate(list_rate):
        list_rate[index] = abs(key - 3)
    index = list_rate.index(min(list_rate))  # index���������ǣ���list_rate�д���index�����е����ݣ��򷵻��������ַ���������ֵ
    return index


def location(img):
    # ��ȡͼƬ��ͳһ�ߴ�
    img_resize = cv.resize(img, (640, 480), )
    # ��˹ģ��+��ֵ�˲�
    img_gaus = cv.GaussianBlur(img_resize, (5, 5), 0)  # ��˹ģ��
    img_med = cv.medianBlur(img_gaus, 5)  # ��ֵ�˲�

    # HSVģ�ʹ���ֱ����ֵ��
    # ת��ΪHSVģ��
    img_hsv = cv.cvtColor(img_med, cv.COLOR_BGR2HSV)  # hsvģ��
    lower_blue = np.array([100, 40, 50])
    higher_blue = np.array([140, 255, 255])
    mask = cv.inRange(img_hsv, lower_blue, higher_blue)  # ��Ĥ����
    img_res = cv.bitwise_and(img_med, img_med, mask=mask)

    # �ҶȻ�+��ֵ��
    img_gray_h = cv.cvtColor(img_res, cv.COLOR_BGR2GRAY)  # ת���˻ҶȻ�
    ret1, img_thre_h = cv.threshold(img_gray_h, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # ����Sobel�������㣬ֱ����ֵ��
    img_gray_s = cv.cvtColor(img_med, cv.COLOR_BGR2GRAY)

    # sobel��������
    img_sobel_x = cv.Sobel(img_gray_s, cv.CV_32F, 1, 0, ksize=3)  # x��Sobel����
    img_sobel_y = cv.Sobel(img_gray_s, cv.CV_32F, 0, 1, ksize=3)
    img_ab_y = np.uint8(np.absolute(img_sobel_y))
    img_ab_x = np.uint8(np.absolute(img_sobel_x))  # ���ص�ȡ����ֵ
    img_ab = cv.addWeighted(img_ab_x, 0.5, img_ab_y, 0.5, 0)  # ������ͼ�������һ�𣨰�һ��Ȩֵ��
    # �����ټ�һ�θ�˹ȥ��
    img_gaus_1 = cv.GaussianBlur(img_ab, (5, 5), 0)  # ��˹ģ��

    # ��ֵ������
    ret2, img_thre_s = cv.threshold(img_gaus_1, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)  # ����ֵ��

    # ��ɫ�ռ����Ե���ӵ�ͼ����ɸѡ
    # ͬʱ����������ֵͼƬ�������߾�Ϊ255������255
    img_1 = np.zeros(img_thre_h.shape, np.uint8)  # ���¿���ͼƬ
    height = img_resize.shape[0]  # ����
    width = img_resize.shape[1]  # ����
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

    # ��ֵ�����ͼ����бղ���
    kernel = np.ones((14, 18), np.uint8)
    img_close = cv.morphologyEx(img_1, cv.MORPH_CLOSE, kernel)  # �ղ���
    img_med_2 = cv.medianBlur(img_close, 5)
    # cv.imshow('close',img_med_2)
    # cv.waitKey(0)

    # ��������
    regions = []  # ����
    list_rate = []
    img_input = img_med_2.copy()
    contours, hierarchy = cv.findContours(img_input, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    #   ɸѡ�����С��
    for contour in contours:
        # ��������������
        area = cv.contourArea(contour)
        # ���С�Ķ�ɸѡ��
        if area <= 2100:
            continue
        # ��������,epsilon���Ǵ����������������������롣��һ��׼ȷ�ʲ������õ�epsilon��ѡ����Եõ���ȷ�������True���������Ƿ�պϡ�
        epslion = 1e-3 * cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, epslion, True)  # �������߻�
        # �ҵ���С�ľ��Σ��þ��ο����з���
        rect = cv.minAreaRect(contour)
        # box���ĸ��������
        box = cv.boxPoints(rect)
        box = np.intp(box)
        # ����ߺͿ�
        height = abs(box[0][1] - box[2][1])
        width = abs(box[0][0] - box[2][0])
        # ������������³��߱�Ϊ2-5֮�䣨��ȷһ���Ϊ��2.2,3.6����
        ratio = float(width) / float(height)
        if 2 < ratio < 5:
            regions.append(box)
            list_rate.append(ratio)
    # ������Ƶ�����
    print('[INF0]:Detect %d license plates' % len(regions))  # ������Ƴ���ͼ�������
    index = getSatifyestBox(list_rate)
    region = regions[index]
    # �����߻�����Щ�ҵ�������
    # ��������ռ俽������Ϊdrawcontours��ı�ԭͼƬ
    img_2 = np.zeros(img_resize.shape, np.uint8)
    img_2 = img_resize.copy()
    cv.drawContours(img_2, [region], 0, (0, 255, 0), 2)
    # cv.imshow('result',img_2)
    # cv.waitKey(0)

    # ��λ����Գ���ͼ����������ַ��ָ�ȴ��������Ҫ������ͼ�鵥����ȡ��������ȡ����
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

    # �������Լ���һЩ�뷨��ϣ���ܹ��Խ�ȡ���ĳ���ͼ����ϸ�´���һ�£�ʹ����������Ʋ��֣�������ò��Ҳ���󣨿�Ц��
    # �����ٽ���һ��HSV
    img_hsv_1 = cv.cvtColor(img_crop, cv.COLOR_BGR2HSV)  # hsvģ��
    lower_blue_1 = np.array([100, 90, 90])
    higher_blue_1 = np.array([140, 255, 255])
    mask_1 = cv.inRange(img_hsv_1, lower_blue_1, higher_blue_1)  # ��Ĥ����
    img_res_1 = cv.bitwise_and(img_crop, img_crop, mask=mask_1)

    # �ҶȻ�+��ֵ��
    img_gray_1 = cv.cvtColor(img_res_1, cv.COLOR_BGR2GRAY)  # ת���˻ҶȻ�
    ret3, img_thre_1 = cv.threshold(img_gray_1, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    height_2 = img_thre_1.shape[0]  # �������
    width_2 = img_thre_1.shape[1]  # �������
    white_min = []
    white_max = []
    a = 0
    b = 0
    # ��ÿ�п�ʼ��������¼ÿ�е�һ�������һ����ɫ���ص������
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
    # ��ÿһ�п�ʼ��������¼ÿһ�е�һ����ɫ���ص㼰���һ�����ص������
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
    img = cv.imread("C:\\study\\homework\\test_picture\\car10.jpg")  # ����ͼƬ
    cv.imshow("plate", img)
    cv.waitKey(0)
    img_p = location(img)
    cv.imshow("plate", img_p)
    cv.waitKey(0)


