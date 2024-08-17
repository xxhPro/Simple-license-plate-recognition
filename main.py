import os
import sys
import hyperlpr3 as lpr3
import numpy as np
from PyQt5 import QtWidgets, uic
from PyQt5.QtGui import QPixmap
from License_Plate_recognition import Ui_Form
import cv2
from License_location import location


class MyWindow(QtWidgets.QMainWindow, Ui_Form):
    def __init__(self):
        super(MyWindow, self).__init__()
        # 准备label列表
        uic.loadUi('License_Plate_Recognition.ui', self)
        self.pushButton.clicked.connect(self.open_file)
        self.pushButton_2.clicked.connect(self.process_image)
        self.pushButton_3.clicked.connect(self.extract_plate)
        self.pushButton_4.clicked.connect(self.edge_detection)
        self.pushButton_5.clicked.connect(self.segment_characters)
        self.pushButton_6.clicked.connect(self.recognize_characters)
        self.image_path = None

    def open_file(self):
        fileName, fileType = QtWidgets.QFileDialog.getOpenFileName(self, "选取文件", os.getcwd(),
                                                                   "All Files(*);;Text Files(*.txt)")
        print(fileName)
        print(fileType)
        if fileName:
            self.display_image(fileName)
            self.image_path = fileName  # 更新图片路径属性
            print("图片路径已保存:", self.image_path)  # 打印路径以验证

    def display_image(self, file_path):
        pixmap = QPixmap(file_path)
        self.label.setPixmap(pixmap)
        self.label.setScaledContents(True)  # 可选: 根据label的尺寸调整图片大小

    def process_image(self):
        if self.image_path:
            # 调用车牌颜色检测函数
            self.detect_blue_color(self.image_path)

    def detect_blue_color(self, image_path):
        # 读取图片
        image = cv2.imread(image_path)
        # 将图片从BGR颜色空间转换到HSV颜色空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # 定义蓝色在HSV颜色空间中的范围
        lower_blue = np.array([100, 150, 50])
        upper_blue = np.array([140, 255, 255])
        # 根据颜色范围创建掩膜
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        # 对原图像和掩膜进行位运算
        result = cv2.bitwise_and(image, image, mask=mask)
        # 保存结果图片
        cv2.imwrite('blue_plate_detection.png', result)
        # 更新界面上的label_3图片显示
        self.update_label_13('blue_plate_detection.png')

    def update_label_13(self, image_path):
        pixmap = QPixmap(image_path)
        self.label_13.setPixmap(pixmap)
        self.label_13.setScaledContents(True)  # 根据label_3的尺寸调整图片大小

    def extract_plate(self):
        if self.image_path:
            # 调用车牌定位函数
            plate_image = location(cv2.imread(self.image_path))
            # 保存提取的车牌图像
            cv2.imwrite('extracted_plate.png', plate_image)
            # 更新界面上的label_3以显示提取的车牌图像
            self.update_label('extracted_plate.png', self.label_3)

            # 将提取的车牌图像转换为灰度图
            plate_image_gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
            # 应用二值化
            _, plate_image_binary = cv2.threshold(plate_image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # 保存二值化后的车牌图像
            cv2.imwrite('binary_plate.png', plate_image_binary)
            # 更新界面上的label_5以显示二值化后的车牌图像
            self.update_label('binary_plate.png', self.label_5)

    def edge_detection(self):
        if self.label_3.pixmap() is not None:
            image = self.pixmap_to_cv(self.label_3.pixmap())
            edges = cv2.Canny(image, 100, 200)
            cv2.imwrite('edge_detection.png', edges)
            self.update_edge_label('edge_detection.png')

    def pixmap_to_cv(self, pixmap):
        image = pixmap.toImage()
        ptr = image.bits()
        ptr.setsize(image.byteCount())
        arr = np.array(ptr).reshape(image.height(), image.width(), 4)  # Assuming RGBA
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2GRAY)

    def update_edge_label(self, image_path):
        pixmap = QPixmap(image_path)
        self.label_7.setPixmap(pixmap)
        self.label_7.setScaledContents(True)

    def update_label(self, image_path, label):
        pixmap = QPixmap(image_path)
        label.setPixmap(pixmap)
        label.setScaledContents(True)  # 根据label的尺寸调整图片大小

    def segment_characters(self):
        img = cv2.imread('extracted_plate.png', 1)
        cv2.imshow('origin', img)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 灰度化
        cv2.imshow('gray', gray)

        blur = cv2.bilateralFilter(gray, 13, 15, 15)  # 双边滤波
        cv2.imshow('blur', blur)

        _, image_binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imshow('binary', image_binary)

        label_list = [self.label_16, self.label_17, self.label_18, self.label_19, self.label_20, self.label_21,
                      self.label_22]

        # 分割字符
        white = []  # 记录每一列白色像素总和
        black = []  # 记录每一列黑色像素总和
        height = image_binary.shape[0]
        width = image_binary.shape[1]

        white_max = 0
        black_max = 0
        # 计算每一列的黑白色像素总和
        for i in range(width):  # 按列遍历
            s = 0  # 这一列白色总数
            t = 0  # 这一列黑色总数
            for j in range(height):  # 按行遍历
                if image_binary[j][i] == 255:  # 白色像素点
                    s += 1
                if image_binary[j][i] == 0:  # 黑色像素点
                    t += 1
            white_max = max(white_max, s)
            black_max = max(black_max, t)

            white.append(s)
            black.append(t)
            print(s)
            print(t)

        print('white_len:', len(white))  # 273
        print('black_len:', len(black))  # 273

        print('black_max = ', black_max)  # 88
        print('white_max = ', white_max)  # 80

        arg = False  # False表示白底黑字；True表示黑底白字,按黑底白字方式切割
        if black_max > white_max:
            arg = True

        # 分割图像
        def find_end(start_):
            end_ = start_ + 1
            for m in range(start_ + 1, width - 1):
                if (black[m] if arg else white[m]) > (
                        0.95 * black_max if arg else 0.95 * white_max):  # 0.95这个参数请多调整，对应下面的0.05
                    end_ = m
                    break
            return end_

        n = 1
        start = 1
        end = 2
        i = 0
        character_images = []  # 用于存储字符图像的列表
        while n < width - 2:
            n += 1
            if (white[n] if arg else black[n]) > (0.05 * white_max if arg else 0.05 * black_max):
                # 上面这些判断用来辨别是白底黑字还是黑底白字
                # 0.05这个参数请多调整，对应上面的0.95
                start = n
                end = find_end(start)
                n = end
                if end - start > 5:
                    cj = image_binary[1:height, start:end]
                    i = i + 1
                    cv2.imwrite(str(i) + '.jpg', cj)  # 自动保存分割结果

        for index, character_image in enumerate(character_images):
            if index < 7:  # 确保我们不会超出标签的数量
                cv2.imwrite(f'character_{index}.jpg', character_image)  # 保存字符图像
                self.update_label(f'character_{index}.jpg', label_list[index])  # 更新对应的标签

    def recognize_characters(self):
        if self.image_path:
            # Instantiate object
            catcher = lpr3.LicensePlateCatcher()
            # load image
            image = cv2.imread(self.image_path)
            # print result
            result = catcher(image)
            plate_number = result[0][0]
            self.update_recognize_label(plate_number)
            # 新增一个方法来更新label的文本

    def update_recognize_label(self, plate_number):
        self.textEdit.setText(plate_number)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())
