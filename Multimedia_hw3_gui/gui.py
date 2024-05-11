import os
import sys
import cv2
import numpy as np
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import QMainWindow, QLabel, QGridLayout, QWidget
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtCore import QSize
from hw3_gui import Ui_Dialog

import matplotlib.pyplot as plt

class MainWindow(QMainWindow, Ui_Dialog):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("HW3 Image Processing")
        # Edge Detection Button
        self.btn_edge.clicked.connect(self.on_btn_edge_clicked)
        self.btn_histEqu.clicked.connect(self.on_btn_histEqu_clicked)

    def Convolution2D(self, p_gaussian, kernel, padding = 1):

        # 宣告一個p_gaussian大小的0
        out = np.zeros(p_gaussian.shape)
        p_padding = np.zeros((p_gaussian.shape[0] + padding * 2, p_gaussian.shape[1] + padding * 2))
        p_padding[padding:-1*padding, padding:-1*padding] = p_gaussian

        for y in range(p_gaussian.shape[1]):
            for x in range(p_gaussian.shape[0]):
                out[x, y] = (kernel * p_padding[x: x + kernel.shape[0], y: y + kernel.shape[1]]).sum()
        return out

    # ==================== Q3_3 ======================
    def comb_thres(self, gray_image):
        # Sobel Edge Detection 針對 X軸的垂直邊緣偵測
        sobel_x_operator = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ])

        # Sobel Edge Detection 針對 y軸的水平邊緣偵測
        sobel_y_operator = np.array([
            [1, 2, 1],
            [0, 0, 0],
            [-1, -2, -1]
        ])

        # Apply Gaussian smoothing to the grayscale image (you can adjust the kernel size and standard deviation as needed)
        smoothed_image = cv2.GaussianBlur(gray_image, (3, 3), 0)

        x = self.Convolution2D(smoothed_image, sobel_x_operator)
        y = self.Convolution2D(smoothed_image, sobel_y_operator)

        # comb = np.sqrt(x * x + y * y)
        combined = np.sqrt(x.astype(np.float64)**2 + y.astype(np.float64)**2)

        # Given threshold
        threshold = 128

        # Apply thresholding
        thresholded_result = np.where(combined < threshold, 0, 255).astype(np.uint8)

        cv2.imshow('Combined Sobel', abs(combined).astype('uint8'))
        # cv2.imshow('Thresholded Result', thresholded_result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def Hist(self, img_rgb, img_grey):
        num_list = range(256)
        hist = cv2.calcHist([img_rgb], [0], None, [256], [0, 256])

        # Apply histogram equalization
        equ = cv2.equalizeHist(img_grey)
        hist_euq = cv2.calcHist([equ], [0], None, [256], [0, 256])

        plt.figure(figsize=(20, 10))
        # Original image
        plt.subplot(2, 3, 1)
        plt.imshow(img_rgb)
        plt.title('Original Image')
        plt.subplot(2, 3, 4)
        plt.bar(num_list, hist.flatten())
        plt.title('Histogram of Original')
        plt.xlabel('Grey Scale')
        plt.ylabel('Frequency')
        # Equalize with OpenCV
        plt.subplot(2, 3, 2)
        plt.imshow(equ, cmap='gray')
        cv2.imwrite('Histogram_Equalization_OpenCV.bmp', equ)
        plt.title('Equalize with OpenCV')
        plt.subplot(2, 3, 5)
        # plt.hist(equ.flatten(), 256, [0, 256], color='b')
        plt.bar(num_list, hist_euq.flatten())
        plt.title('Histogram of Equalize with OpenCV')
        plt.xlabel('Grey Scale')
        plt.ylabel('Frequency')
        # Equalize with my function
        equ_my = self.HistogramEqualization(img_grey)
        my_hist_euq = cv2.calcHist([equ_my], [0], None, [256], [0, 256])
        plt.subplot(2, 3, 3)
        plt.imshow(equ_my, cmap='gray')
        cv2.imwrite('Histogram_Equalization_manual.bmp', equ_my)
        plt.title('Equalized Manually')
        plt.subplot(2, 3, 6)
        # plt.hist(equ_my.flatten(), 256, [0, 256], color='b')
        plt.bar(num_list, my_hist_euq.flatten())
        plt.title('Histogram of Equalized (Manual)')
        plt.xlabel('Grey Scale')
        plt.ylabel('Frequency')
        plt.show()

    def HistogramEqualization(self, img):
        # Get the height and width of the image
        height, width = img.shape
        # Calculate the number of pixels
        num_pixels = height * width
        # Calculate the histogram of the image
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        # Calculate the cumulative histogram
        cum_hist = np.cumsum(hist)
        # Calculate the cumulative histogram
        cum_hist = cum_hist / num_pixels
        # Calculate the cumulative histogram
        cum_hist = cum_hist * 255
        # Calculate the cumulative histogram
        cum_hist = np.uint8(cum_hist)
        # Calculate the cumulative histogram
        equ = cum_hist[img]
        return equ

    
    def on_btn_edge_clicked(self):      
        img = cv2.imread('Histogram+Edge.bmp')
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.comb_thres(img_gray)

    def on_btn_histEqu_clicked(self):
        img = cv2.imread('Histogram+Edge.bmp')
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.Hist(img_rgb, img_gray)
        

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec_())