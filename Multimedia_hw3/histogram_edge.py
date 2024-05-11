import io
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

'''Histogram Equalization'''
def Hist(img, img_grey):
    num_list = range(256)
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])

    # Apply histogram equalization
    equ = cv2.equalizeHist(img_grey)
    hist_euq = cv2.calcHist([equ], [0], None, [256], [0, 256])

    plt.figure(figsize=(20, 10))
    # Original image
    plt.subplot(2, 3, 1)
    plt.imshow(img)
    plt.title('Original Image')

    plt.subplot(2, 3, 4)
    plt.bar(num_list, hist.flatten())
    plt.title('Histogram of Original')
    plt.xlabel('Grey Scale')
    plt.ylabel('Frequency')
    # Equalize with OpenCV
    plt.subplot(2, 3, 2)
    plt.imshow(equ, cmap='grey')
    cv2.imwrite('./images/Histogram_Equalization_OpenCV.bmp', equ)
    plt.title('Equalize with OpenCV')

    plt.subplot(2, 3, 5)
    # plt.hist(equ.flatten(), 256, [0, 256], color='b')
    plt.bar(num_list, hist_euq.flatten())
    plt.title('Histogram of Equalize with OpenCV')
    plt.xlabel('Grey Scale')
    plt.ylabel('Frequency')

    # Equalize with my function
    equ_my = HistogramEqualization(img_grey)
    my_hist_euq = cv2.calcHist([equ_my], [0], None, [256], [0, 256])
    plt.subplot(2, 3, 3)
    plt.imshow(equ_my, cmap='grey')
    cv2.imwrite('./images/Histogram_Equalization_manual.bmp', equ_my)
    plt.title('Equalized Manually')
    plt.subplot(2, 3, 6)
    # plt.hist(equ_my.flatten(), 256, [0, 256], color='b')
    plt.bar(num_list, my_hist_euq.flatten())
    plt.title('Histogram of Equalized (Manual)')
    plt.xlabel('Grey Scale')
    plt.ylabel('Frequency')
    plt.show()

def HistogramEqualization(img):
    # Get the height and width of the image
    height, width = img.shape
    # Calculate the number of pixels
    num_pixels = height * width
    # Calculate the histogram of the image
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    # Calculate the cumulative histogram
    cum_hist = np.cumsum(hist)
    cum_hist = cum_hist / num_pixels
    cum_hist = cum_hist * 255
    cum_hist = np.uint8(cum_hist)
    equ = cum_hist[img]
    return equ

'''Histogram Equalization End'''

'''Edge Detection'''
def Convolution2D(p_gaussian, kernel, padding = 1):
    # 宣告一個p_gaussian大小的0
    out = np.zeros(p_gaussian.shape)
    
    p_padding = np.zeros((p_gaussian.shape[0] + padding * 2, p_gaussian.shape[1] + padding * 2))
    
    p_padding[padding:-1*padding, padding:-1*padding] = p_gaussian

    for y in range(p_gaussian.shape[1]):
        for x in range(p_gaussian.shape[0]):
            out[x, y] = (kernel * p_padding[x: x + kernel.shape[0], y: y + kernel.shape[1]]).sum()
    return out

def comb_thres(gray_image):
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

    x = Convolution2D(smoothed_image, sobel_x_operator)
    y = Convolution2D(smoothed_image, sobel_y_operator)

    # comb = np.sqrt(x * x + y * y)
    combined = np.sqrt(x.astype(np.float64)**2 + y.astype(np.float64)**2)

    cv2.imshow('Combined Sobel', abs(combined).astype('uint8'))
    cv2.imwrite('./images/Edge_detection.bmp', abs(combined).astype('uint8'))
    print('============= Edge Detection Done! =============')
    cv2.waitKey(0)
    cv2.destroyAllWindows()

'''Edge Detection End'''

if __name__ == '__main__':
    img = cv2.imread('./images/Histogram+Edge.bmp')
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    start = False
    action = input('Please enter the action you want to perform (eq {Equalization}/ed {Edge Detection}): ')

    start = True
    while start:
        if action == 'eq': # Q1 : Histogram Equalization
            Hist(img_rgb, img_grey)
        elif action == 'ed': # Q2 : Edge Detection
            comb_thres(img_grey)
        else: 
            while(action != 'eq' and action != 'ed'):
                print('Action not found!')
                print('Please enter the correct action\n')
                break

        decision = input('Do you want to perform another action? (y/n): ')
        if(decision != 'y'):
            start = False
        else:
            action = input('Please enter the action you want to perform (eq {Equalization}/ed {Edge Detection}): ')
        

    