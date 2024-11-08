import matplotlib.pyplot as plt
import cv2 
import numpy as np 
import os
import time

img = cv2.imread("pictures\edifice.jpeg", cv2.IMREAD_GRAYSCALE)
img = cv2.medianBlur(img,5)
_, th1 = cv2.threshold(img, 128,255,cv2.THRESH_BINARY)


"""
---------------*--------------------*-------------------*

#Media 
#cv.adaptativeThreshold(image,vmax,cv.ADAPTATIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,tamaño_kernel_constant)
"""
th2= cv2.adaptiveThreshold(img,255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,20)

#Gaussian
th3= cv2.adaptiveThreshold(img,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

plt.figure(figsize=(8,8))

plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title("Picture")
plt.axis('on')  # Para ocultar los ejes

plt.subplot(2, 2, 2)
plt.imshow(th1, cmap='gray', vmin= 0 , vmax= 255)
plt.title("Thresh Binary")
plt.axis('on')

plt.subplot(2, 2, 3)
plt.imshow(th2, cmap='gray', vmin= 0 , vmax= 255)
plt.title("Adaptative mean")
plt.axis('on')

plt.subplot(2, 2, 4)
plt.imshow(th3, cmap='gray', vmin= 0 , vmax= 255)
plt.title("Adaptative mean")
plt.axis('on')
plt.show()


