import matplotlib.pyplot as plt
import cv2 
import os
import time

img = cv2.imread("pictures\edifice.jpeg")
gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

ret, thres1 =cv2.threshold(blurred_img,127,255,cv2.THRESH_BINARY) #Threshold with 127, 255
ret, thres2 =cv2.threshold(blurred_img,127,255,cv2.THRESH_BINARY_INV) # Inverted values with threshhold
ret, thres3 =cv2.threshold(blurred_img,127,255,cv2.THRESH_TRUNC) # Between those ranges
ret, thres4 =cv2.threshold(blurred_img,127,255,cv2.THRESH_TOZERO) # 
ret, thres5 =cv2.threshold(blurred_img,127,255,cv2.THRESH_TOZERO_INV) # 

plt.figure(figsize=(4,4))

plt.subplot(2, 3, 1)
plt.imshow(gray_img, cmap='gray')
plt.title("Picture")
plt.axis('on')  # Para ocultar los ejes

plt.subplot(2, 3, 2)
plt.imshow(thres1, cmap='gray')
plt.title("Thresh Binary")
plt.axis('on')


plt.subplot(2, 3, 3)
plt.imshow(thres2, cmap='gray')
plt.title("Tresh Inverted")
plt.axis('on')


plt.subplot(2, 3, 4)
plt.imshow(thres3, cmap='gray')
plt.title("Tresh Trun")
plt.axis('on')

plt.subplot(2, 3, 5)
plt.imshow(thres4, cmap='gray')
plt.title("Tresh to Zero ")
plt.axis('on')


plt.subplot(2, 3, 6)
plt.imshow(thres5, cmap='gray')
plt.title("Tresh to Zero inverted ")
plt.axis('on')

plt.show()

