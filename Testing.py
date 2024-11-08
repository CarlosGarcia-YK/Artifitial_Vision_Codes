import matplotlib.pyplot as plt
import cv2 
import numpy as np 
import os
import time

img = cv2.imread("pictures\puerta.jpeg")
gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
blurred_img2 = cv2.medianBlur(gray_img, 5)
blurred_img3 = cv2.bilateralFilter(gray_img, 9, 75, 75)
blurred_img4 = cv2.blur(gray_img, (5, 5))
denoised_img5 = cv2.fastNlMeansDenoising(gray_img, None, 30, 7, 21)

edges = cv2.Canny(blurred_img2, 50, 150)
edges2 = cv2.Canny(blurred_img3, 50, 150)
edges3 = cv2.Canny(blurred_img4, 50, 150)
edges4 = cv2.Canny(denoised_img5, 50, 150)




start_time = time.time()
# Original Canny with blur
edges = cv2.Canny(blurred_img, 50, 150)
end_time = time.time()
print(f"Execution time for Canny with blur: {end_time - start_time} seconds")

start_time = time.time()
# Canny without blur
edges2 = cv2.Canny(img, 50, 150)
end_time = time.time()
print(f"Execution time for Canny without blur: {end_time - start_time} seconds")


plt.subplot(2, 2, 1)
plt.imshow(edges, cmap='gray')
plt.title("Median Blur")
plt.axis('off')


plt.subplot(2, 2, 2)
plt.imshow(edges2, cmap='gray')
plt.title("Bilaterial Filter ")
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(edges3, cmap='gray')
plt.title("Blur")
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(edges4, cmap='gray')
plt.title("Mean Denoising")
plt.axis('off')

plt.show()