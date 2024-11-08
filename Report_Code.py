import cv2
import matplotlib.pyplot as plt

# Function to display images using a 2x3 grid for up to 6 images
def display_images(images, titles):
    plt.figure(figsize=(10, 8))
    for i in range(len(images)):
        plt.subplot(2, 3, i + 1)  # Using 2 rows and 3 columns to accommodate up to 6 images
        plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()  # Optional: For better spacing between plots
    plt.show()

# Load the image and convert to grayscale (adjust this path to your actual image)
image_path = "pictures\edifice.jpeg"
original_image = cv2.imread(image_path)
grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

# Parameters to be modified
tam_kernel_1 = 11  # Block size (must be odd)
tam_kernel_2 = 25  # Larger block size
constant_1 = 2  # Constant subtracted from the mean
constant_2 = 10  # Larger constant to see the effect

# Apply adaptive thresholding with modified parameters
adaptive_thresh_mean_1 = cv2.adaptiveThreshold(grayscale_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                               cv2.THRESH_BINARY, tam_kernel_1, constant_1)
adaptive_thresh_mean_2 = cv2.adaptiveThreshold(grayscale_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                               cv2.THRESH_BINARY, tam_kernel_2, constant_2)

adaptive_thresh_gauss_1 = cv2.adaptiveThreshold(grayscale_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                cv2.THRESH_BINARY, tam_kernel_1, constant_1)
adaptive_thresh_gauss_2 = cv2.adaptiveThreshold(grayscale_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                cv2.THRESH_BINARY, tam_kernel_2, constant_2)

# Display the images in a 2x3 grid
images = [grayscale_image, adaptive_thresh_mean_1, adaptive_thresh_mean_2, 
          adaptive_thresh_gauss_1, adaptive_thresh_gauss_2]
titles = ['Grayscale Image', 'Adaptive Mean (Block=11, C=2)', 'Adaptive Mean (Block=25, C=10)', 
          'Adaptive Gaussian (Block=11, C=2)', 'Adaptive Gaussian (Block=25, C=10)']

display_images(images, titles)
