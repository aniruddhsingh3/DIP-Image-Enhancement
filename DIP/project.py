import cv2
import matplotlib.pyplot as plt
import numpy as np 

image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Error: Image not found! Make sure 'image.jpg' exists in the project folder.")
    exit()  

print("Image loaded successfully!")

cv2.imshow("Original Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis("off")  
plt.show()
equalized_image = cv2.equalizeHist(image)

min_val = np.min(image)
max_val = np.max(image)
stretched_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(equalized_image, cmap='gray')
plt.title("Histogram Equalized Image")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(stretched_image, cmap='gray')
plt.title("Contrast Stretched Image")
plt.axis("off")

plt.show()

plt.figure(figsize=(12,6))

plt.subplot(1,3,1)
plt.hist(image.ravel(), 256, [0,256])
plt.title('Original Histogram')

plt.subplot(1,3,2)
plt.hist(equalized_image.ravel(), 256, [0,256])
plt.title('Equalized Histogram')

plt.subplot(1,3,3)
plt.hist(stretched_image.ravel(), 256, [0,256])
plt.title('Stretched Histogram')

plt.show()

cv2.imwrite('equalized_image.jpg', equalized_image)
cv2.imwrite('stretched_image.jpg', stretched_image)
