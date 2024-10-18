#image equalization

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('image.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Perform histogram equalization
equalized_gray = cv2.equalizeHist(gray_image)
equalized_rgb = cv2.cvtColor(equalized_gray, cv2.COLOR_GRAY2RGB)

# Display original and equalized images and histograms
plt.figure(figsize=(12, 8))

# Original Image
plt.subplot(2, 2, 1)
plt.imshow(image_rgb)
plt.title('Original Image')
plt.axis('off')

# Equalized Image
plt.subplot(2, 2, 2)
plt.imshow(equalized_rgb)
plt.title('Equalized Image')
plt.axis('off')

# Histogram of Original Image
plt.subplot(2, 2, 3)
plt.hist(gray_image.ravel(), bins=256, color='gray')
plt.title('Histogram of Original Image')

# Histogram of Equalized Image
plt.subplot(2, 2, 4)
plt.hist(equalized_gray.ravel(), bins=256, color='gray')
plt.title('Histogram of Equalized Image')

plt.tight_layout()
plt.show()

##########################################################################

# Load the image
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Compute negative
negative_image = 255 - image

# Display original and negative images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(negative_image, cmap='gray')
plt.title('Negative Image')
plt.axis('off')

plt.tight_layout()
plt.show()

###########################################################################################

# Load image
image = cv2.imread('image.jpg')

# Get image dimensions
(h, w) = image.shape[:2]
center = (w // 2, h // 2)

# Rotate the image by 45 degrees
M = cv2.getRotationMatrix2D(center, 45, 1.0)
rotated = cv2.warpAffine(image, M, (w, h))

# Display original and rotated images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))
plt.title('Rotated Image')
plt.axis('off')

plt.tight_layout()
plt.show()

##########################################################################################

# Load image
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Compute gradients
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
sobel = cv2.magnitude(sobel_x, sobel_y)

# Display the gradient images
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(sobel_x, cmap='gray')
plt.title('Sobel X')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(sobel_y, cmap='gray')
plt.title('Sobel Y')
plt.axis('off')

plt.tight_layout()
plt.show()

########################################################################################

# Load image
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Split into bit planes
bit_planes = [((image >> i) & 1) * 255 for i in range(8)]

# Display the bit planes
plt.figure(figsize=(10, 8))
for i, bit_plane in enumerate(bit_planes):
    plt.subplot(2, 4, i + 1)
    plt.imshow(bit_plane, cmap='gray')
    plt.title(f'Bit Plane {i}')
    plt.axis('off')

plt.tight_layout()
plt.show()

###########################################################################################

# Load image
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Compute 2D FFT
f = np.fft.fft2(image)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20 * np.log(np.abs(fshift))

# Display FFT
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(np.fft.fft(image.flatten()))
plt.title('1-D FFT')
plt.xlabel('Frequency')
plt.ylabel('Magnitude')

plt.subplot(1, 2, 2)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('2-D FFT')
plt.axis('off')

plt.tight_layout()
plt.show()

#####################################################################################

# Load image
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Calculate mean and standard deviation
mean, std_dev = cv2.meanStdDev(image)

# Correlation coefficient (between the image and itself, will be 1)
correlation_coefficient = np.corrcoef(image.flatten())

print(f"Mean: {mean[0][0]}, Standard Deviation: {std_dev[0][0]}")
print(f"Correlation Coefficient: {correlation_coefficient[0][1]}")
