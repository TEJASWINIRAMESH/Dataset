import cv2, numpy as np
from google.colab import files
from IPython.display import Image, display

# Upload video
video_path = next(iter(files.upload()))

# Load video and first frame
cap = cv2.VideoCapture(video_path)
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# Initialize tracking points
p0 = cv2.goodFeaturesToTrack(old_gray, 100, 0.3, 7)
mask = np.zeros_like(old_frame)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Compute Optical Flow
    p1, st, _ = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, winSize=(15, 15), maxLevel=2,
                                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    good_new, good_old = p1[st == 1], p0[st == 1]

    # Draw tracks
    for (new, old) in zip(good_new, good_old):
        mask = cv2.line(mask, tuple(new.ravel().astype(int)), tuple(old.ravel().astype(int)), (0, 255, 0), 2)
        frame = cv2.circle(frame, tuple(new.ravel().astype(int)), 5, (0, 0, 255), -1)

    img = cv2.add(frame, mask)
    cv2.imwrite('frame.jpg', img)
    display(Image('frame.jpg'))

    old_gray, p0 = frame_gray.copy(), good_new.reshape(-1, 1, 2)

cap.release()

#######################################################################################################################

from ultralytics import YOLO

# Step 1: Load YOLOv8 pretrained model (replace with your desired model, e.g., 'yolov8n')
model = YOLO('yolov8n.pt')  # Pretrained model

# Step 2: Specify the path to your dataset configuration file (in YOLO format)
config_path = r"C:\Users\dell\Desktop\yolo\config.yaml"  # Update this with the path to your dataset.yaml

# Step 3: Train the model with custom dataset and configurations
model.train(
    data=config_path,  # Path to your dataset configuration file
    epochs=300,                   # Number of epochs to train
    patience=300,                 # Patience for early stopping
    batch=16,                     # Batch size
    lr0=0.001,                    # Initial learning rate
    augment=True                   # Enable data augmentations
)

# Step 4: Validate the model to get performance metrics after training
print("\n--- Running Validation ---")
results = model.val()  # This will run the validation on your dataset

# Step 5: Print validation results (mean Average Precision, precision, recall, etc.)
print("\n--- Validation Results ---")
print(results)  # Results contain metrics like mAP, precision, recall

# Step 6: Test the model on a specific test image after training
test_image_path = r"C:\Users\dell\Desktop\yolo\images\train\download.jpg"

# Run inference/prediction on a test image using the trained model
results_pretrained = model.predict(test_image_path, save=True)

#########################################################################################################

import cv2
import numpy as np
from matplotlib import pyplot as plt
from google.colab import files

# Step 1: Upload the image file
uploaded = files.upload()

# Step 2: Load the uploaded image
image_path = next(iter(uploaded))
img = cv2.imread(image_path)

# Step 3: Apply different variations of a single filter (e.g., Gaussian Blur)
variations = [cv2.GaussianBlur(img, (k, k), 0) for k in range(3, 13, 2)]  # k=3, 5, 7, 9, 11

# Step 4: Plot each variation
plt.figure(figsize=(10, 6))

for i, var in enumerate(variations):
    plt.subplot(1, len(variations), i+1)
    plt.imshow(cv2.cvtColor(var, cv2.COLOR_BGR2RGB))
    plt.title(f'k={3 + 2*i}')
    plt.xticks([]), plt.yticks([])

plt.show()

################################################################################################################

# Import necessary libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Function to plot the results
def plot_images(images, titles):
    plt.figure(figsize=(15, 8))
    for i, (image, title) in enumerate(zip(images, titles)):
        plt.subplot(2, 5, i + 1)  # Adjusted to handle 7 images
        plt.imshow(image, cmap='gray')
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Load an image (you can replace 'image_path' with the actual path of your image)
image_path = '/EdgeDetectors_Original.png'  # Replace with your image path
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if not os.path.exists(image_path):
    print(f"Image path does not exist: {image_path}")
else:
    print(f"Image path exists: {image_path}")
# Sobel Edge Detection
sobel1 = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # Horizontal edges with 3x3 kernel
sobel2 = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)  # Horizontal edges with 5x5 kernel
sobel_combined = cv2.magnitude(sobel1, cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5))

# Prewitt Edge Detection with different kernels
prewitt_horizontal = cv2.filter2D(image, cv2.CV_32F, np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]))  # Horizontal
prewitt_vertical = cv2.filter2D(image, cv2.CV_32F, np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]))  # Vertical

# Canny Edge Detection with varying thresholds
canny1 = cv2.Canny(image, 50, 150)  # Lower thresholds
canny2 = cv2.Canny(image, 100, 200)  # Standard thresholds

# Laplacian Edge Detection with different kernel sizes
laplacian1 = cv2.Laplacian(image, cv2.CV_64F, ksize=3)  # 3x3 kernel
laplacian2 = cv2.Laplacian(image, cv2.CV_64F, ksize=5)  # 5x5 kernel

# Roberts Edge Detection
robertsx = cv2.filter2D(image, cv2.CV_32F, np.array([[1, 0], [0, -1]]))
robertsy = cv2.filter2D(image, cv2.CV_32F, np.array([[0, 1], [-1, 0]]))
roberts_combined = cv2.magnitude(robertsx, robertsy)

# Scharr Edge Detection
scharrx = cv2.Scharr(image, cv2.CV_64F, 1, 0)
scharry = cv2.Scharr(image, cv2.CV_64F, 0, 1)
scharr_combined = cv2.magnitude(scharrx, scharry)

# Prepare titles and images for plotting
titles = [
    'Sobel ksize=3', 'Sobel ksize=5', 'Prewitt Horizontal', 'Prewitt Vertical',
    'Canny (50, 150)', 'Canny (100, 200)', 'Laplacian ksize=3', 'Laplacian ksize=5',
    'Roberts', 'Scharr'
]
images = [
    sobel1, sobel2, prewitt_horizontal, prewitt_vertical,
    canny1, canny2, laplacian1, laplacian2, roberts_combined, scharr_combined
]

# Plot the images with titles
plot_images(images, titles)

##########################################################################################################

# Import necessary libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Function to plot the results
def plot_images(images, titles):
    plt.figure(figsize=(15, 8))
    for i, (image, title) in enumerate(zip(images, titles)):
        plt.subplot(2, 5, i + 1)  # Adjusted to handle 7 images
        plt.imshow(image, cmap='gray')
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Load an image (you can replace 'image_path' with the actual path of your image)
image_path = '/EdgeDetectors_Original.png'  # Replace with your image path
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if not os.path.exists(image_path):
    print(f"Image path does not exist: {image_path}")
else:
    print(f"Image path exists: {image_path}")
# Sobel Edge Detection
sobel1 = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # Horizontal edges with 3x3 kernel
sobel2 = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)  # Horizontal edges with 5x5 kernel
#sobel_combined = cv2.magnitude(sobel1, cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5))

# Prewitt Edge Detection with different kernels
prewitt_horizontal = cv2.filter2D(image, cv2.CV_32F, np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]))  # Horizontal
prewitt_vertical = cv2.filter2D(image, cv2.CV_32F, np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]))  # Vertical

# Canny Edge Detection with varying thresholds
canny1 = cv2.Canny(image, 50, 150)  # Lower thresholds
canny2 = cv2.Canny(image, 100, 200)  # Standard thresholds

# Laplacian Edge Detection with different kernel sizes
laplacian1 = cv2.Laplacian(image, cv2.CV_64F, ksize=3)  # 3x3 kernel
laplacian2 = cv2.Laplacian(image, cv2.CV_64F, ksize=5)  # 5x5 kernel

# Roberts Edge Detection
robertsx = cv2.filter2D(image, cv2.CV_32F, np.array([[1, 0], [0, -1]]))
robertsy = cv2.filter2D(image, cv2.CV_32F, np.array([[0, 1], [-1, 0]]))
roberts_combined = cv2.magnitude(robertsx, robertsy)

# Scharr Edge Detection
scharrx = cv2.Scharr(image, cv2.CV_64F, 1, 0)
scharry = cv2.Scharr(image, cv2.CV_64F, 0, 1)
scharr_combined = cv2.magnitude(scharrx, scharry)

# Prepare titles and images for plotting
titles = [
    'Sobel ksize=3', 'Sobel ksize=5', 'Prewitt Horizontal', 'Prewitt Vertical',
    'Canny (50, 150)', 'Canny (100, 200)', 'Laplacian ksize=3', 'Laplacian ksize=5',
    'Roberts', 'Scharr'
]
images = [
    sobel1, sobel2, prewitt_horizontal, prewitt_vertical,
    canny1, canny2, laplacian1, laplacian2, roberts_combined, scharr_combined
]

# Plot the images with titles
plot_images(images, titles)