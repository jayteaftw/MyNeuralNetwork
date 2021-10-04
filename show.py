import matplotlib.pyplot as plt 
import sys
import cv2

image_data = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
image_data = cv2.resize(image_data, (28, 28))
plt.imshow(image_data, cmap='gray')
plt.show()