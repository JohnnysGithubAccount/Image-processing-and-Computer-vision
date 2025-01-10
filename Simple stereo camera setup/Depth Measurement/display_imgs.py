import cv2
import matplotlib.pyplot as plt


img = cv2.imread('images/left.jpg')
img = cv2.resize(img, (512, 512))

plt.imshow(img)
plt.show()

# x1, y1 = 286, 120
# x2, y2 = 400, 323

