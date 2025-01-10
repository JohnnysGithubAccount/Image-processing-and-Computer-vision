import matplotlib.pyplot as plt


plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.imshow(plt.imread('images/left.jpg'))
plt.title('left')
plt.subplot(1, 2, 2)
plt.imshow(plt.imread('images/right.jpg'))
plt.title('right')
plt.show()
