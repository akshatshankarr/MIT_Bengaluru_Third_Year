import cv2
import numpy as np 
import matplotlib.pyplot as plt 

curr = cv2.imread(r"E:\pics\twt\turtleHUH.jpg")
gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

sobel_x = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)

edges = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
edges = np.uint8(np.absolute(edges))
_, edges = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)

plt.figure(figsize=(10, 10))

plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(curr[:, :, ::-1])

plt.subplot(1, 2, 2)
plt.title("Edges")
plt.imshow(edges, cmap='gray')
plt.show()