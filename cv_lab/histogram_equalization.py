import cv2 as cv 
import numpy as np
import matplotlib.pyplot as plt

curr = cv.imread(r"E:\pics\twt\turtleHUH.jpg")
gray = cv.cvtColor(curr, cv.COLOR_BGR2GRAY)

plt.figure(figsize=(10,10))
plt.subplot(221)
plt.imshow(curr[:,:,::-1])

plt.subplot(222)
histed = cv.calcHist([gray], [0], None, [256], [0,256] )

plt.plot(histed)

plt.subplot(223)
plt.hist(gray.ravel(), 256, [0,256])

eq = cv.equalizeHist(gray)

plt.subplot(224)
plt.imshow(eq, cmap='gray')

plt.show()