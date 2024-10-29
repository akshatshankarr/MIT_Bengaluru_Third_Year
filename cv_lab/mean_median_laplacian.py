import cv2 as cv 
import numpy as np
import matplotlib.pyplot as plt

curr = cv.imread(r"E:\pics\twt\turtleHUH.jpg")
gray = cv.cvtColor(curr, cv.COLOR_BGR2GRAY)

kernel = np.ones((5,5) , np.float32)/25

mean_filtered = cv.filter2D(gray, -1, kernel=kernel)
median_filtered = cv.medianBlur(gray, 5)
laplacian = cv.Laplacian(gray, cv.CV_64F)
laplacian = cv.convertScaleAbs(gray)

cv.imshow('mean', mean_filtered)
cv.imshow('median', median_filtered)
cv.imshow('lapl', laplacian)

cv.waitKey(0)
cv.destroyAllWindows()