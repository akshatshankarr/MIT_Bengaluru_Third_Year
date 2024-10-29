import cv2 as cv 
import numpy as np

curr = cv.imread(r"E:\pics\twt\turtleHUH.jpg")
gray = cv.cvtColor(curr, cv.COLOR_BGR2GRAY)
C=5
logged = C*np.log(1+gray.astype(np.float64))

norm = cv.normalize(logged, None, 0, 255, cv.NORM_MINMAX)

norm = cv.convertScaleAbs(norm)

cv.imshow('norm',norm)
cv.waitKey(0)