import cv2 as cv 
import numpy as np

curr = cv.imread(r"E:\pics\twt\turtleHUH.jpg")
gray = cv.cvtColor(curr, cv.COLOR_BGR2GRAY)

neg = cv.bitwise_not(gray)

cv.imshow('original',curr)
cv.imshow('negative',neg)
cv.waitKey(0)
cv.destroyAllWindows()