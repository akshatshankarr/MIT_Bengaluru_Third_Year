import cv2 as cv 
import numpy as np

curr = cv.imread(r"E:\pics\twt\turtleHUH.jpg")
angle = 45
h, w = curr.shape[:2]

M = cv.getRotationMatrix2D((w/2,h/2), angle, 1)
rotated = cv.warpAffine(curr, M, (w,h), None)

cv.imshow('og', curr)
cv.imshow('rot', rotated)

cv.waitKey(0)