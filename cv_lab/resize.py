import cv2 as cv 
import numpy as np

curr = cv.imread(r"E:\pics\twt\turtleHUH.jpg")

shrinked = cv.resize(curr, (200,100), interpolation=cv.INTER_AREA)
enlarged = cv.resize(shrinked, None, fx=1.5, fy=1.5, interpolation=cv.INTER_CUBIC)

cv.imshow('shrinked', shrinked)
cv.imshow('enlarged', enlarged)

cv.waitKey(0)
