import cv2 as cv 
import numpy as np 

curr = cv.imread(r"E:\wallpaper\throfinn.jpg", 0)
row, col = curr.shape

M = np.float32([[1,0,100], [0,1,50]])
dst = cv.warpAffine(curr, M, (col,row))

M2 = np.float32([[1,0.5,0], [0,1,0], [0,0,1]])
sheared = cv.warpPerspective(curr, M2, (int(col*1.5), int(row*1.5)))

cv.imshow('translated', dst)
cv.imshow('sheared', sheared)

cv.waitKey(0)
cv.destroyAllWindows()