import cv2 as cv 
import numpy as np 

curr = cv.imread(r"E:\wallpaper\throfinn.jpg", 0)
row, column = curr.shape

sliced = np.zeros((row, column), dtype='uint8')
min = 80
max = 140

for i in range(row):
    for j in range(column):
        if curr[i,j]>min and curr[i,j]<max:
            sliced[i,j]= 255
        else:
            if i>0 and j>0:
                sliced[i,j]=curr[i-1,j-1]
            else:
                sliced[i,j]=curr[i,j]

cv.imshow('pog', sliced)
cv.waitKey(0)
cv.destroyAllWindows()