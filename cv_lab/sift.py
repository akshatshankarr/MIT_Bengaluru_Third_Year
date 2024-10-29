import cv2 as cv 
import matplotlib.pyplot as plt
import numpy as np

curr = cv.imread(r"E:\wallpaper\throfinn.jpg",)

train = cv.cvtColor(curr, cv.COLOR_BGR2RGB)
trn_gray = cv.cvtColor(train, cv.COLOR_RGB2GRAY)

test = cv.pyrDown(train)
test = cv.pyrDown(test)

row, col = test.shape[:2]

M = cv.getRotationMatrix2D((col/2, row/2), 30, 1)

test = cv.warpAffine(test, M, (col,row))

tst_gray = cv.cvtColor(test, cv.COLOR_RGB2GRAY)

fig, plots = plt.subplots(1, 2, figsize=(20,10))

plots[0].set_title("training image")
plots[0].imshow(train)
plots[0].axis('off')

plots[1].set_title("test image")
plots[1].imshow(test)
plots[1].axis('off')

plt.show()

sift = cv.SIFT.create()
keypoints_train, decs_test = sift.detectAndCompute(trn_gray, None)
keypts_test, decs_test = sift.detectAndCompute(tst_gray, None)

trn_img_keypts = cv.drawKeypoints(train, keypoints_train, None)
tst_img_keypts = cv.drawKeypoints(test, keypts_test, None)

cv.imshow('ahbdjhab', trn_img_keypts)
cv.imshow('ahdbahdsa', tst_img_keypts)
cv.waitKey(0)