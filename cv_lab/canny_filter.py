import cv2 as cv 
import matplotlib.pyplot as plt

curr = cv.imread(r"E:\wallpaper\throfinn.jpg")
gray = cv.cvtColor(curr, cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(gray, (5,5),0)
edges = cv.Canny(blur, 100, 200)

plt.figure(figsize=(10,10))
plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(curr[:,:,::-1])

plt.subplot(1,2,2)
plt.title("Edges")
plt.imshow(edges, cmap='gray')
plt.show()
