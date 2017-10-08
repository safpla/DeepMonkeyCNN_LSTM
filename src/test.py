import cv2 as cv
import matplotlib.pyplot as plt
img_path = '/mnt/hgfs/VM_share/8c.jpg'

img = cv.imread(img_path)
img = cv.resize(img, (128,256))
cv.imshow('img', img)
cv.waitKey(0)
cv.destroyAllWindows()

