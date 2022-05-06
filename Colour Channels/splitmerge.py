import cv2 as cv
import numpy as np

img = cv.imread("Photos/raccoon.jpg")

blank = np.zeros(img.shape[:2], dtype="uint8")

# Spliting of Image into different colours
b, g, r = cv.split(img)

blue = cv.merge([b, blank, blank])
green = cv.merge([blank, g, blank])
red = cv.merge([blank, blank, r])

cv.imshow("Blues", blue)
cv.imshow("Reds", red)
cv.imshow("Greens", green)

#cv.imshow("Blue", b)
#cv.imshow("Red", r)
#cv.imshow("Green", g)

cv.waitKey(0)