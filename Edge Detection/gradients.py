import cv2 as cv
import numpy as np

img = cv.imread('Photos/kid.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Laplacian
'''
Laplacian Operator is also a derivative operator which is used to find edges in an image.
It is a second order derivative mask.
In this mask we have two further classifications one is Positive Laplacian Operator and other is Negative Laplacian Operator.

Unlike other operators Laplacian did not take out edges in any particular direction but it takes out edges in following classification.
-> Inward Edges
-> Outward Edges

You can perform Laplacian Transform operation on an image using the Laplacian() method.
'''
lap = cv.Laplacian(gray, cv.CV_64F)
lap = np.uint8(np.absolute(lap))
cv.imshow('Laplacian', lap)

# Sobel 
'''
The Sobel Operator is a discrete differentiation operator.
It computes an approximation of the gradient of an image intensity function.
The Sobel Operator combines Gaussian smoothing and differentiation.
'''
sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0)
sobely = cv.Sobel(gray, cv.CV_64F, 0, 1)
combined_sobel = cv.bitwise_or(sobelx, sobely)

cv.imshow('Sobel X', sobelx)
cv.imshow('Sobel Y', sobely)
cv.imshow('Combined Sobel', combined_sobel)

canny = cv.Canny(gray, 150, 175)
cv.imshow('Canny', canny)

cv.waitKey(0)